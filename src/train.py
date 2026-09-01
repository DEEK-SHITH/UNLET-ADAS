"""
UNLET-ADAS: Training Script
=============================
Usage:
    python src/train.py

Or in Colab:
    !python src/train.py --data_root /content/lol_dataset \
                         --save_dir  /content/drive/MyDrive/UNLET_Project/checkpoints \
                         --epochs 100

Optionally, mix in extra *unpaired* low-light images alongside LOL's
485 paired ones (see src/prepare_extra_lowlight.py for building this
directory from real driving footage and/or ExDark) -- LOL alone is
mostly indoor/urban, so this broadens the model past that:
    python src/train.py --extra_low_dirs /content/extra_lowlight/night_drive_frames \
                                          /content/extra_lowlight/exdark
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from glob import glob
from PIL import Image

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import build_model
from src.losses import UNLETLoss

try:
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    from skimage.metrics import structural_similarity  as calc_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print('scikit-image not found — PSNR/SSIM will not be computed')


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def _glob_images(directory):
    return sorted(
        glob(os.path.join(directory, '*.png')) +
        glob(os.path.join(directory, '*.jpg')) +
        glob(os.path.join(directory, '*.jpeg')))


class LOLDataset(Dataset):
    """
    LOL (Low-Light) dataset loader with augmentation. Supports mixing
    in additional *unpaired* low-light-only images via extra_low_dirs
    (e.g. hilly/rural driving frames, ExDark photos) alongside LOL's
    485 paired low/high images -- LOL alone is mostly indoor/urban,
    which is exactly the generalization gap the project's own paper
    names as future work.

    Unpaired images get no matching 'high' ground truth, so
    __getitem__ returns a zero tensor for them; UNLETLoss already
    treats an all-zero target as "no ground truth" and falls back to
    its unsupervised losses only for that sample (color constancy /
    exposure / spatial consistency / smoothness) -- no change needed
    there for this to work correctly.
    """

    def __init__(self, low_dir, high_dir=None, size=256,
                 extra_low_dirs=None):
        self.size = size
        self.lows = _glob_images(low_dir)
        self.hmap = {}
        if high_dir and os.path.exists(high_dir):
            highs = _glob_images(high_dir)
            self.hmap = {
                os.path.basename(p): p for p in highs}
        print(f'  {len(self.lows)} paired images '
              f'from {os.path.basename(low_dir)}')

        extra_count = 0
        for extra_dir in (extra_low_dirs or []):
            if not extra_dir or not os.path.isdir(extra_dir):
                print(f'  WARNING: extra_low_dir not found, '
                      f'skipping: {extra_dir}')
                continue
            extra_imgs = _glob_images(extra_dir)
            # Any filename that happens to collide with an LOL 'high'
            # name would wrongly pair unrelated images -- exclude it
            # from this unpaired set rather than risk a bad pairing.
            extra_imgs = [p for p in extra_imgs
                          if os.path.basename(p) not in self.hmap]
            self.lows.extend(extra_imgs)
            extra_count += len(extra_imgs)
            print(f'  {len(extra_imgs)} unpaired images '
                  f'from {os.path.basename(extra_dir.rstrip("/"))} '
                  '(unsupervised losses only)')
        if extra_count:
            print(f'  Total: {len(self.lows)} images '
                  f'({len(self.lows) - extra_count} paired + '
                  f'{extra_count} unpaired)')

    def _to_tensor(self, path):
        img = Image.open(path).convert('RGB').resize(
            (self.size, self.size), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __len__(self):
        return len(self.lows)

    def __getitem__(self, idx):
        lp   = self.lows[idx]
        low  = self._to_tensor(lp)
        name = os.path.basename(lp)
        high = (self._to_tensor(self.hmap[name])
                if name in self.hmap
                else torch.zeros_like(low))

        # Augmentation
        if torch.rand(1) > 0.5:
            low  = torch.flip(low,  [-1])
            high = torch.flip(high, [-1])
        if torch.rand(1) > 0.5:
            low  = torch.flip(low,  [-2])
            high = torch.flip(high, [-2])
        k    = int(torch.randint(0, 4, (1,)))
        low  = torch.rot90(low,  k, [-2, -1])
        high = torch.rot90(high, k, [-2, -1])

        return low, high


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(enh_t, high_t):
    if not HAS_SKIMAGE:
        return 0.0, 0.0
    psnrs, ssims = [], []
    en = enh_t.detach().cpu().permute(0,2,3,1).numpy()
    hn = high_t.detach().cpu().permute(0,2,3,1).numpy()
    for e, h in zip(en, hn):
        e = e.clip(0, 1); h = h.clip(0, 1)
        psnrs.append(calc_psnr(h, e, data_range=1.0))
        ssims.append(calc_ssim(
            h, e, channel_axis=2, data_range=1.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets
    print('\nLoading datasets...')
    train_ds = LOLDataset(
        os.path.join(args.data_root, 'our485', 'low'),
        os.path.join(args.data_root, 'our485', 'high'),
        extra_low_dirs=args.extra_low_dirs)
    # Validation stays on LOL's own eval15 only -- mixing in unpaired
    # images here would make PSNR/SSIM incomparable across runs (they
    # can only be computed where a real ground truth exists).
    val_ds   = LOLDataset(
        os.path.join(args.data_root, 'eval15', 'low'),
        os.path.join(args.data_root, 'eval15', 'high'))

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(
        val_ds, batch_size=4,
        shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model     = build_model().to(DEVICE)
    criterion = UNLETLoss(device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val  = float('inf')
    patience  = 0
    history   = {'train': [], 'val': [], 'psnr': [], 'ssim': []}
    weights   = os.path.join(args.save_dir, 'zerodce_cbam_best.pt')

    print(f'\nTraining: {args.epochs} epochs | '
          f'batch={args.batch_size} | lr={args.lr}')
    print(f'Patience : {args.patience}')
    print('-' * 65)

    t0 = time.time()

    for epoch in range(args.epochs):
        # Train
        model.train()
        tl = []
        for low, high in train_dl:
            low, high = low.to(DEVICE), high.to(DEVICE)
            optimizer.zero_grad()
            enh, curves = model(low)
            norm = high if high.sum() > 0 else None
            loss = criterion(enh, curves, low, norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            tl.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        vl, vp, vs = [], [], []
        with torch.no_grad():
            for low, high in val_dl:
                low, high = low.to(DEVICE), high.to(DEVICE)
                enh, curves = model(low)
                norm = high if high.sum() > 0 else None
                vl.append(criterion(
                    enh, curves, low, norm).item())
                if high.sum() > 0:
                    p, s = compute_metrics(enh, high)
                    vp.append(p); vs.append(s)

        tl_m = float(np.mean(tl))
        vl_m = float(np.mean(vl))
        vp_m = float(np.mean(vp)) if vp else 0.0
        vs_m = float(np.mean(vs)) if vs else 0.0
        lr   = scheduler.get_last_lr()[0]
        elapsed = (time.time() - t0) / 60

        history['train'].append(tl_m)
        history['val'].append(vl_m)
        history['psnr'].append(vp_m)
        history['ssim'].append(vs_m)

        if vl_m < best_val:
            best_val = vl_m
            patience = 0
            torch.save(model.state_dict(), weights)
            marker = (f'  SAVED  '
                      f'PSNR={vp_m:.2f}dB '
                      f'SSIM={vs_m:.4f}')
        else:
            patience += 1
            marker = f'  patience {patience}/{args.patience}'

        print(f'Ep {epoch+1:3d}/{args.epochs} | '
              f'loss={tl_m:.3f} | val={vl_m:.3f} | '
              f'lr={lr:.1e} | {elapsed:.1f}m{marker}')

        if patience >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Save history
    hist_path = os.path.join(args.save_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    total_t = (time.time() - t0) / 60
    print(f'\nTraining complete!')
    print(f'Time       : {total_t:.1f} min')
    print(f'Best val   : {best_val:.4f}')
    if history['psnr']:
        print(f'Best PSNR  : {max(history["psnr"]):.2f} dB')
        print(f'Best SSIM  : {max(history["ssim"]):.4f}')
    print(f'Weights    : {weights}')
    return history


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Train UNLET-ADAS enhancement model')
    p.add_argument('--data_root',
                   default='./data/lol_dataset')
    p.add_argument('--extra_low_dirs', nargs='+', default=None,
                   help='Optional extra directories of unpaired '
                        'low-light-only images (no ground truth) to '
                        'train on alongside LOL\'s 485 paired images '
                        '-- e.g. real driving footage frames or '
                        'ExDark photos, to generalize beyond LOL\'s '
                        'mostly indoor/urban scenes. See '
                        'src/prepare_extra_lowlight.py.')
    p.add_argument('--save_dir',
                   default='./checkpoints')
    p.add_argument('--epochs',
                   type=int, default=100)
    p.add_argument('--batch_size',
                   type=int, default=8)
    p.add_argument('--lr',
                   type=float, default=2e-4)
    p.add_argument('--patience',
                   type=int, default=20)
    p.add_argument('--image_size',
                   type=int, default=256)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)