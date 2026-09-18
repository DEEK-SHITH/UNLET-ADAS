"""
UNLET-ADAS: Standalone Evaluation on LOL eval15
================================================
The repo previously had no standalone evaluation entry point -- PSNR/SSIM
could only be reproduced by re-running a Colab notebook cell. This script
exists so every reported enhancement number is reproducible with one command
and leaves a JSON artifact behind.

It evaluates three arms against the LOL eval15 ground truth:
    dark input | AutoContrast | UNLET-ADAS

and does so at BOTH resolutions, because they are not the same experiment:

  --mode proxy    both images resized to 256x256, model(x) forward.
                  This is the training resolution and matches the number
                  historically reported for this project.
  --mode fullres  native resolution, model.enhance_full_res(x).
                  This is what the deployed pipeline actually does, and is
                  the number that should be reported alongside any claim
                  about full-resolution curve application.

IMPORTANT: LOL eval15 is also the split used for early stopping in
src/train.py. Scores from it are validation scores, not held-out test
scores. Use --split to point at a genuinely held-out set once one exists.

Usage:
    python src/evaluate.py --data_root ./data/LOL
    python src/evaluate.py --data_root ./data/LOL --mode fullres
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import ZeroDCECBAM  # noqa: E402


def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    # Older checkpoints named the output layer 'out'.
    if any(k.startswith('out.') for k in state):
        state = {('curve_out.' + k[4:] if k.startswith('out.') else k): v
                 for k, v in state.items()}
    # A no-CBAM ablation checkpoint has no cb*.* keys; match it.
    use_cbam = any(k.startswith('cb1.') for k in state)
    if not use_cbam:
        print('  [ablation checkpoint: CBAM disabled]')
    model = ZeroDCECBAM(num_iters=8, channels=32,
                        use_cbam=use_cbam).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _pairs(low_dir, high_dir):
    lows = sorted(glob.glob(os.path.join(low_dir, '*.png')) +
                  glob.glob(os.path.join(low_dir, '*.jpg')))
    highs = {os.path.basename(p): p for p in
             glob.glob(os.path.join(high_dir, '*.png')) +
             glob.glob(os.path.join(high_dir, '*.jpg'))}
    return [(p, highs[os.path.basename(p)]) for p in lows
            if os.path.basename(p) in highs]


@torch.no_grad()
def evaluate(model, device, low_dir, high_dir, mode, proxy):
    acc = {k: {'psnr': [], 'ssim': []}
           for k in ('Dark input', 'AutoContrast', 'UNLET-ADAS')}

    for lp, hp in _pairs(low_dir, high_dir):
        low_img = Image.open(lp).convert('RGB')
        high_img = Image.open(hp).convert('RGB')
        if mode == 'proxy':
            low_img = low_img.resize((proxy, proxy))
            high_img = high_img.resize((proxy, proxy))

        low = np.asarray(low_img, dtype=np.float32) / 255.0
        high = np.asarray(high_img, dtype=np.float32) / 255.0
        auto = np.asarray(ImageOps.autocontrast(low_img),
                          dtype=np.float32) / 255.0

        t = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).to(device)
        if mode == 'proxy':
            enh, _ = model(t)
        else:
            enh, _ = model.enhance_full_res(t, proxy_size=proxy)
        enh = enh[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        for name, img in (('Dark input', low), ('AutoContrast', auto),
                          ('UNLET-ADAS', enh)):
            acc[name]['psnr'].append(calc_psnr(high, img, data_range=1.0))
            acc[name]['ssim'].append(calc_ssim(high, img, channel_axis=2,
                                               data_range=1.0))

    return {k: {'psnr_mean': float(np.mean(v['psnr'])),
                'psnr_std': float(np.std(v['psnr'])),
                'ssim_mean': float(np.mean(v['ssim'])),
                'ssim_std': float(np.std(v['ssim'])),
                'n': len(v['psnr'])} for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='./data/LOL')
    ap.add_argument('--split', default='eval15')
    ap.add_argument('--weights', default='app/zerodce_cbam_best.pt')
    ap.add_argument('--mode', choices=['proxy', 'fullres'], default='proxy')
    ap.add_argument('--proxy', type=int, default=256)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(os.path.join(ROOT, args.weights), device)

    low_dir = os.path.join(args.data_root, args.split, 'low')
    high_dir = os.path.join(args.data_root, args.split, 'high')
    for d in (low_dir, high_dir):
        if not os.path.isdir(d):
            raise SystemExit(f'Not found: {d}')

    res = evaluate(model, device, low_dir, high_dir, args.mode, args.proxy)

    print(f"\nLOL {args.split} | mode={args.mode} | device={device}"
          f"{' | proxy=%d' % args.proxy if args.mode == 'proxy' else ''}")
    if args.split == 'eval15':
        print("  NOTE: eval15 is also used for early stopping in "
              "src/train.py -- these are validation scores.")
    print(f"  {'Method':<14} {'PSNR (dB)':>16} {'SSIM':>16}")
    for k, v in res.items():
        print(f"  {k:<14} {v['psnr_mean']:>9.2f} +/-{v['psnr_std']:<5.2f}"
              f" {v['ssim_mean']:>9.4f} +/-{v['ssim_std']:<5.4f}")
    u, d, a = res['UNLET-ADAS'], res['Dark input'], res['AutoContrast']
    print(f"  Gain over dark input  : "
          f"{u['psnr_mean'] - d['psnr_mean']:+.2f} dB")
    print(f"  Gain over AutoContrast: "
          f"{u['psnr_mean'] - a['psnr_mean']:+.2f} dB")

    out = args.out or f'results/eval_{args.split}_{args.mode}.json'
    os.makedirs(os.path.join(ROOT, os.path.dirname(out)), exist_ok=True)
    with open(os.path.join(ROOT, out), 'w') as f:
        json.dump({'split': args.split, 'mode': args.mode,
                   'proxy': args.proxy, 'weights': args.weights,
                   'device': device, 'results': res}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
