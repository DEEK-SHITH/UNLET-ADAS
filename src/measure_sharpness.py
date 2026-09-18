"""
UNLET-ADAS: Sharpness Measurement (resize pipeline vs full-resolution curves)
=============================================================================
Quantifies the blur cost of the resize round-trip that an earlier version of
this pipeline used, against the current full-resolution curve application.

Both arms use the SAME trained weights (app/zerodce_cbam_best.pt) and the same
frames. Only the inference path differs:

  resize arm    : downscale frame -> proxy, model(proxy), upscale result back
                  to full resolution  (pixel content is downscaled)
  full-res arm  : estimate curves on the proxy, upsample the CURVE MAPS, apply
                  them to the original pixels  (pixel content never downscaled)

Sharpness metric: variance of the Laplacian of the grayscale image, the
standard no-reference blur measure. Higher = sharper.

Usage:
    python src/measure_sharpness.py
    python src/measure_sharpness.py --video results/original_night_drive.mp4 \
        --frames 30 --proxy 256 --out results/sharpness_results.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import ZeroDCECBAM  # noqa: E402


def laplacian_variance(rgb_uint8):
    """Standard no-reference blur measure: variance of the Laplacian."""
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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


@torch.no_grad()
def enhance_resize(model, t_full, proxy_size):
    """Old path: downscale the IMAGE, enhance, upscale the result."""
    proxy = F.interpolate(t_full, size=(proxy_size, proxy_size),
                          mode='bilinear', align_corners=False)
    enhanced_small, _ = model(proxy)
    return F.interpolate(enhanced_small, size=t_full.shape[2:],
                         mode='bilinear', align_corners=False)


@torch.no_grad()
def enhance_fullres(model, t_full, proxy_size):
    """Current path: upsample the CURVES, apply to original pixels."""
    enhanced, _ = model.enhance_full_res(t_full, proxy_size=proxy_size)
    return enhanced


def to_uint8(t):
    arr = t[0].permute(1, 2, 0).cpu().numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='results/original_night_drive.mp4')
    ap.add_argument('--weights', default='app/zerodce_cbam_best.pt')
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--proxy', type=int, default=256)
    ap.add_argument('--out', default='results/sharpness_results.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(os.path.join(ROOT, args.weights), device)

    cap = cv2.VideoCapture(os.path.join(ROOT, args.video))
    rows, idx = [], 0
    while len(rows) < args.frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue
        idx += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        rows.append({
            'frame': idx - 1,
            'height': rgb.shape[0], 'width': rgb.shape[1],
            'input': laplacian_variance(rgb),
            'resize': laplacian_variance(to_uint8(enhance_resize(
                model, t, args.proxy))),
            'fullres': laplacian_variance(to_uint8(enhance_fullres(
                model, t, args.proxy))),
        })
    cap.release()

    if not rows:
        raise SystemExit(f'No frames read from {args.video}')

    def col(k):
        return np.array([r[k] for r in rows])

    res, full = col('resize'), col('fullres')
    ratio = full / res
    summary = {
        'video': args.video, 'weights': args.weights,
        'device': device, 'proxy_size': args.proxy,
        'n_frames': len(rows),
        'resolution': f"{rows[0]['width']}x{rows[0]['height']}",
        'input_mean': float(col('input').mean()),
        'resize_mean': float(res.mean()), 'resize_std': float(res.std()),
        'fullres_mean': float(full.mean()), 'fullres_std': float(full.std()),
        'ratio_mean': float(ratio.mean()), 'ratio_std': float(ratio.std()),
        'ratio_min': float(ratio.min()), 'ratio_max': float(ratio.max()),
        'per_frame': rows,
    }

    print(f"\nSharpness (variance of Laplacian), n={len(rows)} frames "
          f"@ {summary['resolution']}, proxy={args.proxy}, device={device}")
    print(f"  Raw input                    : {summary['input_mean']:8.2f}")
    print(f"  Resize round-trip            : {summary['resize_mean']:8.2f} "
          f"+/- {summary['resize_std']:.2f}")
    print(f"  Full-resolution curves       : {summary['fullres_mean']:8.2f} "
          f"+/- {summary['fullres_std']:.2f}")
    print(f"  Ratio (full-res / resize)    : {summary['ratio_mean']:8.2f}x "
          f"+/- {summary['ratio_std']:.2f} "
          f"[{summary['ratio_min']:.2f}-{summary['ratio_max']:.2f}]")

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
