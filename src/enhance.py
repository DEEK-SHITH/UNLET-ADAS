"""
UNLET-ADAS: Enhancement Functions
===================================
Core functions for single image and batch video enhancement.

Curves are estimated on a small proxy resolution for speed, then
applied directly to the full-resolution frame (see
model.enhance_full_res). This avoids the old resize-down/resize-up
round trip that softened fine detail and hurt detection of small or
distant objects.

Enhancement strength is also scene-adaptive: dark frames (night,
tunnels, shaded hillside roads) get full enhancement, while
already well-lit frames (daylight) are left close to untouched so
the system helps at night and in hilly terrain without washing out
or over-brightening daytime footage.
"""

import cv2
import numpy as np
import torch
from PIL import Image

from src.model import build_model


def load_enhancer(weights_path, device='cuda'):
    """Load trained UNLET-ADAS model."""
    device = torch.device(
        device if torch.cuda.is_available() else 'cpu')
    model  = build_model().to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device))
    model.eval()
    print(f'Model loaded from: {weights_path}')
    print(f'Device: {device}')
    return model, device


def scene_blend_weight(luminance, dark_thresh=0.35, bright_thresh=0.55):
    """
    Blend factor between original and enhanced frame based on
    scene brightness (0..1 average luminance):
      - <= dark_thresh   : 1.0 (full enhancement — night / tunnel / shade)
      - >= bright_thresh : 0.0 (no enhancement — daylight)
      - in between       : smooth ramp (dusk, hillside shadow patches)
    """
    if luminance <= dark_thresh:
        return 1.0
    if luminance >= bright_thresh:
        return 0.0
    return (bright_thresh - luminance) / (bright_thresh - dark_thresh)


def correct_color_cast(frame_uint8, strength=0.6, lum_pctl=60, max_shift=18):
    """
    Mild white-balance correction in LAB space, using only the
    brightest ~40% of pixels to estimate the tint.

    Plain whole-image gray-world fails on night frames: the huge
    near-black background (sky, unlit distance) dominates the color
    statistics, so the correction computed from it gets applied to
    the bright, informative road surface too — overshooting into a
    new (often blue/purple) cast instead of removing the original
    one. Estimating the tint from just the well-lit pixels, shifting
    only LAB's a/b (color) channels partially back toward neutral
    (L, i.e. brightness, is left untouched), and capping the shift
    keeps this a gentle tint fix rather than a full renormalization.
    """
    lab = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    mask = L >= np.percentile(L, lum_pctl)
    if mask.sum() < 50:
        mask = np.ones_like(L, dtype=bool)

    a_dev = A[mask].mean() - 128.0
    b_dev = B[mask].mean() - 128.0
    a_shift = np.clip(-a_dev * strength, -max_shift, max_shift)
    b_shift = np.clip(-b_dev * strength, -max_shift, max_shift)

    lab[:,:,1] = np.clip(A + a_shift, 0, 255)
    lab[:,:,2] = np.clip(B + b_shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


@torch.no_grad()
def enhance_image(model, device, image_path,
                  size=256, output_path=None, adaptive=True):
    """
    Enhance a single low-light image at full resolution.
    Returns PIL Image of enhanced result.
    """
    orig = Image.open(image_path).convert('RGB')
    W, H = orig.size

    arr    = np.array(orig, dtype=np.float32) / 255.0
    t_full = torch.from_numpy(arr).permute(
        2, 0, 1).unsqueeze(0).to(device)

    enh_full, _ = model.enhance_full_res(t_full, proxy_size=size)

    if adaptive:
        alpha    = scene_blend_weight(float(arr.mean()))
        enh_full = t_full * (1 - alpha) + enh_full * alpha

    enh_np = (enh_full[0].permute(1, 2, 0).cpu().numpy()
              * 255).clip(0, 255).astype(np.uint8)
    enh_np = correct_color_cast(enh_np)
    result = Image.fromarray(enh_np)

    if output_path:
        result.save(output_path)
        print(f'Saved: {output_path}')

    return result


@torch.no_grad()
def enhance_frame_batch(model, device, frames_rgb,
                        size=256, adaptive=True):
    """
    Enhance a batch of full-resolution video frames.
    Input : list of (H,W,3) uint8 RGB arrays, all the same size
    Output: list of (H,W,3) uint8 RGB arrays at the same resolution
    """
    arr_full = np.stack(frames_rgb).astype(np.float32) / 255.0
    t_full   = torch.from_numpy(arr_full).permute(
        0, 3, 1, 2).to(device)

    enh_full, _ = model.enhance_full_res(t_full, proxy_size=size)

    if adaptive:
        lum    = t_full.mean(dim=[1, 2, 3])
        alphas = torch.tensor(
            [scene_blend_weight(float(l)) for l in lum],
            device=device).view(-1, 1, 1, 1)
        enh_full = t_full * (1 - alphas) + enh_full * alphas

    out = (enh_full.permute(0, 2, 3, 1).cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)

    return [correct_color_cast(frame) for frame in out]


def enhance_video(model, device,
                  input_path, output_path,
                  original_path=None,
                  comparison_path=None,
                  batch_size=4,
                  size=256,
                  adaptive=True):
    """
    Enhance a full video.

    Outputs:
    - output_path      : enhanced video only
    - original_path    : original video copy (optional)
    - comparison_path  : side-by-side comparison (optional)
    """
    import time

    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Writers
    enh_writer  = cv2.VideoWriter(
        output_path, fourcc, fps, (W, H))
    orig_writer = (cv2.VideoWriter(
        original_path, fourcc, fps, (W, H))
        if original_path else None)
    cmp_writer  = (cv2.VideoWriter(
        comparison_path, fourcc, fps, (W * 2, H))
        if comparison_path else None)

    print(f'Input    : {W}x{H} @ {fps:.0f}fps | {total} frames')
    print(f'Enhanced : {output_path}')
    if original_path:
        print(f'Original : {original_path}')
    if comparison_path:
        print(f'Comparison: {comparison_path}')
    print('-' * 50)

    frame_buf, orig_buf = [], []
    count = 0
    t0    = time.time()

    def flush_buffer(frames_rgb, origs_bgr):
        enhanced = enhance_frame_batch(
            model, device, frames_rgb, size, adaptive)
        for orig_bgr, enh_rgb in zip(origs_bgr, enhanced):
            enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
            enh_writer.write(enh_bgr)
            if orig_writer:
                orig_writer.write(orig_bgr)
            if cmp_writer:
                combined = np.hstack([orig_bgr, enh_bgr])
                cv2.line(combined,
                         (W, 0), (W, H), (255,255,255), 3)
                cmp_writer.write(combined)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_buf.append(frame_rgb)
        orig_buf.append(frame_bgr.copy())
        count += 1

        if len(frame_buf) >= batch_size:
            flush_buffer(frame_buf, orig_buf)
            frame_buf.clear()
            orig_buf.clear()

        if count % 60 == 0:
            elapsed = time.time() - t0
            pct     = 100 * count / max(total, 1)
            eta     = elapsed / count * (total - count)
            print(f'  {count:4d}/{total} ({pct:.0f}%)'
                  f'  ETA: {eta:.0f}s')

    if frame_buf:
        flush_buffer(frame_buf, orig_buf)

    cap.release()
    enh_writer.release()
    if orig_writer: orig_writer.release()
    if cmp_writer:  cmp_writer.release()

    elapsed = time.time() - t0
    print(f'\nDone! {count} frames in {elapsed/60:.1f} min')
    return count
