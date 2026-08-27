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

    results = []
    for frame in out:
        # Gray-world color balance to correct residual tint
        f = frame.astype(np.float32)
        r = f[:,:,0].mean()
        g = f[:,:,1].mean()
        b = f[:,:,2].mean()
        avg = (r + g + b) / 3.0
        if r > 0: f[:,:,0] = f[:,:,0] * (avg / r)
        if g > 0: f[:,:,1] = f[:,:,1] * (avg / g)
        if b > 0: f[:,:,2] = f[:,:,2] * (avg / b)
        results.append(np.clip(f, 0, 255).astype(np.uint8))

    return results


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
