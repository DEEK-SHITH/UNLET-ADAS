"""
UNLET-ADAS: Enhancement Functions
===================================
Core functions for single image and batch video enhancement.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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


@torch.no_grad()
def enhance_image(model, device, image_path,
                  size=256, output_path=None):
    """
    Enhance a single low-light image.
    Returns PIL Image of enhanced result.
    """
    orig = Image.open(image_path).convert('RGB')
    W, H = orig.size

    resized = orig.resize((size, size), Image.BICUBIC)
    arr     = np.array(resized, dtype=np.float32) / 255.0
    t       = torch.from_numpy(arr).permute(
        2, 0, 1).unsqueeze(0).to(device)

    enh, _ = model(t)
    enh_np = (enh[0].permute(1, 2, 0).cpu().numpy()
              * 255).clip(0, 255).astype(np.uint8)

    result = Image.fromarray(enh_np).resize(
        (W, H), Image.BICUBIC)

    if output_path:
        result.save(output_path)
        print(f'Saved: {output_path}')

    return result


@torch.no_grad()
def enhance_frame_batch(model, device, frames_rgb, size=512):
    """
    Enhance a batch of video frames.
    Uses size=512 for better quality on HD video.
    Input : list of (H,W,3) uint8 RGB arrays
    Output: list of (H,W,3) uint8 RGB arrays
    """
    orig_sizes = [(f.shape[1], f.shape[0]) for f in frames_rgb]

    # Use higher resolution for less blur
    resized = np.stack([
        cv2.resize(f, (size, size),
                   interpolation=cv2.INTER_LANCZOS4)
        for f in frames_rgb
    ]).astype(np.float32) / 255.0

    t      = torch.from_numpy(resized).permute(
        0, 3, 1, 2).to(device)
    enh, _ = model(t)
    out    = (enh.permute(0, 2, 3, 1).cpu().numpy()
              * 255).clip(0, 255).astype(np.uint8)

    results = []
    for i, frame in enumerate(out):
        # Color balance
        f = frame.astype(np.float32)
        r = f[:,:,0].mean()
        g = f[:,:,1].mean()
        b = f[:,:,2].mean()
        avg = (r + g + b) / 3.0
        if r > 0: f[:,:,0] = f[:,:,0] * (avg / r)
        if g > 0: f[:,:,1] = f[:,:,1] * (avg / g)
        if b > 0: f[:,:,2] = f[:,:,2] * (avg / b)
        frame = np.clip(f, 0, 255).astype(np.uint8)

        # Resize back with high quality
        result = cv2.resize(
            frame, orig_sizes[i],
            interpolation=cv2.INTER_LANCZOS4)

        # Slight sharpening to reduce upscale blur
        kernel = np.array([
            [ 0, -0.5,  0],
            [-0.5,  3, -0.5],
            [ 0, -0.5,  0]])
        sharp  = cv2.filter2D(result, -1, kernel)
        result = np.clip(sharp, 0, 255).astype(np.uint8)

        results.append(result)

    return results

def enhance_video(model, device,
                  input_path, output_path,
                  original_path=None,
                  comparison_path=None,
                  batch_size=4,
                  size=512):
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
            model, device, frames_rgb, size)
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