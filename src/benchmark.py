"""
UNLET-ADAS: CPU/GPU Throughput Benchmark
===========================================
Measures actual per-frame latency for the enhancement pass alone, and
enhancement + YOLOv8 detection, against real frames from this repo's
own sample footage. Exists so the numbers quoted in README.md's
"Measured Performance" section are reproducible, not just asserted --
run this yourself and compare against your own hardware rather than
assuming the published numbers apply to your machine.

Usage:
    python src/benchmark.py
    python src/benchmark.py --video path/to/other_video.mp4 --frames 30
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.enhance import correct_color_cast, scene_blend_weight
from src.model import build_model


def _enhance_pil(model, device, pil_image, adaptive=True, proxy_size=256):
    # Mirrors app/streamlit_app.py's enhance_pil exactly -- that file
    # can't be imported directly outside a Streamlit runtime context,
    # so this benchmark keeps its own copy of the same logic.
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    enh, _ = model.enhance_full_res(t, proxy_size=proxy_size)
    if adaptive:
        alpha = scene_blend_weight(float(arr.mean()))
        enh = t * (1 - alpha) + enh * alpha
    out = (enh[0].permute(1, 2, 0).cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(correct_color_cast(out))


def _load_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f'Could not read any frames from {video_path}')
    return frames


def _bench(label, fn, frames, n):
    for f in frames[:3]:  # warmup, excluded from the timed average
        fn(f)
    t0 = time.time()
    for i in range(n):
        fn(frames[i % len(frames)])
    ms = (time.time() - t0) / n * 1000
    print(f'{label:55s}: {ms:7.1f} ms/frame  ({1000 / ms:5.2f} FPS)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--video', default=os.path.join(
            ROOT, 'results', 'original_night_drive.mp4'),
        help='Video to draw benchmark frames from (default: this '
             "repo's own sample night-drive footage).")
    p.add_argument('--frames', type=int, default=30,
                    help='Frames to load from --video for warmup/timing.')
    p.add_argument('--iters', type=int, default=15,
                    help='Timed iterations per stage (after warmup).')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device).eval()
    weights = os.path.join(ROOT, 'app', 'zerodce_cbam_best.pt')
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))

    try:
        from ultralytics import YOLO
        yolo = YOLO('yolov8n.pt')
        has_yolo = True
    except Exception as e:
        print(f'YOLO unavailable ({e}) -- skipping detection benchmarks.')
        has_yolo = False

    frames = _load_frames(args.video, args.frames)
    print(f'Device: {device} | CPU cores: {os.cpu_count()} | '
          f'torch threads: {torch.get_num_threads()}')
    print(f'Loaded {len(frames)} frames from {args.video} '
          f'({frames[0].shape[1]}x{frames[0].shape[0]})')

    def enh_only(frame_rgb):
        return np.array(_enhance_pil(model, device, Image.fromarray(frame_rgb)))

    def enh_plus_yolo(imgsz):
        def _run(frame_rgb):
            enh = enh_only(frame_rgb)
            yolo(enh, conf=0.25, imgsz=imgsz, verbose=False)
            return enh
        return _run

    _bench('Enhancement only', enh_only, frames, args.iters)
    if has_yolo:
        _bench('Enhancement + YOLOv8n detect (imgsz=640)',
               enh_plus_yolo(640), frames, args.iters)
        _bench("Enhancement + YOLOv8n detect (imgsz=320, Live Stream's cap)",
               enh_plus_yolo(320), frames, args.iters)


if __name__ == '__main__':
    main()
