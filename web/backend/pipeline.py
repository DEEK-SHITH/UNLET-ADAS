"""
UNLET-ADAS API backend — model loading and the enhance/detect pipeline.

This wraps the shared modules in `src/` (`src/enhance.py`,
`src/detection.py`, `src/lane_detection.py`, `src/depth.py`) so the
FastAPI backend and the Streamlit app run the exact same computer
vision pipeline.
"""
import hashlib
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.join(REPO_ROOT, 'app')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.detection import (  # noqa: E402
    ADAS_CLASSES,
    LOWLIGHT_CLASSES,
    RISK_COLORS,
    _draw_detection,
    _plausible_traffic_light_shape,
    detect_and_draw,
    detect_single_class,
    estimate_risk,
    estimate_risk_depth,
    estimate_risk_geometry,
    redraw_cached_detections,
    track_and_draw,
)
from src.enhance import (  # noqa: E402
    correct_color_cast,
    enhance_frame_batch,
    load_enhancer,
    scene_aware_conf,
    scene_blend_weight,
    transcode_for_browser,
)
from src.lane_detection import (  # noqa: E402
    LaneTracker,
    detect_lanes,
    draw_lanes,
    estimate_dashboard_cutoff,
)

_SIGN_PALETTE = [
    (220, 0, 0), (0, 130, 220), (200, 200, 0), (255, 165, 0),
    (180, 0, 180), (0, 180, 0), (0, 200, 200), (255, 0, 128),
    (180, 100, 0), (0, 0, 255), (100, 255, 100), (130, 0, 0),
]


def detect_signs(sign_yolo, image_rgb, conf, imgsz):
    results = sign_yolo(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_s = float(box.conf[0])
        cls_name = results.names[int(box.cls[0])]
        digest = hashlib.md5(str(cls_name).encode('utf-8')).digest()
        color = _SIGN_PALETTE[digest[0] % len(_SIGN_PALETTE)]
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, f'{cls_name} {conf_s:.0%}', (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out.append({'name': str(cls_name), 'conf': round(conf_s, 3), 'box': [x1, y1, x2, y2]})
    return image_rgb, out


@dataclass
class Models:
    enhancer: object = None
    device: object = None
    yolo: object = None
    has_yolo: bool = False
    yolo_fast: object = None
    has_yolo_fast: bool = False
    pothole_yolo: object = None
    has_pothole: bool = False
    sign_yolo: object = None
    has_signs: bool = False
    midas_model: object = None
    midas_transform: object = None
    has_depth: bool = False
    depth_load_attempted: bool = False


MODELS = Models()


def load_all_models():
    """Called once at FastAPI startup."""
    weights = os.path.join(APP_DIR, 'zerodce_cbam_best.pt')
    MODELS.enhancer, MODELS.device = load_enhancer(weights, device='cuda')

    try:
        from ultralytics import YOLO
        yolo_s = os.path.join(REPO_ROOT, 'yolov8s.pt')
        try:
            MODELS.yolo = YOLO(yolo_s if os.path.exists(yolo_s) else 'yolov8s.pt')
        except Exception:
            MODELS.yolo = YOLO(os.path.join(REPO_ROOT, 'yolov8n.pt'))
        MODELS.has_yolo = True

        yolo_n = os.path.join(REPO_ROOT, 'yolov8n.pt')
        if os.path.exists(yolo_n):
            MODELS.yolo_fast = YOLO(yolo_n)
            MODELS.has_yolo_fast = True
    except Exception as e:
        print(f'ADAS detector not loaded: {e}')

    pothole_weights = os.path.join(APP_DIR, 'pothole_best.pt')
    if os.path.exists(pothole_weights):
        try:
            from ultralytics import YOLO
            MODELS.pothole_yolo = YOLO(pothole_weights)
            MODELS.has_pothole = True
        except Exception as e:
            print(f'Pothole detector not loaded: {e}')

    sign_weights = os.path.join(APP_DIR, 'signs_best.pt')
    if os.path.exists(sign_weights):
        try:
            from ultralytics import YOLO
            MODELS.sign_yolo = YOLO(sign_weights)
            MODELS.has_signs = True
        except Exception as e:
            print(f'Sign detector not loaded: {e}')


def ensure_depth_model():
    """Lazy MiDaS load on first use."""
    if MODELS.depth_load_attempted:
        return MODELS.has_depth
    MODELS.depth_load_attempted = True
    from src.depth import load_midas
    model, transform, ok = load_midas(MODELS.device)
    MODELS.midas_model, MODELS.midas_transform, MODELS.has_depth = model, transform, ok
    return ok


@dataclass
class ImageOptions:
    adaptive: bool = True
    det_conf: float = 0.25
    det_imgsz: int = 640
    enable_detect: bool = True
    enable_pothole: bool = False
    enable_signs: bool = False
    enable_lanes: bool = False
    use_depth_risk: bool = False


def run_image_pipeline(pil_image: Image.Image, opts: ImageOptions):
    """Full enhance + detect pipeline for a single image."""
    t0 = time.time()
    orig_rgb = np.array(pil_image.convert('RGB'))
    orig_luminance = float(orig_rgb.astype(np.float32).mean() / 255)

    enhanced_list = enhance_frame_batch(
        MODELS.enhancer, MODELS.device, [orig_rgb], size=256, adaptive=opts.adaptive)
    enh_rgb = enhanced_list[0]

    eff_conf = scene_aware_conf(opts.det_conf, orig_luminance, night_boost=0.05)

    depth_map = None
    if opts.use_depth_risk and ensure_depth_model():
        from src.depth import estimate_depth_map
        depth_map = estimate_depth_map(MODELS.midas_model, MODELS.midas_transform,
                                        MODELS.device, enh_rgb)

    enh_luminance = float(enh_rgb.astype(np.float32).mean() / 255)
    result = {
        'brightness': round(orig_luminance, 4),
        'enhanced_brightness': round(enh_luminance, 4),
        'effective_confidence': round(eff_conf, 4),
        'detections': [], 'potholes': [], 'signs': [],
        'counts': {}, 'high_risk': False,
        'lanes_found': False,
        'depth_risk_used': depth_map is not None,
    }

    if opts.enable_lanes:
        left, right = detect_lanes(enh_rgb)
        if left is not None or right is not None:
            enh_rgb = draw_lanes(enh_rgb, left, right)
            result['lanes_found'] = True

    if opts.enable_pothole and MODELS.has_pothole:
        enh_rgb, potholes = detect_single_class(
            MODELS.pothole_yolo, enh_rgb, eff_conf, opts.det_imgsz,
            'Pothole', (255, 165, 0))
        result['potholes'] = potholes

    if opts.enable_signs and MODELS.has_signs:
        enh_rgb, signs = detect_signs(MODELS.sign_yolo, enh_rgb, eff_conf, opts.det_imgsz)
        result['signs'] = signs

    if opts.enable_detect and MODELS.has_yolo:
        enh_rgb, dets, counts, high_risk = detect_and_draw(
            MODELS.yolo, enh_rgb, conf=eff_conf, imgsz=opts.det_imgsz, depth_map=depth_map)
        result['detections'] = dets
        result['counts'] = counts
        result['high_risk'] = high_risk

    result['processing_ms'] = round((time.time() - t0) * 1000, 1)
    return enh_rgb, result


import subprocess

# Optimize PyTorch CPU multi-threading
try:
    import torch
    import multiprocessing
    torch.set_num_threads(min(multiprocessing.cpu_count(), 8))
except Exception:
    pass


class DirectFFmpegVideoWriter:
    """
    Direct-to-FFmpeg H.264 video writer via standard input pipe.
    Avoids OpenCV's intermediate mp4v write + subsequent re-transcode pass.
    Writes single-pass, browser-compatible H.264 with +faststart.
    """
    def __init__(self, output_path: str, width: int, height: int, fps: float,
                 crf: int = 20, preset: str = 'veryfast'):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.proc = None
        self._fallback_writer = None
        self.needs_transcode = False
        self._init_pipe()

    def _init_pipe(self):
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-i', 'pipe:0',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', self.preset,
                '-crf', str(self.crf),
                '-movflags', '+faststart',
                self.output_path
            ]
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"DirectFFmpegVideoWriter pipe failed ({e}); falling back to cv2.VideoWriter")
            self.proc = None
            self.needs_transcode = True
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._fallback_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )

    def write(self, frame_bgr):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(frame_bgr.tobytes())
                return
            except Exception:
                pass
        if self._fallback_writer:
            self._fallback_writer.write(frame_bgr)

    def release(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=60)
            except Exception:
                self.proc.kill()
            self.proc = None
        if self._fallback_writer:
            self._fallback_writer.release()
            self._fallback_writer = None


@dataclass
class VideoJob:
    id: str
    status: str = 'queued'          # queued | processing | done | error | cancelled
    progress: int = 0
    frames_done: int = 0
    frames_total: int = 0
    output_path: Optional[str] = None
    input_path: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    counts: dict = field(default_factory=dict)
    high_risk: bool = False
    created_at: float = field(default_factory=time.time)


JOBS: dict[str, VideoJob] = {}


@dataclass
class VideoOptions:
    adaptive: bool = True
    det_conf: float = 0.25
    det_imgsz: int = 640
    enable_detect: bool = True
    enable_pothole: bool = False
    enable_signs: bool = False
    enable_lanes: bool = False
    use_depth_risk: bool = False
    fast_mode: bool = True          # detect every 2nd frame, like the Streamlit Video tab
    max_frames: int = 900           # ~30s at 30fps safety cap for the demo API
    batch_size: int = 4
    quality_preset: str = 'high'    # turbo | high | ultra


@dataclass
class LiveOptions:
    adaptive: bool = True
    det_conf: float = 0.25
    det_imgsz: int = 384
    enable_detect: bool = True
    enable_lanes: bool = False
    interleaved: bool = True
    quality_mode: str = 'turbo'     # turbo (384p, ~15-20 FPS) | balanced (480p, ~10-15 FPS) | quality (640p)


LIVE_DET_CACHE = {
    'last_dets': [],
    'last_counts': {},
    'last_high_risk': False,
    'frame_idx': 0
}


def run_live_pipeline(image_bytes: bytes, opts: LiveOptions) -> tuple[bytes, dict]:
    """
    Ultra-low-latency enhancement + perception for live camera feeds.
    - Decodes direct from memory buffer (avoids PIL / file IO)
    - Zero-DCE proxy tailored to live quality mode
    - Interleaved YOLO detection for 2x-3x higher FPS
    - Encodes enhanced frame directly to JPEG (avoids base64 overhead)
    """
    t0 = time.time()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Could not decode image frame")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    orig_lum = float(frame_rgb.astype(np.float32).mean() / 255)

    # Resolution-aware proxy size for real-time speed
    if opts.quality_mode == 'turbo':
        proxy_sz = 112
        live_imgsz = min(opts.det_imgsz, 256)
        jpeg_q = 75
        det_model = MODELS.yolo_fast if MODELS.has_yolo_fast else MODELS.yolo
    elif opts.quality_mode == 'balanced':
        proxy_sz = 160
        live_imgsz = min(opts.det_imgsz, 384)
        jpeg_q = 82
        det_model = MODELS.yolo
    else:  # 'quality'
        proxy_sz = 256
        live_imgsz = opts.det_imgsz
        jpeg_q = 90
        det_model = MODELS.yolo

    enhanced_list = enhance_frame_batch(
        MODELS.enhancer, MODELS.device, [frame_rgb],
        size=proxy_sz, adaptive=opts.adaptive
    )
    enh_rgb = enhanced_list[0]
    enh_lum = float(enh_rgb.astype(np.float32).mean() / 255)

    dets = []
    counts = {}
    high_risk = False

    if opts.enable_lanes:
        left, right = detect_lanes(enh_rgb)
        if left is not None or right is not None:
            enh_rgb = draw_lanes(enh_rgb, left, right)

    if opts.enable_detect and MODELS.has_yolo:
        LIVE_DET_CACHE['frame_idx'] += 1
        # In interleaved mode, detect on odd frames, use fast redrawing on even frames
        run_heavy = (not opts.interleaved) or (LIVE_DET_CACHE['frame_idx'] % 2 == 1)
        eff_conf = scene_aware_conf(opts.det_conf, orig_lum, night_boost=0.05)

        if run_heavy or not LIVE_DET_CACHE['last_dets']:
            enh_rgb, dets, counts, high_risk = detect_and_draw(
                det_model, enh_rgb, conf=eff_conf, imgsz=live_imgsz
            )
            LIVE_DET_CACHE['last_dets'] = dets
            LIVE_DET_CACHE['last_counts'] = counts
            LIVE_DET_CACHE['last_high_risk'] = high_risk
        else:
            enh_rgb = redraw_cached_detections(enh_rgb, LIVE_DET_CACHE['last_dets'])
            dets = LIVE_DET_CACHE['last_dets']
            counts = LIVE_DET_CACHE['last_counts']
            high_risk = LIVE_DET_CACHE['last_high_risk']
    else:
        LIVE_DET_CACHE['last_dets'] = []
        LIVE_DET_CACHE['last_counts'] = {}
        LIVE_DET_CACHE['last_high_risk'] = False

    # Fast JPEG encode of enhanced frame
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    encoded_bytes = buf.tobytes() if ok else b''

    proc_ms = round((time.time() - t0) * 1000, 1)
    stats = {
        'brightness': round(orig_lum, 4),
        'enhanced_brightness': round(enh_lum, 4),
        'detections': dets,
        'counts': counts,
        'high_risk': high_risk,
        'processing_ms': proc_ms,
        'fps': round(1000.0 / max(proc_ms, 1.0), 1),
    }
    return encoded_bytes, stats


def cleanup_old_jobs(max_age_sec: float = 3600.0):
    """Evict finished/failed/cancelled jobs older than max_age_sec and
    remove their temporary video files."""
    now = time.time()
    expired = [
        jid for jid, j in list(JOBS.items())
        if j.status in ('done', 'error', 'cancelled')
        and (now - j.created_at) > max_age_sec
    ]
    for jid in expired:
        j = JOBS.pop(jid, None)
        if not j:
            continue
        for p in (j.input_path, j.output_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def new_job() -> VideoJob:
    cleanup_old_jobs()
    job = VideoJob(id=uuid.uuid4().hex[:12])
    JOBS[job.id] = job
    return job


def run_video_job(job_id: str, input_path: str, output_dir: str, opts: VideoOptions):
    """Runs inside a background worker thread.
    Enhances frames in batches, tracks objects with ByteTrack to report
    unique object counts rather than inflating counts across frames,
    and streams directly to a single-pass H.264 FFmpeg pipe for maximum speed and quality."""
    job = JOBS[job_id]
    job.input_path = input_path
    job.status = 'processing'
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    job.frames_total = min(total, opts.max_frames) if total > 0 else opts.max_frames

    # Determine processing speed vs quality trade-offs based on preset
    qp = getattr(opts, 'quality_preset', 'high')
    if qp == 'turbo':
        crf = 22
        proxy_sz = 128
        eff_imgsz = min(opts.det_imgsz, 480)
        eff_fast_mode = True
    elif qp == 'ultra':
        crf = 18
        proxy_sz = 256
        eff_imgsz = opts.det_imgsz
        eff_fast_mode = False
    else:  # 'high' (default)
        crf = 20
        proxy_sz = 192
        eff_imgsz = opts.det_imgsz
        eff_fast_mode = opts.fast_mode

    out_path = os.path.join(output_dir, f'{job_id}_enhanced.mp4')
    # Use direct-to-ffmpeg pipe: writes browser-compatible H.264 with +faststart
    # in a single pass, completely bypassing the slow second transcode step!
    writer = DirectFFmpegVideoWriter(out_path, W, H, fps, crf=crf, preset='veryfast')

    has_depth = ensure_depth_model() if opts.use_depth_risk else False
    lane_tracker = LaneTracker() if opts.enable_lanes else None
    lane_roi_bottom: Optional[float] = None

    last_det_list = []
    last_pot_boxes = []
    last_sign_boxes = []
    seen_track_ids_by_class: dict[str, set[int]] = {}
    peak_untracked_by_class: dict[str, int] = {}
    proc_idx = 0
    count = 0
    batch_size = max(1, int(opts.batch_size))

    try:
        while count < job.frames_total:
            if job.cancel_requested:
                job.status = 'cancelled'
                break

            fb_rgb, fb_bgr = [], []
            while len(fb_rgb) < batch_size and count < job.frames_total:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                fb_bgr.append(frame_bgr)
                fb_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                count += 1

            if not fb_rgb:
                break

            if opts.enable_lanes and lane_roi_bottom is None:
                lane_roi_bottom = (
                    estimate_dashboard_cutoff(fb_rgb) if len(fb_rgb) >= 3 else 1.0
                )

            enhanced_batch = enhance_frame_batch(
                MODELS.enhancer, MODELS.device, fb_rgb,
                size=proxy_sz, adaptive=opts.adaptive)

            for orig_bgr, enh_rgb in zip(fb_bgr, enhanced_batch):
                if job.cancel_requested:
                    job.status = 'cancelled'
                    break

                run_heavy = (not eff_fast_mode) or (proc_idx % 2 == 0)
                orig_lum = orig_bgr.astype(np.float32).mean() / 255
                frame_conf = scene_aware_conf(opts.det_conf, orig_lum)

                if opts.enable_lanes:
                    left, right = detect_lanes(
                        enh_rgb,
                        roi_bottom=lane_roi_bottom if lane_roi_bottom is not None else 1.0,
                        tracker=lane_tracker)
                    if left is not None or right is not None:
                        enh_rgb = draw_lanes(enh_rgb, left, right)

                if opts.enable_pothole and MODELS.has_pothole:
                    if run_heavy or not last_pot_boxes:
                        pot_res = MODELS.pothole_yolo(
                            enh_rgb, conf=frame_conf, imgsz=eff_imgsz, verbose=False)[0]
                        last_pot_boxes = [
                            (*map(int, box.xyxy[0]), float(box.conf[0]))
                            for box in pot_res.boxes
                        ]
                    for x1, y1, x2, y2, pconf in last_pot_boxes:
                        cv2.rectangle(enh_rgb, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        cv2.putText(enh_rgb, f'Pothole {pconf:.0%}', (x1, max(y1 - 8, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

                if opts.enable_signs and MODELS.has_signs:
                    if run_heavy or not last_sign_boxes:
                        sign_res = MODELS.sign_yolo(
                            enh_rgb, conf=frame_conf, imgsz=eff_imgsz, verbose=False)[0]
                        last_sign_boxes = [
                            (*map(int, box.xyxy[0]), float(box.conf[0]),
                             sign_res.names[int(box.cls[0])])
                            for box in sign_res.boxes
                        ]
                    for x1, y1, x2, y2, sconf, cls_name in last_sign_boxes:
                        digest = hashlib.md5(str(cls_name).encode('utf-8')).digest()
                        color = _SIGN_PALETTE[digest[0] % len(_SIGN_PALETTE)]
                        cv2.rectangle(enh_rgb, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(enh_rgb, f'{cls_name} {sconf:.0%}', (x1, max(y1 - 8, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if opts.enable_detect and MODELS.has_yolo:
                    depth_map = None
                    if run_heavy and opts.use_depth_risk and has_depth:
                        from src.depth import estimate_depth_map
                        depth_map = estimate_depth_map(
                            MODELS.midas_model, MODELS.midas_transform,
                            MODELS.device, enh_rgb)

                    if run_heavy or not last_det_list:
                        try:
                            (enh_rgb, frame_counts, _frame_ids,
                             frame_risk, last_det_list) = track_and_draw(
                                MODELS.yolo, enh_rgb, conf=frame_conf,
                                imgsz=eff_imgsz, depth_map=depth_map)
                        except Exception:
                            (enh_rgb, last_det_list, frame_counts,
                             frame_risk) = detect_and_draw(
                                MODELS.yolo, enh_rgb, conf=frame_conf,
                                imgsz=eff_imgsz, depth_map=depth_map)
                        job.high_risk = job.high_risk or frame_risk
                        for d in last_det_list:
                            tid = d.get('track_id')
                            if tid is not None:
                                seen_track_ids_by_class.setdefault(
                                    d['name'], set()).add(tid)
                        for k, v in frame_counts.items():
                            if v > peak_untracked_by_class.get(k, 0):
                                peak_untracked_by_class[k] = v
                        all_classes = set(seen_track_ids_by_class) | set(peak_untracked_by_class)
                        job.counts = {
                            cls: max(
                                len(seen_track_ids_by_class.get(cls, set())),
                                peak_untracked_by_class.get(cls, 0),
                            )
                            for cls in all_classes
                        }
                    else:
                        enh_rgb = redraw_cached_detections(enh_rgb, last_det_list)

                proc_idx += 1
                writer.write(cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR))

            job.frames_done = count
            job.progress = int(100 * count / max(job.frames_total, 1))

        cap.release()
        writer.release()

        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

        if job.status != 'cancelled':
            if getattr(writer, 'needs_transcode', False):
                transcode_for_browser(out_path, crf=crf)
            job.output_path = out_path
            job.status = 'done'
            job.progress = 100
        elif os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        try:
            cap.release()
            writer.release()
        except Exception:
            pass
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        try:
            cap.release()
            writer.release()
        except Exception:
            pass
