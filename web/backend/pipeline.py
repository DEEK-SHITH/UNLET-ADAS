"""
UNLET-ADAS API backend — model loading and the enhance/detect pipeline.

This wraps the shared modules in `src/` (`src/enhance.py`,
`src/detection.py`, `src/lane_detection.py`, `src/depth.py`) so the
FastAPI backend and the Streamlit app run the exact same computer
vision pipeline.
"""
import hashlib
import os
import subprocess
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
os.environ.setdefault('YOLO_OFFLINE', 'true')

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
    sharpen_frame,
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


def _draw_header_badge(img, text, origin, bg_color, text_color, dot_color=None):
    """Draws an elegant, modern pill badge onto comparison composite frames."""
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    pad_x, pad_y = 10, 6
    dot_space = 14 if dot_color else 0
    bx1, by1 = x - pad_x, y - th - pad_y
    bx2, by2 = x + tw + pad_x + dot_space, y + pad_y
    cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (60, 60, 60), 1)
    if dot_color:
        cv2.circle(img, (x + 2, y - th // 2), 4, dot_color, -1)
        cv2.putText(img, text, (x + dot_space + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)


def _compose_comparison_frame(orig_bgr, enh_bgr, target_w=None, target_h=None):
    """Creates a high-resolution side-by-side comparison frame with status badges."""
    h, w = orig_bgr.shape[:2]
    if enh_bgr.shape[:2] != (h, w):
        enh_bgr = cv2.resize(enh_bgr, (w, h))

    if target_w is not None and target_h is not None:
        half_w = target_w // 2
        orig_part = cv2.resize(orig_bgr, (half_w, target_h), interpolation=cv2.INTER_AREA)
        enh_part = cv2.resize(enh_bgr, (target_w - half_w, target_h), interpolation=cv2.INTER_AREA)
        comp = np.hstack([orig_part, enh_part])
        cv2.line(comp, (half_w, 0), (half_w, target_h), (50, 50, 50), 2)
        _draw_header_badge(comp, 'ORIGINAL INPUT', (14, 26),
                           (25, 25, 25), (220, 220, 220), dot_color=(0, 0, 220))
        _draw_header_badge(comp, 'UNLET-ADAS ENHANCED', (half_w + 14, 26),
                           (20, 32, 24), (100, 255, 180), dot_color=(0, 220, 100))
        return comp

    comp = np.hstack([orig_bgr, enh_bgr])
    _draw_header_badge(comp, 'ORIGINAL INPUT', (18, 28),
                       (25, 25, 25), (220, 220, 220), dot_color=(0, 0, 220))
    _draw_header_badge(comp, 'UNLET-ADAS ENHANCED', (w + 18, 28),
                       (20, 32, 24), (100, 255, 180), dot_color=(0, 220, 100))
    return comp


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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
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
    """Full enhance + detect pipeline for a single image with side-by-side composite."""
    t0 = time.time()
    orig_rgb = np.array(pil_image.convert('RGB'))
    orig_lum = float(orig_rgb[::16, ::16].mean() / 255.0)

    enhanced_list = enhance_frame_batch(
        MODELS.enhancer, MODELS.device, [orig_rgb], size=256, adaptive=opts.adaptive)
    enh_rgb = enhanced_list[0]

    eff_conf = scene_aware_conf(opts.det_conf, orig_lum, night_boost=0.0)

    depth_map = None
    if opts.use_depth_risk and ensure_depth_model():
        from src.depth import estimate_depth_map
        depth_map = estimate_depth_map(MODELS.midas_model, MODELS.midas_transform,
                                        MODELS.device, enh_rgb)

    enh_lum = float(enh_rgb[::16, ::16].mean() / 255.0)
    result = {
        'brightness': round(orig_lum, 4),
        'enhanced_brightness': round(enh_lum, 4),
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
            MODELS.pothole_yolo, enh_rgb, max(eff_conf, 0.30), opts.det_imgsz,
            'Pothole', (255, 165, 0), min_y_ratio=0.35)
        result['potholes'] = potholes

    if opts.enable_signs and MODELS.has_signs:
        enh_rgb, signs = detect_signs(MODELS.sign_yolo, enh_rgb, max(eff_conf, 0.35), opts.det_imgsz)
        result['signs'] = signs

    if opts.enable_detect and MODELS.has_yolo:
        enh_rgb, dets, counts, high_risk = detect_and_draw(
            MODELS.yolo, enh_rgb, conf=eff_conf, imgsz=opts.det_imgsz, depth_map=depth_map)
        result['detections'] = dets
        result['counts'] = counts
        result['high_risk'] = high_risk

    # Compose side-by-side comparison image
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
    comp_bgr = _compose_comparison_frame(orig_bgr, enh_bgr)
    result['comparison_rgb'] = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)

    result['processing_ms'] = round((time.time() - t0) * 1000, 1)
    return enh_rgb, result


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
                 crf: int = 18, preset: str = 'veryfast'):
        # Ensure dimensions are strictly even for H.264 / yuv420p
        self.width = width if width % 2 == 0 else width - 1
        self.height = height if height % 2 == 0 else height - 1
        self.output_path = output_path
        self.fps = round(float(fps), 2)
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
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
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
            if self.needs_transcode and os.path.exists(self.output_path):
                try:
                    from src.enhance import transcode_for_browser
                    transcode_for_browser(self.output_path, crf=self.crf, preset=self.preset)
                except Exception as e:
                    print(f"Fallback transcode warning: {e}")


@dataclass
class VideoJob:
    id: str
    status: str = 'queued'          # queued | processing | done | error | cancelled
    progress: int = 0
    frames_done: int = 0
    frames_total: int = 0
    output_path: Optional[str] = None
    comparison_path: Optional[str] = None
    has_comparison: bool = False
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
    fast_mode: bool = True          # Stride detection across frames for high throughput
    max_frames: int = 900           # ~30s at 30fps safety cap for demo
    batch_size: int = 6
    quality_preset: str = 'high'    # turbo | high | ultra


@dataclass
class LiveOptions:
    adaptive: bool = True
    det_conf: float = 0.25
    det_imgsz: int = 384
    enable_detect: bool = True
    enable_lanes: bool = False
    interleaved: bool = True
    quality_mode: str = 'turbo'     # turbo (384p) | balanced (480p) | quality (640p)


LIVE_DET_CACHE = {
    'last_dets': [],
    'last_counts': {},
    'last_high_risk': False,
    'frame_idx': 0
}


def run_live_pipeline(image_bytes: bytes, opts: LiveOptions) -> tuple[bytes, dict]:
    """Ultra-low-latency enhancement + perception for live camera feeds."""
    t0 = time.time()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Could not decode image frame")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    orig_lum = float(frame_rgb[::16, ::16].mean() / 255.0)

    # Resolution-aware proxy size for real-time speed
    if opts.quality_mode == 'turbo':
        proxy_sz = 128
        live_imgsz = min(opts.det_imgsz, 256)
        jpeg_q = 76
        det_model = MODELS.yolo_fast if MODELS.has_yolo_fast else MODELS.yolo
        interleave_stride = 3
    elif opts.quality_mode == 'balanced':
        proxy_sz = 192
        live_imgsz = min(opts.det_imgsz, 320)
        jpeg_q = 82
        det_model = MODELS.yolo
        interleave_stride = 2
    else:  # 'quality'
        proxy_sz = 256
        live_imgsz = opts.det_imgsz
        jpeg_q = 88
        det_model = MODELS.yolo
        interleave_stride = 2

    # If adaptive and frame is already well-lit (indoor room or daylight >= 0.38),
    # bypass neural DCE-Net completely to save ~40ms CPU time and preserve natural raw feed.
    if opts.adaptive and orig_lum >= 0.38:
        enh_rgb = frame_rgb.copy()
        enh_lum = orig_lum
    else:
        enhanced_list = enhance_frame_batch(
            MODELS.enhancer, MODELS.device, [frame_rgb],
            size=proxy_sz, adaptive=opts.adaptive
        )
        enh_rgb = enhanced_list[0]
        enh_lum = float(enh_rgb[::16, ::16].mean() / 255.0)

    dets = []
    counts = {}
    high_risk = False

    if opts.enable_lanes:
        left, right = detect_lanes(enh_rgb)
        if left is not None or right is not None:
            enh_rgb = draw_lanes(enh_rgb, left, right)

    if opts.enable_detect and MODELS.has_yolo:
        LIVE_DET_CACHE['frame_idx'] += 1
        run_heavy = (not opts.interleaved) or (LIVE_DET_CACHE['frame_idx'] % interleave_stride == 1)
        eff_conf = scene_aware_conf(opts.det_conf, orig_lum, night_boost=0.0)

        if run_heavy or not LIVE_DET_CACHE['last_dets']:
            enh_rgb, dets, counts, high_risk = detect_and_draw(
                det_model, enh_rgb, conf=eff_conf, imgsz=live_imgsz
            )
            # In live webcam view, the person sitting at the camera is the operator, not a collision hazard
            for d in dets:
                if d.get('name') == 'Person' and d.get('risk') == 'HIGH':
                    d['risk'] = 'SAFE'
                    d['label'] = 'Person · SAFE'
            high_risk = any(d.get('risk') == 'HIGH' for d in dets)

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
        for p in (j.input_path, j.output_path, getattr(j, 'comparison_path', None)):
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
    """
    Optimized background video processing engine:
    1. Batched Zero-DCE curve estimation with 256px resolution for maximum edge fidelity.
    2. Single-pass dual H.264 FFmpeg streaming for BOTH enhanced-only and side-by-side comparison videos.
    3. Strided heavy detection with ByteTrack tracking for 2x-3x speedup.
    4. Adaptive unsharp mask detail enhancement preventing video blur.
    """
    job = JOBS[job_id]
    job.input_path = input_path
    job.status = 'processing'
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    job.frames_total = min(total, opts.max_frames) if total > 0 else opts.max_frames

    # Determine processing speed vs quality trade-offs based on preset
    qp = getattr(opts, 'quality_preset', 'high')
    if qp == 'turbo':
        crf = 18          # Visually lossless H.264 matching native uploaded video quality
        preset = 'ultrafast'
        proxy_sz = 128    # Ultra-fast Zero-DCE proxy estimation
        eff_imgsz = 384   # High accuracy, low CPU latency
        eff_fast_mode = True
        det_stride = 4    # Run detection every 4th frame, redraw cached boxes on intermediate
        batch_size = 8    # Vectorized PyTorch batching
        interp_mode = 'nearest'
        fast_color = True
    elif qp == 'ultra':
        crf = 16
        preset = 'veryfast'
        proxy_sz = 256
        eff_imgsz = opts.det_imgsz
        eff_fast_mode = False
        det_stride = 1
        batch_size = 4
        interp_mode = 'bilinear'
        fast_color = False
    else:  # 'high' / 'balanced'
        crf = 18
        preset = 'ultrafast'
        proxy_sz = 192
        eff_imgsz = min(opts.det_imgsz, 512)
        eff_fast_mode = opts.fast_mode
        det_stride = 3 if eff_fast_mode else 1
        batch_size = 6
        interp_mode = 'bilinear'
        fast_color = True

    active_yolo = MODELS.yolo_fast if (qp == 'turbo' and MODELS.has_yolo_fast) else MODELS.yolo

    out_path = os.path.join(output_dir, f'{job_id}_enhanced.mp4')
    comp_path = os.path.join(output_dir, f'{job_id}_comparison.mp4')

    # Dual single-pass FFmpeg writers (both standard W x H resolution)
    writer = DirectFFmpegVideoWriter(out_path, W, H, fps, crf=crf, preset=preset)
    writer_comp = DirectFFmpegVideoWriter(comp_path, W, H, fps, crf=crf, preset=preset)

    has_depth = ensure_depth_model() if opts.use_depth_risk else False
    lane_tracker = LaneTracker() if opts.enable_lanes else None
    lane_roi_bottom: Optional[float] = None

    last_det_list = []
    last_pot_boxes = []
    last_sign_boxes = []
    last_depth_map = None
    last_lane_lines = None
    seen_track_ids_by_class: dict[str, set[int]] = {}
    peak_untracked_by_class: dict[str, int] = {}
    proc_idx = 0
    count = 0
    batch_size = max(2, min(int(opts.batch_size), 8))

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

            with torch.inference_mode():
                enhanced_batch = enhance_frame_batch(
                    MODELS.enhancer, MODELS.device, fb_rgb,
                    size=proxy_sz, adaptive=opts.adaptive,
                    interp_mode=interp_mode, fast_color=fast_color)

            for orig_bgr, enh_rgb in zip(fb_bgr, enhanced_batch):
                if job.cancel_requested:
                    job.status = 'cancelled'
                    break

                run_heavy = (not eff_fast_mode) or (proc_idx % det_stride == 0)
                orig_lum = float(orig_bgr[::16, ::16].mean() / 255.0)
                frame_conf = scene_aware_conf(opts.det_conf, orig_lum, night_boost=0.0)

                if opts.enable_lanes:
                    if (proc_idx % (det_stride if eff_fast_mode else 1) == 0) or last_lane_lines is None:
                        left, right = detect_lanes(
                            enh_rgb,
                            roi_bottom=lane_roi_bottom if lane_roi_bottom is not None else 1.0,
                            tracker=lane_tracker)
                        last_lane_lines = (left, right)
                    else:
                        left, right = last_lane_lines
                    if left is not None or right is not None:
                        enh_rgb = draw_lanes(enh_rgb, left, right)

                if opts.enable_pothole and MODELS.has_pothole:
                    if run_heavy or not last_pot_boxes:
                        pot_res = MODELS.pothole_yolo(
                            enh_rgb, conf=max(frame_conf, 0.30), imgsz=eff_imgsz, verbose=False)[0]
                        last_pot_boxes = []
                        h_enh = enh_rgb.shape[0]
                        for box in pot_res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Restrict pothole to roadway surface
                            if y2 >= int(h_enh * 0.35):
                                last_pot_boxes.append((x1, y1, x2, y2, float(box.conf[0])))

                    for x1, y1, x2, y2, pconf in last_pot_boxes:
                        cv2.rectangle(enh_rgb, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        cv2.putText(enh_rgb, f'Pothole {pconf:.0%}', (x1, max(y1 - 8, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 165, 0), 2)

                if opts.enable_signs and MODELS.has_signs:
                    if run_heavy or not last_sign_boxes:
                        sign_res = MODELS.sign_yolo(
                            enh_rgb, conf=max(frame_conf, 0.35), imgsz=eff_imgsz, verbose=False)[0]
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

                if opts.enable_detect and MODELS.has_yolo:
                    depth_map = None
                    if opts.use_depth_risk and has_depth:
                        if proc_idx % 4 == 0 or last_depth_map is None:
                            from src.depth import estimate_depth_map
                            last_depth_map = estimate_depth_map(
                                MODELS.midas_model, MODELS.midas_transform,
                                MODELS.device, enh_rgb)
                        depth_map = last_depth_map

                    if run_heavy or not last_det_list:
                        try:
                            (enh_rgb, frame_counts, _frame_ids,
                             frame_risk, last_det_list) = track_and_draw(
                                active_yolo, enh_rgb, conf=frame_conf,
                                imgsz=eff_imgsz, depth_map=depth_map)
                        except Exception:
                            (enh_rgb, last_det_list, frame_counts,
                             frame_risk) = detect_and_draw(
                                active_yolo, enh_rgb, conf=frame_conf,
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
                enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
                writer.write(enh_bgr)
                comp_frame = _compose_comparison_frame(orig_bgr, enh_bgr, target_w=W, target_h=H)
                writer_comp.write(comp_frame)

            job.frames_done = count
            job.progress = int(100 * count / max(job.frames_total, 1))

        cap.release()
        writer.release()
        writer_comp.release()

        if os.path.exists(input_path) and ('unlet_adas_api_' in input_path or 'tmp' in input_path.lower()):
            try:
                os.remove(input_path)
            except OSError:
                pass

        if job.status != 'cancelled':
            if getattr(writer, 'needs_transcode', False):
                transcode_for_browser(out_path, crf=crf)
            if getattr(writer_comp, 'needs_transcode', False):
                transcode_for_browser(comp_path, crf=crf)
            job.output_path = out_path
            job.comparison_path = comp_path
            job.has_comparison = True
            job.status = 'done'
            job.progress = 100
        else:
            for p in (out_path, comp_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        try:
            cap.release()
            writer.release()
            writer_comp.release()
        except Exception:
            pass
