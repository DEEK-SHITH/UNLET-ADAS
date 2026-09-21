"""
UNLET-ADAS API backend — model loading and the enhance/detect pipeline.

This wraps the same underlying code the Streamlit app uses
(src/enhance.py, src/lane_detection.py, src/depth.py) rather than
reimplementing enhancement or lane detection. The detection-drawing
and risk-estimation helpers below are ported from app/streamlit_app.py
instead of imported from it, because that module runs
st.set_page_config() and other Streamlit-runtime calls at import time
and is only safe to import inside an actual Streamlit script.
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
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.join(REPO_ROOT, 'app')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.enhance import (  # noqa: E402
    load_enhancer, enhance_frame_batch, scene_blend_weight,
    scene_aware_conf, correct_color_cast, transcode_for_browser,
)
from src.lane_detection import detect_lanes, draw_lanes  # noqa: E402

ADAS_CLASSES = {
    0:  ('Person',        (50,  205,  50)),
    1:  ('Bicycle',       (255, 165,   0)),
    2:  ('Car',           (30,  144, 255)),
    3:  ('Motorcycle',    (255, 100, 100)),
    5:  ('Bus',           (138,  43, 226)),
    7:  ('Truck',         (255,  20, 147)),
    9:  ('Traffic Light', (255, 215,   0)),
    11: ('Stop Sign',     (220,  20,  60)),
}

RISK_COLORS = {
    'HIGH':   (220,  20,  60),
    'MEDIUM': (255, 165,   0),
    'LOW':    (50,  205,  50),
}

_SIGN_PALETTE = [
    (220, 0, 0), (0, 130, 220), (200, 200, 0), (255, 165, 0),
    (180, 0, 180), (0, 180, 0), (0, 200, 200), (255, 0, 128),
    (180, 100, 0), (0, 0, 255), (100, 255, 100), (130, 0, 0),
]


def estimate_risk_geometry(x1, y1, x2, y2, frame_w, frame_h):
    box_h_frac = (y2 - y1) / max(frame_h, 1)
    cx = (x1 + x2) / 2
    in_path = 0.2 * frame_w <= cx <= 0.8 * frame_w
    if box_h_frac > 0.35 and in_path:
        return 'HIGH'
    if box_h_frac > 0.18 or (in_path and box_h_frac > 0.10):
        return 'MEDIUM'
    return 'LOW'


def estimate_risk(x1, y1, x2, y2, frame_w, frame_h, depth_map=None):
    if depth_map is not None:
        from src.depth import sample_proximity, classify_proximity
        proximity = sample_proximity(depth_map, x1, y1, x2, y2)
        return classify_proximity(proximity)
    return estimate_risk_geometry(x1, y1, x2, y2, frame_w, frame_h)


def _plausible_traffic_light_shape(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    return (w / h) <= 0.75


def _draw_box(img_bgr, x1, y1, x2, y2, label, color, thickness=2):
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img_bgr, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def detect_and_draw(yolo, image_np, conf=0.25, imgsz=640, depth_map=None,
                     class_map=None):
    class_map = class_map or ADAS_CLASSES
    results = yolo(image_np, conf=conf, imgsz=imgsz, verbose=False)[0]
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]
    counts = {}
    det_list = []
    high_risk = False
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in class_map:
            continue
        name, _ = class_map[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if name == 'Traffic Light' and not _plausible_traffic_light_shape(x1, y1, x2, y2):
            continue
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map)
        high_risk = high_risk or risk == 'HIGH'
        _draw_box(img_bgr, x1, y1, x2, y2, f'{name} · {risk}',
                  RISK_COLORS[risk], thickness=3 if risk == 'HIGH' else 2)
        counts[name] = counts.get(name, 0) + 1
        det_list.append({
            'name': name, 'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2], 'risk': risk,
        })
    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, det_list, counts, high_risk


def detect_single_class(yolo, image_rgb, conf, imgsz, label_prefix, color):
    """Shared helper for the pothole detector and any other
    single/no-class-map YOLO model: draws every box the model returns,
    labelled '<label_prefix> <conf%>'."""
    results = yolo(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_s = float(box.conf[0])
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, f'{label_prefix} {conf_s:.0%}', (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out.append({'name': label_prefix, 'conf': round(conf_s, 3), 'box': [x1, y1, x2, y2]})
    return image_rgb, out


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
    """Called once at FastAPI startup. Mirrors app/streamlit_app.py's
    @st.cache_resource loaders, minus the Streamlit caching decorator
    and minus the MiDaS depth model (loaded lazily on first request
    that asks for it — it needs a torch.hub download on first use and
    shouldn't block API startup)."""
    weights = os.path.join(APP_DIR, 'zerodce_cbam_best.pt')
    MODELS.enhancer, MODELS.device = load_enhancer(weights, device='cuda')

    try:
        from ultralytics import YOLO
        yolo_weights = os.path.join(REPO_ROOT, 'yolov8s.pt')
        if not os.path.exists(yolo_weights):
            yolo_weights = os.path.join(REPO_ROOT, 'yolov8n.pt')
        MODELS.yolo = YOLO(yolo_weights)
        MODELS.has_yolo = True
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
    """Lazy MiDaS load on first use — see load_all_models docstring."""
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
    """Full enhance + detect pipeline for a single image, mirroring the
    Streamlit app's Image Enhancement tab. Returns a dict of results
    plus the annotated RGB numpy array."""
    t0 = time.time()
    orig_rgb = np.array(pil_image.convert('RGB'))
    orig_luminance = float(orig_rgb.astype(np.float32).mean() / 255)

    enhanced_list = enhance_frame_batch(
        MODELS.enhancer, MODELS.device, [orig_rgb], size=256, adaptive=opts.adaptive)
    enh_rgb = enhanced_list[0]

    eff_conf = scene_aware_conf(opts.det_conf, orig_luminance)

    depth_map = None
    if opts.use_depth_risk and ensure_depth_model():
        from src.depth import estimate_depth_map
        depth_map = estimate_depth_map(MODELS.midas_model, MODELS.midas_transform,
                                        MODELS.device, enh_rgb)

    result = {
        'brightness': round(orig_luminance, 4),
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


@dataclass
class VideoJob:
    id: str
    status: str = 'queued'          # queued | processing | done | error | cancelled
    progress: int = 0
    frames_done: int = 0
    frames_total: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    counts: dict = field(default_factory=dict)
    high_risk: bool = False


JOBS: dict[str, VideoJob] = {}


@dataclass
class VideoOptions:
    adaptive: bool = True
    det_conf: float = 0.25
    det_imgsz: int = 640
    enable_detect: bool = True
    enable_pothole: bool = False
    enable_lanes: bool = False
    fast_mode: bool = True          # detect every 2nd frame, like the Streamlit Video tab
    max_frames: int = 900           # ~30s at 30fps safety cap for the demo API


def new_job() -> VideoJob:
    job = VideoJob(id=uuid.uuid4().hex[:12])
    JOBS[job.id] = job
    return job


def run_video_job(job_id: str, input_path: str, output_dir: str, opts: VideoOptions):
    """Runs synchronously inside a FastAPI BackgroundTasks worker.
    Frame-skip cadence (every 2nd frame for the heavy detector passes)
    matches the Streamlit Video tab's fast mode — see
    app/streamlit_app.py's process_video_chunk and the "detect objects
    every 2nd frame" change made there."""
    job = JOBS[job_id]
    job.status = 'processing'
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    job.frames_total = min(total, opts.max_frames) if total > 0 else opts.max_frames

    out_path = os.path.join(output_dir, f'{job_id}_enhanced.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    from ultralytics import YOLO  # noqa: F401 (already loaded on MODELS)

    last_det_list = []
    last_pot_boxes = []
    proc_idx = 0
    count = 0

    try:
        while count < job.frames_total:
            if job.cancel_requested:
                job.status = 'cancelled'
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            count += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            enh_rgb = enhance_frame_batch(
                MODELS.enhancer, MODELS.device, [frame_rgb],
                size=128, adaptive=opts.adaptive)[0]

            run_heavy = (not opts.fast_mode) or (proc_idx % 2 == 0)
            orig_lum = frame_bgr.astype(np.float32).mean() / 255
            frame_conf = scene_aware_conf(opts.det_conf, orig_lum)

            if opts.enable_lanes:
                left, right = detect_lanes(enh_rgb)
                if left is not None or right is not None:
                    enh_rgb = draw_lanes(enh_rgb, left, right)

            if opts.enable_pothole and MODELS.has_pothole:
                if run_heavy or not last_pot_boxes:
                    pot_res = MODELS.pothole_yolo(
                        enh_rgb, conf=frame_conf, imgsz=opts.det_imgsz, verbose=False)[0]
                    last_pot_boxes = [
                        (*map(int, box.xyxy[0]), float(box.conf[0])) for box in pot_res.boxes]
                for x1, y1, x2, y2, pconf in last_pot_boxes:
                    cv2.rectangle(enh_rgb, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(enh_rgb, f'Pothole {pconf:.0%}', (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

            if opts.enable_detect and MODELS.has_yolo:
                if run_heavy or not last_det_list:
                    enh_rgb, last_det_list, frame_counts, frame_risk = detect_and_draw(
                        MODELS.yolo, enh_rgb, conf=frame_conf, imgsz=opts.det_imgsz)
                    job.high_risk = job.high_risk or frame_risk
                    for k, v in frame_counts.items():
                        job.counts[k] = job.counts.get(k, 0) + v
                else:
                    img_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
                    for d in last_det_list:
                        x1, y1, x2, y2 = d['box']
                        _draw_box(img_bgr, x1, y1, x2, y2, f"{d['name']} · {d['risk']}",
                                  RISK_COLORS[d['risk']])
                        job.counts[d['name']] = job.counts.get(d['name'], 0) + 1
                    enh_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            proc_idx += 1
            writer.write(cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR))
            job.frames_done = count
            job.progress = int(100 * count / max(job.frames_total, 1))

        cap.release()
        writer.release()

        if job.status != 'cancelled':
            transcode_for_browser(out_path)
            job.output_path = out_path
            job.status = 'done'
            job.progress = 100
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        try:
            cap.release()
            writer.release()
        except Exception:
            pass
