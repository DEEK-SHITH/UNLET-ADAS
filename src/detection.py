"""
UNLET-ADAS: Object Detection, Tracking, and Risk Estimation
=============================================================
Framework-agnostic ADAS object detection, ByteTrack multi-object
tracking, proximity risk estimation (box geometry + MiDaS depth),
and OpenCV annotation helpers shared by the Streamlit app and the
FastAPI web backend.
"""

import cv2

from src.depth import classify_proximity, sample_proximity

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

# Class map for the optional low-light-specialized detector (see
# src/train_lowlight.py) — uses its own 0..4 class-id scheme (NOT COCO
# ids), covering the 5 ADAS classes ExDark provides night training data for.
LOWLIGHT_CLASSES = {
    0: ('Person',     (50,  205,  50)),
    1: ('Bicycle',    (255, 165,   0)),
    2: ('Car',        (30,  144, 255)),
    3: ('Motorcycle', (255, 100, 100)),
    4: ('Bus',        (138,  43, 226)),
}

RISK_COLORS = {
    'HIGH':   (220,  20,  60),   # red
    'MEDIUM': (255, 165,   0),   # amber
    'LOW':    (50,  205,  50),   # green
}


def estimate_risk_geometry(x1, y1, x2, y2, frame_w, frame_h):
    """
    Collision-proximity estimate from bounding-box geometry alone: a
    box filling a large fraction of frame height is close to the
    camera, and one centered in the middle corridor (20%..80% of width)
    is in the ego vehicle's path.
    """
    box_h_frac = (y2 - y1) / max(frame_h, 1)
    cx = (x1 + x2) / 2
    in_path = 0.2 * frame_w <= cx <= 0.8 * frame_w
    if box_h_frac > 0.35 and in_path:
        return 'HIGH'
    if box_h_frac > 0.18 or (in_path and box_h_frac > 0.10):
        return 'MEDIUM'
    return 'LOW'


def estimate_risk_depth(depth_map, x1, y1, x2, y2, frame_w=None):
    """
    Proximity-risk estimate from a per-frame normalized MiDaS depth map
    (0..1, higher = closer). When frame_w is provided, objects outside
    the ego vehicle's central corridor (0.15..0.85 of frame width) are
    capped at MEDIUM risk to avoid false HIGH collision alerts from
    roadside objects.
    """
    proximity = sample_proximity(depth_map, x1, y1, x2, y2)
    in_ego_path = True
    if frame_w is not None and frame_w > 0:
        cx = (x1 + x2) / 2
        in_ego_path = 0.15 * frame_w <= cx <= 0.85 * frame_w
    return classify_proximity(proximity, in_ego_path=in_ego_path)


def estimate_risk(x1, y1, x2, y2, frame_w, frame_h, depth_map=None):
    """
    Dispatches to the MiDaS depth-based estimate when a depth map is
    available for the frame, otherwise falls back to box geometry.
    """
    if depth_map is not None:
        return estimate_risk_depth(depth_map, x1, y1, x2, y2, frame_w=frame_w)
    return estimate_risk_geometry(x1, y1, x2, y2, frame_w, frame_h)


def _plausible_traffic_light_shape(x1, y1, x2, y2):
    """
    Suppresses round streetlight glare blobs misclassified as 'Traffic
    Light' by COCO-pretrained YOLOv8: a vertical traffic-light housing
    is taller than it is wide (w/h <= 0.75).
    """
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    return (w / h) <= 0.75


def _draw_detection(img_bgr, x1, y1, x2, y2, label, risk):
    """Draw a risk-colored bounding box and filled label banner on BGR image."""
    r, g, b = RISK_COLORS[risk]
    color = (b, g, r)  # OpenCV draws in BGR order on img_bgr
    thickness = 3 if risk == 'HIGH' else 2
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_top = max(y1 - th - 8, 0)
    y_bot = max(y1, th + 8)
    cv2.rectangle(
        img_bgr, (x1, y_top), (x1 + tw + 8, y_bot), color, -1)
    cv2.putText(
        img_bgr, label, (x1 + 4, y_bot - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def detect_and_draw(yolo, image_np, conf=0.25, imgsz=640, depth_map=None,
                     class_map=None):
    """
    Single-frame ADAS object detection and annotation.
    Returns (annotated_rgb, det_list, counts, high_risk).
    """
    if class_map is None:
        class_map = ADAS_CLASSES
    h, w = image_np.shape[:2]
    yolo_in = (
        cv2.GaussianBlur(image_np, (3, 3), 0)
        if max(h, w) > imgsz and min(h, w) >= 3
        else image_np
    )
    results = yolo(yolo_in, conf=conf, imgsz=imgsz, verbose=False)[0]
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
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
        if name == 'Traffic Light' and not _plausible_traffic_light_shape(
                x1, y1, x2, y2):
            continue
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map)
        high_risk = high_risk or risk == 'HIGH'
        label = f'{name} · {risk}'
        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        det_list.append({
            'name': name, 'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2], 'risk': risk, 'label': label,
        })
    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, det_list, counts, high_risk


def track_and_draw(yolo, image_np, conf=0.25, imgsz=640, depth_map=None,
                    class_map=None):
    """
    Multi-frame ADAS detection + ByteTrack tracking.
    Returns (annotated_rgb, counts, track_ids, high_risk, det_list).
    """
    if class_map is None:
        class_map = ADAS_CLASSES
    h, w = image_np.shape[:2]
    yolo_in = (
        cv2.GaussianBlur(image_np, (3, 3), 0)
        if max(h, w) > imgsz and min(h, w) >= 3
        else image_np
    )
    results = yolo.track(
        yolo_in, conf=conf, imgsz=imgsz, persist=True,
        tracker='bytetrack.yaml', verbose=False)[0]
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    counts = {}
    track_ids = set()
    high_risk = False
    det_list = []
    ids = results.boxes.id
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        if cls_id not in class_map:
            continue
        name, _ = class_map[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if name == 'Traffic Light' and not _plausible_traffic_light_shape(
                x1, y1, x2, y2):
            continue
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map)
        high_risk = high_risk or risk == 'HIGH'
        track_id = int(ids[i]) if ids is not None else None
        id_bit = f'#{track_id} ' if track_id is not None else ''
        label = f'{name} {id_bit}· {risk}'
        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        if track_id is not None:
            track_ids.add(track_id)
        det_list.append({
            'name': name, 'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2], 'risk': risk,
            'label': label, 'track_id': track_id,
        })
    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, counts, track_ids, high_risk, det_list


def redraw_cached_detections(image_np, det_list):
    """
    Redraw a previously computed set of detection boxes onto a new
    frame without re-running YOLO (used on skipped frames in fast mode).
    """
    if not det_list:
        return image_np
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for d in det_list:
        x1, y1, x2, y2 = d['box']
        label = d.get('label') or f"{d['name']} · {d['risk']}"
        _draw_detection(img_bgr, x1, y1, x2, y2, label, d['risk'])
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def detect_single_class(yolo, image_rgb, conf, imgsz, label_prefix, color):
    """
    Run a single-class YOLO detector (e.g. pothole detector) and draw
    boxes onto image_rgb. Returns (annotated_rgb, detections_list).
    """
    results = yolo(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_s = float(box.conf[0])
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_rgb, f'{label_prefix} {conf_s:.0%}',
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out.append({
            'name': label_prefix,
            'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2],
        })
    return image_rgb, out
