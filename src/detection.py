"""
UNLET-ADAS: Object Detection, Tracking, and Risk Estimation
=============================================================
Framework-agnostic ADAS object detection, ByteTrack multi-object
tracking, proximity risk estimation (box geometry + MiDaS depth),
and OpenCV annotation helpers shared by the Streamlit app and the
FastAPI web backend.
"""

import cv2
import numpy as np

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

LOWLIGHT_CLASSES = {
    0: ('Person',     (50,  205,  50)),
    1: ('Bicycle',    (255, 165,   0)),
    2: ('Car',        (30,  144, 255)),
    3: ('Motorcycle', (255, 100, 100)),
    4: ('Bus',        (138,  43, 226)),
}

# Physical obstacle classes capable of presenting a vehicular collision hazard
COLLISION_OBSTACLE_CLASSES = {
    'Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck', 'Pothole'
}

# Regulatory/traffic signals (informational, not collision impact hazards)
SIGNAL_CLASSES = {
    'Traffic Light', 'Stop Sign'
}

RISK_COLORS = {
    'HIGH':   (220,  20,  60),   # Crimson Red
    'MEDIUM': (255, 165,   0),   # Amber Orange
    'LOW':    (50,  205,  50),   # Lime Green
    'SIGNAL': (255, 215,   0),   # Golden Yellow
    'INFO':   (0,   191, 255),   # Deep Sky Blue
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
    in_path = 0.20 * frame_w <= cx <= 0.80 * frame_w
    if box_h_frac > 0.35 and in_path:
        return 'HIGH'
    if box_h_frac > 0.18 or (in_path and box_h_frac > 0.10):
        return 'MEDIUM'
    return 'LOW'


def estimate_risk_depth(depth_map, x1, y1, x2, y2, frame_w=None):
    """
    Proximity-risk estimate from a per-frame normalized MiDaS depth map
    (0..1, higher = closer). Objects outside the ego vehicle's central
    corridor (0.15..0.85 of frame width) are capped at MEDIUM risk to
    avoid false HIGH collision alerts from roadside infrastructure.
    """
    proximity = sample_proximity(depth_map, x1, y1, x2, y2)
    in_ego_path = True
    if frame_w is not None and frame_w > 0:
        cx = (x1 + x2) / 2
        in_ego_path = 0.15 * frame_w <= cx <= 0.85 * frame_w
    return classify_proximity(proximity, in_ego_path=in_ego_path)


def estimate_risk(x1, y1, x2, y2, frame_w, frame_h, depth_map=None, class_name=None):
    """
    Dispatches risk estimation for objects. Regulatory signals (traffic lights,
    stop signs) return 'SIGNAL' status so they do not trigger false vehicular
    collision alarms.
    """
    if class_name in SIGNAL_CLASSES:
        return 'SIGNAL'
    if depth_map is not None:
        return estimate_risk_depth(depth_map, x1, y1, x2, y2, frame_w=frame_w)
    return estimate_risk_geometry(x1, y1, x2, y2, frame_w, frame_h)


def _plausible_traffic_light_shape(x1, y1, x2, y2):
    """
    Suppresses round streetlight glare blobs misclassified as 'Traffic Light':
    vertical traffic-light housings are typically taller than they are wide (w/h <= 0.85).
    """
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    return (w / h) <= 0.85


def _disambiguate_vehicle_class(name, x1, y1, x2, y2, frame_w, frame_h):
    """
    Resolves false 'Bus' or 'Truck' classifications on passenger cars in front of
    the ego vehicle. When a car's rear, roofline, or hood is close and truncated
    at the bottom edge of a dashcam frame, its aspect ratio becomes wide, confusing
    YOLO into predicting 'Bus'. If the box is truncated near the bottom and has
    car-like proportions (height < 45% of frame height), it is reclassified as 'Car'.
    """
    box_w = x2 - x1
    box_h = y2 - y1
    if name in ('Bus', 'Truck'):
        # In dashcam views, a vehicle whose bottom touches the lower frame with
        # height < 45% of frame and wide aspect ratio is a passenger car.
        if y2 >= 0.70 * frame_h and box_h <= 0.45 * frame_h and box_w >= 1.2 * box_h:
            return 'Car'
        # Any box of small absolute height (< 22% frame height) is not a massive bus/truck
        if box_h < 0.22 * frame_h:
            return 'Car'
    return name


def _merge_overlapping_traffic_lights(det_list):
    """
    Merges duplicate or fragmented traffic light bounding boxes (e.g. top red/yellow
    signal and bottom green arrows detected as separate overlapping boxes on the
    same fixture) into a single cohesive bounding box with maximum confidence.
    """
    tl_items = [d for d in det_list if d['name'] == 'Traffic Light']
    if len(tl_items) <= 1:
        return det_list

    non_tl = [d for d in det_list if d['name'] != 'Traffic Light']
    merged_tl = []
    used = set()

    for i, d1 in enumerate(tl_items):
        if i in used:
            continue
        x1, y1, x2, y2 = d1['box']
        conf = d1['conf']
        track_id = d1.get('track_id')

        for j, d2 in enumerate(tl_items):
            if j <= i or j in used:
                continue
            ox1, oy1, ox2, oy2 = d2['box']
            # If horizontally aligned (same pole/housing) and vertically overlapping or adjacent
            if abs(x1 - ox1) < 25 and abs(x2 - ox2) < 25:
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
                conf = max(conf, d2['conf'])
                if track_id is None:
                    track_id = d2.get('track_id')
                used.add(j)
        used.add(i)

        merged_item = dict(d1)
        merged_item['box'] = [x1, y1, x2, y2]
        merged_item['conf'] = conf
        if track_id is not None:
            merged_item['track_id'] = track_id
        merged_tl.append(merged_item)

    return non_tl + merged_tl


def _filter_duplicate_vehicle_boxes(det_list):
    """
    Suppresses redundant vehicle boxes that overlap significantly on the same object
    (e.g., both Car and Bus predicted on the same car).
    """
    if len(det_list) <= 1:
        return det_list

    def _box_iou(b1, b2):
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / max(union, 1e-6)

    keep = []
    # Sort by confidence descending
    sorted_dets = sorted(det_list, key=lambda d: d.get('conf', 0), reverse=True)
    suppressed = set()

    for i, d1 in enumerate(sorted_dets):
        if i in suppressed:
            continue
        keep.append(d1)
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            d2 = sorted_dets[j]
            # If two vehicle detections overlap heavily
            if d1['name'] in ('Car', 'Bus', 'Truck') and d2['name'] in ('Car', 'Bus', 'Truck'):
                if _box_iou(d1['box'], d2['box']) > 0.40:
                    suppressed.add(j)

    return keep


def sharpen_frame(img_rgb, amount=0.25):
    """
    Fast, subtle unsharp-mask filter to restore crisp high-frequency edges,
    signs, and lane markers without amplifying background noise.
    """
    if amount <= 0:
        return img_rgb
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), 1.5)
    return cv2.addWeighted(img_rgb, 1.0 + amount, blurred, -amount, 0)


def _draw_detection(img_bgr, x1, y1, x2, y2, label, risk):
    """Draw a risk-colored bounding box and filled label banner on BGR image."""
    r, g, b = RISK_COLORS.get(risk, (50, 205, 50))
    color = (b, g, r)  # OpenCV draws in BGR order on img_bgr
    thickness = 3 if risk == 'HIGH' else 2
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    y_top = max(y1 - th - 8, 0)
    y_bot = max(y1, th + 8)
    cv2.rectangle(
        img_bgr, (x1, y_top), (x1 + tw + 8, y_bot), color, -1)
    cv2.putText(
        img_bgr, label, (x1 + 4, y_bot - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


def detect_and_draw(yolo, image_np, conf=0.25, imgsz=640, depth_map=None,
                     class_map=None):
    """
    Single-frame ADAS object detection and annotation with vehicle disambiguation
    and traffic signal coherence. Returns (annotated_rgb, det_list, counts, high_risk).
    """
    if class_map is None:
        class_map = ADAS_CLASSES
    h, w = image_np.shape[:2]
    results = yolo(image_np, conf=conf, imgsz=imgsz, verbose=False)[0]

    raw_candidates = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in class_map:
            continue
        name, _ = class_map[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name == 'Traffic Light' and not _plausible_traffic_light_shape(x1, y1, x2, y2):
            continue

        name = _disambiguate_vehicle_class(name, x1, y1, x2, y2, w, h)
        raw_candidates.append({
            'name': name,
            'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2],
        })

    # Coherence filters: merge fragmented traffic lights and remove overlapping duplicate boxes
    filtered_dets = _merge_overlapping_traffic_lights(raw_candidates)
    filtered_dets = _filter_duplicate_vehicle_boxes(filtered_dets)

    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    counts = {}
    det_list = []
    high_risk = False

    for item in filtered_dets:
        name = item['name']
        conf_s = item['conf']
        x1, y1, x2, y2 = item['box']
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map, class_name=name)

        # High risk alarms are reserved for physical collision hazards
        if name in COLLISION_OBSTACLE_CLASSES and risk == 'HIGH':
            high_risk = True

        label = f'{name} · {risk}'
        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        det_list.append({
            'name': name,
            'conf': conf_s,
            'box': [x1, y1, x2, y2],
            'risk': risk,
            'label': label,
        })

    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, det_list, counts, high_risk


def track_and_draw(yolo, image_np, conf=0.25, imgsz=640, depth_map=None,
                    class_map=None):
    """
    Multi-frame ADAS detection + ByteTrack tracking with vehicle disambiguation
    and signal coherence. Returns (annotated_rgb, counts, track_ids, high_risk, det_list).
    """
    if class_map is None:
        class_map = ADAS_CLASSES
    h, w = image_np.shape[:2]
    results = yolo.track(
        image_np, conf=conf, imgsz=imgsz, persist=True,
        tracker='bytetrack.yaml', verbose=False)[0]

    raw_candidates = []
    ids = results.boxes.id
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        if cls_id not in class_map:
            continue
        name, _ = class_map[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name == 'Traffic Light' and not _plausible_traffic_light_shape(x1, y1, x2, y2):
            continue

        name = _disambiguate_vehicle_class(name, x1, y1, x2, y2, w, h)
        track_id = int(ids[i]) if ids is not None else None
        raw_candidates.append({
            'name': name,
            'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2],
            'track_id': track_id,
        })

    filtered_dets = _merge_overlapping_traffic_lights(raw_candidates)
    filtered_dets = _filter_duplicate_vehicle_boxes(filtered_dets)

    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    counts = {}
    track_ids = set()
    high_risk = False
    det_list = []

    for item in filtered_dets:
        name = item['name']
        conf_s = item['conf']
        x1, y1, x2, y2 = item['box']
        track_id = item.get('track_id')
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map, class_name=name)

        if name in COLLISION_OBSTACLE_CLASSES and risk == 'HIGH':
            high_risk = True

        id_bit = f'#{track_id} ' if track_id is not None else ''
        label = f'{name} {id_bit}· {risk}'
        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        if track_id is not None:
            track_ids.add(track_id)
        det_list.append({
            'name': name,
            'conf': conf_s,
            'box': [x1, y1, x2, y2],
            'risk': risk,
            'label': label,
            'track_id': track_id,
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
        label = d.get('label') or f"{d['name']} · {d.get('risk', 'LOW')}"
        _draw_detection(img_bgr, x1, y1, x2, y2, label, d.get('risk', 'LOW'))
    return cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)


def detect_single_class(yolo, image_rgb, conf, imgsz, label_prefix, color,
                        min_y_ratio=0.35):
    """
    Run a single-class YOLO detector (e.g. pothole detector) and draw
    boxes onto image_rgb. Includes road plane ROI check to avoid sky/tree false positives.
    """
    results = yolo(image_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    h = image_rgb.shape[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_s = float(box.conf[0])

        # Roadway surface constraint: potholes must sit on the road surface
        if min_y_ratio is not None and y2 < int(h * min_y_ratio):
            continue

        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_rgb, f'{label_prefix} {conf_s:.0%}',
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out.append({
            'name': label_prefix,
            'conf': round(conf_s, 3),
            'box': [x1, y1, x2, y2],
            'risk': 'MEDIUM',
        })
    return image_rgb, out
