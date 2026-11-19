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

from collections import Counter, deque
from src.depth import classify_proximity, sample_proximity

# Sliding-window class history per ByteTrack ID to prevent label flipping (Truck / Bus / Car)
_TRACK_CLASS_HISTORY: dict[int, deque] = {}
_MAX_HISTORY_LEN = 7


# Bounding-box height history per track ID for Optical Time-To-Collision (TTC) estimation
_TRACK_BBOX_HISTORY: dict[int, deque] = {}

def estimate_ttc(track_id: int, y1: int, y2: int, fps: float = 30.0) -> tuple[str, float]:
    """
    Estimates Time-To-Collision (TTC) using the optical expansion rate of bounding box height.
    Returns: (risk_level, ttc_seconds)
    """
    if track_id is None:
        return 'LOW', 99.0
    
    h = max(y2 - y1, 1)
    if track_id not in _TRACK_BBOX_HISTORY:
        _TRACK_BBOX_HISTORY[track_id] = deque(maxlen=8)
    
    _TRACK_BBOX_HISTORY[track_id].append(h)
    history = _TRACK_BBOX_HISTORY[track_id]
    
    if len(history) >= 4:
        dh = history[-1] - history[0]
        dt = (len(history) - 1) / max(fps, 1.0)
        growth_rate = dh / max(dt, 0.01)
        if growth_rate > 8.0:
            ttc = h / max(growth_rate, 0.1)
            if ttc < 2.5:
                return 'HIGH', round(ttc, 1)
            elif ttc < 4.5:
                return 'MEDIUM', round(ttc, 1)
    return 'LOW', 99.0


# Known real-world obstacle physical widths (in meters) for monocular pinhole distance estimation
_REAL_WORLD_WIDTHS_M = {
    'Car': 1.82,
    'Bus': 2.55,
    'Truck': 2.50,
    'Auto-Rickshaw': 1.30,
    'Motorcycle': 0.80,
    'Bicycle': 0.65,
    'Person': 0.52,
    'Cattle / Cow': 1.65,
    'Cow': 1.65,
    'Dog': 0.45,
    'Cat': 0.35,
    'Horse': 1.50,
    'Sheep': 0.65,
    'Elephant': 2.80,
    'Bird': 0.28,
    'Traffic Light': 0.40,
    'Stop Sign': 0.75,
}

def estimate_metric_distance(x1: int, y1: int, x2: int, y2: int,
                             frame_w: int, frame_h: int,
                             class_name: str = 'Car',
                             camera_height_m: float = 1.30,
                             horizon_ratio: float = 0.50,
                             focal_length_px: float = None) -> float:
    """
    Computes true metric physical distance (in meters) using ground-plane inverse perspective
    geometry and class-specific optical bounding box priors. Costs $0 without needing LiDAR.
    """
    if focal_length_px is None:
        focal_length_px = float(frame_w) * 0.88
    
    box_w = max(x2 - x1, 1)
    real_w = _REAL_WORLD_WIDTHS_M.get(class_name, 1.80)
    d_pinhole = float((focal_length_px * real_w) / box_w)
    
    y_horizon = frame_h * horizon_ratio
    dy = max(y2 - y_horizon, 2.0)
    d_ground = float(camera_height_m * (focal_length_px / dy))
    
    # Fused estimate: ground-plane for nearby objects, pinhole width for distant objects
    ground_weight = float(np.clip((y2 - y_horizon) / (0.35 * max(frame_h, 1)), 0.20, 0.80))
    dist_m = ground_weight * d_ground + (1.0 - ground_weight) * d_pinhole
    return float(np.clip(dist_m, 1.5, 120.0))


def reset_track_history():
    """Reset temporal tracking history (called when starting a new video/stream)."""
    global _TRACK_CLASS_HISTORY, _TRACK_BBOX_HISTORY
    _TRACK_CLASS_HISTORY.clear()
    _TRACK_BBOX_HISTORY.clear()

def _get_stabilized_class(track_id: int, current_name: str) -> str:
    """
    Sliding-window majority voting over past frames for active ByteTrack IDs.
    Eliminates rapid single-frame label toggles (e.g. Truck vs Bus vs Car).
    """
    if track_id is None:
        return current_name
    if track_id not in _TRACK_CLASS_HISTORY:
        _TRACK_CLASS_HISTORY[track_id] = deque(maxlen=_MAX_HISTORY_LEN)
    _TRACK_CLASS_HISTORY[track_id].append(current_name)
    counts = Counter(_TRACK_CLASS_HISTORY[track_id])
    most_common_class, most_common_count = counts.most_common(1)[0]
    if most_common_count >= 2 or len(_TRACK_CLASS_HISTORY[track_id]) <= 2:
        return most_common_class
    return current_name


ADAS_CLASSES = {
    0:  ('Person',        (50,  205,  50)),   # Lime Green
    1:  ('Bicycle',       (255, 165,   0)),   # Amber Orange
    2:  ('Car',           (30,  144, 255)),   # Dodger Blue
    3:  ('Motorcycle',    (255, 100, 100)),   # Light Coral
    5:  ('Bus',           (138,  43, 226)),   # Blue Violet
    6:  ('Auto-Rickshaw', (255, 191,   0)),   # Warm Amber / Gold (COCO train mislabel on roads)
    7:  ('Truck',         (255,  20, 147)),   # Deep Pink
    9:  ('Traffic Light', (255, 215,   0)),   # Golden Yellow
    11: ('Stop Sign',     (220,  20,  60)),   # Crimson Red
    14: ('Bird',          (0,   206, 209)),   # Dark Turquoise
    15: ('Cat',           (255, 182, 193)),   # Light Pink
    16: ('Dog',           (255, 140,   0)),   # Dark Orange (Stray dog)
    17: ('Horse',         (160,  82,  45)),   # Saddle Brown
    18: ('Sheep',         (240, 230, 140)),   # Khaki (Sheep / Goat)
    19: ('Cattle / Cow',  (205, 133,  63)),   # Peru Brown (Indian cattle / cow / bull)
    20: ('Elephant',      (112, 128, 144)),   # Slate Gray
}

LOWLIGHT_CLASSES = {
    0:  ('Person',        (50,  205,  50)),
    1:  ('Bicycle',       (255, 165,   0)),
    2:  ('Car',           (30,  144, 255)),
    3:  ('Motorcycle',    (255, 100, 100)),
    4:  ('Bus',           (138,  43, 226)),
    6:  ('Auto-Rickshaw', (255, 191,   0)),
    14: ('Bird',          (0,   206, 209)),
    15: ('Cat',           (255, 182, 193)),
    16: ('Dog',           (255, 140,   0)),
    17: ('Horse',         (160,  82,  45)),
    18: ('Sheep',         (240, 230, 140)),
    19: ('Cattle / Cow',  (205, 133,  63)),
    20: ('Elephant',      (112, 128, 144)),
}

# Physical obstacle classes capable of presenting a vehicular collision hazard
COLLISION_OBSTACLE_CLASSES = {
    'Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck', 'Auto-Rickshaw',
    'Cattle / Cow', 'Cow', 'Dog', 'Cat', 'Horse', 'Sheep', 'Elephant', 'Bird', 'Pothole'
}

# Regulatory/traffic signals (informational, not collision impact hazards)
SIGNAL_CLASSES = {
    'Traffic Light', 'Stop Sign'
}

RISK_COLORS = {
    'HIGH':   (220,  20,  60),   # Crimson Red
    'MEDIUM': (255, 165,   0),   # Amber Orange
    'LOW':    (50,  205,  50),   # Lime Green
    'SAFE':   (50,  205,  50),   # Lime Green
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
    return (w / h) <= 0.75


def _is_rear_cargo_truck(patch):
    """
    Identifies cargo trucks / goods carriers (e.g., Tata, Ashok Leyland, box trucks)
    that COCO YOLO frequently confuses with 'Bus' from behind due to rectangular rear bodies.
    Trucks feature distinct hanging mudflaps, high undercarriage clearance (axle gap),
    or square/wide cargo tailboards.
    """
    if patch is None or patch.size == 0:
        return False
    h, w = patch.shape[:2]
    if h < 30 or w < 30:
        return False
    aspect = w / h
    # Commercial cargo trucks have squarish rear bodies (aspect ratio ~0.80 - 1.35)
    # Transit buses are significantly taller and narrower (w/h typically < 0.82)
    if aspect < 0.80 or aspect > 1.35:
        return False
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    bottom_strip = gray[int(0.82 * h):, :]
    if bottom_strip.size == 0:
        return False
    col_means = np.mean(bottom_strip, axis=0)
    col_var = float(np.var(col_means))
    sobel_h = cv2.Sobel(gray[int(0.40 * h):, :], cv2.CV_64F, 0, 1, ksize=3)
    h_edge = float(np.mean(np.abs(sobel_h)))
    if col_var >= 800.0 or (aspect >= 0.88 and h_edge >= 35.0):
        return True
    return False


def _is_auto_rickshaw(patch, box_w, box_h, original_class, conf=1.0, frame_w=1920):
    """
    Identifies Indian Auto-Rickshaws (3-wheelers / Bajaj RE, TVS King, Ape, e-rickshaws)
    that standard COCO YOLO misclassifies as 'Truck' (or 'Car'/'Motorcycle').

    Distinguishing features of Indian auto-rickshaws:
    1. Aspect Ratio: Tall, upright, compact profile. Width/Height between 0.60 and 1.35.
       (Standard cars have W/H > 1.30; freight trucks have wide chassis > 2.2m with dual wheels).
    2. Color Palette: Distinctive high-visibility livery across Indian cities:
       - Top canopy: Bright auto yellow, green, or black canvas hood.
       - Lower body: Dark CNG green (Delhi/Gujarat/Mumbai) or jet black (Mumbai/Kolkata/Hyderabad)
         or bright yellow (Bangalore/Chennai/AP).
    3. Structural Silhouette: Tapered upper canopy, open-side passenger cabin,
       and absence of heavy dual-axle freight truck clearance.
    """
    if box_h <= 0 or box_w <= 0:
        return False

    aspect = box_w / float(box_h)

    # Auto-rickshaws strictly fall within 0.60 <= W/H <= 1.35
    if aspect < 0.60 or aspect > 1.35:
        return False

    # A massive vehicle (> 50% of frame width) is a heavy bus or truck right in front
    if box_w > 0.50 * frame_w:
        return False

    has_rickshaw_colors = False

    if patch is not None and patch.size >= 100:
        h_p, w_p = patch.shape[:2]
        # Check for heavy multi-axle freight cargo truck features
        if _is_rear_cargo_truck(patch):
            return False

        # Color analysis in HSV
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV) if len(patch.shape) == 3 else None
        if hsv is not None:
            H = hsv[:, :, 0]
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]

            # Yellow mask (Auto-rickshaw yellow canopy or yellow body): H in [14, 38], S >= 35, V >= 40
            yellow_mask = (H >= 14) & (H <= 38) & (S >= 35) & (V >= 40)
            yellow_ratio = float(np.mean(yellow_mask))

            # Green mask (CNG Auto green body): H in [38, 85], S >= 28, V >= 30
            green_mask = (H >= 38) & (H <= 85) & (S >= 28) & (V >= 30)
            green_ratio = float(np.mean(green_mask))

            # Dark lower chassis / black hood: V < 45
            dark_ratio = float(np.mean(V < 45))

            # Upper canopy yellow (top 45% of patch)
            top_hsv = hsv[:int(0.45 * h_p), :]
            top_yellow = float(np.mean((top_hsv[:, :, 0] >= 14) & (top_hsv[:, :, 0] <= 38) &
                                       (top_hsv[:, :, 1] >= 35) & (top_hsv[:, :, 2] >= 40))) if top_hsv.size > 0 else 0.0

            if yellow_ratio >= 0.04 or green_ratio >= 0.045 or top_yellow >= 0.04:
                has_rickshaw_colors = True
            elif (yellow_ratio + green_ratio) >= 0.035 and dark_ratio >= 0.20:
                has_rickshaw_colors = True

    # Disambiguation logic based on COCO YOLO initial prediction:
    if original_class == 'Truck':
        # COCO consistently mislabels auto-rickshaws as 'Truck' due to their open rear and boxy canopy.
        # If it has rickshaw colors, or if it is an upright/compact vehicle (0.62 <= W/H <= 1.25)
        # without heavy truck cargo rear characteristics:
        if has_rickshaw_colors:
            return True
        if 0.62 <= aspect <= 1.25 and box_w <= 0.45 * frame_w:
            return True

    elif original_class == 'Car':
        # Cars have low, wide aspect ratios (typically > 1.25). An upright box with rickshaw colors is an auto.
        if 0.65 <= aspect <= 1.18 and has_rickshaw_colors:
            return True

    elif original_class == 'Motorcycle':
        # Motorcycles are very narrow (aspect < 0.60). A wide vehicle with rickshaw colors or canopy is an auto.
        if aspect >= 0.65 and (has_rickshaw_colors or box_w >= 50):
            return True

    return False


def _disambiguate_vehicle_class(name, x1, y1, x2, y2, frame_w, frame_h, conf=1.0, image_np=None):
    """
    Resolves vehicle class ambiguities in automotive dashcam perspectives:
    1. Truncated passenger car hoods at the bottom edge (misleadingly wide, conf < 0.50) -> 'Car'.
    2. Indian Auto-Rickshaws misclassified by COCO YOLO as 'Truck', 'Car', or 'Motorcycle' -> 'Auto-Rickshaw'.
    3. Indian cargo trucks / goods carriers confused by COCO YOLO as 'Bus' -> 'Truck'.
    4. Distant/mid-range buses and trucks driving down the road -> preserved strictly.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    if name in ('Bus', 'Truck'):
        is_bottom_truncated = (y2 >= 0.88 * frame_h)
        is_flat_slice = (box_h > 0 and box_w >= 2.8 * box_h)
        is_marginal_conf = (conf < 0.50)
        if is_bottom_truncated and is_flat_slice and is_marginal_conf:
            return 'Car'

    patch = None
    if image_np is not None:
        px1, py1 = max(0, x1), max(0, y1)
        px2, py2 = min(frame_w, x2), min(frame_h, y2)
        if px2 > px1 and py2 > py1:
            patch = image_np[py1:py2, px1:px2]

    # Auto-Rickshaw disambiguation
    if name in ('Truck', 'Car', 'Motorcycle'):
        if _is_auto_rickshaw(patch, box_w, box_h, name, conf=conf, frame_w=frame_w):
            return 'Auto-Rickshaw'

    # Indian cargo trucks confused by COCO YOLO as 'Bus'
    if name == 'Bus' and patch is not None:
        if _is_rear_cargo_truck(patch):
            return 'Truck'

    return name


def _disambiguate_animal_class(name, x1, y1, x2, y2, frame_w, frame_h, conf=1.0):
    """
    Resolves animal taxonomy in Indian road environments:
    - Stray dogs moving on Indian roads are commonly mislabeled as 'Sheep' or 'Cat' by COCO YOLO.
    - Large cattle (cows, bulls, buffaloes) are standardized to 'Cattle / Cow'.
    """
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    aspect = box_w / float(box_h)

    if name in ('Sheep', 'Cat'):
        # On Indian paved roadways, free-roaming medium quadrupeds are stray dogs
        if 0.70 <= aspect <= 2.30:
            return 'Dog'
    elif name == 'Cow':
        return 'Cattle / Cow'
    return name


def _merge_overlapping_traffic_lights(det_list):
    """
    Merges duplicate or multi-aspect traffic light bounding boxes (such as top
    red/yellow signal heads and bottom green directional arrows detected as separate
    overlapping or contiguous boxes on the same fixture) into a single cohesive
    bounding box with the maximum confidence of the fixture.
    """
    tl_items = [d for d in det_list if d['name'] == 'Traffic Light']
    if not tl_items:
        return det_list

    non_tl = [d for d in det_list if d['name'] != 'Traffic Light']
    confirmed_tls = [d for d in tl_items if d.get('confirmed', True)]
    aux_tls = [d for d in tl_items if not d.get('confirmed', True)]

    # If no confirmed traffic light passes the threshold, drop auxiliary candidates
    if not confirmed_tls:
        return non_tl

    merged_tl = []
    for i, c_tl in enumerate(confirmed_tls):
        x1, y1, x2, y2 = c_tl['box']
        conf = c_tl['conf']
        track_id = c_tl.get('track_id')

        # 1. Merge with any other confirmed traffic lights on the same pole
        for j, o_tl in enumerate(confirmed_tls):
            if j <= i:
                continue
            ox1, oy1, ox2, oy2 = o_tl['box']
            cx, ocx = (x1 + x2) / 2, (ox1 + ox2) / 2
            bw, obw = x2 - x1, ox2 - ox1
            bh, obh = y2 - y1, oy2 - oy1
            h_overlap = min(x2, ox2) - max(x1, ox1)
            v_gap = max(0, max(y1, oy1) - min(y2, oy2))
            if (h_overlap > 0.20 * min(bw, obw) or abs(cx - ocx) < 0.60 * max(bw, obw)) and v_gap < 0.50 * max(bh, obh):
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
                conf = max(conf, o_tl['conf'])
                if track_id is None:
                    track_id = o_tl.get('track_id')

        # 2. Merge with any auxiliary candidates (e.g. green arrows or full fixture sub-detections)
        for k, a_tl in enumerate(aux_tls):
            ax1, ay1, ax2, ay2 = a_tl['box']
            cx, acx = (x1 + x2) / 2, (ax1 + ax2) / 2
            bw, abw = x2 - x1, ax2 - ax1
            bh, abh = y2 - y1, ay2 - ay1
            h_overlap = min(x2, ax2) - max(x1, ax1)
            v_gap = max(0, max(y1, ay1) - min(y2, ay2))
            if (h_overlap > 0.20 * min(bw, abw) or abs(cx - acx) < 0.65 * max(bw, abw)) and v_gap < 0.60 * max(bh, abh):
                x1 = min(x1, ax1)
                y1 = min(y1, ay1)
                x2 = max(x2, ax2)
                y2 = max(y2, ay2)
                conf = max(conf, a_tl['conf'])
                if track_id is None:
                    track_id = a_tl.get('track_id')

        merged_item = dict(c_tl)
        merged_item['box'] = [x1, y1, x2, y2]
        merged_item['conf'] = conf
        if track_id is not None:
            merged_item['track_id'] = track_id
        merged_tl.append(merged_item)

    # De-duplicate any confirmed merged items that now encompass each other
    final_tl = []
    used_tl = set()
    for i, m1 in enumerate(merged_tl):
        if i in used_tl:
            continue
        bx1, by1, bx2, by2 = m1['box']
        m_conf = m1['conf']
        m_tid = m1.get('track_id')
        for j in range(i + 1, len(merged_tl)):
            if j in used_tl:
                continue
            m2 = merged_tl[j]
            obx1, oby1, obx2, oby2 = m2['box']
            inter = max(0, min(bx2, obx2) - max(bx1, obx1)) * max(0, min(by2, oby2) - max(by1, oby1))
            a1 = (bx2 - bx1) * (by2 - by1)
            a2 = (obx2 - obx1) * (oby2 - oby1)
            if inter > 0.35 * min(a1, a2):
                bx1 = min(bx1, obx1)
                by1 = min(by1, oby1)
                bx2 = max(bx2, obx2)
                by2 = max(by2, oby2)
                m_conf = max(m_conf, m2['conf'])
                if m_tid is None:
                    m_tid = m2.get('track_id')
                used_tl.add(j)
        used_tl.add(i)
        f_item = dict(m1)
        f_item['box'] = [bx1, by1, bx2, by2]
        f_item['conf'] = m_conf
        if m_tid is not None:
            f_item['track_id'] = m_tid
        final_tl.append(f_item)

    return non_tl + final_tl

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
            if d1['name'] in ('Car', 'Bus', 'Truck', 'Auto-Rickshaw') and d2['name'] in ('Car', 'Bus', 'Truck', 'Auto-Rickshaw'):
                if _box_iou(d1['box'], d2['box']) > 0.40:
                    suppressed.add(j)

    return keep


def _filter_duplicate_animal_boxes(det_list):
    """
    Suppresses redundant overlapping bounding boxes for animals
    (e.g., both Dog and Sheep predicted on the same stray dog).
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

    sorted_dets = sorted(det_list, key=lambda d: d.get('conf', 0), reverse=True)
    suppressed = set()
    animal_set = {'Dog', 'Sheep', 'Cattle / Cow', 'Cow', 'Cat', 'Horse', 'Bird'}

    keep = []
    for i, d1 in enumerate(sorted_dets):
        if i in suppressed:
            continue
        keep.append(d1)
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            d2 = sorted_dets[j]
            if d1['name'] in animal_set and d2['name'] in animal_set:
                if _box_iou(d1['box'], d2['box']) > 0.35:
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
    scan_conf = min(conf, 0.12)
    results = yolo(image_np, conf=scan_conf, imgsz=imgsz, verbose=False)[0]

    raw_candidates = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in class_map:
            continue
        name, _ = class_map[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name == 'Traffic Light':
            if not _plausible_traffic_light_shape(x1, y1, x2, y2):
                continue
            raw_candidates.append({
                'name': name,
                'conf': round(conf_s, 3),
                'box': [x1, y1, x2, y2],
                'confirmed': conf_s >= conf,
            })
        else:
            is_animal = name in ('Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle / Cow', 'Elephant')
            # Street animals and road surface hazards use higher sensitivity (lower threshold)
            threshold = min(conf, 0.18) if is_animal else conf
            if conf_s < threshold:
                continue
            name = _disambiguate_vehicle_class(name, x1, y1, x2, y2, w, h, conf=conf_s, image_np=image_np)
            name = _disambiguate_animal_class(name, x1, y1, x2, y2, w, h, conf=conf_s)
            raw_candidates.append({
                'name': name,
                'conf': round(conf_s, 3),
                'box': [x1, y1, x2, y2],
                'confirmed': True,
            })

    # Coherence filters: merge fragmented traffic lights and remove overlapping duplicate boxes
    filtered_dets = _merge_overlapping_traffic_lights(raw_candidates)
    filtered_dets = _filter_duplicate_vehicle_boxes(filtered_dets)
    filtered_dets = _filter_duplicate_animal_boxes(filtered_dets)

    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    counts = {}
    det_list = []
    high_risk = False

    for item in filtered_dets:
        name = item['name']
        conf_s = item['conf']
        x1, y1, x2, y2 = item['box']
        dist_m = estimate_metric_distance(x1, y1, x2, y2, w, h, class_name=name)
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map, class_name=name)

        # Automatic metric distance risk escalation
        if name in COLLISION_OBSTACLE_CLASSES:
            cx = (x1 + x2) / 2
            in_ego = 0.20 * w <= cx <= 0.80 * w
            if dist_m < 8.5 and in_ego:
                risk = 'HIGH'
            elif dist_m < 22.0 and risk == 'LOW' and in_ego:
                risk = 'MEDIUM'

        if name in COLLISION_OBSTACLE_CLASSES and risk == 'HIGH':
            high_risk = True

        dist_str = f'{dist_m:.1f}m'
        label = f'{name} · {dist_str} · {risk}'
        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        det_list.append({
            'name': name,
            'conf': conf_s,
            'box': [x1, y1, x2, y2],
            'risk': risk,
            'dist_m': round(dist_m, 1),
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
    scan_conf = min(conf, 0.12)
    results = yolo.track(
        image_np, conf=scan_conf, imgsz=imgsz, persist=True,
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

        if name == 'Traffic Light':
            if not _plausible_traffic_light_shape(x1, y1, x2, y2):
                continue
            track_id = int(ids[i]) if ids is not None else None
            raw_candidates.append({
                'name': name,
                'conf': round(conf_s, 3),
                'box': [x1, y1, x2, y2],
                'track_id': track_id,
                'confirmed': conf_s >= conf,
            })
        else:
            is_animal = name in ('Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle / Cow', 'Elephant')
            threshold = min(conf, 0.18) if is_animal else conf
            if conf_s < threshold:
                continue
            name = _disambiguate_vehicle_class(name, x1, y1, x2, y2, w, h, conf=conf_s, image_np=image_np)
            name = _disambiguate_animal_class(name, x1, y1, x2, y2, w, h, conf=conf_s)
            track_id = int(ids[i]) if ids is not None else None
            raw_candidates.append({
                'name': name,
                'conf': round(conf_s, 3),
                'box': [x1, y1, x2, y2],
                'track_id': track_id,
                'confirmed': True,
            })

    filtered_dets = _merge_overlapping_traffic_lights(raw_candidates)
    filtered_dets = _filter_duplicate_vehicle_boxes(filtered_dets)
    filtered_dets = _filter_duplicate_animal_boxes(filtered_dets)

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
        if track_id is not None:
            name = _get_stabilized_class(track_id, name)
            item['name'] = name
        dist_m = estimate_metric_distance(x1, y1, x2, y2, w, h, class_name=name)
        risk = estimate_risk(x1, y1, x2, y2, w, h, depth_map=depth_map, class_name=name)
        ttc_sec = 99.0
        if track_id is not None and name in COLLISION_OBSTACLE_CLASSES:
            ttc_risk, ttc_sec = estimate_ttc(track_id, y1, y2, fps=30.0)
            if ttc_risk == 'HIGH':
                risk = 'HIGH'

        # Automatic metric distance risk escalation
        if name in COLLISION_OBSTACLE_CLASSES:
            cx = (x1 + x2) / 2
            in_ego = 0.20 * w <= cx <= 0.80 * w
            if dist_m < 8.5 and in_ego:
                risk = 'HIGH'
            elif dist_m < 22.0 and risk == 'LOW' and in_ego:
                risk = 'MEDIUM'

        if name in COLLISION_OBSTACLE_CLASSES and risk == 'HIGH':
            high_risk = True

        dist_str = f'{dist_m:.1f}m'
        id_bit = f'#{track_id} ' if track_id is not None else ''
        if ttc_sec < 4.5:
            label = f'{name} {id_bit}· {dist_str} (TTC {ttc_sec:.1f}s) · {risk}'
        else:
            label = f'{name} {id_bit}· {dist_str} · {risk}'

        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        if track_id is not None:
            track_ids.add(track_id)
        det_list.append({
            'name': name,
            'conf': conf_s,
            'box': [x1, y1, x2, y2],
            'risk': risk,
            'dist_m': round(dist_m, 1),
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
