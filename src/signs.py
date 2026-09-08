"""
UNLET-ADAS: Road Sign Detection Drawing
==========================================
Rendering helper for the dedicated road-sign detector (see
src/train_signs.py — COCO/YOLOv8 has no generic road-sign classes, so
this runs as its own model, separate from the main ADAS detector and
the single-class pothole detector).

Kept as a small shared module, rather than inlined at each call site
like the single-class pothole drawing code: with 4 distinct sign
classes needing their own colors, duplicating that per call site
(Image tab, Video tab, Live tab) would drift out of sync over time.
"""

import cv2

# BGR (OpenCV drawing convention), chosen for visual distinction from
# the main detector's default green boxes and the pothole detector's
# orange boxes.
SIGN_CLASS_COLORS = {
    'stop':         (0, 0, 220),      # red
    'speedlimit':   (220, 130, 0),    # blue
    'crosswalk':    (0, 200, 200),    # yellow
    'trafficlight': (0, 165, 255),    # orange
}
DEFAULT_COLOR = (200, 200, 200)  # gray, for any class not in the map above


def draw_sign_detections(frame_rgb, results, class_names):
    """
    Draw the sign detector's boxes (with per-class colors and
    "<class> <conf%>" labels) onto an RGB frame.

    frame_rgb    : (H,W,3) uint8 RGB array to draw onto (mutated via a
                   BGR round-trip, same pattern as the rest of this
                   codebase's cv2 drawing calls).
    results      : one ultralytics Results object (results[0] from a
                   YOLO(...) call).
    class_names  : the model's names dict/list (results.names), so
                   labels/colors follow the class the model actually
                   predicted rather than a hardcoded index assumption.

    Returns (annotated_frame_rgb, count).
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    count = 0
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = class_names.get(cls_id, str(cls_id)) \
            if isinstance(class_names, dict) else class_names[cls_id]
        color = SIGN_CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr, f'{cls_name} {conf:.0%}', (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        count += 1

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), count
