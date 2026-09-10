"""
UNLET-ADAS: Road Sign Detection Drawing
==========================================
Rendering helper for the dedicated road-sign detector (see
src/train_signs.py — COCO/YOLOv8 has no generic road-sign classes, so
this runs as its own model, separate from the main ADAS detector and
the single-class pothole detector).

Kept as a small shared module, rather than inlined at each call site
like the single-class pothole drawing code: with the dataset's actual
class list running into the dozens (see src/train_signs.py), keeping
the color logic in one place matters more than it would for a small
fixed set.

Colors are assigned by hashing the class name rather than a hardcoded
per-class dict: the dataset this project trains against (currently
indiantrafficsigns/indian-traffic-signs1, see src/train_signs.py)
turned out to be a large, real-world taxonomy of specific sign types
rather than a small handful of English labels, so the class list
can't be enumerated by name up front. Hashing still gives each class
a distinct, stable color across runs without needing to know the
class list ahead of time.
"""

import hashlib

import cv2

# BGR (OpenCV drawing convention). A small palette of visually distinct
# colors; classes are mapped onto it deterministically by name so the
# same class always gets the same color across runs and processes.
_PALETTE = [
    (0, 0, 220),      # red
    (220, 130, 0),    # blue
    (0, 200, 200),    # yellow
    (0, 165, 255),    # orange
    (180, 0, 180),    # magenta
    (0, 180, 0),      # green
    (200, 200, 0),    # cyan
    (128, 0, 255),    # pink
    (0, 100, 180),    # brown-ish
    (255, 0, 0),      # deep blue
    (100, 255, 100),  # light green
    (0, 0, 130),      # dark red
]


def _color_for_class(cls_name):
    """Deterministic BGR color for a class name. Uses md5 rather than
    the builtin hash(), which is randomized per-process by default and
    would make colors flicker between runs of the same app."""
    digest = hashlib.md5(str(cls_name).encode('utf-8')).digest()
    return _PALETTE[digest[0] % len(_PALETTE)]


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
        color = _color_for_class(cls_name)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr, f'{cls_name} {conf:.0%}', (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        count += 1

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), count
