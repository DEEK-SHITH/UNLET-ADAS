"""
UNLET-ADAS: Lane Detection
=============================
Classical computer-vision lane detection (Canny edges + a road-shaped
region of interest + probabilistic Hough transform), run on the
enhanced frame. No training data required, which keeps it consistent
with the rest of the pipeline's lightweight, real-time design —
unlike object/sign detection, lane markings are geometric, high-
contrast lines that classical edge detection handles well once the
frame has been brightened.
"""

import cv2
import numpy as np


def _region_of_interest(edges):
    """Mask everything outside a trapezoid covering the road ahead."""
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0,            h),
        (w,            h),
        (int(w * 0.58), int(h * 0.55)),
        (int(w * 0.42), int(h * 0.55)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def _average_slope_line(lines, h, y_top_frac=0.55):
    """
    Collapse a cluster of nearby Hough segments (all left-lane or
    all right-lane) into one representative line, extrapolated from
    the bottom of the frame up to y_top_frac * height.
    """
    if not lines:
        return None
    slopes, intercepts = [], []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        slopes.append(slope)
        intercepts.append(intercept)
    if not slopes:
        return None
    # Median, not mean — a couple of noisy/outlier Hough segments
    # (common on faint night-time markings) shouldn't swing the
    # extrapolated line's vanishing point.
    slope, intercept = np.median(slopes), np.median(intercepts)
    y1 = h
    y2 = int(h * y_top_frac)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)


def detect_lanes(frame_rgb, canny_lo=50, canny_hi=150,
                  hough_threshold=25, min_line_len=30, max_line_gap=80):
    """
    Detect left/right lane boundary lines on an (already enhanced)
    RGB frame.

    Returns (left_line, right_line), each either None or an
    (x1, y1, x2, y2) tuple in pixel coordinates.
    """
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    roi_edges = _region_of_interest(edges)

    segments = cv2.HoughLinesP(
        roi_edges, rho=2, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_len, maxLineGap=max_line_gap)

    left, right = [], []
    if segments is not None:
        for seg in segments:
            x1, y1, x2, y2 = seg.reshape(-1)
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.35:      # near-horizontal — not a lane edge
                continue
            (left if slope < 0 else right).append((x1, y1, x2, y2))

    return (_average_slope_line(left, h), _average_slope_line(right, h))


def draw_lanes(frame_rgb, left_line, right_line,
               color=(0, 255, 60), thickness=6, fill_alpha=0.25):
    """Overlay detected lane lines (and the lane area, if both found) on frame_rgb."""
    overlay = frame_rgb.copy()

    if left_line is not None:
        cv2.line(overlay, left_line[:2], left_line[2:], color, thickness)
    if right_line is not None:
        cv2.line(overlay, right_line[:2], right_line[2:], color, thickness)

    if left_line is not None and right_line is not None:
        lane_poly = np.array([[
            left_line[0:2], left_line[2:4],
            right_line[2:4], right_line[0:2],
        ]], dtype=np.int32)
        fill = frame_rgb.copy()
        cv2.fillPoly(fill, lane_poly, color)
        overlay = cv2.addWeighted(fill, fill_alpha, overlay, 1 - fill_alpha, 0)

    return overlay
