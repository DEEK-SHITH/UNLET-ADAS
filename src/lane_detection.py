"""
UNLET-ADAS: Lane Detection & Lane Departure Warning (LDW)
=========================================================
Classical computer-vision lane detection (Canny edges + a road-shaped
region of interest + probabilistic Hough transform + 2nd order polynomial
curve fitting), run on the enhanced frame. Includes a real-time Lane
Departure Warning (LDW) module that monitors vehicle lateral drift
within the lane corridor and alerts the driver.
"""

import cv2
import numpy as np


def estimate_dashboard_cutoff(frames_rgb, min_static_frac=0.12,
                               max_static_frac=0.6, var_thresh=6.0):
    """
    Auto-detect a static dashboard/steering-wheel region at the bottom
    of the frame from a short sequence of consecutive frames.
    """
    if len(frames_rgb) < 3:
        return 1.0
    stack = np.stack([f.astype(np.float32) for f in frames_rgb], axis=0)
    row_var = stack.var(axis=0).mean(axis=(1, 2))
    h = row_var.shape[0]

    cutoff_row = h
    for y in range(h - 1, -1, -1):
        if row_var[y] < var_thresh:
            cutoff_row = y
        else:
            break

    exclude_frac = (h - cutoff_row) / h
    if exclude_frac < min_static_frac or exclude_frac > max_static_frac:
        return 1.0
    return cutoff_row / h


def _region_of_interest(edges, roi_top=0.55, roi_bottom=1.0):
    """Mask everything outside a trapezoid covering the road ahead."""
    h, w = edges.shape
    mask = np.zeros_like(edges)
    y_top, y_bottom = int(h * roi_top), int(h * roi_bottom)
    polygon = np.array([[
        (0,             y_bottom),
        (w,             y_bottom),
        (int(w * 0.58), y_top),
        (int(w * 0.42), y_top),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def _fit_lane_curve(lines, h, y_top_frac=0.55, y_bottom_frac=1.0, num_pts=24):
    """
    Fits a smooth 2nd-order polynomial curve (x = a*y^2 + b*y + c) to Hough
    line segments. Returns (x1, y1, x2, y2, curve_pts), where curve_pts
    is an (N, 2) int32 array for smooth polyline and corridor drawing.
    Falls back to robust linear median extrapolation if points are collinear
    or fewer than 3 segments exist.
    """
    if not lines:
        return None

    all_x = []
    all_y = []
    min_y = h
    slopes, intercepts = [], []

    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        slopes.append(slope)
        intercepts.append(intercept)
        min_y = min(min_y, y1, y2)
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])

    if not slopes or not all_x:
        return None

    y1 = int(h * y_bottom_frac)
    y2 = max(int(h * y_top_frac), min_y)
    y_samples = np.linspace(y1, y2, num_pts)

    curve_pts = None
    if len(all_x) >= 6:
        try:
            poly = np.polyfit(all_y, all_x, deg=2)
            # Sanity check curvature: second derivative 2*poly[0] should be moderate
            if abs(poly[0]) < 0.003:
                x_samples = np.polyval(poly, y_samples)
                curve_pts = np.stack([x_samples.astype(np.int32), y_samples.astype(np.int32)], axis=1)
        except Exception:
            curve_pts = None

    # Linear fallback / baseline
    slope, intercept = np.median(slopes), np.median(intercepts)
    lx1 = int((y1 - intercept) / slope)
    lx2 = int((y2 - intercept) / slope)

    if curve_pts is None:
        x_linear = np.linspace(lx1, lx2, num_pts)
        curve_pts = np.stack([x_linear.astype(np.int32), y_samples.astype(np.int32)], axis=1)

    return (lx1, y1, lx2, y2, curve_pts)


def check_lane_departure(left_lane, right_lane, frame_w, threshold_ratio=0.18):
    """
    Computes lateral ego-vehicle offset relative to the lane center.
    Assumes vehicle center is at frame_w / 2.
    Returns: (status, offset_px, lane_width_px)
      status: 'CENTERED' | 'DRIFT_LEFT' | 'DRIFT_RIGHT' | 'UNKNOWN'
    """
    if left_lane is None or right_lane is None:
        return 'UNKNOWN', 0.0, 0.0

    lx_bot = left_lane[0]
    rx_bot = right_lane[0]
    lane_width = rx_bot - lx_bot
    if lane_width <= 40:
        return 'UNKNOWN', 0.0, 0.0

    lane_center = (lx_bot + rx_bot) / 2.0
    ego_center = frame_w / 2.0
    offset = ego_center - lane_center

    dist_to_left = ego_center - lx_bot
    dist_to_right = rx_bot - ego_center
    thresh = threshold_ratio * lane_width

    if dist_to_left < thresh:
        return 'DRIFT_LEFT', round(offset, 1), round(lane_width, 1)
    elif dist_to_right < thresh:
        return 'DRIFT_RIGHT', round(offset, 1), round(lane_width, 1)

    return 'CENTERED', round(offset, 1), round(lane_width, 1)


class LaneTracker:
    """
    Temporal smoother (Exponential Moving Average + short dropout hold)
    for left/right lane lines and curved polylines across video frames.
    """
    def __init__(self, alpha=0.45, max_missing=4):
        self.alpha = float(alpha)
        self.max_missing = int(max_missing)
        self._left = None
        self._right = None
        self._miss_left = 0
        self._miss_right = 0

    def _update_side(self, current, prev, miss_count):
        if current is not None:
            if prev is None:
                return current, 0
            a = self.alpha
            # Smooth 4-tuple coords
            smoothed_coords = tuple(
                int(round(a * c + (1.0 - a) * p))
                for c, p in zip(current[:4], prev[:4])
            )
            # Smooth curve points if present
            smoothed_pts = None
            if len(current) > 4 and current[4] is not None and len(prev) > 4 and prev[4] is not None:
                if len(current[4]) == len(prev[4]):
                    smoothed_pts = (a * current[4] + (1.0 - a) * prev[4]).astype(np.int32)
                else:
                    smoothed_pts = current[4]
            elif len(current) > 4:
                smoothed_pts = current[4]

            res = smoothed_coords + ((smoothed_pts,) if smoothed_pts is not None else ())
            return res, 0

        if prev is not None and miss_count < self.max_missing:
            return prev, miss_count + 1
        return None, miss_count + 1

    def update(self, left_line, right_line):
        self._left, self._miss_left = self._update_side(
            left_line, self._left, self._miss_left)
        self._right, self._miss_right = self._update_side(
            right_line, self._right, self._miss_right)

        if self._left is not None and self._right is not None:
            if self._left[0] >= self._right[0] or self._left[2] >= self._right[2]:
                self._left, self._right = None, None

        return self._left, self._right

    def reset(self):
        self._left = None
        self._right = None
        self._miss_left = 0
        self._miss_right = 0


def detect_lanes(frame_rgb, canny_lo=50, canny_hi=150,
                  hough_threshold=25, min_line_len=30, max_line_gap=80,
                  roi_top=0.55, roi_bottom=1.0, tracker=None):
    """
    Detect left/right curved lane boundaries on an enhanced RGB frame.
    Returns (left_lane, right_lane), each either None or a tuple:
    (x1, y1, x2, y2, curve_pts).
    """
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    roi_edges = _region_of_interest(edges, roi_top=roi_top, roi_bottom=roi_bottom)

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
            if abs(slope) < 0.35:
                continue
            (left if slope < 0 else right).append((x1, y1, x2, y2))

    left_lane = _fit_lane_curve(left, h, y_top_frac=roi_top, y_bottom_frac=roi_bottom)
    right_lane = _fit_lane_curve(right, h, y_top_frac=roi_top, y_bottom_frac=roi_bottom)

    if left_lane is not None and right_lane is not None:
        if left_lane[0] >= right_lane[0] or left_lane[2] >= right_lane[2]:
            left_lane, right_lane = None, None

    if tracker is not None:
        return tracker.update(left_lane, right_lane)

    return (left_lane, right_lane)


def draw_lanes(frame_rgb, left_lane, right_lane,
               color=(0, 230, 80), thickness=4, fill_alpha=0.14,
               show_ldw=True, frame_idx=0):
    """
    Overlay detected curved lane boundaries and semi-transparent drivable
    navigation corridor on frame_rgb.
    Includes Lane Departure Warning (LDW) HUD alert if vehicle drifts.
    """
    h, w = frame_rgb.shape[:2]
    overlay = frame_rgb.copy()

    ldw_status, offset, lane_w = check_lane_departure(left_lane, right_lane, w)

    # 1. Drivable Corridor Polygon Fill
    if left_lane is not None and right_lane is not None and fill_alpha > 0:
        l_pts = left_lane[4] if len(left_lane) > 4 and left_lane[4] is not None else None
        r_pts = right_lane[4] if len(right_lane) > 4 and right_lane[4] is not None else None

        if l_pts is not None and r_pts is not None:
            # Corridor polygon: left down to bottom, across to right bottom, up along right
            corridor_poly = np.vstack([l_pts, r_pts[::-1]])
            corridor_mask = frame_rgb.copy()
            corridor_color = (0, 210, 100) if ldw_status == 'CENTERED' else (255, 140, 0)
            cv2.fillPoly(corridor_mask, [corridor_poly], corridor_color)
            overlay = cv2.addWeighted(corridor_mask, fill_alpha, overlay, 1.0 - fill_alpha, 0)
        else:
            lane_poly = np.array([[
                left_lane[0:2], left_lane[2:4],
                right_lane[2:4], right_lane[0:2],
            ]], dtype=np.int32)
            corridor_mask = frame_rgb.copy()
            cv2.fillPoly(corridor_mask, lane_poly, (0, 210, 100))
            overlay = cv2.addWeighted(corridor_mask, fill_alpha, overlay, 1.0 - fill_alpha, 0)

    # 2. Draw Lane Boundaries (Polylines or Lines)
    left_color = (255, 60, 60) if ldw_status == 'DRIFT_LEFT' else color
    right_color = (255, 60, 60) if ldw_status == 'DRIFT_RIGHT' else color

    if left_lane is not None:
        if len(left_lane) > 4 and left_lane[4] is not None:
            cv2.polylines(overlay, [left_lane[4]], isClosed=False, color=left_color, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(overlay, left_lane[:2], left_lane[2:4], left_color, thickness, lineType=cv2.LINE_AA)

    if right_lane is not None:
        if len(right_lane) > 4 and right_lane[4] is not None:
            cv2.polylines(overlay, [right_lane[4]], isClosed=False, color=right_color, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(overlay, right_lane[:2], right_lane[2:4], right_color, thickness, lineType=cv2.LINE_AA)

    # 3. Lane Departure Warning HUD Banner
    if show_ldw and ldw_status in ('DRIFT_LEFT', 'DRIFT_RIGHT'):
        # Flashing HUD banner
        flash = (frame_idx // 6) % 2 == 0 if frame_idx > 0 else True
        if flash:
            drift_text = "LANE DEPARTURE: DRIFTING LEFT" if ldw_status == 'DRIFT_LEFT' else "LANE DEPARTURE: DRIFTING RIGHT"
            (tw, th), _ = cv2.getTextSize(drift_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            bx1 = (w - tw) // 2 - 16
            by1 = 40
            bx2 = bx1 + tw + 32
            by2 = by1 + th + 18
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 69, 255), 2)  # Amber-red border
            cv2.putText(overlay, f"WARNING: {drift_text}", (bx1 + 10, by1 + th + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)

    return overlay
