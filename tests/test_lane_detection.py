"""
Unit tests for src/lane_detection.py -- classical CV lane detection
(Canny + Hough), previously with zero test coverage anywhere in the
suite (only exercised indirectly by the e2e Video tab test, which
doesn't assert anything about the lane geometry itself).

Uses synthetic frames with two straight lines drawn on a plain
background rather than real road photos, since the point here is
testing the geometry/math (averaging, extrapolation, the
never-cross invariant), not the underlying edge/line detection.
"""
import numpy as np
import pytest

from src.lane_detection import (
    _average_slope_line, detect_lanes, draw_lanes,
)


def _synthetic_lane_frame(w=400, h=300):
    """A dark road with two bright, converging lane lines painted on
    it -- Canny+Hough should pick these up as one segment cluster per
    side without needing a real photo."""
    import cv2
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    # Left line: bottom-left up to top-center-ish.
    cv2.line(frame, (40, h - 1), (int(w * 0.45), int(h * 0.55)),
              (220, 220, 220), 4)
    # Right line: bottom-right up to top-center-ish.
    cv2.line(frame, (w - 40, h - 1), (int(w * 0.55), int(h * 0.55)),
              (220, 220, 220), 4)
    return frame


def test_detect_lanes_finds_both_sides_on_a_clean_synthetic_frame():
    frame = _synthetic_lane_frame()
    left, right = detect_lanes(frame)

    assert left is not None
    assert right is not None
    # Left line's x-coordinates should stay left of right's, both ends.
    assert left[0] < right[0]
    assert left[2] < right[2]


def test_detect_lanes_returns_none_none_on_a_blank_frame():
    frame = np.full((300, 400, 3), 30, dtype=np.uint8)
    left, right = detect_lanes(frame)
    assert left is None
    assert right is None


def test_detect_lanes_does_not_crash_on_tiny_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    left, right = detect_lanes(frame)
    assert left is None
    assert right is None


def test_average_slope_line_none_for_empty_input():
    assert _average_slope_line([], h=300) is None


def test_average_slope_line_uses_median_not_mean():
    # One wild outlier segment shouldn't swing the extrapolated line
    # far off from where the consistent segments point.
    h = 300
    consistent = [(100, 300, 120, 200)] * 5
    outlier = [(100, 300, 500, 0)]  # near-horizontal outlier slope
    line_with_outlier = _average_slope_line(consistent + outlier, h)
    line_without_outlier = _average_slope_line(consistent, h)

    assert line_with_outlier is not None
    assert line_without_outlier is not None
    # Bottom point (y1=h) should match closely since the median slope
    # is barely perturbed by a single outlier among five consistent
    # segments.
    assert line_with_outlier[0] == pytest.approx(
        line_without_outlier[0], abs=2)


def test_average_slope_line_never_extrapolates_above_topmost_segment():
    # y_top_frac would put the line very high up, but no segment was
    # seen that high -- the line must stop at the topmost evidence,
    # not stretch past it (the "lines crossing high in the sky" defect
    # this function's docstring specifically calls out).
    h = 300
    lines = [(100, 300, 110, 280)]  # only seen down to y=280
    result = _average_slope_line(lines, h, y_top_frac=0.1)  # would be y=30
    assert result is not None
    _, _, _, y2 = result
    assert y2 >= 280


def test_never_cross_invariant_suppresses_a_crossing_pair():
    # Directly construct a left/right pair that crosses (left's x
    # ends up right of right's) and confirm detect_lanes' own
    # never-cross check would reject it -- exercised indirectly via a
    # frame engineered to trigger a bad Hough fit is unreliable, so
    # this pins the documented invariant at the unit level instead.
    from src.lane_detection import detect_lanes as _dl
    # Use a frame where left/right segments are deliberately swapped
    # in screen position while keeping their "left-slope"/"right-slope"
    # classification the same, so the crossing check has to fire.
    import cv2
    h, w = 300, 400
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    # A left-sloping (slope < 0) segment placed on the right side, and
    # a right-sloping (slope > 0) segment placed on the left side --
    # this is exactly the "X" crossing geometry the invariant guards.
    cv2.line(frame, (w - 40, h - 1), (w - 10, int(h * 0.55)),
              (220, 220, 220), 4)
    cv2.line(frame, (40, h - 1), (10, int(h * 0.55)),
              (220, 220, 220), 4)
    left, right = _dl(frame)
    # Whatever detect_lanes concludes, it must never report a pair
    # that crosses.
    if left is not None and right is not None:
        assert left[0] < right[0]
        assert left[2] < right[2]


def test_draw_lanes_draws_something_when_lines_present():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = draw_lanes(frame, (40, 299, 180, 165), (360, 299, 220, 165))
    assert out.sum() > 0
    assert out.shape == frame.shape


def test_draw_lanes_no_op_with_no_lines():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = draw_lanes(frame, None, None)
    assert np.array_equal(out, frame)


def test_draw_lanes_fill_alpha_requires_both_lines():
    # fill_alpha > 0 with only one line present shouldn't crash or
    # attempt to build a polygon from a missing line.
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    out = draw_lanes(frame, (40, 299, 180, 165), None, fill_alpha=0.2)
    assert out.shape == frame.shape
