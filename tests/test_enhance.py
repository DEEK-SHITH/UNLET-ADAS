"""
Unit tests for src/enhance.py's scene-aware confidence/blend helpers.

Regression coverage for a real bug: scene_aware_conf()'s result is
fed straight into YOLO's conf= argument at every call site in
app/streamlit_app.py. luminance is always a numpy float32 in
production (the mean of a frame array), which made the arithmetic --
and so the returned value -- inherit that numpy dtype. A newer
ultralytics release started strictly validating conf= as a native
Python int/float and rejecting numpy scalar subtypes, crashing every
video/image/live detection call with:
    TypeError: 'conf=0.259...' is of invalid type float32.
This had zero test coverage before, which is exactly why it went
unnoticed until it broke in production.
"""
import numpy as np
import pytest

from src.enhance import scene_aware_conf, scene_blend_weight


def test_scene_aware_conf_returns_native_float_with_numpy_luminance():
    """The actual production case: luminance from np.mean() of a
    frame array, not a plain Python float."""
    luminance = np.float32(0.1)  # dark frame, like the real crash
    result = scene_aware_conf(0.25, luminance)
    assert type(result) is float
    assert not isinstance(result, np.generic)


def test_scene_aware_conf_returns_native_float_with_python_luminance():
    result = scene_aware_conf(0.25, 0.1)
    assert type(result) is float


def test_scene_blend_weight_full_at_or_below_dark_thresh():
    assert scene_blend_weight(0.1, dark_thresh=0.35, bright_thresh=0.55) == 1.0
    assert scene_blend_weight(np.float32(0.35), dark_thresh=0.35,
                               bright_thresh=0.55) == 1.0


def test_scene_blend_weight_zero_at_or_above_bright_thresh():
    assert scene_blend_weight(0.9, dark_thresh=0.35, bright_thresh=0.55) == 0.0
    assert scene_blend_weight(np.float32(0.55), dark_thresh=0.35,
                               bright_thresh=0.55) == 0.0


def test_scene_blend_weight_smooth_ramp_between_thresholds():
    mid = scene_blend_weight(0.45, dark_thresh=0.35, bright_thresh=0.55)
    assert 0.0 < mid < 1.0


def test_scene_aware_conf_boosts_at_night():
    dark = scene_aware_conf(0.25, np.float32(0.1))
    bright = scene_aware_conf(0.25, np.float32(0.9))
    assert dark > 0.25
    assert dark == pytest.approx(0.35, abs=1e-6)  # matches the paper's 0.25 -> 0.35
    assert bright == pytest.approx(0.25, abs=1e-6)


def test_scene_aware_conf_clamped_at_0_95():
    result = scene_aware_conf(0.9, np.float32(0.0), night_boost=0.5)
    assert result <= 0.95
