"""
Unit tests for the pure-Python logic in src/depth.py (proximity
sampling and risk classification). These don't need the actual MiDaS
model or network access — that path is exercised indirectly by the
e2e tests in test_app.py, which tolerate MiDaS being unavailable
(has_depth=False) since it requires a first-run download.
"""
import numpy as np
import pytest

from src.depth import classify_proximity, sample_proximity


def test_sample_proximity_prefers_closer_regions():
    depth_map = np.zeros((100, 100), dtype=np.float32)
    depth_map[80:100, :] = 1.0   # bottom strip = "close"
    depth_map[0:20, :] = 0.0     # top strip = "far"

    near = sample_proximity(depth_map, 0, 80, 100, 100)
    far = sample_proximity(depth_map, 0, 0, 100, 20)
    assert near > far
    assert near == pytest.approx(1.0)
    assert far == pytest.approx(0.0)


def test_sample_proximity_uses_central_patch_not_full_box():
    # A box whose edges are far (0) but whose center is close (1)
    # should read as close — proves it samples the interior, not an
    # average that includes the noisy/background edge pixels.
    depth_map = np.zeros((100, 100), dtype=np.float32)
    depth_map[40:60, 40:60] = 1.0
    val = sample_proximity(depth_map, 0, 0, 100, 100)
    assert val > 0.05  # the center patch pulls the average up
    assert val < 1.0   # but it's not purely the center either


def test_sample_proximity_handles_degenerate_box():
    depth_map = np.random.rand(50, 50).astype(np.float32)
    # zero-width/height box shouldn't crash or divide by zero
    val = sample_proximity(depth_map, 10, 10, 10, 10)
    assert 0.0 <= val <= 1.0


def test_classify_proximity_thresholds():
    assert classify_proximity(0.9) == 'HIGH'
    assert classify_proximity(0.65) == 'HIGH'
    assert classify_proximity(0.64) == 'MEDIUM'
    assert classify_proximity(0.35) == 'MEDIUM'
    assert classify_proximity(0.34) == 'LOW'
    assert classify_proximity(0.0) == 'LOW'


def test_classify_proximity_custom_thresholds():
    assert classify_proximity(0.5, high_thresh=0.4, med_thresh=0.2) == 'HIGH'
    assert classify_proximity(0.3, high_thresh=0.4, med_thresh=0.2) == 'MEDIUM'
    assert classify_proximity(0.1, high_thresh=0.4, med_thresh=0.2) == 'LOW'
