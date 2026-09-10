"""
Unit tests for src/signs.py's draw_sign_detections().

No prior coverage existed for this kind of per-class detection
drawing anywhere in the codebase (the pothole detector's equivalent
logic is inlined at each call site rather than a testable function),
so this is new coverage, not a regression test.
"""
import numpy as np
import pytest

from src.signs import draw_sign_detections, _color_for_class, _PALETTE


class _FakeBox:
    """Mimics the one ultralytics.engine.results.Boxes attributes
    draw_sign_detections actually reads: .xyxy, .conf, .cls."""
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.float32(conf)]
        self.cls = [np.float32(cls)]


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


def _blank_frame(size=(100, 120)):
    h, w = size
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_sign_detections_returns_correct_count():
    class_names = {0: 'stop', 1: 'speedlimit'}
    results = _FakeResults([
        _FakeBox([10, 10, 30, 30], 0.9, 0),
        _FakeBox([40, 40, 60, 60], 0.6, 1),
    ])
    frame = _blank_frame()

    out, count = draw_sign_detections(frame, results, class_names)
    assert count == 2
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_draw_sign_detections_draws_something_for_each_box():
    class_names = {0: 'stop'}
    results = _FakeResults([_FakeBox([10, 10, 50, 50], 0.9, 0)])
    frame = _blank_frame()

    out, count = draw_sign_detections(frame, results, class_names)
    assert count == 1
    # A box was actually drawn -- the frame is no longer all-zero.
    assert out.sum() > 0


def test_draw_sign_detections_zero_boxes_leaves_frame_unchanged():
    class_names = {0: 'stop'}
    results = _FakeResults([])
    frame = _blank_frame()

    out, count = draw_sign_detections(frame, results, class_names)
    assert count == 0
    assert np.array_equal(out, frame)


def test_color_for_class_is_deterministic():
    # The real dataset's class list (currently indian-traffic-signs1,
    # see src/train_signs.py) runs into several dozen specific sign
    # types rather than a small fixed set of English names, so colors
    # are hashed from the class name instead of looked up in a
    # hardcoded dict. Same name must always map to the same color.
    assert _color_for_class('stop') == _color_for_class('stop')
    assert _color_for_class('stop') in _PALETTE


def test_color_for_class_varies_across_different_names():
    # Not a strict no-collision guarantee (the palette is finite and
    # names are hashed), just a sanity check that a handful of
    # different names don't all collapse onto the same color.
    names = ['stop', 'speedlimit', 'crosswalk', 'trafficlight',
             'larangan-berhenti', 'peringatan-simpang-tiga']
    colors = {_color_for_class(n) for n in names}
    assert len(colors) > 1


def test_unknown_class_falls_back_to_default_color_without_crashing():
    class_names = {0: 'some_new_class'}
    results = _FakeResults([_FakeBox([5, 5, 15, 15], 0.5, 0)])
    frame = _blank_frame()

    out, count = draw_sign_detections(frame, results, class_names)
    assert count == 1


def test_class_names_as_list_instead_of_dict():
    # ultralytics' Results.names is sometimes a list rather than a
    # dict depending on how the model was constructed/loaded.
    class_names = ['stop', 'speedlimit']
    results = _FakeResults([_FakeBox([10, 10, 30, 30], 0.9, 1)])
    frame = _blank_frame()

    out, count = draw_sign_detections(frame, results, class_names)
    assert count == 1
