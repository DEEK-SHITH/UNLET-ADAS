"""
Unit tests for src/detection.py and web/backend/pipeline.py.
"""
import time

import numpy as np
import pytest

from src.detection import (
    _plausible_traffic_light_shape,
    detect_and_draw,
    estimate_risk,
    estimate_risk_geometry,
    redraw_cached_detections,
    track_and_draw,
)


class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.float32(conf)]
        self.cls = [np.float32(cls)]


class _FakeBoxesWithId(list):
    def __init__(self, boxes, ids=None):
        super().__init__(boxes)
        self.id = np.array(ids, dtype=np.float32) if ids is not None else None


class _FakeResult:
    def __init__(self, boxes, ids=None):
        self.boxes = _FakeBoxesWithId(boxes, ids=ids)


class _FakeYOLO:
    def __init__(self, boxes, ids=None):
        self._res = [_FakeResult(boxes, ids=ids)]

    def __call__(self, *args, **kwargs):
        return self._res

    def track(self, *args, **kwargs):
        return self._res


def test_estimate_risk_geometry_tiers():
    w, h = 400, 300
    # Large box in central corridor -> HIGH
    assert estimate_risk_geometry(150, 100, 250, 250, w, h) == 'HIGH'
    # Moderate box in central corridor -> MEDIUM
    assert estimate_risk_geometry(180, 100, 220, 145, w, h) == 'MEDIUM'
    # Small distant box -> LOW
    assert estimate_risk_geometry(190, 100, 210, 115, w, h) == 'LOW'


def test_estimate_risk_depth_respects_ego_corridor():
    depth_map = np.full((300, 400), 0.9, dtype=np.float32)
    # High depth in central corridor -> HIGH
    assert estimate_risk(150, 100, 250, 200, 400, 300, depth_map=depth_map) == 'HIGH'
    # High depth on far peripheral edge (x in 0..30 on 400px frame) -> capped at MEDIUM
    assert estimate_risk(5, 100, 30, 200, 400, 300, depth_map=depth_map) == 'MEDIUM'


def test_plausible_traffic_light_shape_filters_wide_glare():
    # Vertical housing (w=20, h=50 -> w/h=0.4 <= 0.75) passes
    assert _plausible_traffic_light_shape(10, 10, 30, 60) is True
    # Square/wide glare blob (w=40, h=40 -> w/h=1.0 > 0.75) rejected
    assert _plausible_traffic_light_shape(10, 10, 50, 50) is False


def test_detect_and_draw_suppresses_square_traffic_light_glare():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # cls 9 = Traffic Light (square glare -> should be filtered), cls 2 = Car (kept)
    yolo = _FakeYOLO([
        _FakeBox([10, 10, 50, 50], 0.8, 9),
        _FakeBox([100, 80, 200, 180], 0.9, 2),
    ])
    ann, dets, counts, high_risk = detect_and_draw(yolo, frame)
    assert counts == {'Car': 1}
    assert len(dets) == 1
    assert dets[0]['name'] == 'Car'
    assert high_risk is True
    assert ann.sum() > 0


def test_track_and_draw_and_redraw_cached_detections():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    yolo = _FakeYOLO([_FakeBox([100, 80, 200, 180], 0.9, 2)], ids=[42])
    ann, counts, track_ids, high_risk, det_list = track_and_draw(yolo, frame)
    assert counts == {'Car': 1}
    assert track_ids == {42}
    assert high_risk is True
    assert det_list[0]['track_id'] == 42

    blank = np.zeros((200, 300, 3), dtype=np.uint8)
    redrawn = redraw_cached_detections(blank, det_list)
    assert redrawn.sum() > 0


def test_web_pipeline_cleanup_old_jobs_removes_expired(tmp_path):
    from web.backend import pipeline as pl

    pl.JOBS.clear()
    out_file = tmp_path / 'old_out.mp4'
    out_file.write_bytes(b'dummy')

    old_job = pl.VideoJob(
        id='expired123',
        status='done',
        output_path=str(out_file),
        created_at=time.time() - 7200,
    )
    active_job = pl.VideoJob(
        id='active456',
        status='processing',
        created_at=time.time() - 7200,
    )
    pl.JOBS[old_job.id] = old_job
    pl.JOBS[active_job.id] = active_job

    pl.cleanup_old_jobs(max_age_sec=3600)

    assert 'expired123' not in pl.JOBS
    assert not out_file.exists()
    assert 'active456' in pl.JOBS
    pl.JOBS.clear()
