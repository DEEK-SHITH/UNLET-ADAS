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
import os

import cv2
import numpy as np
import pytest

from src.enhance import scene_aware_conf, scene_blend_weight, transcode_for_browser


def _decode_fourcc(v):
    v = int(v)
    return ''.join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def _write_mp4v_video(path, n_frames=6, size=(64, 48)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = size
    writer = cv2.VideoWriter(str(path), fourcc, 8, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


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


def test_transcode_for_browser_converts_mp4v_to_h264(tmp_path):
    """The actual production bug: cv2.VideoWriter's 'mp4v' fourcc
    (MPEG-4 Part 2) is not decodable by browsers' HTML5 <video>, so
    st.video() on an untouched file shows a blank/unplayable player --
    the download button becomes the only way to watch the result.
    transcode_for_browser() must turn it into an H.264 file in place."""
    path = tmp_path / 'clip.mp4'
    _write_mp4v_video(path)

    # The exact fourcc OpenCV reports back for a 'mp4v' request varies by
    # build/platform (this environment's ffmpeg negotiates it as 'FMP4',
    # not a literal echo of 'mp4v') -- the invariant that actually
    # matters, and the one production depends on, is that it is NOT the
    # browser-playable H.264 family before transcoding.
    before = cv2.VideoCapture(str(path))
    fourcc_before = _decode_fourcc(before.get(cv2.CAP_PROP_FOURCC))
    before.release()
    assert fourcc_before not in ('avc1', 'h264', 'H264')

    ok = transcode_for_browser(path)
    assert ok is True

    after = cv2.VideoCapture(str(path))
    assert after.isOpened()
    fourcc_after = _decode_fourcc(after.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(after.get(cv2.CAP_PROP_FRAME_COUNT))
    after.release()
    assert fourcc_after in ('avc1', 'h264', 'H264')
    assert frame_count > 0


def test_transcode_for_browser_falls_back_on_failure(tmp_path, monkeypatch):
    """A missing/broken ffmpeg must never break video processing --
    the original (still downloadable, just not inline-playable) file
    should be left in place and the function should report failure
    rather than raising."""
    path = tmp_path / 'clip.mp4'
    _write_mp4v_video(path)
    original_bytes = path.read_bytes()

    import subprocess as _subprocess

    def _boom(*args, **kwargs):
        raise _subprocess.CalledProcessError(1, 'ffmpeg')

    monkeypatch.setattr('src.enhance.subprocess.run', _boom)

    ok = transcode_for_browser(path)
    assert ok is False
    assert path.read_bytes() == original_bytes
    assert not os.path.exists(str(path) + '.h264.mp4')
