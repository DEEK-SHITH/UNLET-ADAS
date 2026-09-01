"""
Unit tests for src/prepare_extra_lowlight.py -- fully offline. Builds
a tiny synthetic video for extract_video_frames() and a synthetic
ExDark-shaped image folder (no annotations needed) for
flatten_exdark_images().
"""
import os

import cv2
import numpy as np
import pytest

from src.prepare_extra_lowlight import (
    extract_video_frames,
    flatten_exdark_images,
)
from src.train_lowlight import EXDARK_SOURCE_CLASSES


@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / 'tiny.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 10, (32, 32))
    for i in range(20):
        frame = np.full((32, 32, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_video_frames_uses_stride(tiny_video, tmp_path):
    out_dir = str(tmp_path / 'frames')
    n = extract_video_frames(tiny_video, out_dir, every_n=5)
    # 20 frames, every 5th (indices 0,5,10,15) -> 4 frames
    assert n == 4
    files = sorted(os.listdir(out_dir))
    assert len(files) == 4
    assert all(f.endswith('.jpg') for f in files)


def test_extract_video_frames_respects_max_frames(tiny_video, tmp_path):
    out_dir = str(tmp_path / 'frames_capped')
    n = extract_video_frames(tiny_video, out_dir, every_n=1, max_frames=3)
    assert n == 3
    assert len(os.listdir(out_dir)) == 3


def test_extract_video_frames_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_video_frames(str(tmp_path / 'nope.mp4'), str(tmp_path / 'out'))


def _make_fake_exdark_images_only(root):
    for c in EXDARK_SOURCE_CLASSES:
        d = os.path.join(root, c)
        os.makedirs(d, exist_ok=True)
        for i in range(2):
            path = os.path.join(d, f'{c.lower()}_{i}.jpg')
            with open(path, 'wb') as f:
                f.write(b'fake-jpeg-bytes')


def test_flatten_exdark_images_copies_all_and_prefixes(tmp_path):
    images_root = str(tmp_path / 'exdark_images')
    _make_fake_exdark_images_only(images_root)
    out_dir = str(tmp_path / 'flat')

    copied = flatten_exdark_images(images_root, out_dir)
    assert copied == len(EXDARK_SOURCE_CLASSES) * 2

    names = os.listdir(out_dir)
    assert len(names) == copied
    # Every filename is prefixed with its source class folder, so
    # same-named images from different classes can't collide.
    for c in EXDARK_SOURCE_CLASSES:
        assert f'{c}_{c.lower()}_0.jpg' in names


def test_flatten_exdark_images_handles_missing_class_folders(tmp_path):
    images_root = str(tmp_path / 'partial_exdark')
    os.makedirs(os.path.join(images_root, 'Car'), exist_ok=True)
    with open(os.path.join(images_root, 'Car', 'car_0.jpg'), 'wb') as f:
        f.write(b'x')

    out_dir = str(tmp_path / 'flat_partial')
    copied = flatten_exdark_images(images_root, out_dir)
    assert copied == 1
    assert os.listdir(out_dir) == ['Car_car_0.jpg']
