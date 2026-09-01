"""
Unit tests for LOLDataset's extra_low_dirs mixing logic in
src/train.py -- the dataset-loading half of "expand training data
beyond LOL's 485 pairs" (extra unpaired low-light images, e.g. real
driving footage or ExDark, folded in alongside LOL's paired images).
Fully offline, synthetic images only.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from src.train import LOLDataset


def _write_image(path, color=(40, 40, 40), size=(32, 32)):
    Image.fromarray(
        np.full((size[1], size[0], 3), color, dtype=np.uint8)
    ).save(path)


@pytest.fixture
def paired_dirs(tmp_path):
    low_dir = tmp_path / 'low'
    high_dir = tmp_path / 'high'
    low_dir.mkdir()
    high_dir.mkdir()
    for i in range(2):
        _write_image(str(low_dir / f'img{i}.png'), color=(10, 10, 10))
        _write_image(str(high_dir / f'img{i}.png'), color=(200, 200, 200))
    return str(low_dir), str(high_dir)


@pytest.fixture
def extra_dir(tmp_path):
    d = tmp_path / 'extra'
    d.mkdir()
    for i in range(3):
        _write_image(str(d / f'extra{i}.jpg'), color=(15, 15, 15))
    return str(d)


def test_extra_images_added_to_low_list(paired_dirs, extra_dir):
    low_dir, high_dir = paired_dirs
    ds = LOLDataset(low_dir, high_dir, extra_low_dirs=[extra_dir])
    assert len(ds) == 2 + 3


def test_paired_images_get_real_high_tensor(paired_dirs, extra_dir):
    low_dir, high_dir = paired_dirs
    ds = LOLDataset(low_dir, high_dir, extra_low_dirs=[extra_dir])
    # The two paired images are the first two (sorted low_dir glob).
    for i in range(2):
        low, high = ds[i]
        assert high.sum().item() > 0
        # high should reflect the bright (200,200,200) fixture image,
        # not an all-zero placeholder.
        assert high.mean().item() > 0.5


def test_extra_images_get_zero_high_tensor(paired_dirs, extra_dir):
    low_dir, high_dir = paired_dirs
    ds = LOLDataset(low_dir, high_dir, extra_low_dirs=[extra_dir])
    for i in range(2, len(ds)):
        low, high = ds[i]
        assert high.sum().item() == 0
        assert low.sum().item() > 0


def test_missing_extra_dir_is_skipped_not_crashed(paired_dirs, tmp_path):
    low_dir, high_dir = paired_dirs
    missing = str(tmp_path / 'does_not_exist')
    ds = LOLDataset(low_dir, high_dir, extra_low_dirs=[missing])
    assert len(ds) == 2  # only the paired images


def test_no_extra_dirs_behaves_like_before(paired_dirs):
    low_dir, high_dir = paired_dirs
    ds = LOLDataset(low_dir, high_dir)
    assert len(ds) == 2


def test_filename_collision_with_high_is_excluded(paired_dirs, tmp_path):
    """An extra-dir image that happens to share a filename with one
    of LOL's own 'high' images must NOT be treated as paired with it
    -- that would silently feed the model a mismatched ground truth
    from a completely unrelated photo."""
    low_dir, high_dir = paired_dirs
    colliding_dir = tmp_path / 'colliding'
    colliding_dir.mkdir()
    _write_image(str(colliding_dir / 'img0.png'), color=(99, 99, 99))
    _write_image(str(colliding_dir / 'unique.png'), color=(15, 15, 15))

    ds = LOLDataset(low_dir, high_dir, extra_low_dirs=[str(colliding_dir)])
    # Only the non-colliding extra image should have been added.
    assert len(ds) == 2 + 1
    basenames = [os.path.basename(p) for p in ds.lows]
    assert basenames.count('img0.png') == 1  # not duplicated
