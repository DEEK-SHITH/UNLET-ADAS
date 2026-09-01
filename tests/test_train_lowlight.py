"""
Unit tests for the pure-Python dataset filtering/remapping logic in
src/train_lowlight.py. Fully offline — builds a synthetic fake
ExDark-shaped YOLO export rather than needing the real dataset or
network access.
"""
import os

import pytest

from src.train_lowlight import LOWLIGHT_CLASSES, filter_and_remap_dataset

# 12 ExDark-like source classes, deliberately mixed casing/spelling to
# exercise CLASS_ALIASES matching against real-world re-export variance.
SOURCE_NAMES = ['Bicycle', 'Boat', 'bottle', 'Bus', 'car', 'Cat',
                 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']


def _make_fake_dataset(root, per_split=(('train', 8), ('valid', 3))):
    for split, n in per_split:
        img_dir = os.path.join(root, split, 'images')
        lbl_dir = os.path.join(root, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            name = f'{split}_{i}'
            open(os.path.join(img_dir, name + '.jpg'), 'w').write('x')
            adas_id = [0, 3, 4, 9, 10][i % 5]     # Bicycle/Bus/car/Motorbike/People
            junk_id = [1, 2, 5, 6, 7, 8, 11][i % 7]
            open(os.path.join(lbl_dir, name + '.txt'), 'w').write(
                f'{adas_id} 0.5 0.5 0.2 0.2\n{junk_id} 0.3 0.3 0.1 0.1\n')
        # one hard-negative image with no label file at all
        open(os.path.join(img_dir, f'{split}_nolabel.jpg'), 'w').write('x')

    with open(os.path.join(root, 'data.yaml'), 'w') as f:
        f.write(f'train: {root}/train/images\n')
        f.write(f'val: {root}/valid/images\n')
        f.write(f'nc: {len(SOURCE_NAMES)}\n')
        f.write(f'names: {SOURCE_NAMES}\n')


@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / 'raw'
    _make_fake_dataset(str(root))
    return str(root)


def test_filters_out_non_adas_classes(fake_dataset, tmp_path):
    out_dir = str(tmp_path / 'filtered')
    filter_and_remap_dataset(fake_dataset, out_dir)

    for split in ('train', 'valid'):
        for lbl_name in os.listdir(os.path.join(out_dir, split, 'labels')):
            for line in open(os.path.join(out_dir, split, 'labels', lbl_name)):
                if not line.strip():
                    continue
                cls_id = int(line.split()[0])
                assert 0 <= cls_id < len(LOWLIGHT_CLASSES), (
                    f'class id {cls_id} out of the 5-class range in {lbl_name}')


def test_keeps_all_images_including_hard_negatives(fake_dataset, tmp_path):
    out_dir = str(tmp_path / 'filtered')
    filter_and_remap_dataset(fake_dataset, out_dir)

    train_images = os.listdir(os.path.join(out_dir, 'train', 'images'))
    assert len(train_images) == 9  # 8 labeled + 1 hard negative
    nolabel_txt = os.path.join(out_dir, 'train', 'labels', 'train_nolabel.txt')
    assert os.path.exists(nolabel_txt)
    assert open(nolabel_txt).read() == ''


def test_remapped_box_counts_match_source(fake_dataset, tmp_path):
    out_dir = str(tmp_path / 'filtered')
    filter_and_remap_dataset(fake_dataset, out_dir)

    counts = {name: 0 for name in LOWLIGHT_CLASSES}
    for split in ('train', 'valid'):
        for lbl_name in os.listdir(os.path.join(out_dir, split, 'labels')):
            for line in open(os.path.join(out_dir, split, 'labels', lbl_name)):
                if line.strip():
                    counts[LOWLIGHT_CLASSES[int(line.split()[0])]] += 1

    # 8 train + 3 valid = 11 images total, each contributing exactly
    # one ADAS-relevant box (id cycles through all 5 classes evenly
    # enough that none should be zero for this fixture).
    assert sum(counts.values()) == 11
    assert all(v >= 0 for v in counts.values())
    assert counts['Car'] > 0 and counts['Person'] > 0


def test_output_data_yaml_has_five_classes(fake_dataset, tmp_path):
    out_dir = str(tmp_path / 'filtered')
    out_yaml = filter_and_remap_dataset(fake_dataset, out_dir)

    content = open(out_yaml).read()
    assert 'nc: 5' in content
    for name in LOWLIGHT_CLASSES:
        assert name in content


def test_missing_class_produces_no_crash(tmp_path):
    # A source with only 1 of the 5 relevant classes should still
    # filter cleanly (just with zero boxes for the other 4), not raise.
    root = tmp_path / 'raw_partial'
    img_dir = root / 'train' / 'images'
    lbl_dir = root / 'train' / 'labels'
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / 'a.jpg').write_text('x')
    (lbl_dir / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n')
    (root / 'data.yaml').write_text("train: x\nval: x\nnc: 1\nnames: ['Bicycle']\n")

    out_dir = str(tmp_path / 'filtered_partial')
    out_yaml = filter_and_remap_dataset(str(root), out_dir)
    assert os.path.exists(out_yaml)
