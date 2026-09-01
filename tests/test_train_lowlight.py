"""
Unit tests for the pure-Python dataset filtering/remapping logic in
src/train_lowlight.py. Fully offline — builds a synthetic fake
ExDark-shaped YOLO export rather than needing the real dataset or
network access.
"""
import os

import pytest
from PIL import Image

from src.train_lowlight import (
    EXDARK_SOURCE_CLASSES,
    LOWLIGHT_CLASSES,
    _find_class_folders_root,
    convert_exdark_official,
    filter_and_remap_dataset,
)

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


def _make_fake_official_dataset(root):
    """Builds a synthetic dataset in the *official* ExDark layout: 12
    per-class folders under images/ and groundtruth/, one real small
    image + one bbGt-style .txt annotation file per image, following
    the exact line format confirmed against
    https://github.com/Yb1t/ExDark2Yolo (header line to skip, then
    '<ClassName> <left_px> <top_px> <width_px> <height_px> <occ...>'
    per object)."""
    images_root = os.path.join(root, 'images')
    gt_root = os.path.join(root, 'groundtruth')
    for c in EXDARK_SOURCE_CLASSES:
        os.makedirs(os.path.join(images_root, c), exist_ok=True)
        os.makedirs(os.path.join(gt_root, c), exist_ok=True)

    def write_image(class_folder, name, size=(200, 100)):
        path = os.path.join(images_root, class_folder, name)
        Image.new('RGB', size, color=(10, 10, 10)).save(path)
        return path

    def write_ann(class_folder, name, object_lines):
        path = os.path.join(gt_root, class_folder, name + '.txt')
        with open(path, 'w') as f:
            f.write('%  bbGt version=3\n')
            for line in object_lines:
                f.write(line + '\n')

    # 1) A Car-dominant image with a Car box AND a People box
    #    (multi-object, multi-class annotation in one file).
    write_image('Car', 'car_0001.jpg', size=(200, 100))
    write_ann('Car', 'car_0001.jpg', [
        'Car 20 10 100 50 0 0 0 0 0 0 0',    # left=20 top=10 w=100 h=50
        'People 5 5 20 40 0 0 0 0 0 0 0',
    ])

    # 2) A Cat-dominant image (non-ADAS class) that still has a Person
    #    annotated in the background -- must be kept, and the Person
    #    box must survive filtering even though the folder is 'Cat'.
    write_image('Cat', 'cat_0001.jpg', size=(200, 100))
    write_ann('Cat', 'cat_0001.jpg', [
        'Cat 0 0 50 50 0 0 0 0 0 0 0',
        'People 100 20 40 60 0 0 0 0 0 0 0',
    ])

    # 3) A Boat image with zero ADAS-relevant boxes -- hard negative.
    write_image('Boat', 'boat_0001.jpg', size=(200, 100))
    write_ann('Boat', 'boat_0001.jpg', [
        'Boat 0 0 100 100 0 0 0 0 0 0 0',
    ])

    return images_root, gt_root


@pytest.fixture
def fake_official_dataset(tmp_path):
    root = tmp_path / 'official_raw'
    return _make_fake_official_dataset(str(root))


def test_official_converts_official_bbgt_format(fake_official_dataset, tmp_path):
    images_root, gt_root = fake_official_dataset
    out_dir = str(tmp_path / 'filtered_official')
    out_yaml = convert_exdark_official(images_root, gt_root, out_dir,
                                        val_fraction=0.0)
    assert os.path.exists(out_yaml)

    all_labels = {}
    for split in ('train', 'valid'):
        lbl_dir = os.path.join(out_dir, split, 'labels')
        if not os.path.isdir(lbl_dir):
            continue
        for name in os.listdir(lbl_dir):
            all_labels[name] = open(os.path.join(lbl_dir, name)).read()

    # All 3 source images produced a label file (hard negative included).
    assert set(all_labels) == {
        'car_0001.txt', 'cat_0001.txt', 'boat_0001.txt'}

    # Boat image: zero ADAS boxes -> empty label file, not dropped.
    assert all_labels['boat_0001.txt'].strip() == ''

    # Car image: 2 boxes (Car + People), both ADAS-relevant.
    car_lines = [l for l in all_labels['car_0001.txt'].splitlines() if l]
    assert len(car_lines) == 2
    car_id = LOWLIGHT_CLASSES.index('Car')
    person_id = LOWLIGHT_CLASSES.index('Person')
    ids_found = {int(l.split()[0]) for l in car_lines}
    assert ids_found == {car_id, person_id}

    # Verify the pixel->normalized-center conversion for the Car box
    # (left=20 top=10 w=100 h=50 on a 200x100 image):
    #   x_center = (20 + 100/2) / 200 = 0.35, y_center = (10+50/2)/100 = 0.35
    #   w_norm = 100/200 = 0.5, h_norm = 50/100 = 0.5
    car_line = next(l for l in car_lines if int(l.split()[0]) == car_id)
    _, xc, yc, w, h = car_line.split()
    assert abs(float(xc) - 0.35) < 1e-4
    assert abs(float(yc) - 0.35) < 1e-4
    assert abs(float(w) - 0.5) < 1e-4
    assert abs(float(h) - 0.5) < 1e-4

    # Cat-dominant image: the co-occurring Person box must survive
    # filtering even though the image lives in the non-ADAS 'Cat'
    # folder -- only the Cat box itself should be dropped.
    cat_lines = [l for l in all_labels['cat_0001.txt'].splitlines() if l]
    assert len(cat_lines) == 1
    assert int(cat_lines[0].split()[0]) == person_id


def test_official_train_valid_split(fake_official_dataset, tmp_path):
    images_root, gt_root = fake_official_dataset
    out_dir = str(tmp_path / 'filtered_official')
    convert_exdark_official(images_root, gt_root, out_dir, val_fraction=1.0)

    # val_fraction=1.0 -> every image goes to valid, none to train.
    valid_images = os.listdir(os.path.join(out_dir, 'valid', 'images'))
    train_dir = os.path.join(out_dir, 'train', 'images')
    train_images = os.listdir(train_dir) if os.path.isdir(train_dir) else []
    assert len(valid_images) == 3
    assert len(train_images) == 0


def test_find_class_folders_root_handles_nesting(tmp_path):
    # Some zip exports nest the class folders one level deeper than
    # the extraction root (e.g. under an extra 'ExDark/' folder).
    nested = tmp_path / 'extracted' / 'ExDark_v1' / 'inner'
    for c in EXDARK_SOURCE_CLASSES:
        (nested / c).mkdir(parents=True)

    found = _find_class_folders_root(str(tmp_path / 'extracted'))
    assert found == str(nested)


def test_find_class_folders_root_returns_none_when_absent(tmp_path):
    (tmp_path / 'unrelated').mkdir()
    assert _find_class_folders_root(str(tmp_path)) is None


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
