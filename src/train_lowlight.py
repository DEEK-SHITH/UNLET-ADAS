"""
UNLET-ADAS: Low-Light-Specialized Detector Training
======================================================
Fine-tunes a small YOLOv8 model on ExDark (Exclusively Dark Image
Dataset — Loh & Chan, CVIU 2019, already cited in the project's
literature survey [19]) so detection is trained on real night-time
appearance, not just stock COCO daylight images run on enhanced
frames.

Why this is a SEPARATE model rather than a fine-tune of the main
8-class ADAS detector: ExDark has 12 classes (Bicycle, Boat, Bottle,
Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table), only 5 of
which overlap with the app's ADAS classes (Person, Bicycle, Car,
Motorcycle, Bus). Ultralytics' training resizes the detection head to
match whatever class list you train on — fine-tuning directly on
ExDark's 12 classes would silently DROP Traffic Light / Stop Sign /
Truck detection entirely, not just leave them unchanged, since the
fine-tuned model would never see a single example of them. Instead,
this trains a dedicated 5-class low-light specialist
(Person/Bicycle/Car/Motorcycle/Bus) that the app offers as an
alternative Detector Model choice alongside the stock COCO one, which
keeps full 8-class daylight coverage. See app/streamlit_app.py's
LOWLIGHT_CLASSES / class_map wiring.

Dataset sourcing: by default this downloads ExDark directly from its
authoritative source — the Google Drive links published by the
dataset's authors on
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset — and
converts the official bounding-box annotation format (Piotr's
Computer Vision Matlab Toolbox / "bbGt" style .txt files, one per
image, organized under 12 per-class folders) straight to YOLO format.
This needs no API key and no manual mirror-hunting. Non-commercial
research use only, per the dataset's own license.

If that download is ever unreachable (Google Drive rate limits or
blocks anonymous downloads from some networks/IP ranges), you can
instead point this script at a YOLO-format export from a Roboflow
Universe mirror with --roboflow_workspace/--roboflow_project
--roboflow_version (search "ExDark" at
https://universe.roboflow.com, pick a project whose image count is
close to the real dataset's ~7,363, and read the three values off its
"Download Dataset -> YOLOv8 -> Show download code" snippet).

This script also prints the image/label counts it actually finds
after download+filtering, so an incomplete source is caught
immediately rather than producing a silently-undertrained model.

Usage:
    pip install gdown ultralytics
    python src/train_lowlight.py
    # or, using a Roboflow mirror instead of the official source:
    pip install roboflow ultralytics
    python src/train_lowlight.py --source roboflow \
        --roboflow_key YOUR_FREE_API_KEY \
        --roboflow_workspace WORKSPACE --roboflow_project PROJECT \
        --roboflow_version N

Or in Colab (free GPU, no local install needed):
    Open notebooks/UNLET_ADAS_Lowlight_YOLO_Colab.ipynb in Google
    Colab, fill in the config cell, and run top to bottom.

Output: lowlight_best.pt — drop it in app/ as yolov8_lowlight.pt next
to zerodce_cbam_best.pt to add the "Low-Light Detector" option to the
Streamlit app's Detector Model sidebar choice.
"""

import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Fixed order — index in this list IS the class id the fine-tuned
# model will use (0=Person, 1=Bicycle, ...). app/streamlit_app.py's
# LOWLIGHT_CLASSES must stay in this exact order to match.
LOWLIGHT_CLASSES = ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus']

# Different ExDark YOLO re-exports spell/case class names differently
# (e.g. "People" vs "Person", "Motorbike" vs "Motorcycle") — map every
# spelling variant seen in the wild to our canonical name.
CLASS_ALIASES = {
    'person': 'Person', 'people': 'Person',
    'bicycle': 'Bicycle', 'bike': 'Bicycle',
    'car': 'Car',
    'motorbike': 'Motorcycle', 'motorcycle': 'Motorcycle',
    'bus': 'Bus',
}

# The official ExDark distribution's own 12 class folder names (source
# names, not our 5-class ADAS subset — see LOWLIGHT_CLASSES above).
EXDARK_SOURCE_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table',
]

# Google Drive file IDs published by the dataset authors on
# https://github.com/cs-chan/Exclusively-Dark-Image-Dataset (see that
# repo's Dataset/README.md and Groundtruth/README.md).
EXDARK_IMAGES_GDRIVE_ID = '1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx'
EXDARK_GROUNDTRUTH_GDRIVE_ID = '1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i'


def _find_class_folders_root(search_root):
    """Find the directory that directly contains all 12 ExDark class
    subfolders (Bicycle/, Boat/, ..., Table/), searching however deep
    the zip happens to nest them — different exports of this dataset
    have used different internal layouts."""
    for dirpath, dirnames, _ in os.walk(search_root):
        if all(c in dirnames for c in EXDARK_SOURCE_CLASSES):
            return dirpath
    return None


def download_exdark_official(dest_dir, retries=3):
    """
    Download the ExDark dataset directly from its authoritative
    source (the Google Drive links the dataset's authors publish on
    https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) rather
    than requiring a hand-found Roboflow mirror of uncertain
    completeness.

    Returns (images_root, groundtruth_root): the directories that
    each directly contain the 12 per-class subfolders.
    """
    import time
    import zipfile
    import gdown

    os.makedirs(dest_dir, exist_ok=True)
    targets = [
        ('images', EXDARK_IMAGES_GDRIVE_ID,
         os.path.join(dest_dir, 'images.zip')),
        ('groundtruth', EXDARK_GROUNDTRUTH_GDRIVE_ID,
         os.path.join(dest_dir, 'groundtruth.zip')),
    ]

    roots = {}
    for label, file_id, zip_path in targets:
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    gdown.download(id=file_id, output=zip_path, quiet=False)
                    if (os.path.exists(zip_path)
                            and os.path.getsize(zip_path) > 0):
                        break
                except Exception as e:
                    last_err = e
                print(f'{label} download attempt {attempt}/{retries} '
                      f'failed{": " + str(last_err) if last_err else ""}')
                if attempt < retries:
                    time.sleep(3)
            if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                raise RuntimeError(
                    f'Failed to download ExDark {label} from Google Drive '
                    f'after {retries} attempts: {last_err}. Google '
                    'occasionally rate-limits or blocks anonymous '
                    'downloads of this file from some networks. Work '
                    'around it by downloading it yourself from '
                    'https://github.com/cs-chan/Exclusively-Dark-Image-'
                    f'Dataset, extracting it to {dest_dir}/{label}/, and '
                    're-running this script (it will find and reuse '
                    "what's already on disk) -- or pass --source "
                    'roboflow with a mirror instead.')

        extract_dir = os.path.join(dest_dir, label)
        if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
            print(f'Extracting {label}...')
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        root = _find_class_folders_root(extract_dir)
        if root is None:
            raise RuntimeError(
                f"Extracted ExDark {label} to '{extract_dir}' but "
                "couldn't find the expected 12 class subfolders "
                '(Bicycle, Boat, ...) anywhere inside it -- the zip '
                "layout may have changed since this script was written.")
        roots[label] = root

    return roots['images'], roots['groundtruth']


def convert_exdark_official(images_root, groundtruth_root, out_dir,
                             val_fraction=0.15):
    """
    Convert the official ExDark bounding-box annotations into a
    filtered, remapped YOLOv8 dataset — same output shape
    filter_and_remap_dataset() produces from a Roboflow export, so
    train() can use either source interchangeably.

    Annotation line format (confirmed against a public ExDark->YOLO
    converter, https://github.com/Yb1t/ExDark2Yolo, since the official
    repo's own docs describe the columns but don't show a literal
    example line): each per-image .txt file has one header line to
    skip, then one line per object:
        <ClassName> <left_px> <top_px> <width_px> <height_px> <occ...>
    where left/top are the pixel position of the box's top-left
    corner (not its center).

    Every one of the 12 official class folders is walked (not just
    the 5 ADAS-relevant ones), so an image whose *dominant* label is
    e.g. Cat or Dog but which still has a Person or Car annotated
    somewhere in it isn't silently discarded, and images left with
    zero ADAS-relevant boxes are kept as hard negatives — same policy
    as filter_and_remap_dataset.
    """
    from collections import Counter
    from PIL import Image

    out_img_train = os.path.join(out_dir, 'train', 'images')
    out_lbl_train = os.path.join(out_dir, 'train', 'labels')
    out_img_valid = os.path.join(out_dir, 'valid', 'images')
    out_lbl_valid = os.path.join(out_dir, 'valid', 'labels')
    for d in (out_img_train, out_lbl_train, out_img_valid, out_lbl_valid):
        os.makedirs(d, exist_ok=True)

    before_counts = Counter()
    after_counts = Counter({c: 0 for c in LOWLIGHT_CLASSES})
    images_kept = 0
    images_total = 0
    images_skipped = 0
    val_every = round(1 / val_fraction) if 0 < val_fraction <= 1 else 0

    for class_folder in EXDARK_SOURCE_CLASSES:
        gt_dir = os.path.join(groundtruth_root, class_folder)
        img_dir = os.path.join(images_root, class_folder)
        if not os.path.isdir(gt_dir) or not os.path.isdir(img_dir):
            continue

        txt_names = sorted(
            f for f in os.listdir(gt_dir) if f.lower().endswith('.txt'))
        for i, txt_name in enumerate(txt_names):
            images_total += 1
            img_name = txt_name[:-4]  # 'foo.jpg.txt' -> 'foo.jpg'
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                # some mirrors upper-case the extension inconsistently
                stem, ext = os.path.splitext(img_name)
                alt_path = os.path.join(img_dir, stem + ext.upper())
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    images_skipped += 1
                    continue

            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception:
                images_skipped += 1
                continue

            new_lines = []
            with open(os.path.join(gt_dir, txt_name)) as f:
                lines = f.readlines()
            for line in lines[1:]:  # first line is a tool header
                parts = line.split()
                if not parts:
                    continue
                src_name = parts[0]
                before_counts[src_name] += 1
                canonical = CLASS_ALIASES.get(src_name.strip().lower())
                if canonical is None:
                    continue
                try:
                    l, t, w, h = (float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]))
                except (IndexError, ValueError):
                    continue
                if w <= 0 or h <= 0:
                    continue
                new_id = LOWLIGHT_CLASSES.index(canonical)
                after_counts[canonical] += 1
                x_center = (l + w / 2) / width
                y_center = (t + h / 2) / height
                new_lines.append(' '.join([
                    str(new_id),
                    format(min(max(x_center, 0.0), 1.0), '.6f'),
                    format(min(max(y_center, 0.0), 1.0), '.6f'),
                    format(min(w / width, 1.0), '.6f'),
                    format(min(h / height, 1.0), '.6f'),
                ]))

            is_valid = val_every and (i % val_every == 0)
            out_img_dir = out_img_valid if is_valid else out_img_train
            out_lbl_dir = out_lbl_valid if is_valid else out_lbl_train

            shutil.copy(img_path, os.path.join(out_img_dir, img_name))
            lbl_out_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join(out_lbl_dir, lbl_out_name), 'w') as f:
                f.write('\n'.join(new_lines))
            images_kept += 1

    print(f'\nImages processed: {images_total}, kept: {images_kept}, '
          f'skipped (missing/unreadable file): {images_skipped}')
    print(f'Boxes per source class (all 12, before filtering): '
          f'{dict(before_counts)}')
    print(f'Boxes per ADAS class (after filtering+remapping): '
          f'{dict(after_counts)}')
    missing = [c for c in LOWLIGHT_CLASSES if after_counts[c] == 0]
    if missing:
        print(f'WARNING: zero boxes found for: {missing}')
    if images_kept < 5000:
        print(f'WARNING: only {images_kept} images found -- the full '
              'official ExDark dataset has ~7,363. The download or '
              'extraction may be incomplete.')

    out_yaml = os.path.join(out_dir, 'data.yaml')
    with open(out_yaml, 'w') as f:
        f.write(f"train: {out_img_train}\n")
        f.write(f"val: {out_img_valid}\n")
        f.write(f'nc: {len(LOWLIGHT_CLASSES)}\n')
        f.write(f'names: {LOWLIGHT_CLASSES}\n')

    return out_yaml


def download_dataset(api_key, dest_dir, workspace, project, version,
                      retries=3):
    """
    Download an ExDark YOLO-format export from Roboflow. Unlike the
    pothole dataset, there's no single verified canonical mirror to
    default to, so workspace/project/version must be supplied
    explicitly (see this module's docstring for how to find one).
    """
    import time
    from roboflow import Roboflow

    os.makedirs(dest_dir, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    # Same resilience as the pothole downloader: Roboflow's download
    # occasionally stalls or drops the connection mid-transfer with
    # no exception raised.
    dataset = None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            dataset = ver.download('yolov8', location=dest_dir)
            break
        except Exception as e:
            last_err = e
            print(f'Download attempt {attempt}/{retries} failed: {e}')
            if attempt < retries:
                time.sleep(3)
    if dataset is None:
        raise RuntimeError(
            f'Roboflow download failed after {retries} attempts: {last_err}')

    location = dataset.location
    if os.path.exists(os.path.join(location, 'data.yaml')):
        return location
    for root, _, files in os.walk(location):
        if 'data.yaml' in files:
            return root

    raise RuntimeError(
        f"Roboflow reported the dataset was downloaded to '{location}' "
        "but no data.yaml was found anywhere under it. This usually "
        "means the download itself failed silently rather than the "
        "file just being in an unexpected subfolder — double check "
        "your Roboflow API key and network connection, then look at "
        "the download log printed above this error for the real cause.")


def _read_yaml_names(data_yaml_path):
    """Read just the `names` list/dict out of a YOLO data.yaml,
    without requiring pyyaml (Roboflow's exports are simple enough to
    parse by hand, and this avoids adding a new dependency)."""
    import re
    with open(data_yaml_path) as f:
        text = f.read()
    m = re.search(r'names:\s*(\[.*?\]|\{.*?\})', text, re.DOTALL)
    if not m:
        raise ValueError(f'Could not find a names list in {data_yaml_path}')
    names_literal = m.group(1)
    import ast
    names = ast.literal_eval(names_literal)
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names


def filter_and_remap_dataset(dataset_dir, out_dir):
    """
    Take a downloaded ExDark YOLO export (12 classes, whatever the
    source mirror named them) and produce a filtered copy containing
    only LOWLIGHT_CLASSES, remapped to ids 0..4 in that fixed order.
    Boxes for the other 7 classes (Boat, Bottle, Cat, Chair, Cup, Dog,
    Table) are dropped; images that end up with zero boxes are KEPT
    (as hard negatives — real night scenes with none of our 5 target
    classes in them are useful negative training signal, not just
    dead weight).

    Prints per-class box counts before and after so an incomplete or
    mislabeled source mirror is obvious immediately, rather than
    silently producing an undertrained model.
    """
    src_yaml = os.path.join(dataset_dir, 'data.yaml')
    source_names = _read_yaml_names(src_yaml)

    # source class id -> our canonical LOWLIGHT_CLASSES id, or None
    # to drop
    id_map = {}
    unmatched = []
    for src_id, src_name in enumerate(source_names):
        canonical = CLASS_ALIASES.get(src_name.strip().lower())
        if canonical is None:
            unmatched.append(src_name)
            continue
        id_map[src_id] = LOWLIGHT_CLASSES.index(canonical)

    matched = sorted({CLASS_ALIASES[source_names[i].strip().lower()]
                       for i in id_map})
    print(f'Source classes: {source_names}')
    print(f'Matched to ADAS classes: {matched}')
    if unmatched:
        print(f'Dropped (not ADAS-relevant): {unmatched}')
    missing = [c for c in LOWLIGHT_CLASSES if c not in matched]
    if missing:
        print(f'WARNING: this source has NO data for: {missing} — '
              'the fine-tuned model will be weak/blind on these '
              'classes. Consider a different Roboflow mirror.')

    os.makedirs(out_dir, exist_ok=True)
    from collections import Counter
    before_counts = Counter()
    after_counts = Counter({c: 0 for c in LOWLIGHT_CLASSES})
    images_kept = 0
    images_total = 0

    for split in ('train', 'valid', 'val', 'test'):
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        img_dir = os.path.join(split_dir, 'images')
        lbl_dir = os.path.join(split_dir, 'labels')
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        out_split = 'valid' if split in ('val', 'valid') else split
        out_img_dir = os.path.join(out_dir, out_split, 'images')
        out_lbl_dir = os.path.join(out_dir, out_split, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for img_name in os.listdir(img_dir):
            stem, _ = os.path.splitext(img_name)
            lbl_path = os.path.join(lbl_dir, stem + '.txt')
            images_total += 1

            new_lines = []
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        src_id = int(parts[0])
                        before_counts[source_names[src_id]] += 1
                        if src_id not in id_map:
                            continue
                        new_id = id_map[src_id]
                        after_counts[LOWLIGHT_CLASSES[new_id]] += 1
                        new_lines.append(
                            ' '.join([str(new_id)] + parts[1:]))

            shutil.copy(
                os.path.join(img_dir, img_name),
                os.path.join(out_img_dir, img_name))
            with open(os.path.join(out_lbl_dir, stem + '.txt'), 'w') as f:
                f.write('\n'.join(new_lines))
            images_kept += 1

    print(f'\nImages copied: {images_kept}/{images_total}')
    print(f'Boxes per source class (all 12, before filtering): '
          f'{dict(before_counts)}')
    print(f'Boxes per ADAS class (after filtering+remapping): '
          f'{dict(after_counts)}')
    if images_kept < 500:
        print(f'WARNING: only {images_kept} images found — the real '
              'ExDark dataset has ~7,363. This mirror may be a small '
              'partial subset; check you picked the right Roboflow '
              'project before training on it.')

    out_yaml = os.path.join(out_dir, 'data.yaml')
    val_split = 'valid' if os.path.isdir(
        os.path.join(out_dir, 'valid')) else 'train'
    with open(out_yaml, 'w') as f:
        f.write(f"train: {os.path.join(out_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(out_dir, val_split, 'images')}\n")
        f.write(f'nc: {len(LOWLIGHT_CLASSES)}\n')
        f.write(f'names: {LOWLIGHT_CLASSES}\n')

    return out_yaml


def train(args):
    from ultralytics import YOLO

    os.makedirs(args.save_dir, exist_ok=True)

    if args.data_yaml:
        data_yaml = args.data_yaml
        print(f'Using existing filtered dataset: {data_yaml}')
    elif args.source == 'roboflow':
        if not args.roboflow_key:
            raise SystemExit(
                'Pass --roboflow_key YOUR_KEY (free account at '
                'https://app.roboflow.com) or --data_yaml to point at '
                'an already-filtered dataset in YOLOv8 format.')
        if not (args.roboflow_workspace and args.roboflow_project
                and args.roboflow_version):
            raise SystemExit(
                '--roboflow_workspace, --roboflow_project and '
                '--roboflow_version are all required — see this '
                "script's module docstring for how to find them on "
                'Roboflow Universe (search "ExDark").')
        print('Downloading ExDark dataset from Roboflow...')
        raw_dir = download_dataset(
            args.roboflow_key, args.dataset_dir,
            args.roboflow_workspace, args.roboflow_project,
            args.roboflow_version)
        print('\nFiltering to ADAS-relevant classes and remapping IDs...')
        filtered_dir = os.path.join(args.dataset_dir, 'filtered')
        data_yaml = filter_and_remap_dataset(raw_dir, filtered_dir)
    else:
        print('Downloading the official ExDark dataset '
              '(images + groundtruth, direct from the authors)...')
        images_root, groundtruth_root = download_exdark_official(
            args.dataset_dir)
        print(f'Images     : {images_root}')
        print(f'Groundtruth: {groundtruth_root}')
        print('\nConverting to YOLO format, filtering to ADAS-relevant '
              'classes and remapping IDs...')
        filtered_dir = os.path.join(args.dataset_dir, 'filtered')
        data_yaml = convert_exdark_official(
            images_root, groundtruth_root, filtered_dir)

    print(f'\nFine-tuning YOLOv8{args.model_size} for low-light detection')
    print(f'Data     : {data_yaml}')
    print(f'Classes  : {LOWLIGHT_CLASSES}')
    print(f'Epochs   : {args.epochs}')
    print(f'Image sz : {args.image_size}')
    print('-' * 50)

    model = YOLO(f'yolov8{args.model_size}.pt')
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        patience=args.patience,
        project=args.save_dir,
        name='lowlight_run',
        exist_ok=True,
    )

    best = os.path.join(
        args.save_dir, 'lowlight_run', 'weights', 'best.pt')
    out = os.path.join(args.save_dir, 'lowlight_best.pt')
    if os.path.exists(best):
        shutil.copy(best, out)
        print(f'\nTraining complete! Weights saved to: {out}')
        print('Copy this file to app/yolov8_lowlight.pt to add the '
              '"Low-Light Detector" option in the Streamlit app.')
    else:
        print(f'\nExpected weights at {best} but they were not found — '
              'check the training log above for errors.')

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune a YOLOv8 low-light specialist detector '
                    '(5 ADAS classes) on ExDark for UNLET-ADAS')
    p.add_argument('--source', default='exdark_official',
                   choices=['exdark_official', 'roboflow'],
                   help='Where to get the dataset from. '
                        "'exdark_official' (default) downloads directly "
                        "from the dataset authors' Google Drive, no API "
                        "key needed. 'roboflow' uses a hand-found "
                        'Roboflow Universe mirror instead (see '
                        '--roboflow_* args).')
    p.add_argument('--roboflow_key', default=None,
                   help='Free Roboflow API key. Only used with '
                        '--source roboflow.')
    p.add_argument('--roboflow_workspace', default=None,
                   help='Roboflow workspace slug for an ExDark YOLO '
                        'export — no default; see module docstring.')
    p.add_argument('--roboflow_project', default=None,
                   help='Roboflow project slug for an ExDark YOLO export.')
    p.add_argument('--roboflow_version', type=int, default=None,
                   help='Roboflow dataset version number.')
    p.add_argument('--data_yaml', default=None,
                   help='Path to an already-downloaded-and-filtered '
                        "dataset's data.yaml (skips download+filter).")
    p.add_argument('--dataset_dir', default='./data/lowlight_dataset')
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--model_size', default='n', choices=['n', 's', 'm'],
                   help='YOLOv8 size to fine-tune. n = fastest/smallest, '
                        'matches the lightweight theme of this project.')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--image_size', type=int, default=640)
    p.add_argument('--patience', type=int, default=15)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
