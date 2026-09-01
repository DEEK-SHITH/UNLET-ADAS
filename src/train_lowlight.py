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

Dataset sourcing: ExDark's official distribution (see
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) is not a
ready-made YOLO export, and several independent YOLO-format mirrors
of varying completeness exist on Roboflow Universe. There's no single
canonical one to hardcode with confidence, so — unlike the pothole
detector's dataset — this script requires you to supply the exact
workspace/project/version yourself:
  1. Go to https://universe.roboflow.com and search "ExDark"
  2. Pick a project whose image count is close to the real dataset's
     ~7,363 images (a much smaller count usually means a partial
     subset, e.g. person-only)
  3. Click "Download Dataset" -> YOLOv8 -> "Show download code" and
     read off the workspace/project/version from the generated
     rf.workspace(...).project(...).version(...) snippet
This script also prints the image/label counts it actually finds
after download+filtering, so an incomplete mirror is caught
immediately rather than producing a silently-undertrained model.

Usage:
    pip install roboflow ultralytics
    python src/train_lowlight.py --roboflow_key YOUR_FREE_API_KEY \
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
    else:
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
    p.add_argument('--roboflow_key', default=None,
                   help='Free Roboflow API key. Omit if using --data_yaml.')
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
