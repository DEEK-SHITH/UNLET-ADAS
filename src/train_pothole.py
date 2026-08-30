"""
UNLET-ADAS: Pothole Detector Training
========================================
Fine-tunes a small YOLOv8 model as a dedicated single-class pothole
detector, trained separately from the main ADAS YOLOv8 pass (person /
car / bus / traffic light / stop sign, etc. — COCO classes). COCO has
no "pothole" class, so this can't be added by just flipping a flag on
the existing detector; it needs its own model trained on a labeled
pothole dataset.

Dataset: "Pothole Object Detection Dataset" on Roboflow's curated
Public Datasets collection (not a random community upload) —
665 road images with pothole bounding boxes, originally created and
shared by Atikur Rahman Chitholian as part of an undergraduate
thesis. https://public.roboflow.com/object-detection/pothole

Usage:
    pip install roboflow ultralytics
    python src/train_pothole.py --roboflow_key YOUR_FREE_API_KEY

Or in Colab (free GPU, no local install needed):
    Open notebooks/UNLET_ADAS_Pothole_Colab.ipynb in Google Colab,
    paste your API key into the config cell, and run top to bottom.

    Equivalent manual command if you'd rather run this script directly
    in a Colab cell instead of using the notebook:
    !python src/train_pothole.py --roboflow_key YOUR_FREE_API_KEY \
                                  --save_dir /content/drive/MyDrive/UNLET_Project/checkpoints \
                                  --epochs 100

A free Roboflow account/API key is required to download the dataset
(https://app.roboflow.com — Settings -> API Keys). This script does
not bundle or auto-fetch any dataset without one.

Output: pothole_best.pt — drop it in app/ next to
zerodce_cbam_best.pt to enable the "Pothole Detection" toggle in the
Streamlit app.
"""

import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def download_dataset(api_key, dest_dir, retries=3):
    """
    Download the Roboflow public pothole dataset in YOLOv8 format.
    This is the same 665-image Chitholian pothole dataset listed at
    public.roboflow.com/object-detection/pothole, mirrored on
    Universe under Roboflow's own account.
    """
    import time
    from roboflow import Roboflow

    os.makedirs(dest_dir, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace('brad-dwyer').project('pothole-voxrl')
    # Fallback: if this ever 404s (Universe slugs can be renamed),
    # open https://universe.roboflow.com/brad-dwyer/pothole-voxrl,
    # click "Download Dataset" -> YOLOv8, and copy the exact
    # rf.workspace(...).project(...) snippet Roboflow generates there.
    version = project.version(1)

    # Roboflow's download occasionally stalls or drops the connection
    # mid-transfer with no exception raised, silently leaving an
    # incomplete/empty export on disk. Retry a few times before
    # concluding the download itself is broken.
    dataset = None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            dataset = version.download('yolov8', location=dest_dir)
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
    # Roboflow's SDK has, across versions, sometimes placed the export
    # directly in `location` and sometimes nested it one level deeper
    # (e.g. location/<project>-<version>/data.yaml) despite the
    # explicit `location=` argument above. Search for data.yaml rather
    # than assume where it landed, so a download that actually
    # succeeded doesn't look like a failure just because of a path
    # mismatch.
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


def train(args):
    from ultralytics import YOLO

    os.makedirs(args.save_dir, exist_ok=True)

    if args.data_yaml:
        data_yaml = args.data_yaml
        print(f'Using existing dataset: {data_yaml}')
    else:
        if not args.roboflow_key:
            raise SystemExit(
                'Pass --roboflow_key YOUR_KEY (free account at '
                'https://app.roboflow.com) or --data_yaml to point at '
                'an already-downloaded dataset in YOLOv8 format.')
        print('Downloading pothole dataset from Roboflow...')
        dataset_dir = download_dataset(args.roboflow_key, args.dataset_dir)
        data_yaml = os.path.join(dataset_dir, 'data.yaml')

    print(f'\nFine-tuning YOLOv8{args.model_size} for pothole detection')
    print(f'Data     : {data_yaml}')
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
        name='pothole_run',
        exist_ok=True,
    )

    best = os.path.join(
        args.save_dir, 'pothole_run', 'weights', 'best.pt')
    out = os.path.join(args.save_dir, 'pothole_best.pt')
    if os.path.exists(best):
        shutil.copy(best, out)
        print(f'\nTraining complete! Weights saved to: {out}')
        print('Copy this file to app/pothole_best.pt to enable it '
              'in the Streamlit app.')
    else:
        print(f'\nExpected weights at {best} but they were not found — '
              'check the training log above for errors.')

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune a YOLOv8 pothole detector for UNLET-ADAS')
    p.add_argument('--roboflow_key', default=None,
                   help='Free Roboflow API key, used to download the '
                        'public pothole dataset. Omit if using --data_yaml.')
    p.add_argument('--data_yaml', default=None,
                   help='Path to an already-downloaded dataset\'s '
                        'data.yaml (skips the Roboflow download).')
    p.add_argument('--dataset_dir', default='./data/pothole_dataset')
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--model_size', default='n', choices=['n', 's', 'm'],
                   help='YOLOv8 size to fine-tune. n = fastest/smallest, '
                        'matches the lightweight theme of this project.')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--image_size', type=int, default=640)
    p.add_argument('--patience', type=int, default=20)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
