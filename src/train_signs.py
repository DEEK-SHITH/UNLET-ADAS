"""
UNLET-ADAS: Road Sign Detector Training
========================================
Fine-tunes a small YOLOv8 model as a dedicated road-sign detector,
trained separately from the main ADAS YOLOv8 pass (person / car / bus /
traffic light / stop sign, etc. — COCO classes). COCO has no generic
"warning sign" / "speed limit sign" / "crosswalk sign" class, so — same
situation as the pothole detector (src/train_pothole.py) — this can't
be added by just flipping a flag on the existing detector; it needs its
own model trained on a labeled road-sign dataset.

Dataset: defaults to Roboflow's "indian-traffic-signs1" project,
version 4 (https://universe.roboflow.com/indiantrafficsigns/
indian-traffic-signs1, ~70 Indian-context sign classes, ~7.7k images
-- workspace/project/version confirmed via that project's own
"Download Dataset -> YOLOv8" snippet), but --roboflow_workspace/
--roboflow_project/--roboflow_version point this at any Roboflow
road-sign project instead. An earlier default, Roboflow-100's
"road-signs-6ih4y", turned out to be a real-world, Indonesian-language
sign taxonomy that does not generalize to Indian/Vienna-Convention-
style sign designs (confirmed directly: the detector it produced had
zero candidate detections, even at near-zero confidence, on a real
dashcam frame with a clearly visible warning sign). train() reads the
actual class list back from the downloaded data.yaml rather than
assuming one regardless of which project is used, and the app's
sign-drawing code (src/signs.py) assigns colors by hashing the class
name for the same reason — see those files for details.

Usage:
    pip install roboflow ultralytics
    python src/train_signs.py --roboflow_key YOUR_FREE_API_KEY
    # or, for a different dataset (e.g. a region-specific one):
    python src/train_signs.py --roboflow_key YOUR_FREE_API_KEY \
        --roboflow_workspace WORKSPACE --roboflow_project PROJECT \
        --roboflow_version N

Or in Colab (free GPU, no local install needed):
    Open notebooks/UNLET_ADAS_Signs_Colab.ipynb in Google Colab,
    paste your API key into the config cell, and run top to bottom.

    Equivalent manual command if you'd rather run this script directly
    in a Colab cell instead of using the notebook:
    !python src/train_signs.py --roboflow_key YOUR_FREE_API_KEY \
                                --save_dir /content/drive/MyDrive/UNLET_Project/checkpoints \
                                --epochs 100

A free Roboflow account/API key is required to download the dataset
(https://app.roboflow.com — Settings -> API Keys). This script does
not bundle or auto-fetch any dataset without one.

Output: signs_best.pt — drop it in app/ next to zerodce_cbam_best.pt
and pothole_best.pt to enable the "Road Sign Detection" toggle in the
Streamlit app.
"""

import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def download_dataset(api_key, dest_dir, retries=5,
                      workspace='indiantrafficsigns', project_slug='indian-traffic-signs1',
                      version_num=4):
    """
    Download a Roboflow road-sign dataset in YOLOv8 format. Defaults to
    the "indian-traffic-signs1" project, but any Roboflow project can
    be used instead by passing workspace/project_slug/version_num (see
    the --roboflow_workspace/--roboflow_project/--roboflow_version CLI
    args) -- an earlier default's class list turned out not to match
    every use case (see the module docstring's history), so the script
    no longer hardcodes a single project.
    """
    import time
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    # Fallback: if this ever 404s (Universe slugs can be renamed), open
    # the project's Universe page, click "Download Dataset" -> YOLOv8,
    # and copy the exact rf.workspace(...).project(...) snippet
    # Roboflow generates there -- then pass those as
    # --roboflow_workspace/--roboflow_project/--roboflow_version.
    version = project.version(version_num)

    # Roboflow's download occasionally stalls or drops the connection
    # mid-transfer with NO exception raised, silently leaving an
    # incomplete/empty export on disk -- confirmed in practice with
    # this exact dataset: the identical version.download('yolov8')
    # call has succeeded outright in one run and produced an export
    # with no data.yaml anywhere in it in another, back to back, same
    # key, same code. So the data.yaml check below has to be *inside*
    # the retry loop, not just the exception handling -- a "successful"
    # call that produced nothing useful is exactly the failure mode
    # this needs to retry past, not raise on immediately.
    #
    # Deliberately NOT passing location=dest_dir: an earlier version of
    # this function did, and that turned out to reliably produce empty
    # exports; letting the SDK pick its own default location and moving
    # the result into dest_dir ourselves afterward avoids that.
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            dataset = version.download('yolov8')
        except Exception as e:
            last_err = e
            print(f'Download attempt {attempt}/{retries} raised: {e}')
            if attempt < retries:
                time.sleep(5)
            continue

        location = dataset.location
        # Roboflow's SDK has, across versions, sometimes placed the
        # export directly in `location` and sometimes nested it one
        # level deeper. Search for data.yaml rather than assume where
        # it landed, so a download that actually succeeded doesn't
        # look like a failure just because of a path mismatch.
        found = None
        if os.path.exists(os.path.join(location, 'data.yaml')):
            found = location
        else:
            for root, _, files in os.walk(location):
                if 'data.yaml' in files:
                    found = root
                    break

        if found:
            # Move the verified-good download into the
            # caller-requested dest_dir, so callers can still point
            # this at a stable path (e.g. Google Drive) without us
            # having to trust the SDK's location= arg.
            dest_dir = os.path.abspath(dest_dir)
            if os.path.abspath(found) != dest_dir:
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.move(found, dest_dir)
                found = dest_dir
            return found

        contents = (os.listdir(location) if os.path.isdir(location)
                    else '<directory does not exist>')
        last_err = RuntimeError(
            f"reported location '{location}' but no data.yaml "
            f"anywhere under it (contents: {contents})")
        print(f'Download attempt {attempt}/{retries} produced no '
              f'data.yaml: {last_err}')
        if attempt < retries:
            time.sleep(5)

    raise RuntimeError(
        f'Roboflow download failed after {retries} attempts: {last_err}\n'
        'Roboflow downloads have shown intermittent server-side '
        'failures in practice -- the exact same call can succeed once '
        'and come back empty the next time. If it keeps failing after '
        'several retries, wait a few minutes and try again, or '
        f'download it manually from https://universe.roboflow.com/'
        f'{workspace}/{project_slug} (Download Dataset -> YOLOv8 -> '
        'zip), extract it, and pass --data_yaml pointing at the '
        'extracted data.yaml instead of --roboflow_key.')


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
        print(f'Downloading road-sign dataset from Roboflow '
              f'({args.roboflow_workspace}/{args.roboflow_project} '
              f'v{args.roboflow_version})...')
        dataset_dir = download_dataset(
            args.roboflow_key, args.dataset_dir,
            workspace=args.roboflow_workspace,
            project_slug=args.roboflow_project,
            version_num=args.roboflow_version)
        data_yaml = os.path.join(dataset_dir, 'data.yaml')

    # Read the real class list back from the dataset itself rather than
    # hardcoding an assumed one here -- Roboflow Universe listings can
    # differ from what a dataset's name/description implies, and this
    # way the log always reflects what's actually about to be trained,
    # whatever that turns out to be.
    real_classes = '(unknown -- see data.yaml)'
    try:
        import yaml
        with open(data_yaml) as f:
            real_classes = ', '.join(yaml.safe_load(f).get('names', []))
    except Exception:
        pass

    print(f'\nFine-tuning YOLOv8{args.model_size} for road-sign detection')
    print(f'Data     : {data_yaml}')
    print(f'Classes  : {real_classes}')
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
        name='signs_run',
        exist_ok=True,
    )

    best = os.path.join(
        args.save_dir, 'signs_run', 'weights', 'best.pt')
    out = os.path.join(args.save_dir, 'signs_best.pt')
    if os.path.exists(best):
        shutil.copy(best, out)
        print(f'\nTraining complete! Weights saved to: {out}')
        print('Copy this file to app/signs_best.pt to enable it '
              'in the Streamlit app.')
    else:
        print(f'\nExpected weights at {best} but they were not found — '
              'check the training log above for errors.')

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune a YOLOv8 road-sign detector for UNLET-ADAS')
    p.add_argument('--roboflow_key', default=None,
                   help='Free Roboflow API key, used to download the '
                        'public road-sign dataset. Omit if using --data_yaml.')
    p.add_argument('--data_yaml', default=None,
                   help='Path to an already-downloaded dataset\'s '
                        'data.yaml (skips the Roboflow download).')
    p.add_argument('--roboflow_workspace', default='indiantrafficsigns',
                   help='Roboflow workspace slug. Defaults to the '
                        'indiantrafficsigns account; override to train '
                        'on a different road-sign dataset without '
                        'editing this file.')
    p.add_argument('--roboflow_project', default='indian-traffic-signs1',
                   help='Roboflow project slug (see --roboflow_workspace).')
    p.add_argument('--roboflow_version', type=int, default=4,
                   help='Roboflow dataset version number (see '
                        '--roboflow_workspace).')
    p.add_argument('--dataset_dir', default='./data/signs_dataset')
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
