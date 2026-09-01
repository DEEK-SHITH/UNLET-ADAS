"""
UNLET-ADAS: Extra Low-Light Training Data
============================================
Builds directories of *unpaired* low-light images to train the
enhancement model (src/train.py's --extra_low_dirs) on, alongside
LOL's 485 paired low/high images. The project's own paper names this
in its conclusion: LOL is mostly indoor/urban, so real driving
footage and a broader, more diverse low-light photo set should
generalize better, especially on hilly/rural terrain.

No ground truth is needed for these images -- UNLETLoss already runs
in unsupervised-only mode (color constancy / exposure / spatial
consistency / smoothness) for any sample with no paired 'high' image,
so this only broadens what low-light *appearance* the model sees, not
a bottleneck requiring paired data collection.

Two sources, both able to run with zero manual data hunting:

1. extract_video_frames() -- pulls frames straight out of this
   repo's own real night-drive footage (results/original_night_drive.mp4
   by default), sparsely sampled to avoid near-duplicate consecutive
   frames. Zero network dependency; always available.

2. build_exdark_extra() -- reuses src/train_lowlight.py's
   download_exdark_official() (already built for the low-light
   detector) to pull the ~7,363-image ExDark set and flattens it into
   one directory, ignoring its bounding-box annotations entirely
   since enhancement training needs no labels at all -- just a large,
   genuinely diverse set of real low-light photos across 10 lighting
   conditions and both indoor/outdoor scenes.

Usage:
    python src/prepare_extra_lowlight.py --out_dir ./data/extra_lowlight
    # add --include_exdark to also pull ExDark (~1.5GB download)
"""

import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_VIDEO = os.path.join(ROOT, 'results', 'original_night_drive.mp4')


def extract_video_frames(video_path, out_dir, every_n=30, max_frames=None):
    """
    Pull frames out of a video at a fixed stride (default: every 30th
    frame, ~1 per second of most dashcam footage) so consecutive
    frames aren't near-duplicates of each other -- more stride, more
    genuinely distinct scenes per frame extracted.

    Returns the number of frames written.
    """
    import cv2

    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n == 0:
            out_path = os.path.join(
                out_dir, f'{stem}_frame{frame_idx:06d}.jpg')
            cv2.imwrite(out_path, frame)
            written += 1
            if max_frames and written >= max_frames:
                break
        frame_idx += 1
    cap.release()

    print(f'Extracted {written} frames from {os.path.basename(video_path)} '
          f'(every {every_n} of {frame_idx} total) -> {out_dir}')
    return written


def flatten_exdark_images(images_root, out_dir):
    """
    Copy every image out of ExDark's 12 per-class subfolders into one
    flat directory -- enhancement training has no use for the class
    labels, only the raw low-light images themselves. Prefixes each
    filename with its source class folder to avoid any collision
    between images that happen to share a base filename across
    classes.
    """
    from src.train_lowlight import EXDARK_SOURCE_CLASSES

    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for class_folder in EXDARK_SOURCE_CLASSES:
        src_dir = os.path.join(images_root, class_folder)
        if not os.path.isdir(src_dir):
            continue
        for name in os.listdir(src_dir):
            src_path = os.path.join(src_dir, name)
            if not os.path.isfile(src_path):
                continue
            dst_name = f'{class_folder}_{name}'
            shutil.copy(src_path, os.path.join(out_dir, dst_name))
            copied += 1

    print(f'Flattened {copied} ExDark images -> {out_dir}')
    return copied


def build_exdark_extra(dest_dir, download_dir=None):
    """
    Download the official ExDark dataset (reusing
    src/train_lowlight.py's downloader) and flatten it into a single
    unpaired low-light image directory. Needs real internet access to
    Google Drive -- if that's blocked on this network, download it
    manually (see src/train_lowlight.py's module docstring) and pass
    that path directly as one of train.py's --extra_low_dirs instead
    of using this function.
    """
    from src.train_lowlight import download_exdark_official

    download_dir = download_dir or os.path.join(dest_dir, '_exdark_raw')
    images_root, _groundtruth_root = download_exdark_official(download_dir)
    out_dir = os.path.join(dest_dir, 'exdark')
    flatten_exdark_images(images_root, out_dir)
    return out_dir


def build_extra_lowlight_set(dest_dir, video_paths=None, every_n=30,
                              include_exdark=False):
    """
    Orchestrates both sources into dest_dir's subfolders. Returns the
    list of directories actually populated (ready to pass straight to
    train.py's --extra_low_dirs).
    """
    os.makedirs(dest_dir, exist_ok=True)
    built_dirs = []

    video_paths = video_paths if video_paths is not None else [DEFAULT_VIDEO]
    if video_paths:
        frames_dir = os.path.join(dest_dir, 'night_drive_frames')
        total = 0
        for video_path in video_paths:
            if not os.path.exists(video_path):
                print(f'WARNING: video not found, skipping: {video_path}')
                continue
            total += extract_video_frames(video_path, frames_dir, every_n)
        if total:
            built_dirs.append(frames_dir)

    if include_exdark:
        built_dirs.append(build_exdark_extra(dest_dir))

    return built_dirs


def parse_args():
    p = argparse.ArgumentParser(
        description='Build extra unpaired low-light training data '
                    'for src/train.py --extra_low_dirs')
    p.add_argument('--out_dir', default='./data/extra_lowlight')
    p.add_argument('--video', nargs='*', default=None,
                   help='Video(s) to extract frames from. Defaults '
                        "to this repo's own results/"
                        'original_night_drive.mp4.')
    p.add_argument('--every_n', type=int, default=30,
                   help='Keep 1 of every N frames (default 30, so '
                        'consecutive kept frames are genuinely '
                        "distinct rather than near-duplicates).")
    p.add_argument('--include_exdark', action='store_true',
                   help='Also download+flatten the official ExDark '
                        'dataset (~1.5GB, needs Google Drive access) '
                        'as an additional unpaired source.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dirs = build_extra_lowlight_set(
        args.out_dir, video_paths=args.video, every_n=args.every_n,
        include_exdark=args.include_exdark)
    print(f'\nBuilt {len(dirs)} extra low-light dir(s):')
    for d in dirs:
        print(f'  {d}')
    print('\nPass these to src/train.py via --extra_low_dirs, e.g.:')
    print(f"    python src/train.py --extra_low_dirs {' '.join(dirs)}")
