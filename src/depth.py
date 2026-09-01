"""
UNLET-ADAS: Monocular Depth Estimation
=========================================
MiDaS small — a lightweight (~21MB) monocular depth model — run as a
second inference pass on the enhanced frame, giving estimate_risk() an
actual per-pixel relative-depth lookup instead of pure box geometry.

MiDaS predicts *relative* inverse depth (disparity-like: higher value
means closer to the camera), not metric distance — a single
uncalibrated RGB camera can't recover true distance without extra
information (camera intrinsics, a known object size, stereo, etc.).
What it does give is a genuine per-pixel ordering of what's near vs.
far in the scene, which box geometry alone cannot: a large box could
be a big nearby car OR a large but distant truck, and box size alone
can't tell them apart.
"""
import numpy as np
import torch
import torch.nn.functional as F


def load_midas(device):
    """
    Load MiDaS_small via torch.hub — the lightweight variant, chosen
    to keep this a second CPU-friendly pass alongside the enhancer
    and YOLO detector, not a third heavyweight network. Returns
    (model, transform, True) on success, or (None, None, False) if
    the load fails for any reason (no internet, torch.hub API
    changes, first-run download blocked, etc.), so the app falls
    back to the geometry heuristic instead of crashing — same
    graceful-degradation pattern as the optional pothole detector.
    """
    try:
        model = torch.hub.load(
            'intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        model.to(device).eval()
        transforms = torch.hub.load(
            'intel-isl/MiDaS', 'transforms', trust_repo=True)
        return model, transforms.small_transform, True
    except Exception:
        return None, None, False


@torch.no_grad()
def estimate_depth_map(model, transform, device, image_rgb):
    """
    Run MiDaS on image_rgb (HxWx3 uint8) and return a per-pixel
    relative depth map at the image's own resolution, min-max
    normalized to 0..1 per frame (1.0 = nearest point in this frame,
    0.0 = farthest). MiDaS's raw output is unitless/relative rather
    than metric, so this per-frame normalization is what makes a
    fixed proximity threshold meaningful frame to frame.
    """
    h, w = image_rgb.shape[:2]
    inp = transform(image_rgb).to(device)
    pred = model(inp)
    pred = F.interpolate(
        pred.unsqueeze(1), size=(h, w),
        mode='bicubic', align_corners=False).squeeze(1).squeeze(0)
    depth = pred.cpu().numpy()
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return ((depth - lo) / (hi - lo)).astype(np.float32)


def sample_proximity(depth_map, x1, y1, x2, y2):
    """
    Average depth (0..1, higher = closer) over the central half of a
    detection box rather than a single pixel or the full box — more
    robust to noise and to box edges that often land on background
    rather than the object itself.
    """
    h, w = depth_map.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px1 = int(np.clip(x1 + bw * 0.25, 0, w - 1))
    px2 = int(np.clip(x2 - bw * 0.25, px1 + 1, w))
    py1 = int(np.clip(y1 + bh * 0.25, 0, h - 1))
    py2 = int(np.clip(y2 - bh * 0.25, py1 + 1, h))
    patch = depth_map[py1:py2, px1:px2]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def classify_proximity(proximity, high_thresh=0.65, med_thresh=0.35):
    """
    Bucket a 0..1 proximity value (see sample_proximity) into
    LOW/MEDIUM/HIGH risk. Thresholds are against the per-frame
    min-max-normalized depth, not a metric distance.
    """
    if proximity >= high_thresh:
        return 'HIGH'
    if proximity >= med_thresh:
        return 'MEDIUM'
    return 'LOW'
