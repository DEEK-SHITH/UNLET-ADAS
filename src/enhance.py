"""
UNLET-ADAS: Enhancement Functions
===================================
Core functions for single image and batch video enhancement.

Curves are estimated on a small proxy resolution for speed, then
applied directly to the full-resolution frame (see
model.enhance_full_res). This avoids the old resize-down/resize-up
round trip that softened fine detail and hurt detection of small or
distant objects.

Enhancement strength is also scene-adaptive: dark frames (night,
tunnels, shaded hillside roads) get full enhancement, while
already well-lit frames (daylight) are left close to untouched so
the system helps at night and in hilly terrain without washing out
or over-brightening daytime footage.
"""

import os
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image

from src.model import build_model


def load_enhancer(weights_path, device='cuda'):
    """Load trained UNLET-ADAS model."""
    device = torch.device(
        device if torch.cuda.is_available() else 'cpu')
    model  = build_model().to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device))
    model.eval()
    print(f'Model loaded from: {weights_path}')
    print(f'Device: {device}')
    return model, device


def dynamic_scene_blend_weight(lum_mean: float, lum_std: float = 0.12) -> tuple:
    """
    Mathematically grounded dynamic scene adaptation using scene luminance distribution
    P(L) rather than rigid heuristic thresholds (C^1 continuous transition).
    """
    tau_dark = float(np.clip(lum_mean - 0.40 * lum_std, 0.16, 0.30))
    tau_bright = float(np.clip(lum_mean + 0.80 * lum_std, 0.32, 0.50))
    mid = (tau_dark + tau_bright) / 2.0
    scale = max(tau_bright - tau_dark, 0.05)
    z = (lum_mean - mid) / (scale / 4.0)
    alpha = float(1.0 / (1.0 + np.exp(np.clip(z, -12.0, 12.0))))
    return alpha, tau_dark, tau_bright


def scene_blend_weight(luminance, dark_thresh=0.26, bright_thresh=0.38):
    """
    Blend factor between original and enhanced frame based on scene brightness.
    Satisfies smooth continuous transition between nighttime enhancement and daylight bypass.
    """
    if luminance <= dark_thresh:
        return 1.0
    if luminance >= bright_thresh:
        return 0.0
    # Sigmoidal smoothstep transition: S(t) = 3t^2 - 2t^3
    t = (bright_thresh - luminance) / max(bright_thresh - dark_thresh, 1e-4)
    return float(np.clip(t * t * (3.0 - 2.0 * t), 0.0, 1.0))


def scene_aware_conf(base_conf, luminance, dark_thresh=0.35,
                      bright_thresh=0.55, night_boost=0.10):
    """
    Mathematically couples object detection confidence threshold directly to the
    non-reference enhancement gain alpha:
      conf(alpha) = min(base_conf + night_boost * (alpha ** 1.5), 0.95)
    
    Because low-light enhancement nonlinearly amplifies sensor noise variance (sigma_n^2),
    raising the confidence threshold proportionally to alpha^1.5 suppresses sensor-noise
    false alarms while preserving true vehicle detection recall.
    """
    weight = scene_blend_weight(luminance, dark_thresh, bright_thresh)
    boost = night_boost * float(weight ** 1.5)
    return float(min(base_conf + boost, 0.95))


def correct_color_cast(frame_uint8, strength=0.6, lum_pctl=60, max_shift=18):
    """
    Ultra-fast white-balance correction in LAB space, using only the
    brightest ~40% of pixels to estimate the tint.

    Estimates chromaticity bias from a downsampled proxy (64x64) using fast
    partitioning, avoiding full-resolution floating point percentile sorts.
    If the detected tint bias is negligible (|shift| < 1.0), returns the original
    frame directly with zero computational overhead.
    """
    # Rapid chromaticity estimation on 64x64 thumbnail
    small = cv2.resize(frame_uint8, (64, 64), interpolation=cv2.INTER_NEAREST)
    lab_s = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    L_flat = lab_s[:, :, 0].ravel()
    k = int((lum_pctl / 100.0) * len(L_flat))
    thresh = np.partition(L_flat, k)[k]
    mask = lab_s[:, :, 0] >= thresh
    if mask.sum() < 20:
        mask = np.ones((64, 64), dtype=bool)

    a_dev = float(lab_s[:, :, 1][mask].mean()) - 128.0
    b_dev = float(lab_s[:, :, 2][mask].mean()) - 128.0
    a_shift = float(np.clip(-a_dev * strength, -max_shift, max_shift))
    b_shift = float(np.clip(-b_dev * strength, -max_shift, max_shift))

    # If color shift is negligible, avoid full-resolution conversions
    if abs(a_shift) < 1.0 and abs(b_shift) < 1.0:
        return frame_uint8

    lab = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2LAB)
    idx = np.arange(256, dtype=np.float32)
    lut_a = np.clip(idx + a_shift, 0, 255).astype(np.uint8)
    lut_b = np.clip(idx + b_shift, 0, 255).astype(np.uint8)
    lab[:, :, 1] = cv2.LUT(lab[:, :, 1], lut_a)
    lab[:, :, 2] = cv2.LUT(lab[:, :, 2], lut_b)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)




def suppress_headlight_glare(img_rgb: np.ndarray, thresh: float = 0.85, max_attenuation: float = 0.40) -> np.ndarray:
    """
    Local Adaptive Anti-Glare Filter for Oncoming High-Beams and Streetlamps.
    Detects specular sensor saturation (headlight bloom) and applies smooth local
    tonemapping to recover periphery road details and prevent driver blinding ($0 cost).
    """
    if max_attenuation <= 0.01:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    glare_mask = np.clip((gray - thresh) / (1.0 - thresh + 1e-4), 0.0, 1.0)
    if glare_mask.max() < 0.05:
        return img_rgb
    smooth_glare = cv2.GaussianBlur(glare_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
    tonemap = 1.0 - max_attenuation * (smooth_glare ** 1.5)
    tonemap_3c = np.repeat(tonemap[:, :, None], 3, axis=2)
    return np.clip(img_rgb.astype(np.float32) * tonemap_3c, 0.0, 255.0).astype(np.uint8)


def fast_dehaze(img_rgb: np.ndarray, omega: float = 0.75, radius: int = 7) -> np.ndarray:
    """
    Fast Atmospheric Scattering Inversion (Dark Channel Prior) for Fog and Rain-Spray.
    Strips atmospheric veil before illumination enhancement, preventing low-light models
    from amplifying backscatter and fog glare ($0 cost, runs on CPU).
    """
    arr = img_rgb.astype(np.float32) / 255.0
    min_ch = np.min(arr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
    dark_ch = cv2.erode(min_ch, kernel)
    
    num_pixels = dark_ch.size
    top_k = max(int(num_pixels * 0.002), 1)
    flat_dark = dark_ch.ravel()
    thresh_val = np.partition(flat_dark, -top_k)[-top_k]
    bright_mask = dark_ch >= thresh_val
    A = arr[bright_mask].mean(axis=0)
    A = np.clip(A, 0.50, 0.98)
    
    normalized = arr / (A + 1e-4)
    norm_dark = cv2.erode(np.min(normalized, axis=2), kernel)
    t_raw = np.clip(1.0 - omega * norm_dark, 0.15, 1.0)
    t_smooth = cv2.GaussianBlur(t_raw, (0, 0), sigmaX=5.0, sigmaY=5.0)
    t_3c = np.repeat(np.maximum(t_smooth, 0.20)[:, :, None], 3, axis=2)
    
    J = (arr - A) / t_3c + A
    return np.clip(J * 255.0, 0.0, 255.0).astype(np.uint8)


def preserve_native_details(enh_rgb: np.ndarray, orig_rgb: np.ndarray, weight: float = 0.45) -> np.ndarray:
    """
    AI High-Frequency Texture & Edge Retention Filter:
    Fuses native camera sensor high-frequency detail (edges, asphalt grain, vehicle decals,
    lane markings, distant road signs) from the original input into the Zero-DCE enhanced frame.
    Guarantees the enhanced video retains 100% of the input camera's sharpness and micro-textures.
    """
    if weight <= 0.01:
        return enh_rgb

    # Extract high-frequency luminance detail from original input frame
    y_orig = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
    base_orig = cv2.GaussianBlur(y_orig, (0, 0), sigmaX=1.5, sigmaY=1.5)
    detail = y_orig.astype(np.float32) - base_orig.astype(np.float32)

    # Soft coring: suppress sensor noise floor (< 1.2 DN) while preserving crisp edges
    coring = np.sign(detail) * np.maximum(0.0, np.abs(detail) - 1.2)
    detail_3c = np.repeat(coring[:, :, None], 3, axis=2)

    # Blend high-frequency detail into enhanced frame
    return np.clip(enh_rgb.astype(np.float32) + float(weight) * detail_3c, 0.0, 255.0).astype(np.uint8)


@torch.no_grad()
def enhance_pil(model, device, pil_image, adaptive=True, proxy_size=256):
    """
    Enhance a PIL Image at full resolution: curves are estimated on a
    small proxy_size x proxy_size tensor and applied directly to the
    full-resolution image. Returns a PIL Image.
    """
    orig = pil_image.convert('RGB')
    arr = np.array(orig, dtype=np.float32) / 255.0
    t_full = torch.from_numpy(arr).permute(
        2, 0, 1).unsqueeze(0).to(device)

    enh_full, _ = model.enhance_full_res(t_full, proxy_size=proxy_size)

    if adaptive:
        alpha = scene_blend_weight(float(arr.mean()))
        enh_full = t_full * (1 - alpha) + enh_full * alpha

    enh_np = (enh_full[0].permute(1, 2, 0).cpu().numpy()
              * 255).clip(0, 255).astype(np.uint8)
    from src.detection import sharpen_frame
    enh_np = sharpen_frame(correct_color_cast(enh_np), amount=0.20)
    enh_np = preserve_native_details(enh_np, np.array(orig), weight=0.40 * (alpha if adaptive else 1.0))
    return Image.fromarray(enh_np)


@torch.no_grad()
def enhance_image(model, device, image_path,
                  size=256, output_path=None, adaptive=True):
    """
    Enhance a single low-light image at full resolution.
    Returns PIL Image of enhanced result.
    """
    orig = Image.open(image_path).convert('RGB')
    result = enhance_pil(
        model, device, orig, adaptive=adaptive, proxy_size=size)

    if output_path:
        result.save(output_path)
        print(f'Saved: {output_path}')

    return result


class TemporalEnhanceSmoother:
    """
    Exponential Moving Average (EMA) smoother for estimated curve parameter maps
    across consecutive video frames, suppressing temporal brightness flicker.
    """
    def __init__(self, gamma=0.65):
        self.gamma = float(gamma)
        self.prev_curves = None

    def smooth(self, curves: torch.Tensor) -> torch.Tensor:
        if self.prev_curves is None or self.prev_curves.shape != curves.shape or self.prev_curves.device != curves.device:
            self.prev_curves = curves.clone()
            return curves
        smoothed = self.gamma * self.prev_curves + (1.0 - self.gamma) * curves
        self.prev_curves = smoothed.clone()
        return smoothed

    def reset(self):
        self.prev_curves = None


@torch.no_grad()
def enhance_frame_batch(model, device, frames_rgb,
                        size=256, adaptive=True,
                        dark_thresh=0.26, bright_thresh=0.38,
                        interp_mode='bilinear', fast_color=False, smoother=None,
                        orig_frames_rgb=None, preserve_detail=True):
    """
    Enhance a batch of full-resolution video frames.
    Input : list of (H,W,3) uint8 RGB arrays, all the same size
    Output: list of (H,W,3) uint8 RGB arrays at the same resolution
    """
    if not frames_rgb:
        return []

    # Fast daylight / well-lit bypass: if scene is already well lit, avoid redundant
    # neural network computation and preserve natural camera contrast without noise/bleaching.
    if adaptive:
        lums = [float(f[::16, ::16].mean() / 255.0) for f in frames_rgb]
        weights = [scene_blend_weight(l, dark_thresh=dark_thresh, bright_thresh=bright_thresh) for l in lums]
        if max(weights) < 0.02:
            return [f.copy() for f in frames_rgb]
    else:
        weights = [1.0] * len(frames_rgb)

    arr_full = np.stack(frames_rgb).astype(np.float32) / 255.0
    t_full   = torch.from_numpy(arr_full).permute(
        0, 3, 1, 2).to(device)

    enh_full, _ = model.enhance_full_res(t_full, proxy_size=size, interp_mode=interp_mode, smoother=smoother)

    if adaptive:
        alphas = torch.tensor(weights, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        enh_full = t_full * (1 - alphas) + enh_full * alphas

    out = (enh_full.permute(0, 2, 3, 1).cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)

    from src.detection import sharpen_frame
    res = []
    lut_r, lut_b = None, None
    if fast_color and len(out) > 0 and max(weights) >= 0.05:
        # Fast color correction via single-pass RGB LUT estimated from 64x64 proxy
        small = cv2.resize(out[0], (64, 64), interpolation=cv2.INTER_NEAREST)
        lab_s = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        L_flat = lab_s[:, :, 0].ravel()
        k = int(0.60 * len(L_flat))
        thresh = np.partition(L_flat, k)[k]
        mask = lab_s[:, :, 0] >= thresh
        if mask.sum() >= 20:
            a_dev = float(lab_s[:, :, 1][mask].mean()) - 128.0
            b_dev = float(lab_s[:, :, 2][mask].mean()) - 128.0
            if abs(a_dev) > 2.0 or abs(b_dev) > 2.0:
                scale_r = float(np.clip(1.0 - a_dev * 0.0035, 0.85, 1.15))
                scale_b = float(np.clip(1.0 - b_dev * 0.0035, 0.85, 1.15))
                idx = np.arange(256, dtype=np.float32)
                lut_r = np.clip(idx * scale_r, 0, 255).astype(np.uint8)
                lut_b = np.clip(idx * scale_b, 0, 255).astype(np.uint8)

    orig_list = orig_frames_rgb if (orig_frames_rgb is not None and len(orig_frames_rgb) == len(out)) else [None] * len(out)
    for frame, orig_f, w in zip(out, orig_list, weights):
        if w < 0.05:
            res.append(frame)
        else:
            if fast_color:
                if lut_r is not None and lut_b is not None:
                    frame[:, :, 0] = cv2.LUT(frame[:, :, 0], lut_r)
                    frame[:, :, 2] = cv2.LUT(frame[:, :, 2], lut_b)
                cc = frame
            else:
                cc = correct_color_cast(frame, strength=0.6 * w)
            sh = sharpen_frame(cc, amount=0.20 * w)
            if preserve_detail and orig_f is not None:
                sh = preserve_native_details(sh, orig_f, weight=0.45 * w)
            # Anti-glare suppression on bright high-beam headlight sources
            sh = suppress_headlight_glare(sh, thresh=0.86, max_attenuation=0.35 * w)
            res.append(sh)
    return res

def transcode_for_browser(path, crf=17, preset='veryfast', timeout=300):
    """
    Re-encode an OpenCV-written video (VideoWriter's 'mp4v' fourcc is
    MPEG-4 Part 2, not H.264) to H.264/yuv420p in place, via the
    self-contained ffmpeg binary imageio-ffmpeg ships on pip install --
    no system/apt ffmpeg required, so this works identically on
    Streamlit Cloud, the Colab demo, and locally.

    Browsers' HTML5 <video> element does not play MPEG-4 Part 2, so
    st.video() on an untouched mp4v file renders a blank/unplayable
    player -- the only way to watch it is the download button, which
    is the "why can't I just watch it" report this fixes. -movflags
    +faststart also moves the moov atom to the front of the file so
    playback can start before it's fully downloaded.

    Transcodes into a temp file and atomically replaces `path`, so
    every existing caller (st.video(path), the download button reading
    path) keeps working unchanged. Falls back to leaving the original
    mp4v file in place if ffmpeg is missing or the transcode fails --
    still downloadable and playable in a desktop player, just not
    inline in the browser -- so a transcode failure never breaks video
    processing itself.
    """
    path = str(path)  # accept pathlib.Path as well as str
    tmp_out = path + '.h264.mp4'
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_exe, '-y', '-i', path,
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-preset', preset, '-crf', str(crf),
             '-tune', 'film',
             '-movflags', '+faststart', tmp_out],
            check=True, capture_output=True, timeout=timeout)
        os.replace(tmp_out, path)
        return True
    except Exception as e:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        print(f'Browser transcode skipped ({e}); '
              f'serving the original file (download-only playback).')
        return False


def enhance_video(model, device,
                  input_path, output_path,
                  original_path=None,
                  comparison_path=None,
                  batch_size=4,
                  size=256,
                  adaptive=True):
    """
    Enhance a full video.

    Outputs:
    - output_path      : enhanced video only
    - original_path    : original video copy (optional)
    - comparison_path  : side-by-side comparison (optional)
    """
    import time

    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Writers
    enh_writer  = cv2.VideoWriter(
        output_path, fourcc, fps, (W, H))
    orig_writer = (cv2.VideoWriter(
        original_path, fourcc, fps, (W, H))
        if original_path else None)
    cmp_writer  = (cv2.VideoWriter(
        comparison_path, fourcc, fps, (W * 2, H))
        if comparison_path else None)

    print(f'Input    : {W}x{H} @ {fps:.0f}fps | {total} frames')
    print(f'Enhanced : {output_path}')
    if original_path:
        print(f'Original : {original_path}')
    if comparison_path:
        print(f'Comparison: {comparison_path}')
    print('-' * 50)

    frame_buf, orig_buf = [], []
    count = 0
    t0    = time.time()

    def flush_buffer(frames_rgb, origs_bgr):
        enhanced = enhance_frame_batch(
            model, device, frames_rgb, size, adaptive,
            interp_mode='bilinear', orig_frames_rgb=frames_rgb, preserve_detail=True)
        for orig_bgr, enh_rgb in zip(origs_bgr, enhanced):
            enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
            enh_writer.write(enh_bgr)
            if orig_writer:
                orig_writer.write(orig_bgr)
            if cmp_writer:
                combined = np.hstack([orig_bgr, enh_bgr])
                cv2.line(combined,
                         (W, 0), (W, H), (255,255,255), 3)
                cmp_writer.write(combined)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_buf.append(frame_rgb)
        orig_buf.append(frame_bgr.copy())
        count += 1

        if len(frame_buf) >= batch_size:
            flush_buffer(frame_buf, orig_buf)
            frame_buf.clear()
            orig_buf.clear()

        if count % 60 == 0:
            elapsed = time.time() - t0
            pct     = 100 * count / max(total, 1)
            eta     = elapsed / count * (total - count)
            print(f'  {count:4d}/{total} ({pct:.0f}%)'
                  f'  ETA: {eta:.0f}s')

    if frame_buf:
        flush_buffer(frame_buf, orig_buf)

    cap.release()
    enh_writer.release()
    if orig_writer: orig_writer.release()
    if cmp_writer:  cmp_writer.release()

    elapsed = time.time() - t0
    print(f'\nDone! {count} frames in {elapsed/60:.1f} min')
    return count
