"""
UNLET-ADAS: Gradio Web Application
=====================================
Deployed on HuggingFace Spaces
"""

import gradio as gr
import torch
import numpy as np
import cv2
import sys
import os
import tempfile
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import build_model
from src.enhance import scene_blend_weight, correct_color_cast

# ── Load model ────────────────────────────────
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model   = build_model().to(DEVICE)

# Load weights if available
WEIGHTS = os.path.join(os.path.dirname(__file__), 'zerodce_cbam_best.pt')
if os.path.exists(WEIGHTS):
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    print(f'Weights loaded from {WEIGHTS}')
else:
    print('No weights found — using untrained model')
model.eval()


# ── Enhancement functions ─────────────────────
@torch.no_grad()
def enhance_pil(pil_image, adaptive=True):
    """
    Curves are estimated on a small 256px proxy then applied at the
    original resolution, avoiding the blur of a resize round-trip.
    Adaptive blending skips over-brightening already well-lit frames.
    """
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(
        2,0,1).unsqueeze(0).to(DEVICE)
    enh, _ = model.enhance_full_res(t, proxy_size=256)
    if adaptive:
        alpha = scene_blend_weight(float(arr.mean()))
        enh   = t * (1 - alpha) + enh * alpha
    out = (enh[0].permute(1,2,0).cpu().numpy()
           * 255).clip(0,255).astype(np.uint8)
    out = correct_color_cast(out)
    return Image.fromarray(out)


def enhance_image_fn(input_image):
    if input_image is None:
        return None, None, "Please upload an image."
    pil   = Image.fromarray(input_image).convert('RGB')
    enh   = enhance_pil(pil)
    auto  = ImageOps.autocontrast(pil)

    orig_b = np.array(pil).mean() / 255
    enh_b  = np.array(enh).mean() / 255
    impr   = (enh_b - orig_b) / max(orig_b, 0.01) * 100

    info = (f"**Enhancement Results**\n\n"
            f"- Input brightness  : {orig_b:.3f}\n"
            f"- Enhanced brightness: {enh_b:.3f}\n"
            f"- Improvement       : +{impr:.1f}%\n\n"
            f"Model: Zero-DCE++ with CBAM Attention\n"
            f"GitHub: github.com/DEEK-SHITH/UNLET-ADAS")
    return np.array(enh), np.array(auto), info


@torch.no_grad()
def enhance_video_fn(video_path, progress=gr.Progress()):
    if video_path is None:
        return None, "Please upload a video."

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit to 10 seconds for demo
    max_frames = min(total, int(10 * fps))

    out_path = os.path.join(
        tempfile.gettempdir(), 'enhanced_output.mp4')
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    writer   = cv2.VideoWriter(
        out_path, fourcc, fps, (W * 2, H))

    from src.enhance import enhance_frame_batch

    frames, origs = [], []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        origs.append(frame.copy())
        count += 1

        if len(frames) >= 4:
            enhanced = enhance_frame_batch(
                model, DEVICE, frames, size=256, adaptive=True)
            for orig_bgr, enh_rgb in zip(origs, enhanced):
                enh_bgr  = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
                combined = np.hstack([orig_bgr, enh_bgr])
                cv2.line(combined,(W,0),(W,H),(255,255,255),3)
                cv2.putText(combined,'ORIGINAL',
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(80,80,255),2)
                cv2.putText(combined,'UNLET ENHANCED',
                    (W+10,30),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(50,220,80),2)
                writer.write(combined)
            frames.clear(); origs.clear()
            progress(count/max_frames)

    cap.release(); writer.release()
    info = (f"**Video Enhancement Complete**\n\n"
            f"- Frames processed : {count}\n"
            f"- Output format    : Side-by-side comparison\n"
            f"- Left             : Original (dark)\n"
            f"- Right            : UNLET Enhanced\n\n"
            f"Note: Limited to 10 seconds for demo.")
    return out_path, info


# ── Gradio Interface ──────────────────────────
with gr.Blocks(
    title="UNLET-ADAS",
    theme=gr.themes.Base(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="slate")
) as demo:

    gr.Markdown("""
    # 🚗 UNLET-ADAS
    ## Real-Time Low-Light Enhancement for Intelligent Vehicle Systems
    **B.E. Major Project | SJBIT Bengaluru | CSE 2025-26**

    > Upload a dark/night image or video to enhance it using
    > Zero-DCE++ with CBAM Attention — designed for ADAS systems.

    [![GitHub](https://img.shields.io/badge/GitHub-DEEK--SHITH/UNLET--ADAS-black)](https://github.com/DEEK-SHITH/UNLET-ADAS)
    """)

    with gr.Tabs():

        # ── Image Tab ─────────────────────────
        with gr.TabItem("🖼️ Image Enhancement"):
            gr.Markdown("### Upload a low-light image to enhance")
            with gr.Row():
                img_input = gr.Image(
                    label="Input (Low-Light)",
                    type="numpy", height=350)
                img_enh   = gr.Image(
                    label="UNLET Enhanced",
                    type="numpy", height=350)
                img_auto  = gr.Image(
                    label="AutoContrast (Baseline)",
                    type="numpy", height=350)
            img_info = gr.Markdown()
            img_btn  = gr.Button(
                "✨ Enhance Image",
                variant="primary", size="lg")
            img_btn.click(
                fn=enhance_image_fn,
                inputs=img_input,
                outputs=[img_enh, img_auto, img_info])

            gr.Examples(
                examples=[],
                inputs=img_input)

        # ── Video Tab ─────────────────────────
        with gr.TabItem("🎬 Video Enhancement"):
            gr.Markdown("""
            ### Upload a night driving video to enhance
            **Output:** Side-by-side comparison (Original | UNLET Enhanced)
            *Note: Demo limited to first 10 seconds*
            """)
            with gr.Row():
                vid_input  = gr.Video(
                    label="Input Video (Night/Dark)")
                vid_output = gr.Video(
                    label="Enhanced Output (Side-by-Side)")
            vid_info = gr.Markdown()
            vid_btn  = gr.Button(
                "🚀 Enhance Video",
                variant="primary", size="lg")
            vid_btn.click(
                fn=enhance_video_fn,
                inputs=vid_input,
                outputs=[vid_output, vid_info])

        # ── About Tab ─────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About UNLET-ADAS

            ### Architecture
            | Component | Detail |
            |---|---|
            | Enhancement | Zero-DCE++ with CBAM Attention |
            | Attention | Channel + Spatial (CBAM) |
            | Iterations | 8 curve estimation steps |
            | Training Data | LOL Dataset (485 pairs) |

            ### Results
            | Metric | Original | UNLET Enhanced |
            |---|---|---|
            | PSNR | ~8 dB | ~21 dB |
            | SSIM | 0.18 | 0.72 |

            ### How It Works
            1. Input dark video frame
            2. Zero-DCE++ estimates enhancement curves
            3. CBAM attention focuses on dark regions
            4. Iterative curve application brightens the image
            5. Output is a naturally bright, color-accurate frame

            ### Project
            - **College:** SJBIT Bengaluru
            - **Degree:** B.E. Computer Science
            - **Year:** 2025-26
            - **GitHub:** [DEEK-SHITH/UNLET-ADAS](https://github.com/DEEK-SHITH/UNLET-ADAS)
            """)

demo.launch()