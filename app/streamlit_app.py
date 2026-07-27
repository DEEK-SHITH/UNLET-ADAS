"""
UNLET-ADAS: Streamlit Web Application
Real-Time Low-Light Enhancement for Intelligent Vehicle Systems
B.E. Major Project | SJBIT Bengaluru | CSE 2025-26
"""

import streamlit as st
import torch
import numpy as np
import cv2
import sys
import os
import io
import time
from PIL import Image, ImageOps

# Page config
st.set_page_config(
    page_title="UNLET-ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: #0f172a;}
[data-testid="stSidebar"] {background: #1e293b;}
h1,h2,h3,p,label {color: #e2e8f0 !important;}
.metric-box {
    background: #1e293b;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    border: 1px solid #334155;
}
.metric-val {font-size: 2rem; font-weight: 700; color: #22c55e;}
.metric-lbl {font-size: 0.85rem; color: #94a3b8;}
.stButton>button {
    background: linear-gradient(135deg, #22c55e, #0ea5e9);
    color: white; border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────
@st.cache_resource
def load_model():
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.model import build_model

    DEVICE  = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model   = build_model().to(DEVICE)

    weights = os.path.join(
        os.path.dirname(__file__), 'zerodce_cbam_best.pt')

    if os.path.exists(weights):
        model.load_state_dict(
            torch.load(weights, map_location=DEVICE))
        status = f'Model loaded on {DEVICE}'
    else:
        status = 'No weights found — upload zerodce_cbam_best.pt to app/'

    model.eval()
    return model, DEVICE, status


# ── Enhancement ───────────────────────────────
@torch.no_grad()
def enhance_pil(model, device, pil_image):
    orig_size = pil_image.size
    arr = np.array(
        pil_image.resize((256, 256)),
        dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(
        2, 0, 1).unsqueeze(0).to(device)
    enh, _ = model(t)
    out = (enh[0].permute(1, 2, 0).cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out).resize(
        orig_size, Image.BICUBIC)


@torch.no_grad()
def enhance_video(model, device, video_path,
                  max_seconds=10):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_fr = min(total, int(max_seconds * fps))

    enh_path  = '/tmp/enhanced.mp4'
    orig_path = '/tmp/original.mp4'
    cmp_path  = '/tmp/comparison.mp4'
    fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
    enh_w     = cv2.VideoWriter(enh_path,  fourcc, fps, (W, H))
    orig_w    = cv2.VideoWriter(orig_path, fourcc, fps, (W, H))
    cmp_w     = cv2.VideoWriter(cmp_path,  fourcc, fps, (W*2, H))

    fb, ob = [], []
    count  = 0
    prog   = st.progress(0, text='Enhancing frames...')

    while count < max_fr:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fb.append(cv2.resize(rgb, (256, 256)))
        ob.append(frame.copy())
        count += 1

        if len(fb) >= 8 or count == max_fr:
            arr = np.stack(fb).astype(np.float32)/255.0
            t   = torch.from_numpy(arr).permute(
                0,3,1,2).to(device)
            enh, _ = model(t)
            enp    = (enh.permute(0,2,3,1).cpu().numpy()
                      *255).clip(0,255).astype(np.uint8)

            for orig_bgr, enh_rgb in zip(ob, enp):
                ef = cv2.resize(enh_rgb, (W, H))
                eb = cv2.cvtColor(ef, cv2.COLOR_RGB2BGR)

                # Labels
                cv2.putText(orig_bgr,'ORIGINAL',
                    (10,35),cv2.FONT_HERSHEY_DUPLEX,
                    1.0,(80,80,255),2,cv2.LINE_AA)
                cv2.putText(eb,'UNLET ENHANCED',
                    (10,35),cv2.FONT_HERSHEY_DUPLEX,
                    1.0,(50,220,80),2,cv2.LINE_AA)

                enh_w.write(eb)
                orig_w.write(orig_bgr)
                cmp = np.hstack([orig_bgr, eb])
                cv2.line(cmp,(W,0),(W,H),(255,255,255),3)
                cmp_w.write(cmp)

            fb.clear(); ob.clear()
            prog.progress(min(count/max_fr, 1.0),
                         text=f'Processing {count}/{max_fr} frames...')

    cap.release()
    enh_w.release(); orig_w.release(); cmp_w.release()
    prog.progress(1.0, text='Done!')
    return enh_path, orig_path, cmp_path, count


# ── UI ────────────────────────────────────────
# Header
st.markdown("""
<h1 style='text-align:center; color:#22c55e;'>
🚗 UNLET-ADAS
</h1>
<p style='text-align:center; color:#94a3b8; font-size:1.1rem;'>
Real-Time Low-Light Enhancement for Intelligent Vehicle Systems
</p>
<p style='text-align:center; color:#64748b;'>
B.E. Major Project | SJBIT Bengaluru | CSE 2025-26
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model
with st.spinner('Loading UNLET-ADAS model...'):
    model, DEVICE, status = load_model()

# Sidebar
st.sidebar.markdown("## ⚙️ System Info")
if 'cuda' in str(DEVICE):
    st.sidebar.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("💻 Running on CPU")
st.sidebar.markdown(f"`{status}`")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
**Model:** Zero-DCE++ + CBAM Attention

**Components:**
- Channel Attention
- Spatial Attention
- 8-iteration curve enhancement
- Perceptual + SSIM loss

**Training:** LOL Dataset (485 pairs)

**GitHub:** [DEEK-SHITH/UNLET-ADAS](https://github.com/DEEK-SHITH/UNLET-ADAS)
""")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Enhancement",
    "🎬 Video Enhancement",
    "ℹ️ About Project"
])

# ── Image Tab ─────────────────────────────────
with tab1:
    st.header("Image Enhancement")
    st.markdown("Upload a dark/night image to enhance it")

    uploaded = st.file_uploader(
        "Choose an image",
        type=['jpg','jpeg','png','bmp'],
        key='img')

    if uploaded:
        pil = Image.open(uploaded).convert('RGB')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📷 Original")
            st.image(pil, use_container_width=True)
            orig_b = np.array(pil).mean()/255
            st.caption(f"Brightness: {orig_b:.3f}")

        if st.button("✨ Enhance Image", key='enh_img'):
            with st.spinner("Enhancing..."):
                t0     = time.time()
                enh    = enhance_pil(model, DEVICE, pil)
                auto   = ImageOps.autocontrast(pil)
                ms     = (time.time()-t0)*1000
                enh_b  = np.array(enh).mean()/255
                auto_b = np.array(auto).mean()/255

            with col2:
                st.subheader("✨ UNLET Enhanced")
                st.image(enh, use_container_width=True)
                st.caption(f"Brightness: {enh_b:.3f}")

            with col3:
                st.subheader("🔆 AutoContrast")
                st.image(auto, use_container_width=True)
                st.caption(f"Brightness: {auto_b:.3f}")

            # Metrics
            st.markdown("---")
            st.subheader("📊 Enhancement Metrics")
            m1,m2,m3,m4 = st.columns(4)
            impr = (enh_b-orig_b)/max(orig_b,0.01)*100

            with m1:
                st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{orig_b:.3f}</div>
                <div class='metric-lbl'>Input Brightness</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{enh_b:.3f}</div>
                <div class='metric-lbl'>Enhanced Brightness</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>+{impr:.1f}%</div>
                <div class='metric-lbl'>Improvement</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class='metric-box'>
                <div class='metric-val'>{ms:.0f}ms</div>
                <div class='metric-lbl'>Inference Time</div>
                </div>""", unsafe_allow_html=True)

            # Download
            st.markdown("---")
            buf = io.BytesIO()
            enh.save(buf, format='PNG')
            st.download_button(
                "⬇️ Download Enhanced Image",
                data=buf.getvalue(),
                file_name=f"enhanced_{uploaded.name}",
                mime='image/png',
                use_container_width=True)

# ── Video Tab ─────────────────────────────────
with tab2:
    st.header("Video Enhancement")
    st.markdown("""
    Upload a night driving video to enhance it.
    **Output:** 3 separate video files
    - `enhanced.mp4` — enhanced frames only
    - `original.mp4` — original frames only
    - `comparison.mp4` — side by side
    """)

    vid_upload = st.file_uploader(
        "Choose a video (MP4/AVI/MOV)",
        type=['mp4','avi','mov'],
        key='vid')

    max_sec = st.slider(
        "Max seconds to process", 5, 60, 10)

    if vid_upload:
        tmp_path = f'/tmp/{vid_upload.name}'
        with open(tmp_path, 'wb') as f:
            f.write(vid_upload.read())

        cap   = cv2.VideoCapture(tmp_path)
        fps_v = cap.get(cv2.CAP_PROP_FPS)
        W_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tot_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        st.info(f"Video: {W_v}×{H_v} @ {fps_v:.0f}fps | "
                f"{tot_v} frames | "
                f"{tot_v/fps_v:.1f} seconds")

        if st.button("🚀 Enhance Video", key='enh_vid'):
            with st.spinner("Processing video..."):
                enh_p, orig_p, cmp_p, n = enhance_video(
                    model, DEVICE, tmp_path, max_sec)

            st.success(f"Done! Processed {n} frames")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Enhanced Video")
                st.video(enh_p)
                with open(enh_p,'rb') as f:
                    st.download_button(
                        "⬇️ Download Enhanced",
                        f.read(),
                        file_name='enhanced.mp4',
                        mime='video/mp4',
                        use_container_width=True)

            with col_b:
                st.subheader("Side-by-Side Comparison")
                st.video(cmp_p)
                with open(cmp_p,'rb') as f:
                    st.download_button(
                        "⬇️ Download Comparison",
                        f.read(),
                        file_name='comparison.mp4',
                        mime='video/mp4',
                        use_container_width=True)

            with open(orig_p,'rb') as f:
                st.download_button(
                    "⬇️ Download Original",
                    f.read(),
                    file_name='original.mp4',
                    mime='video/mp4',
                    use_container_width=True)

# ── About Tab ─────────────────────────────────
with tab3:
    st.header("About UNLET-ADAS")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Project Details
| Field | Detail |
|---|---|
| **Project** | UNLET-ADAS |
| **College** | SJBIT Bengaluru |
| **Department** | Computer Science |
| **Year** | 2025-26 |
| **Degree** | B.E. Major Project |
        """)

    with col2:
        st.markdown("""
### Technical Details
| Component | Detail |
|---|---|
| **Model** | Zero-DCE++ + CBAM |
| **Training Data** | LOL Dataset |
| **Framework** | PyTorch |
| **PSNR** | ~21 dB |
| **SSIM** | ~0.72 |
        """)

    st.markdown("""
### How It Works
Dark Night Video
↓
Zero-DCE++ CBAM Enhancement
↓
Curve Estimation (8 iterations)
↓
Naturally Bright Output
↓
ADAS Ready Frames

    """)

    st.markdown("---")
    st.markdown(
        "**GitHub:** https://github.com/DEEK-SHITH/UNLET-ADAS")