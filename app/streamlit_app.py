"""
UNLET-ADAS: Streamlit Web Application
Real-Time Low-Light Enhancement for Intelligent Vehicle Systems
B.E. Major Project | SJBIT Bengaluru | CSE 2025-26
"""

import streamlit as st
import torch
import numpy as np
import sys
import os
import io
import time
import tempfile
from PIL import Image, ImageOps

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

st.set_page_config(
    page_title="UNLET-ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0f172a;}
[data-testid="stSidebar"] {background:#1e293b;}
h1,h2,h3,p,label {color:#e2e8f0;}
.metric-box {
    background:#1e293b;
    border-radius:10px;
    padding:18px;
    text-align:center;
    border:1px solid #334155;
    margin:4px;
}
.metric-val {font-size:1.8rem; font-weight:700; color:#22c55e;}
.metric-lbl {font-size:0.82rem; color:#94a3b8; margin-top:4px;}
.stButton>button {
    background:linear-gradient(135deg,#22c55e,#0ea5e9);
    color:white;
    border:none;
    border-radius:8px;
    padding:12px 28px;
    font-weight:600;
    font-size:1rem;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

ADAS_CLASSES = {
    0:  ('Person',        (50,  205,  50)),
    1:  ('Bicycle',       (255, 165,   0)),
    2:  ('Car',           ( 30, 144, 255)),
    3:  ('Motorcycle',    (255, 100, 100)),
    5:  ('Bus',           (138,  43, 226)),
    7:  ('Truck',         (255,  20, 147)),
    9:  ('Traffic Light', (255, 215,   0)),
    11: ('Stop Sign',     (220,  20,  60)),
}


@st.cache_resource
def load_enhancer():
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from src.model import build_model
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(DEVICE)
    weights = os.path.join(
        os.path.dirname(__file__), 'zerodce_cbam_best.pt')
    if os.path.exists(weights):
        model.load_state_dict(
            torch.load(weights, map_location=DEVICE))
        status = f'Model loaded on {DEVICE}'
    else:
        status = 'No weights found'
    model.eval()
    return model, DEVICE, status


@st.cache_resource
def load_detector(weights='yolov8s.pt'):
    try:
        from ultralytics import YOLO
        yolo = YOLO(weights)
        return yolo, True
    except Exception:
        return None, False


@torch.no_grad()
def enhance_pil(model, device, pil_image, adaptive=True):
    """
    Enhance at full resolution: curves are estimated on a small
    256px proxy (cheap) then applied directly to the original
    full-size image, so no detail is lost to a resize round-trip.
    When adaptive=True, already well-lit (daytime) images are
    blended back toward the original instead of being over-brightened.
    """
    from src.enhance import scene_blend_weight
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(
        2, 0, 1).unsqueeze(0).to(device)
    enh, _ = model.enhance_full_res(t, proxy_size=256)
    if adaptive:
        alpha = scene_blend_weight(float(arr.mean()))
        enh   = t * (1 - alpha) + enh * alpha
    out = (enh[0].permute(1, 2, 0).cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def detect_and_draw(yolo, image_np, conf=0.25, imgsz=640):
    results = yolo(image_np, conf=conf, imgsz=imgsz, verbose=False)[0]
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    counts = {}
    det_list = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ADAS_CLASSES:
            continue
        name, color = ADAS_CLASSES[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr, 2)
        label = f'{name} {conf_s:.0%}'
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img_bgr,
            (x1, y1 - th - 8), (x1 + tw + 6, y1), bgr, -1)
        cv2.putText(img_bgr, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2, cv2.LINE_AA)
        counts[name] = counts.get(name, 0) + 1
        det_list.append({
            'name': name,
            'conf': conf_s,
            'box': [x1, y1, x2, y2]
        })
    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, det_list, counts


# Header
st.markdown("""
<h1 style='text-align:center;color:#22c55e;'>
🚗 UNLET-ADAS
</h1>
<p style='text-align:center;color:#94a3b8;font-size:1.1rem;'>
Real-Time Low-Light Enhancement for Intelligent Vehicle Systems
</p>
<p style='text-align:center;color:#64748b;'>
B.E. Major Project | SJBIT Bengaluru | CSE 2025-26
</p>
""", unsafe_allow_html=True)
st.markdown("---")

det_model_choice = st.sidebar.selectbox(
    'Detector Model', ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
    index=1,
    help='Larger models are more accurate but slower. '
         'yolov8s is the default balance of speed/accuracy.')

with st.spinner('Loading models...'):
    model, DEVICE, status = load_enhancer()
    yolo, has_yolo = load_detector(det_model_choice)

# Sidebar
st.sidebar.markdown("## System Info")
if 'cuda' in str(DEVICE):
    st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Running on CPU")
st.sidebar.code(status)
if has_yolo:
    st.sidebar.success(f"{det_model_choice} Detection Ready")
else:
    st.sidebar.warning("YOLOv8 not available")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
**Model:** Zero-DCE++ + CBAM Attention

**Components:**
- Channel Attention
- Spatial Attention
- 8-iteration curve enhancement
- Full-resolution curve application (no blur)
- Scene-adaptive day/night blending
- Perceptual + SSIM + Color loss

**Training:** LOL Dataset (485 pairs)

**Results:**
- PSNR: 18.80 dB
- SSIM: 0.747

**GitHub:** [DEEK-SHITH/UNLET-ADAS](https://github.com/DEEK-SHITH/UNLET-ADAS)
""")

adaptive_mode = st.sidebar.checkbox(
    'Adaptive Day/Night Mode', value=True,
    help='Automatically reduces enhancement strength on '
         'already well-lit (daytime) frames instead of '
         'over-brightening them, while still fully enhancing '
         'dark, night, or shaded hillside footage.')
det_conf = st.sidebar.slider(
    'Detection Confidence', 0.1, 0.9, 0.25, 0.05)
det_imgsz = st.sidebar.select_slider(
    'Detection Resolution', options=[320, 480, 640, 832, 960],
    value=640,
    help='Higher resolution improves detection of small/far '
         'objects (pedestrians, distant vehicles) at some speed cost.')

tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Enhancement",
    "🎬 Video Enhancement",
    "ℹ️ About Project"
])

# IMAGE TAB
with tab1:
    st.header("Image Enhancement + Detection")
    st.markdown(
        "Upload a dark/night image. "
        "UNLET enhances it then YOLOv8 detects objects.")

    use_detection = st.checkbox(
        "Enable YOLOv8 Detection after enhancement",
        value=True)

    uploaded = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key='img_upload')

    if uploaded:
        pil = Image.open(uploaded).convert('RGB')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📷 Original")
            st.image(pil, use_container_width=True)
            orig_b = np.array(pil).mean() / 255
            st.caption(f"Avg brightness: {orig_b:.3f}")

        if st.button("✨ Enhance + Detect", key='btn_img'):
            with st.spinner("Enhancing image..."):
                t0 = time.time()
                enh = enhance_pil(model, DEVICE, pil, adaptive=adaptive_mode)
                ms = (time.time() - t0) * 1000

            enh_arr = np.array(enh)
            enh_b = enh_arr.mean() / 255
            auto = ImageOps.autocontrast(pil)
            auto_b = np.array(auto).mean() / 255

            det_img = enh_arr.copy()
            det_list = []
            counts = {}

            if use_detection and has_yolo and HAS_CV2:
                with st.spinner("Running YOLOv8..."):
                    det_img, det_list, counts = detect_and_draw(
                        yolo, enh_arr, conf=det_conf, imgsz=det_imgsz)

            with col2:
                st.subheader("✨ UNLET Enhanced")
                st.image(enh, use_container_width=True)
                st.caption(f"Avg brightness: {enh_b:.3f}")

            with col3:
                if use_detection and det_list:
                    st.subheader(
                        f"🎯 Detection ({len(det_list)} objects)")
                    st.image(det_img, use_container_width=True)
                else:
                    st.subheader("🔆 AutoContrast")
                    st.image(auto, use_container_width=True)
                    st.caption(f"Avg brightness: {auto_b:.3f}")

            st.markdown("---")
            st.subheader("📊 Results")
            m1, m2, m3, m4, m5 = st.columns(5)
            impr = (enh_b - orig_b) / max(orig_b, 0.01) * 100

            with m1:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{orig_b:.3f}</div>"
                    f"<div class='metric-lbl'>Input Brightness</div>"
                    f"</div>", unsafe_allow_html=True)
            with m2:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{enh_b:.3f}</div>"
                    f"<div class='metric-lbl'>Enhanced Brightness</div>"
                    f"</div>", unsafe_allow_html=True)
            with m3:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>+{impr:.1f}%</div>"
                    f"<div class='metric-lbl'>Brightness Gain</div>"
                    f"</div>", unsafe_allow_html=True)
            with m4:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{ms:.0f}ms</div>"
                    f"<div class='metric-lbl'>Inference Time</div>"
                    f"</div>", unsafe_allow_html=True)
            with m5:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{len(det_list)}</div>"
                    f"<div class='metric-lbl'>Objects Detected</div>"
                    f"</div>", unsafe_allow_html=True)

            if counts:
                st.markdown("**Detected Objects:**")
                cols = st.columns(len(counts))
                for col, (name, cnt) in zip(cols, counts.items()):
                    col.metric(name, cnt)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                buf = io.BytesIO()
                enh.save(buf, format='PNG')
                st.download_button(
                    "⬇️ Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name=f"enhanced_{uploaded.name}",
                    mime='image/png',
                    use_container_width=True)
            with c2:
                if det_list:
                    buf2 = io.BytesIO()
                    Image.fromarray(det_img).save(buf2, 'PNG')
                    st.download_button(
                        "⬇️ Download Detection Result",
                        data=buf2.getvalue(),
                        file_name=f"detection_{uploaded.name}",
                        mime='image/png',
                        use_container_width=True)

# VIDEO TAB
with tab2:
    st.header("Video Enhancement")

    if not HAS_CV2:
        st.warning(
            "Video enhancement requires OpenCV. "
            "Use the Image tab for online demo. "
            "Run locally for video processing.")
        st.code("""
git clone https://github.com/DEEK-SHITH/UNLET-ADAS
cd UNLET-ADAS
pip install -r requirements.txt
streamlit run app/streamlit_app.py
        """)
    else:
        st.markdown(
            "Upload a night driving video. "
            "Output: Enhanced video + side-by-side comparison.")

        vid_upload = st.file_uploader(
            "Choose a video (MP4/AVI/MOV)",
            type=['mp4', 'avi', 'mov'],
            key='vid_upload')

        max_sec = st.slider(
            "Max seconds to process", 5, 60, 10)
        vid_detect = st.checkbox(
            "Run YOLOv8 detection on enhanced frames", value=True,
            key='vid_detect')

        if vid_upload:
            tmp = os.path.join(
                tempfile.gettempdir(), vid_upload.name)
            with open(tmp, 'wb') as f:
                f.write(vid_upload.read())

            cap = cv2.VideoCapture(tmp)
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
            W_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tot_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            st.info(
                f"Video: {W_v}x{H_v} @ {fps_v:.0f}fps | "
                f"{tot_v} frames | "
                f"{tot_v/fps_v:.1f}s total")

            if st.button("🚀 Enhance Video", key='btn_vid'):
                from src.enhance import enhance_frame_batch

                enh_p = os.path.join(
                    tempfile.gettempdir(), 'enhanced.mp4')
                cmp_p = os.path.join(
                    tempfile.gettempdir(), 'comparison.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                enh_w = cv2.VideoWriter(
                    enh_p, fourcc, fps_v, (W_v, H_v))
                cmp_w = cv2.VideoWriter(
                    cmp_p, fourcc, fps_v, (W_v * 2, H_v))

                cap = cv2.VideoCapture(tmp)
                max_fr = min(tot_v, int(max_sec * fps_v))
                prog = st.progress(0, 'Starting...')
                fb = []
                ob = []
                count = 0

                while count < max_fr:
                    ret, frm = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    fb.append(rgb)
                    ob.append(frm.copy())
                    count += 1

                    if len(fb) >= 4 or count == max_fr:
                        # Curves are estimated on a small proxy and
                        # applied at the original W_v x H_v resolution,
                        # so enhanced frames stay sharp instead of
                        # being blurred by a resize round-trip.
                        enhanced = enhance_frame_batch(
                            model, DEVICE, fb,
                            size=256, adaptive=adaptive_mode)

                        for orig_bgr, enh_rgb in zip(ob, enhanced):
                            if vid_detect and has_yolo:
                                enh_rgb, _, _ = detect_and_draw(
                                    yolo, enh_rgb,
                                    conf=det_conf, imgsz=det_imgsz)
                            eb = cv2.cvtColor(
                                enh_rgb, cv2.COLOR_RGB2BGR)
                            cv2.putText(
                                orig_bgr, 'ORIGINAL',
                                (10, 35),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1.0, (80, 80, 255), 2,
                                cv2.LINE_AA)
                            cv2.putText(
                                eb, 'UNLET ENHANCED',
                                (10, 35),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1.0, (50, 220, 80), 2,
                                cv2.LINE_AA)
                            enh_w.write(eb)
                            cmp = np.hstack([orig_bgr, eb])
                            cv2.line(
                                cmp, (W_v, 0), (W_v, H_v),
                                (255, 255, 255), 3)
                            cmp_w.write(cmp)

                        fb.clear()
                        ob.clear()
                        prog.progress(
                            min(count / max_fr, 1.0),
                            f'Frame {count}/{max_fr}')

                cap.release()
                enh_w.release()
                cmp_w.release()
                prog.progress(1.0, 'Done!')

                st.success(f"Processed {count} frames!")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Enhanced Video")
                    st.video(enh_p)
                    with open(enh_p, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Enhanced",
                            f.read(),
                            'enhanced.mp4',
                            'video/mp4',
                            use_container_width=True)
                with col_b:
                    st.subheader("Side-by-Side")
                    st.video(cmp_p)
                    with open(cmp_p, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Comparison",
                            f.read(),
                            'comparison.mp4',
                            'video/mp4',
                            use_container_width=True)

# ABOUT TAB
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
| **Degree** | B.E. Major Project |
| **Year** | 2025-26 |
        """)

    with col2:
        st.markdown("""
### Results (LOL eval15)
| Method | PSNR | SSIM |
|---|---|---|
| Dark Input | 7.80 dB | 0.186 |
| AutoContrast | 13.12 dB | 0.518 |
| **UNLET (Ours)** | **18.80 dB** | **0.747** |

**UNLET beats AutoContrast by +5.68 dB PSNR**
        """)

    st.markdown("---")
    st.markdown("""
### System Architecture
```
Video Input (Night / Hilly / Day)
      ↓
Zero-DCE++ CBAM Curve Estimation (low-res proxy)
      ↓
Full-Resolution Curve Application (no blur)
      ↓
Scene-Adaptive Blend (skips over-brightening daylight)
      ↓
YOLOv8 Detection (n/s/m selectable)
      ↓
Enhanced Output + Detections
```
    """)
    st.markdown("---")
    st.info(
        "Live Demo: "
        "https://unlet-adas-g4xvfhrfamxaqhpfuaqtri.streamlit.app\n\n"
        "GitHub: https://github.com/DEEK-SHITH/UNLET-ADAS")
