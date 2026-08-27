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

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

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


@st.cache_resource
def load_pothole_detector():
    """
    Loads a dedicated single-class pothole detector, if one has been
    trained (see src/train_pothole.py — COCO/YOLOv8 has no pothole
    class, so this is a separate fine-tuned model, not a toggle on
    the main ADAS detector). Returns (None, False) if the weights
    file isn't present, so the app degrades gracefully instead of
    crashing on a fresh checkout.
    """
    weights = os.path.join(
        os.path.dirname(__file__), 'pothole_best.pt')
    if not os.path.exists(weights):
        return None, False
    try:
        from ultralytics import YOLO
        return YOLO(weights), True
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
    from src.enhance import scene_blend_weight, correct_color_cast
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(
        2, 0, 1).unsqueeze(0).to(device)
    enh, _ = model.enhance_full_res(t, proxy_size=256)
    if adaptive:
        alpha = scene_blend_weight(float(arr.mean()))
        enh   = t * (1 - alpha) + enh * alpha
    out = (enh[0].permute(1, 2, 0).cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)
    out = correct_color_cast(out)
    return Image.fromarray(out)


RISK_COLORS = {
    'HIGH':   (220,  20,  60),   # red
    'MEDIUM': (255, 165,   0),   # amber
    'LOW':    ( 50, 205,  50),   # green
}


def estimate_risk(x1, y1, x2, y2, frame_w, frame_h):
    """
    Rough collision-proximity estimate from box geometry alone (no
    depth sensor / calibration available): a box that fills a large
    fraction of the frame's height is close to the camera, and one
    centered in the middle third of the frame is roughly in the
    ego vehicle's path. This is a heuristic, not a measured distance
    — good enough to flag "large and in front of you" for a demo,
    not for real collision avoidance.
    """
    box_h_frac = (y2 - y1) / max(frame_h, 1)
    cx = (x1 + x2) / 2
    in_path = 0.2 * frame_w <= cx <= 0.8 * frame_w
    if box_h_frac > 0.35 and in_path:
        return 'HIGH'
    if box_h_frac > 0.18 or (in_path and box_h_frac > 0.10):
        return 'MEDIUM'
    return 'LOW'


def _draw_detection(img_bgr, x1, y1, x2, y2, label, risk):
    """Box outlined by risk level; class/ID/confidence as the label."""
    color = RISK_COLORS[risk]
    thickness = 3 if risk == 'HIGH' else 2
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img_bgr,
        (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img_bgr, label, (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2, cv2.LINE_AA)


def detect_and_draw(yolo, image_np, conf=0.25, imgsz=640):
    """Single-frame detection (no temporal tracking — see track_and_draw
    for video, which additionally assigns persistent IDs)."""
    results = yolo(image_np, conf=conf, imgsz=imgsz, verbose=False)[0]
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]
    counts = {}
    det_list = []
    high_risk = False
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in ADAS_CLASSES:
            continue
        name, _ = ADAS_CLASSES[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        risk = estimate_risk(x1, y1, x2, y2, w, h)
        high_risk = high_risk or risk == 'HIGH'
        _draw_detection(
            img_bgr, x1, y1, x2, y2,
            f'{name} · {risk}', risk)
        counts[name] = counts.get(name, 0) + 1
        det_list.append({
            'name': name, 'conf': conf_s,
            'box': [x1, y1, x2, y2], 'risk': risk,
        })
    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, det_list, counts, high_risk


def track_and_draw(yolo, image_np, conf=0.25, imgsz=640):
    """
    Like detect_and_draw, but uses YOLOv8's built-in ByteTrack
    (persist=True keeps the tracker's internal state alive across
    calls on the same model instance) to assign each object a stable
    ID across frames, instead of re-detecting from scratch every
    frame with no memory of what was seen before.
    """
    results = yolo.track(
        image_np, conf=conf, imgsz=imgsz, persist=True,
        tracker='bytetrack.yaml', verbose=False)[0]
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]
    counts = {}
    track_ids = set()
    high_risk = False
    det_list = []
    ids = results.boxes.id
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        if cls_id not in ADAS_CLASSES:
            continue
        name, _ = ADAS_CLASSES[cls_id]
        conf_s = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        risk = estimate_risk(x1, y1, x2, y2, w, h)
        high_risk = high_risk or risk == 'HIGH'
        track_id = int(ids[i]) if ids is not None else None
        id_bit = f'#{track_id} ' if track_id is not None else ''
        label = f'{name} {id_bit}· {risk}'
        _draw_detection(img_bgr, x1, y1, x2, y2, label, risk)
        counts[name] = counts.get(name, 0) + 1
        if track_id is not None:
            track_ids.add(track_id)
        det_list.append({
            'name': name, 'box': [x1, y1, x2, y2],
            'risk': risk, 'label': label, 'track_id': track_id,
        })
    ann = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ann, counts, track_ids, high_risk, det_list


def redraw_cached_detections(image_np, det_list):
    """
    Draw a previously-computed set of detection boxes onto a new frame
    without running the model again — used on the frames "fast mode"
    skips, so the video keeps a box on every frame (no flicker) while
    only paying for a fresh YOLO pass every other frame.
    """
    if not det_list:
        return image_np
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for d in det_list:
        x1, y1, x2, y2 = d['box']
        _draw_detection(img_bgr, x1, y1, x2, y2, d['label'], d['risk'])
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def process_video_chunk(job, model, DEVICE, yolo, has_yolo,
                        pothole_yolo, has_pothole, chunk_size=8):
    """
    Process up to chunk_size more frames of an in-progress video job
    (see the Video tab), mutating job in place.

    Video processing is split into small chunks like this — instead of
    one big blocking loop — specifically so it can be cancelled: the
    Video tab calls this once per Streamlit rerun and re-renders the
    Cancel button in between calls, which a single long-running loop
    would never yield control back to. It also means a video isn't
    lost if it stalls partway — whatever was written so far is still
    a valid, downloadable file.

    Returns True if there's more work to do (call again), False once
    the job is finished (ran out of frames, hit its frame cap, or was
    cancelled).
    """
    from src.enhance import enhance_frame_batch
    from src.lane_detection import detect_lanes, draw_lanes

    p = job['params']
    fb, ob = [], []
    while len(fb) < chunk_size and job['count'] < job['max_fr']:
        ret, frm = job['cap'].read()
        if not ret:
            # A single failed read isn't necessarily true end-of-stream
            # — some decoders hiccup on one bad/corrupt frame in an
            # otherwise longer file. Only give up after a few misses
            # in a row, rather than truncating the whole rest of the
            # video on the first stumble.
            job['read_fail_streak'] = job.get('read_fail_streak', 0) + 1
            if job['read_fail_streak'] >= 5:
                job['ended_early'] = True
                break
            continue
        job['read_fail_streak'] = 0
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        fb.append(rgb)
        ob.append(frm.copy())
        job['count'] += 1

    if fb:
        # Curves are estimated on a small proxy and applied at the
        # original W_v x H_v resolution, so enhanced frames stay sharp
        # instead of being blurred by a resize round-trip.
        enhanced = enhance_frame_batch(
            model, DEVICE, fb, size=256, adaptive=p['adaptive_mode'])

        for orig_bgr, enh_rgb in zip(ob, enhanced):
            # In fast mode, the expensive model passes (pothole YOLO,
            # detection + tracking) only run on every other frame; the
            # skipped frame reuses the last frame's boxes so the
            # output still looks fully annotated.
            run_heavy = (not p['vid_fast']) or (job['proc_idx'] % 2 == 0)

            if p['vid_lanes']:
                left_line, right_line = detect_lanes(enh_rgb)
                if left_line is not None or right_line is not None:
                    enh_rgb = draw_lanes(enh_rgb, left_line, right_line)

            if p['vid_pothole'] and has_pothole:
                if run_heavy or not job['last_pot_boxes']:
                    pot_res = pothole_yolo(
                        enh_rgb, conf=p['det_conf'],
                        imgsz=p['det_imgsz'], verbose=False)[0]
                    job['last_pot_boxes'] = [
                        (*map(int, box.xyxy[0]), float(box.conf[0]))
                        for box in pot_res.boxes]
                for x1, y1, x2, y2, pconf in job['last_pot_boxes']:
                    cv2.rectangle(
                        enh_rgb, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(
                        enh_rgb, f'Pothole {pconf:.0%}',
                        (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

            if p['vid_detect'] and has_yolo:
                if run_heavy or not job['last_det_list']:
                    (enh_rgb, frame_counts, frame_ids, frame_risk,
                     job['last_det_list']) = track_and_draw(
                        yolo, enh_rgb, conf=p['det_conf'], imgsz=p['det_imgsz'])
                    job['all_track_ids'] |= frame_ids
                    job['any_high_risk'] = job['any_high_risk'] or frame_risk
                    for k, v in frame_counts.items():
                        job['all_counts'][k] = job['all_counts'].get(k, 0) + v
                else:
                    enh_rgb = redraw_cached_detections(enh_rgb, job['last_det_list'])
                    job['any_high_risk'] = job['any_high_risk'] or any(
                        d['risk'] == 'HIGH' for d in job['last_det_list'])
                    for d in job['last_det_list']:
                        job['all_counts'][d['name']] = job['all_counts'].get(d['name'], 0) + 1

            job['proc_idx'] += 1
            eb = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                orig_bgr, 'ORIGINAL', (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(
                eb, 'UNLET ENHANCED', (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (50, 220, 80), 2, cv2.LINE_AA)
            job['enh_w'].write(eb)
            cmp = np.hstack([orig_bgr, eb])
            cv2.line(cmp, (job['W_v'], 0), (job['W_v'], job['H_v']), (255, 255, 255), 3)
            job['cmp_w'].write(cmp)

    return (not job['cancel_requested']
            and not job.get('ended_early')
            and job['count'] < job['max_fr'])


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
    pothole_yolo, has_pothole = load_pothole_detector()

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
if has_pothole:
    st.sidebar.success("Pothole Detector Ready")
else:
    st.sidebar.info(
        "Pothole Detector not trained — see src/train_pothole.py")
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
- Bright-pixel LAB color-cast correction
- Classical lane detection (Canny + Hough)
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

tab1, tab2, tab_live, tab3 = st.tabs([
    "🖼️ Image Enhancement",
    "🎬 Video Enhancement",
    "📸 Live Camera",
    "ℹ️ About Project"
])

# IMAGE TAB
with tab1:
    st.header("Image Enhancement + Detection")
    st.markdown(
        "Upload a dark/night image. "
        "UNLET enhances it then YOLOv8 detects objects.")

    col_chk1, col_chk2, col_chk3 = st.columns(3)
    with col_chk1:
        use_detection = st.checkbox(
            "Enable YOLOv8 Detection after enhancement",
            value=True)
    with col_chk2:
        use_lanes = st.checkbox(
            "Enable Lane Detection", value=True,
            help="Classical edge-based lane-line detection (Canny + "
                 "Hough transform) run on the enhanced frame — best on "
                 "straight/gently-curved roads with visible markings.")
    with col_chk3:
        use_pothole = st.checkbox(
            "Enable Pothole Detection", value=True,
            disabled=not has_pothole,
            help="Needs a trained pothole model — run "
                 "src/train_pothole.py and drop the resulting "
                 "pothole_best.pt into app/ to enable this."
                 if not has_pothole else
                 "Dedicated single-class YOLOv8 model fine-tuned on "
                 "a public pothole dataset (COCO/YOLOv8 has no "
                 "pothole class, so this runs as a separate pass).")

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
            high_risk = False

            if use_detection and has_yolo and HAS_CV2:
                with st.spinner("Running YOLOv8..."):
                    det_img, det_list, counts, high_risk = detect_and_draw(
                        yolo, enh_arr, conf=det_conf, imgsz=det_imgsz)

            lane_found = False
            if use_lanes and HAS_CV2:
                from src.lane_detection import detect_lanes, draw_lanes
                left_line, right_line = detect_lanes(det_img)
                lane_found = left_line is not None or right_line is not None
                if lane_found:
                    det_img = draw_lanes(det_img, left_line, right_line)

            pothole_count = 0
            if use_pothole and has_pothole and HAS_CV2:
                with st.spinner("Running pothole detector..."):
                    pot_res = pothole_yolo(
                        det_img, conf=det_conf, imgsz=det_imgsz,
                        verbose=False)[0]
                    pot_bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
                    for box in pot_res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        pconf = float(box.conf[0])
                        cv2.rectangle(
                            pot_bgr, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(
                            pot_bgr, f'Pothole {pconf:.0%}', (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                        pothole_count += 1
                    det_img = cv2.cvtColor(pot_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("✨ UNLET Enhanced")
                st.image(enh, use_container_width=True)
                st.caption(f"Avg brightness: {enh_b:.3f}")

            with col3:
                if (use_detection and det_list) or lane_found or pothole_count:
                    label_bits = []
                    if use_detection and det_list:
                        label_bits.append(f"{len(det_list)} objects")
                    if lane_found:
                        label_bits.append("lanes")
                    if pothole_count:
                        label_bits.append(f"{pothole_count} potholes")
                    st.subheader(f"🎯 Detection ({', '.join(label_bits)})")
                    st.image(det_img, use_container_width=True)
                else:
                    st.subheader("🔆 AutoContrast")
                    st.image(auto, use_container_width=True)
                    st.caption(f"Avg brightness: {auto_b:.3f}")

            if high_risk:
                st.error(
                    "⚠️ HIGH proximity risk — an object fills a large "
                    "part of the frame in the vehicle's path. (Estimated "
                    "from box size/position, not a measured distance.)")
            if pothole_count:
                st.warning(
                    f"🕳️ {pothole_count} pothole(s) detected in the road ahead.")

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

        if 'vid_job' not in st.session_state:
            st.session_state.vid_job = None

        if vid_upload and st.session_state.vid_job is None:
            # Only re-parsed/re-written while no job is running — once
            # a job starts, everything it needs (the opened capture,
            # writers, dimensions) already lives in st.session_state,
            # so there's no need to re-write this file to disk and
            # re-probe it on every single chunk-processing rerun below.
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
            total_dur = tot_v / fps_v

            st.info(
                f"Video: {W_v}x{H_v} @ {fps_v:.0f}fps | "
                f"{tot_v} frames | "
                f"{total_dur:.1f}s total")

            # Keyed to this specific file so the default adapts to each
            # newly uploaded video (defaults to the whole thing, up to
            # the 60s cap) instead of always resetting to a fixed 10s —
            # that fixed default was silently truncating every longer
            # upload to its first 10 seconds.
            max_sec = st.slider(
                "Max seconds to process", 5, 60,
                value=min(max(int(total_dur) + 1, 5), 60),
                key=f'max_sec_{vid_upload.name}_{vid_upload.size}',
                help="Processing runs frame-by-frame on CPU/GPU, so "
                     "longer clips take proportionally longer — this "
                     "caps it. Lower it for a quick preview.")
            if max_sec < total_dur:
                st.warning(
                    f"Only the first {max_sec}s of this "
                    f"{total_dur:.1f}s video will be processed — raise "
                    "the slider above to cover more of it.")

            vid_detect = st.checkbox(
                "Run YOLOv8 detection on enhanced frames", value=True,
                key='vid_detect')
            vid_lanes = st.checkbox(
                "Run lane detection on enhanced frames", value=True,
                key='vid_lanes')
            vid_pothole = st.checkbox(
                "Run pothole detection on enhanced frames",
                value=True, disabled=not has_pothole,
                key='vid_pothole',
                help=None if has_pothole else
                "Needs a trained pothole model — see src/train_pothole.py")
            vid_fast = st.checkbox(
                "⚡ Faster processing (detect objects every 2nd frame)",
                value=True, key='vid_fast',
                help="Object detection is the slowest step. In fast "
                     "mode it runs on every other frame and the boxes "
                     "are carried over onto the frame in between, so "
                     "the video still looks fully annotated but takes "
                     "roughly half the time. Turn off to run detection "
                     "on every single frame instead.")

            if st.button("🚀 Enhance Video", key='btn_vid'):
                # Reset ByteTrack state so IDs from a previous run
                # (if any) don't bleed into this one.
                if has_yolo and getattr(yolo, 'predictor', None):
                    try:
                        yolo.predictor.trackers[0].reset()
                    except Exception:
                        pass

                enh_p = os.path.join(
                    tempfile.gettempdir(), 'enhanced.mp4')
                cmp_p = os.path.join(
                    tempfile.gettempdir(), 'comparison.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                st.session_state.vid_job = {
                    'status': 'running',
                    'cap': cv2.VideoCapture(tmp),
                    'enh_w': cv2.VideoWriter(enh_p, fourcc, fps_v, (W_v, H_v)),
                    'cmp_w': cv2.VideoWriter(cmp_p, fourcc, fps_v, (W_v * 2, H_v)),
                    'enh_p': enh_p, 'cmp_p': cmp_p,
                    'W_v': W_v, 'H_v': H_v,
                    'max_fr': min(tot_v, int(max_sec * fps_v)),
                    'tot_v': tot_v, 'max_sec': max_sec, 'total_dur': total_dur,
                    'count': 0, 'proc_idx': 0,
                    'last_det_list': [], 'last_pot_boxes': [],
                    'all_track_ids': set(), 'all_counts': {},
                    'any_high_risk': False,
                    'cancel_requested': False, 'ended_early': False,
                    'params': {
                        'vid_lanes': vid_lanes, 'vid_pothole': vid_pothole,
                        'vid_detect': vid_detect, 'vid_fast': vid_fast,
                        'det_conf': det_conf, 'det_imgsz': det_imgsz,
                        'adaptive_mode': adaptive_mode,
                    },
                }
                st.rerun()

        if st.session_state.vid_job is not None:
            job = st.session_state.vid_job

            if job['status'] == 'running':
                frac = min(job['count'] / job['max_fr'], 1.0) if job['max_fr'] else 1.0
                st.progress(frac, f"Frame {job['count']}/{job['max_fr']}")
                if job['max_fr'] < job['tot_v']:
                    st.caption(
                        f"Processing {job['max_fr']} of {job['tot_v']} frames "
                        f"(~{job['max_sec']}s of {job['total_dur']:.1f}s) — "
                        "set by the 'Max seconds to process' slider before "
                        "you clicked Enhance. Cancel and raise it to cover "
                        "more of the video.")
                if st.button("🛑 Cancel Enhancement", key='btn_cancel_vid'):
                    job['cancel_requested'] = True

                more = process_video_chunk(
                    job, model, DEVICE, yolo, has_yolo,
                    pothole_yolo, has_pothole)

                if more:
                    st.rerun()
                else:
                    job['cap'].release()
                    job['enh_w'].release()
                    job['cmp_w'].release()
                    job['status'] = 'done'
                    st.rerun()
            else:
                if job['cancel_requested']:
                    st.warning(
                        f"Cancelled — kept the {job['count']} frames "
                        "processed before you stopped it.")
                elif job.get('ended_early') and job['count'] < job['max_fr']:
                    st.info(
                        f"Video ended after {job['count']} frames — "
                        f"fewer than the {job['max_fr']} its own "
                        "metadata claimed. This happens with "
                        "variable-frame-rate recordings (a camera "
                        "changing exposure in low light can vary "
                        "its real frame rate even though the file "
                        "header states a fixed one) — what you got "
                        "is the actual decodable content, not a "
                        "processing bug.")
                else:
                    st.success(f"Processed {job['count']} frames!")

                if (not job.get('ended_early')) and job['max_fr'] < job['tot_v']:
                    st.caption(
                        f"That's {job['max_fr']} of the video's {job['tot_v']} "
                        f"total frames (~{job['max_sec']}s of "
                        f"{job['total_dur']:.1f}s) — capped by the "
                        "'Max seconds to process' slider. Raise it before "
                        "your next run to cover more of the video.")

                if job['params']['vid_detect'] and has_yolo:
                    if job['any_high_risk']:
                        st.error(
                            "⚠️ HIGH proximity risk detected in at least "
                            "one frame — an object filled a large part "
                            "of the frame in the vehicle's path.")

                    st.markdown("---")
                    st.subheader("📊 Detection Summary")
                    s1, s2 = st.columns(2)
                    with s1:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-val'>{len(job['all_track_ids'])}</div>"
                            f"<div class='metric-lbl'>Unique Objects Tracked "
                            f"(not just per-frame counts)</div>"
                            f"</div>", unsafe_allow_html=True)
                    with s2:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div class='metric-val'>{sum(job['all_counts'].values())}</div>"
                            f"<div class='metric-lbl'>Total Detections "
                            f"Across All Frames</div>"
                            f"</div>", unsafe_allow_html=True)

                    if job['all_counts'] and not job['all_track_ids']:
                        st.caption(
                            "No object was tracked confidently enough "
                            "across consecutive frames to earn a persistent "
                            "ID (ByteTrack confirms a track over a few "
                            "frames before assigning one) — the detections "
                            "above were real but too brief/sparse in this "
                            "clip to accumulate a stable ID.")

                    if job['all_counts']:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 3))
                        names = list(job['all_counts'].keys())
                        vals = [job['all_counts'][n] for n in names]
                        ax.bar(names, vals, color='#22c55e')
                        ax.set_facecolor('#0f172a')
                        fig.patch.set_facecolor('#0f172a')
                        ax.tick_params(colors='#e2e8f0')
                        ax.spines[:].set_color('#334155')
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_color('#e2e8f0')
                        ax.set_ylabel('Detections', color='#e2e8f0')
                        st.pyplot(fig)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Enhanced Video")
                    st.video(job['enh_p'])
                    with open(job['enh_p'], 'rb') as f:
                        st.download_button(
                            "⬇️ Download Enhanced",
                            f.read(),
                            'enhanced.mp4',
                            'video/mp4',
                            use_container_width=True)
                with col_b:
                    st.subheader("Side-by-Side")
                    st.video(job['cmp_p'])
                    with open(job['cmp_p'], 'rb') as f:
                        st.download_button(
                            "⬇️ Download Comparison",
                            f.read(),
                            'comparison.mp4',
                            'video/mp4',
                            use_container_width=True)

                if st.button("🔁 Process Another Video", key='btn_vid_reset'):
                    st.session_state.vid_job = None
                    st.rerun()

# LIVE CAMERA TAB
with tab_live:
    st.header("Live Camera")

    live_mode = st.radio(
        "Mode", ["📸 Snapshot", "🎥 Live Stream (real-time)"],
        horizontal=True, key='live_mode',
        help="Snapshot always works and is the reliable option for a "
             "demo. Live Stream processes your camera continuously in "
             "real time, but its frame rate depends entirely on your "
             "CPU/GPU — it can be choppy on a laptop with no GPU.")

    if live_mode == "🎥 Live Stream (real-time)":
        st.markdown(
            "Continuously enhances your camera feed and shows the "
            "original and enhanced video **side by side, live** — no "
            "button press needed once the stream starts. Runs entirely "
            "in your browser + this server via WebRTC, so it also "
            "works when the app is deployed on Streamlit Cloud.")

        if not HAS_WEBRTC:
            st.warning(
                "Live Stream mode needs the `streamlit-webrtc` package, "
                "which isn't installed. Run `pip install -r "
                "requirements.txt` (it's included there) and restart "
                "the app, or use Snapshot mode above instead.")
        else:
            live_stream_detect = st.checkbox(
                "Run YOLOv8 detection on the live stream", value=False,
                key='live_stream_detect',
                help="Off by default — object detection roughly halves "
                     "the frame rate again on top of enhancement. Turn "
                     "it on if your machine can keep up.")

            def _live_video_frame_callback(frame):
                img_bgr = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                enh_rgb = np.array(enhance_pil(
                    model, DEVICE, Image.fromarray(rgb),
                    adaptive=adaptive_mode))

                if live_stream_detect and has_yolo:
                    # Capped well below the sidebar's det_imgsz — live
                    # frame-by-frame inference needs to stay fast or
                    # the stream stalls, unlike a one-shot image/video.
                    enh_rgb, _, _, _ = detect_and_draw(
                        yolo, enh_rgb, conf=det_conf,
                        imgsz=min(det_imgsz, 320))

                enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)
                orig_labeled = img_bgr.copy()
                cv2.putText(
                    orig_labeled, 'ORIGINAL', (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    enh_bgr, 'ENHANCED', (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 220, 80), 2, cv2.LINE_AA)
                combined = np.hstack([orig_labeled, enh_bgr])
                return av.VideoFrame.from_ndarray(combined, format="bgr24")

            webrtc_streamer(
                key='live-enhance-stream',
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=_live_video_frame_callback,
                async_processing=True,
            )
            st.caption(
                "Click START above, then allow camera access. If it "
                "stays black, your network may be blocking the WebRTC "
                "connection — Snapshot mode is the fallback.")
        live_shot = None
    else:
        st.markdown(
            "Uses your browser's camera (works on a phone/laptop even "
            "when the app is running on Streamlit Cloud, since capture "
            "happens client-side) — take a photo and it runs through the "
            "same enhancement + detection pipeline as the Image tab. "
            "This is a snap-and-analyze flow, not a continuous live "
            "video feed — switch to Live Stream mode above for that.")

        camera_on = st.checkbox(
            "📷 Turn Camera On", value=False, key='camera_on',
            help="Off by default — the browser only asks for camera "
                 "permission / shows a live preview once you switch "
                 "this on, not just from visiting this tab.")

        if not camera_on:
            st.info("Camera is off. Turn it on above to take a photo.")
            live_shot = None
        else:
            live_shot = st.camera_input("Take a photo")

    if live_shot is not None:
        pil_live = Image.open(live_shot).convert('RGB')
        with st.spinner("Enhancing + detecting..."):
            t0 = time.time()
            enh_live = enhance_pil(model, DEVICE, pil_live, adaptive=adaptive_mode)
            live_arr = np.array(enh_live)

            live_high_risk = False
            live_pothole_count = 0
            if has_yolo and HAS_CV2:
                live_arr, live_dets, _, live_high_risk = detect_and_draw(
                    yolo, live_arr, conf=det_conf, imgsz=det_imgsz)
            else:
                live_dets = []

            if HAS_CV2:
                from src.lane_detection import detect_lanes, draw_lanes
                l_left, l_right = detect_lanes(live_arr)
                if l_left is not None or l_right is not None:
                    live_arr = draw_lanes(live_arr, l_left, l_right)

            if has_pothole and HAS_CV2:
                pot_res = pothole_yolo(
                    live_arr, conf=det_conf, imgsz=det_imgsz, verbose=False)[0]
                pot_bgr = cv2.cvtColor(live_arr, cv2.COLOR_RGB2BGR)
                for box in pot_res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pconf = float(box.conf[0])
                    cv2.rectangle(pot_bgr, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        pot_bgr, f'Pothole {pconf:.0%}', (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                    live_pothole_count += 1
                live_arr = cv2.cvtColor(pot_bgr, cv2.COLOR_BGR2RGB)
            ms_live = (time.time() - t0) * 1000

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.subheader("📷 Captured")
            st.image(pil_live, use_container_width=True)
        with col_l2:
            st.subheader(f"✨ Enhanced + Detected ({ms_live:.0f}ms)")
            st.image(live_arr, use_container_width=True)

        if live_high_risk:
            st.error("⚠️ HIGH proximity risk — an object fills a large "
                      "part of the frame in the vehicle's path.")
        if live_pothole_count:
            st.warning(f"🕳️ {live_pothole_count} pothole(s) detected.")
        if live_dets:
            cols = st.columns(min(len(live_dets), 4))
            for i, d in enumerate(live_dets):
                cols[i % len(cols)].metric(
                    d['name'], f"{d['conf']:.0%}", d['risk'])

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
Bright-Pixel LAB Color-Cast Correction
      ↓
Lane Detection (Canny + Hough)  +  YOLOv8 Detection (n/s/m selectable)
      ↓
Enhanced Output + Lanes + Detections
```
    """)
    st.markdown("---")
    st.info(
        "Live Demo: "
        "https://unlet-adas-g4xvfhrfamxaqhpfuaqtri.streamlit.app\n\n"
        "GitHub: https://github.com/DEEK-SHITH/UNLET-ADAS")
