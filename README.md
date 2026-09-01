# 🚗 UNLET-ADAS
## Real-Time Low-Light Enhancement for Intelligent Vehicle Systems

[![Demo](https://img.shields.io/badge/🌐-Live_Demo-green)](https://deek-shith-unlet-adas-streamlit-app.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-DEEK--SHITH-black)](https://github.com/DEEK-SHITH/UNLET-ADAS)
[![Tests](https://github.com/DEEK-SHITH/UNLET-ADAS/actions/workflows/test.yml/badge.svg)](https://github.com/DEEK-SHITH/UNLET-ADAS/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> **B.E. Major Project | SJBIT Bengaluru | Computer Science | 2025–26**

---

## 🎯 Problem Statement

Night driving causes over 50% of fatal road accidents despite
only 25% of total travel occurring at night. ADAS camera systems
fail in low-light conditions — they cannot reliably detect
vehicles, pedestrians, or lane markings in darkness.

## 💡 Solution

UNLET-ADAS enhances dark video frames in real time using
**Zero-DCE++ with CBAM Attention**, making night scenes
clearly visible for ADAS computer vision pipelines.

---

## 📊 Quantitative Results (LOL eval15 Dataset)

| Method | PSNR (dB) ↑ | SSIM ↑ | Improvement |
|--------|------------|--------|-------------|
| Original (Dark Input) | 7.80 | 0.186 | baseline |
| AutoContrast (Traditional) | 13.12 | 0.518 | +5.32 dB |
| **UNLET-ADAS (Ours)** | **19.20** | **0.753** | **+11.40 dB** |

> UNLET-ADAS outperforms traditional AutoContrast by **+5.68 dB PSNR**
> and achieves **4× better SSIM** than the dark input.

---

## 🆙 Sharper Output & Better Detection

Earlier versions resized every frame down to a small square for the
network and upscaled it back with a sharpening filter — that
resize round-trip was the source of the blurry output. The pipeline
now:

- **Applies enhancement curves at full resolution.** Zero-DCE++
  estimates its per-pixel curves on a small 256px proxy (cheap and
  robust), but those smooth curve maps are then upsampled and applied
  directly to the original frame — no pixel content is ever
  downscaled, so fine detail and edges (lane markings, distant
  vehicles, pedestrians) stay sharp.
- **Adapts to scene brightness automatically.** A blend factor based
  on average luminance gives full enhancement to dark scenes (night,
  tunnels, shaded hillside bends) and fades enhancement out for
  already well-lit daytime frames, so the same pipeline helps at
  night, in hilly terrain with sudden shadow/light changes, **and**
  during the day without washing anything out.
- **Uses a stronger default detector** (YOLOv8s instead of YOLOv8n)
  with a configurable detection resolution (up to 960px) so small or
  distant objects are picked up more reliably — critical for ADAS use
  cases like spotting a pedestrian or oncoming vehicle around a hill
  curve.
- **Corrects color cast without overshooting.** A naive whole-frame
  white balance gets skewed by the huge near-black background in
  night scenes and overcorrects the road surface into a new
  blue/purple tint. Color balance is now estimated only from the
  brightest ~40% of pixels, in LAB space, and applied as a partial,
  capped correction — a mild tint fix, not a full renormalization.
- **Detects lane boundaries** on the enhanced frame using classical
  Canny-edge + Hough-transform lane detection (`src/lane_detection.py`)
  — no additional training data needed, consistent with the project's
  lightweight, real-time design. Works best on straight/gently-curved
  roads with visible markings; a linear line fit can't perfectly hug
  a tight curve.
- **Optional dedicated pothole detector.** COCO/YOLOv8 has no pothole
  class, so this isn't a flag on the existing detector — it's a
  separate, single-class YOLOv8 model you fine-tune yourself with
  `src/train_pothole.py` (see [Train the Pothole Detector](#train-the-pothole-detector)
  below). The app detects the trained weights automatically once
  present and enables the toggle; without them, it stays off and says
  why.
- **Optional low-light-specialized detector.** The main detector runs
  stock COCO weights — trained entirely on daylight photos — on
  enhanced frames. `src/train_lowlight.py` fine-tunes a separate
  YOLOv8 model on ExDark (Exclusively Dark Image Dataset, Loh & Chan
  CVIU 2019) covering the 5 ADAS classes ExDark actually has night
  data for (Person/Bicycle/Car/Motorcycle/Bus); Traffic Light/Stop
  Sign/Truck still use the stock model, since fine-tuning directly on
  ExDark's 12 classes would silently drop those 3 rather than leave
  them unchanged. Once trained, it's selectable from the sidebar's
  Detector Model dropdown alongside the stock n/s/m options — see
  [Train the Low-Light Detector](#train-the-low-light-detector) below.
- **Real-time live camera stream.** The Live Camera tab has a "Live
  Stream" mode (via WebRTC, `streamlit-webrtc`) that continuously
  enhances your camera feed and shows original + enhanced **side by
  side, live** — no button press per frame. Frame rate depends
  entirely on your CPU/GPU; the original snapshot-based mode (one
  photo at a time) is kept as the reliable fallback for slower
  machines or restrictive networks.

---

## 🏗️ System Architecture

Night Video Input
↓
┌─────────────────────────┐
│ Zero-DCE++ CBAM │
│ Enhancement Model │
│ • Channel Attention │
│ • Spatial Attention │
│ • 8-iter curves │
└─────────────────────────┘
↓
Enhanced Video Output
↓
┌─────────────────────────┐
│ ADAS Vision Pipeline │
│ • YOLOv8l Detection │
│ • DeepSORT Tracking │
│ • Lane Detection │
│ • Depth Estimation │
└─────────────────────────┘
↓
Professional HUD Overlay


---

## 🚀 Quick Start

### Clone and Install
```bash
git clone https://github.com/DEEK-SHITH/UNLET-ADAS.git
cd UNLET-ADAS
pip install -r requirements.txt
```

### Run Web App
```bash
streamlit run app/streamlit_app.py
```

### Train Your Own Model
```bash
python src/train.py \
  --data_root ./data/lol_dataset \
  --epochs 100 \
  --batch_size 8
```

### Train the Pothole Detector
Optional — the app runs fine without it, just with the pothole
toggle disabled. COCO/YOLOv8 has no pothole class, so this fine-tunes
a small, dedicated YOLOv8 model on a public pothole dataset rather
than retraining the main ADAS detector.
```bash
pip install roboflow
python src/train_pothole.py --roboflow_key YOUR_FREE_API_KEY
# then: cp checkpoints/pothole_best.pt app/pothole_best.pt
```
Get a free API key at [app.roboflow.com](https://app.roboflow.com)
(Settings → API Keys). Dataset:
[Pothole Object Detection Dataset](https://public.roboflow.com/object-detection/pothole)
(665 images, Roboflow's curated Public Datasets collection).

Prefer a free GPU over local CPU training? Open
[`notebooks/UNLET_ADAS_Pothole_Colab.ipynb`](notebooks/UNLET_ADAS_Pothole_Colab.ipynb)
in Google Colab — same download + training steps, just paste your API
key into the config cell and run top to bottom (~15–25 min on a T4).

### Train the Low-Light Detector
Optional — the app runs fine without it, using the stock COCO
detector on every tab. Fine-tunes a separate 5-class YOLOv8 model
(Person/Bicycle/Car/Motorcycle/Bus) on ExDark, a dataset of real
night-time images, rather than only ever seeing daylight COCO photos.
```bash
pip install roboflow
python src/train_lowlight.py --roboflow_key YOUR_FREE_API_KEY \
    --roboflow_workspace WORKSPACE --roboflow_project PROJECT \
    --roboflow_version N
# then: cp checkpoints/lowlight_best.pt app/yolov8_lowlight.pt
```
Unlike the pothole dataset, there's no single verified ExDark mirror
to default to — search [Roboflow Universe](https://universe.roboflow.com)
for "ExDark", pick a project with an image count close to the real
dataset's ~7,363, then click **Download Dataset → YOLOv8 → Show
download code** to read off `WORKSPACE`/`PROJECT`/`N`. The script
prints a per-class box count after downloading so a partial/incomplete
mirror is obvious immediately rather than silently undertraining.

Prefer a free GPU over local CPU training? Open
[`notebooks/UNLET_ADAS_Lowlight_YOLO_Colab.ipynb`](notebooks/UNLET_ADAS_Lowlight_YOLO_Colab.ipynb)
in Google Colab — same steps, just fill in the config cell and run
top to bottom (~20–35 min on a T4).

### Enhance a Video
```python
from src.model import build_model
from src.enhance import enhance_video
import torch

device = torch.device('cuda')
model  = build_model().to(device)
model.load_state_dict(torch.load('app/zerodce_cbam_best.pt'))
model.eval()

enhance_video(
    model=model,
    device=device,
    input_path='night_drive.mp4',
    output_path='enhanced.mp4',
    original_path='original.mp4',
    comparison_path='comparison.mp4'
)
```

### Run the Automated Tests
End-to-end tests drive the actual Streamlit app in a headless browser
(Playwright) and check that all three tabs — Image, Video, Live
Camera — load and, for Image/Video, genuinely process input through
the real enhancement + detection pipeline. Runs automatically on every
push via [GitHub Actions](.github/workflows/test.yml).
```bash
pip install -r requirements.txt -r requirements-dev.txt
playwright install --with-deps chromium
pytest tests/ -v
```

---

## 📁 Project Structure

UNLET-ADAS/
├── src/
│ ├── model.py # Zero-DCE++ CBAM architecture
│ ├── losses.py # Composite loss functions
│ ├── train.py # Training pipeline
│ └── enhance.py # Enhancement functions
├── app/
│ ├── streamlit_app.py # Web application
│ └── zerodce_cbam_best.pt # Trained weights
├── notebooks/
│ ├── UNLET_ADAS_Colab.ipynb # Main model training (Colab)
│ ├── UNLET_ADAS_Pothole_Colab.ipynb # Pothole detector training (Colab)
│ └── UNLET_ADAS_Lowlight_YOLO_Colab.ipynb # Low-light detector training (Colab)
├── results/
│ ├── metric_comparison.png # PSNR/SSIM charts
│ ├── training_curves.png # Loss curves
│ └── video_frames.png # Before/after frames
├── tests/
│ ├── conftest.py # Test fixtures (app server, browser, test data)
│ └── test_app.py # End-to-end UI tests (Playwright)
├── .github/workflows/
│ └── test.yml # CI — runs tests/ on every push
├── requirements.txt
├── requirements-dev.txt # Extra deps for running tests/ (CI only)
└── README.md


---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Enhancement Model | Zero-DCE++ with CBAM Attention |
| Framework | PyTorch 2.0 |
| Detection | YOLOv8l (Ultralytics) |
| Tracking | DeepSORT |
| Depth Estimation | MiDaS |
| Web App | Streamlit |
| Training Data | LOL Dataset (485 pairs) |
| Deployment | Streamlit Community Cloud |

---

## 📈 Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=2e-4, weight_decay=1e-5) |
| Scheduler | Cosine Annealing (T_max=100) |
| Epochs | 100 |
| Batch Size | 8 |
| Image Size | 256×256 |
| Loss | Color(50) + Exposure(10) + Perceptual(0.1) + SSIM(2) |
| GPU | NVIDIA T4 (Google Colab) |
| Training Time | ~60 minutes |

---

## 🔬 Loss Function Design

The composite loss uses **color constancy weight = 50**
to prevent the green color cast common in Zero-DCE models:

```python
Loss = 50.0 × ColorConstancy
     + 10.0 × Exposure
     +  1.0 × Spatial
     + 200.0 × Smoothness
     +  1.0 × L1
     +  0.1 × Perceptual (VGG19)
     +  2.0 × SSIM
     +  0.1 × Frequency
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{deekshith2026unletadas,
  title     = {UNLET-ADAS: Low-Light Video Enhancement
               for Intelligent Vehicle Systems},
  author    = {Deekshith},
  year      = {2026},
  school    = {SJB Institute of Technology, Bengaluru},
  note      = {B.E. Major Project, CSE Department},
  url       = {https://github.com/DEEK-SHITH/UNLET-ADAS}
}
```

---

## 👤 Author

**Deekshith**
B.E. Computer Science Engineering
SJB Institute of Technology, Bengaluru

- GitHub: [@DEEK-SHITH](https://github.com/DEEK-SHITH)

---

## 📄 License

MIT License — free to use for research and education.