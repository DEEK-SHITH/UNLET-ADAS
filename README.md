# 🚗 UNLET-ADAS
## Real-Time Low-Light Enhancement & Perception for Intelligent Vehicle Systems

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js_16-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![GitHub](https://img.shields.io/badge/GitHub-DEEK--SHITH-black?logo=github)](https://github.com/DEEK-SHITH/UNLET-ADAS)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> **B.E. Major Project | SJB Institute of Technology, Bengaluru | Computer Science & Engineering | 2025–26**
>
> 🌟 **Primary Interface**: Modern Full-Stack Web App (**Next.js + FastAPI**) on port `8000` with 60 FPS hardware canvas live camera streaming, interactive before/after split slider, and single-pass Direct FFmpeg video processing.
>
> 🧪 **Secondary Interface**: Classic **Streamlit** dashboard on port `8501` for rapid research prototyping and experimental benchmarking.

---

## 🎯 Problem Statement

Night driving accounts for over 50% of fatal traffic accidents despite representing only 25% of total travel time. Standard Advanced Driver Assistance Systems (ADAS) camera pipelines fail severely in low-light and adverse weather conditions — they cannot reliably resolve pedestrians in dark clothing, unlit vehicles, or road boundaries.

## 💡 Solution

**UNLET-ADAS** enhances dark video streams in real time using **Zero-DCE++ with CBAM (Convolutional Block Attention Module)**. It estimates dynamic illumination adjustment curves without requiring paired ground-truth images, seamlessly restoring visibility for downstream perception engines (YOLOv8 object tracking, monocular depth collision risk, and lane detection).

---

## 🖥️ Applications Overview

UNLET-ADAS provides two independent interfaces accessing the same core ML pipeline in `src/`:

```
                                  ┌───────────────────────────┐
                                  │   UNLET-ADAS Core Engine  │
                                  │  • Zero-DCE++ CBAM Model  │
                                  │  • YOLOv8 ADAS Perception │
                                  │  • Classical Lane Tracker │
                                  │  • MiDaS Depth Proximity  │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  🌟 PRIMARY: Next.js + FastAPI                                   🧪 SECONDARY: Streamlit
     (Port 8000 / Production)                                        (Port 8501 / Research)
  • Modern animated dashboard (Tailwind + Motion)                 • Rapid research & model evaluation
  • Interactive side-by-side comparison slider                    • Tabbed Image / Video / WebRTC tabs
  • 60 FPS Zero-Lag Live Camera HUD (Canvas)                      • Side-by-side metric benchmarking
  • Single-pass Direct-to-FFmpeg H.264 engine                     • Standalone Python app prototyping
  • Mobile HTTPS & camera flip (🔄 Flip)
```

---

## 📊 Quantitative Results (LOL eval15 Benchmark)

Evaluated against the standard Low-Light (LOL) benchmark dataset:

| Method | 256×256 PSNR (dB) ↑ | 256×256 SSIM ↑ | Full-Res PSNR (dB) ↑ | Full-Res SSIM ↑ | Parameters |
|---|---|---|---|---|---|
| Original (Dark Input) | 7.80 | 0.186 | 7.77 | 0.191 | — |
| AutoContrast (Traditional) | 13.12 | 0.518 | 11.24 | 0.380 | — |
| Zero-DCE++ (w/o CBAM Ablation) | 18.45 | 0.718 | 17.61 | 0.531 | 19,291 |
| **UNLET-ADAS (Ours, w/ CBAM)** | **19.26** | **0.757** | **18.49** | **0.579** | **21,769** |

* UNLET-ADAS outperforms traditional AutoContrast by **+6.14 dB PSNR** (proxy) and **+7.25 dB PSNR** (full-res).
* Achieves **4× higher SSIM** than the dark input.
* Adds **+0.81–0.88 dB PSNR** from the 2,478-parameter CBAM dual-attention mechanism with negligible compute overhead.

---

## ⏱️ Real-Time Performance & Latency Benchmarks

Measured on a standard CPU workstation (Intel Xeon / Core i7, CPU-only execution):

| Pipeline Stage | Latency | Throughput | Notes |
|---|---|---|---|
| **Zero-DCE++ CBAM Proxy (112px)** | **35–42 ms** | **24–28 FPS** | Native PyTorch CPU execution |
| **YOLOv8-Nano ADAS Perception** | **19–27 ms** | **37–52 FPS** | Ultra-lightweight real-time tracking |
| **Live HUD Canvas Engine** | **< 1 ms** | **60 FPS** | Hardware-accelerated GPU render |
| **Full Live Pipeline (Turbo Mode)** | **~90–100 ms** | **10–12+ FPS** | Interleaved detection on CPU (60+ FPS on GPU) |
| **Direct FFmpeg Video Stream** | **1.66× Faster** | Single-pass | Bypasses intermediate disk transcode passes |

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/DEEK-SHITH/UNLET-ADAS.git
cd UNLET-ADAS

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies for both core and web backend
pip install -r requirements.txt -r web/backend/requirements.txt
```

---

### 2. Run Primary Web App (FastAPI + Next.js) 🌟

The primary application serves both the **Next.js animated frontend** and the **FastAPI ML backend** from a unified single server:

```bash
uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

Open your browser:
* 🌐 **Dashboard & Image Comparison**: [http://localhost:8000](http://localhost:8000)
* 📷 **60 FPS Zero-Lag Live Camera**: [http://localhost:8000/live](http://localhost:8000/live)
* 🎬 **Direct FFmpeg Video Enhancement**: [http://localhost:8000/video](http://localhost:8000/video)

#### 📱 Access on Mobile / Share Globally
Mobile devices (iOS Safari and Android Chrome) require a secure HTTPS connection for camera access. In a second terminal, launch a secure Cloudflare Tunnel:

```bash
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000
```
Open the generated `https://xxxx.trycloudflare.com/live` URL on your phone to use the real-time night camera with front/rear camera flip (`🔄 Flip`).

---

### 3. Run Secondary App (Streamlit) 🧪

If you want to use the classic Streamlit research dashboard:

```bash
streamlit run app/streamlit_app.py
```

Open in browser:
* 🎈 **Streamlit UI**: [http://localhost:8501](http://localhost:8501)

---

## ✨ Primary Web App Features

### 1. Zero-Lag 60 FPS Live Camera HUD
* **Instant Hardware Canvas**: Renders incoming camera feed directly through a hardware-accelerated HTML5 `<canvas>` via `requestAnimationFrame` at 30–60 FPS with zero visual delay.
* **On-Device Dynamic Range Illumination**: Applies hardware-accelerated dynamic contrast and brightness expansion in $<1\text{ ms}$.
* **Asynchronous Neural Detection**: Runs deep YOLOv8 detection, road signs, and lane detection in background workers without stalling video playback.
* **Mobile Ready**: Includes camera flip (`🔄 Flip`) to toggle between front and rear lenses on smartphones.

### 2. Interactive Side-by-Side Image Comparison
* Draggable split slider allowing instant inspection between raw night-time frames and enhanced output.
* Synchronized bounding box overlays with risk badges (`LOW`, `MEDIUM`, `HIGH`).
* Quantitative metrics display: luminance levels, effective confidence thresholds, and execution time.

### 3. Single-Pass Direct FFmpeg Video Processing
* **Direct-to-FFmpeg Pipe (`DirectFFmpegVideoWriter`)**: Streams raw frames directly into an `ffmpeg` H.264 process (`-c:v libx264 -pix_fmt yuv420p -preset veryfast -movflags +faststart`).
* Eliminates the OpenCV double-pass re-encoding bottleneck, making output files immediately playable in browsers.
* **Quality Presets**:
  * `⚡ Turbo Fast`: Fastest batch turnaround with CRF 22 and 128px proxy.
  * `🎯 Balanced HD`: High-definition 1080p/720p output with CRF 20.
  * `🔬 Studio Master`: Uncompressed frame tracking with CRF 18.

---

## 🏗️ System Architecture

```
                  Raw Night Input (Image / Video / Camera Stream)
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │         Zero-DCE++ CBAM Enhancement Engine       │
               │  • Spatial + Channel Attention (CBAM)            │
               │  • 8-iteration non-linear curve estimation       │
               │  • Full-resolution curve application             │
               │  • Adaptive day/night luminance blending         │
               │  • LAB color-cast correction                     │
               └────────────────────────┬─────────────────────────┘
                                        │
                                 Enhanced Frame
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │              ADAS Perception Pipeline            │
               │  • YOLOv8 (s/n) Vehicle & Pedestrian Tracking    │
               │  • Classical Canny/Hough Lane Tracker            │
               │  • MiDaS Monocular Depth Proximity Warning       │
               │  • Optional: Pothole & Road Sign Detectors       │
               └────────────────────────┬─────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
  🌟 Next.js + FastAPI Web UI                     🧪 Streamlit Interface
  • Real-time 60 FPS Canvas HUD                   • Tabbed research layout
  • Interactive comparison slider                 • Playwright test fixtures
  • Direct FFmpeg stream writer                   • Matplotlib / Plotly charts
```

---

## 📁 Project Structure

```
UNLET-ADAS/
├── web/                             # 🌟 PRIMARY FULL-STACK APPLICATION
│   ├── backend/                     # FastAPI backend service
│   │   ├── main.py                  # API routes (/api/enhance/*, /api/jobs/*)
│   │   ├── pipeline.py              # Low-latency streaming & Direct FFmpeg writer
│   │   └── requirements.txt         # Backend web dependencies
│   └── frontend/                    # Next.js 16 + React 19 + Tailwind CSS UI
│       ├── app/                     # App router (/, /live, /video)
│       ├── components/              # CompareView, LiveHUD, StatCards, Navbar
│       ├── lib/api.ts               # Type-safe API client
│       └── out/                     # Pre-compiled static export
│
├── app/                             # 🧪 SECONDARY APPLICATION & WEIGHTS
│   ├── streamlit_app.py             # Streamlit research interface
│   ├── zerodce_cbam_best.pt         # Trained Zero-DCE++ CBAM weights (21.7k params)
│   ├── pothole_best.pt              # Trained pothole detector weights
│   └── signs_best.pt                # Trained Indian road-sign detector weights
│
├── src/                             # CORE DEEP LEARNING & PERCEPTION MODULES
│   ├── model.py                     # Zero-DCE++ CBAM architecture definition
│   ├── losses.py                    # Multi-term composite unsupervised loss
│   ├── enhance.py                   # Full-resolution curve application & blending
│   ├── detection.py                 # YOLOv8 tracking & bounding box overlays
│   ├── depth.py                     # MiDaS monocular depth & risk estimation
│   ├── lane_detection.py            # Classical Canny/Hough lane boundary detection
│   ├── signs.py                     # Traffic sign visualization & color hashing
│   ├── benchmark.py                 # CPU/GPU inference throughput benchmark
│   ├── train.py                     # Training script for Zero-DCE++ CBAM
│   ├── train_lowlight.py            # ExDark low-light YOLO fine-tuning
│   ├── train_pothole.py             # Pothole detector fine-tuning (Roboflow)
│   └── train_signs.py               # Road sign detector fine-tuning (Roboflow)
│
├── notebooks/                       # GOOGLE COLAB TRAINING NOTEBOOKS
│   ├── UNLET_ADAS_Colab.ipynb       # Main enhancer training
│   ├── UNLET_ADAS_Pothole_Colab.ipynb
│   ├── UNLET_ADAS_Signs_Colab.ipynb
│   └── UNLET_ADAS_Demo_Colab.ipynb  # Free GPU cloud demo launcher
│
├── tests/                           # AUTOMATED TEST SUITE
│   ├── test_web_api.py              # FastAPI endpoints & live streaming tests
│   ├── test_detection.py            # Perception pipeline unit tests
│   ├── test_enhance.py              # Zero-DCE++ model unit tests
│   └── test_app.py                  # End-to-end Playwright tests
│
├── Dockerfile.web                   # Multi-stage production container (Next.js + FastAPI)
├── Dockerfile                       # Streamlit deployment container
├── render.yaml                      # One-click Render.com blueprint
└── requirements.txt                 # Core ML requirements
```

---

## 🛠️ Tech Stack

| Domain | Primary Web App (Main) | Secondary Interface (Research) |
|---|---|---|
| **Frontend** | **Next.js 16 (React 19)**, TypeScript, Tailwind CSS, Framer Motion | **Streamlit** (Python components) |
| **Backend & APIs** | **FastAPI**, Uvicorn ASGI, Direct FFmpeg Pipe | Python in-memory execution |
| **Core AI Model** | **Zero-DCE++ with CBAM Attention** (21,769 parameters) | Same core model |
| **Perception Engine** | **YOLOv8** (s/n) + ByteTrack object tracking | Same core model |
| **Depth & Risk** | **MiDaS Small** monocular depth estimation | Same core model |
| **Lane Detection** | Classical Canny edge + Hough transform | Same core model |
| **Video Engine** | **Direct-to-FFmpeg H.264 pipe** (single-pass) | OpenCV VideoWriter + transcode |
| **Deployment** | Dockerfile.web (Render / Railway / HF Spaces) | Streamlit Community Cloud |

---

## 🏋️ Training Your Own Models

### Train Enhancer on LOL Dataset
```bash
python src/train.py \
  --data_root ./data/lol_dataset \
  --epochs 100 \
  --batch_size 8
```

If training is interrupted, resume instantly from the latest epoch checkpoint:
```bash
python src/train.py --data_root ./data/lol_dataset --save_dir ./checkpoints --epochs 100 --resume
```

### Train Optional Pothole Detector
```bash
pip install roboflow
python src/train_pothole.py --roboflow_key YOUR_FREE_API_KEY
# Copy trained weights: cp checkpoints/pothole_best.pt app/pothole_best.pt
```

### Train Optional Road Sign Detector
```bash
python src/train_signs.py --roboflow_key YOUR_FREE_API_KEY
# Copy trained weights: cp checkpoints/signs_best.pt app/signs_best.pt
```

---

## 🧪 Automated Testing

Run the automated test suite covering both the ML core and FastAPI web endpoints:

```bash
# Run web API & pipeline tests
pytest tests/test_web_api.py tests/test_detection.py -v

# Run entire test suite
pytest tests/ -v
```

---

## 🚢 Production Deployment

### 1. Render.com / Cloud Containers
The repository includes [`render.yaml`](render.yaml) and [`Dockerfile.web`](Dockerfile.web) for automated deployment:
1. Connect your GitHub repository to [Render.com](https://render.com).
2. Choose **Blueprint** deployment (`render.yaml`).
3. Render automatically builds the Next.js static assets and spins up the FastAPI container.

### 2. Hugging Face Spaces
Create a new Space with **Docker SDK** and link this repository with [`Dockerfile.web`](Dockerfile.web) configured.

---

## 📄 Citation

```bibtex
@misc{deekshith2026unletadas,
  title     = {UNLET-ADAS: Real-Time Low-Light Video Enhancement and Perception for Intelligent Vehicle Systems},
  author    = {Deekshith},
  year      = {2026},
  school    = {SJB Institute of Technology, Bengaluru},
  note      = {B.E. Major Project, Department of Computer Science & Engineering},
  url       = {https://github.com/DEEK-SHITH/UNLET-ADAS}
}
```

---

## 👤 Author

**Deekshith**  
B.E. Computer Science & Engineering  
SJB Institute of Technology, Bengaluru  
GitHub: [@DEEK-SHITH](https://github.com/DEEK-SHITH)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
