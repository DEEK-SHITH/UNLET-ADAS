# 🚗 UNLET-ADAS
## Real-Time Low-Light Enhancement for Intelligent Vehicle Systems

[![Demo](https://img.shields.io/badge/🌐-Live_Demo-green)](https://deek-shith-unlet-adas-streamlit-app.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-DEEK--SHITH-black)](https://github.com/DEEK-SHITH/UNLET-ADAS)
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
│ └── UNLET_ADAS_Colab.ipynb # Google Colab notebook
├── results/
│ ├── metric_comparison.png # PSNR/SSIM charts
│ ├── training_curves.png # Loss curves
│ └── video_frames.png # Before/after frames
├── requirements.txt
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