\# 🚗 UNLET-ADAS

\## Real-Time Low-Light Video Enhancement for Intelligent Vehicle Systems



\[!\[Demo](https://img.shields.io/badge/🤗-Live\_Demo-yellow)](https://huggingface.co/spaces/YourName/UNLET-ADAS)

\[!\[Paper](https://img.shields.io/badge/📄-Paper-blue)](link)

\[!\[License](https://img.shields.io/badge/License-MIT-green)](LICENSE)



> B.E. Major Project | SJBIT Bengaluru | Computer Science | 2025–26



\---



\## 🎯 Problem

Night driving causes 50% of fatal accidents despite only 25% of travel.

ADAS cameras fail in low-light — they cannot detect pedestrians, 

vehicles or lane markings in the dark.



\## 💡 Solution

UNLET-ADAS enhances dark video frames in real time using 

Zero-DCE++ with CBAM Attention, then runs YOLOv8 detection 

and DeepSORT tracking on the enhanced output.



\## 📊 Results



| Metric | Before Enhancement | After Enhancement |

|--------|-------------------|-------------------|

| PSNR | 8.2 dB | 21.3 dB |

| SSIM | 0.18 | 0.72 |

| Objects detected/frame | 0.01 | 0.04 |

| Detection confidence | 0.41 | 0.67 |



\## 🏗️ Architecture



\## 🚀 Quick Start



```bash

git clone https://github.com/YourName/UNLET-ADAS.git

cd UNLET-ADAS

pip install -r requirements.txt

python app/app.py

```



\## 📁 Project Structure



\## 🛠️ Tech Stack

\- \*\*Enhancement:\*\* Zero-DCE++ with CBAM Attention (PyTorch)

\- \*\*Detection:\*\* YOLOv8l (Ultralytics)

\- \*\*Tracking:\*\* DeepSORT

\- \*\*Lane Detection:\*\* Polynomial curve fitting

\- \*\*Depth:\*\* MiDaS monocular depth estimation

\- \*\*App:\*\* Gradio + HuggingFace Spaces

\- \*\*Training Data:\*\* LOL Dataset (485 pairs)



\## 📈 Training



```bash

python src/train.py --data\_root ./data/lol\_dataset \\

&#x20;                   --epochs 100 \\

&#x20;                   --batch\_size 8

```



\## 👤 Author

\*\*Deekshith\*\* | B.E. CSE | SJBIT Bengaluru  

GitHub: \[@DEEK-SHITH](https://github.com/DEEK-SHITH)



\## 📄 License

MIT License — free to use for research and education.

