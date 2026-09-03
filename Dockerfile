# UNLET-ADAS on Hugging Face Spaces (Docker SDK, GPU hardware tier)
# ======================================================================
# Streamlit Community Cloud is CPU-only, which caps live-stream frame
# rate. This image is built for HF Spaces' GPU tiers instead -- the
# app itself already auto-detects CUDA (torch.cuda.is_available()) for
# the enhancer, MiDaS depth model, and YOLO detectors, so no code
# changes were needed; this is purely deployment plumbing.
#
# Setup (one-time, on huggingface.co):
#   1. Create a new Space -> SDK: Docker -> pick a GPU hardware tier
#      (e.g. "T4 small").
#   2. In the Space's Settings, note its full name
#      (<your-hf-username>/<space-name>).
#   3. In this GitHub repo's Settings -> Secrets and variables ->
#      Actions, add:
#        HF_TOKEN     - a Hugging Face access token with write access
#                       (huggingface.co/settings/tokens)
#        HF_SPACE_REPO - <your-hf-username>/<space-name> from step 2
#   4. Push to main -- .github/workflows/deploy-hf-space.yml pushes
#      this repo to the Space automatically, which rebuilds this
#      Dockerfile on HF's GPU runtime.
#
# See README.md's "Deploy to Hugging Face Spaces (GPU)" section for
# the full walkthrough.

FROM python:3.11-slim

# opencv-python-headless, streamlit-webrtc (av/aiortc), and MiDaS/YOLO
# all need a few system libs a bare slim image doesn't ship with.
# build-essential is a defensive include: lap (used for detection
# tracking) and streamlit-webrtc's C-extension dependencies
# (pylibsrtp, etc.) aren't guaranteed to have a prebuilt wheel for
# every platform, and this was written without a way to run a real
# `docker build` to confirm otherwise (see the note in this Space's
# README section on GPU deployment).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces containers run as a non-root user (uid 1000) by
# convention -- torch/ultralytics/streamlit all need a writable HOME
# for their caches (model downloads, .streamlit config, etc.).
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt packages.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .
USER appuser

# HF Spaces' Docker SDK expects the app on port 7860 by default.
EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
