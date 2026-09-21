# UNLET-ADAS Web (React/Next.js + FastAPI)

An additional, independent frontend for the same enhance/detect pipeline
the Streamlit app (`app/streamlit_app.py`) uses — built for more control
over look and feel. **The Streamlit app is untouched**: this is a second
front door onto `src/enhance.py`, `src/lane_detection.py`, etc., not a
replacement.

```
web/
  backend/    FastAPI service wrapping src/enhance.py
  frontend/   Next.js (App Router) + TypeScript + Tailwind UI
```

## Backend

```bash
# From the repo root, with the project's normal Python env active:
pip install -r requirements.txt -r web/backend/requirements.txt
uvicorn web.backend.main:app --reload --port 8000
```

Loads the same weight files the Streamlit app uses
(`app/zerodce_cbam_best.pt`, `app/pothole_best.pt`, `app/signs_best.pt`,
`yolov8s.pt`/`yolov8n.pt`), so nothing needs to be trained or downloaded
separately. Endpoints:

| Method | Path                         | Purpose                                   |
|--------|------------------------------|--------------------------------------------|
| GET    | `/api/health`                | Liveness + device (cpu/cuda)                |
| GET    | `/api/config`                | Which optional detectors are available      |
| POST   | `/api/enhance/image`         | Enhance + detect a single image (synchronous, returns JSON with base64 images) |
| POST   | `/api/enhance/video`         | Submit a video; runs in the background      |
| GET    | `/api/jobs/{id}`             | Poll progress of a video job                 |
| POST   | `/api/jobs/{id}/cancel`      | Cancel an in-progress video job              |
| GET    | `/api/jobs/{id}/video`       | Download the finished, browser-playable video |

Video jobs mirror the Streamlit Video tab's "fast mode" — object
detection runs on every 2nd frame, with boxes carried over onto the
frame in between — and are capped at 900 frames per job in this demo
API (`pipeline.VideoOptions.max_frames`).

## Frontend

```bash
cd web/frontend
cp .env.local.example .env.local   # points at http://localhost:8000 by default
npm install
npm run dev       # http://localhost:3000
```

Three pages: **Image** (`/`), **Video** (`/video`), and **Live Camera**
(`/live`), each with its own controls panel (adaptive enhancement,
detector toggles, confidence/resolution sliders) and a live view of
the backend's response — a drag-to-compare original/enhanced slider
for images and live-camera snapshots, a progress bar + playable/
downloadable result for videos.

Live Camera uses `getUserMedia()` to preview your browser's camera and
captures a snapshot on demand (not a continuous stream — full-quality
enhancement + detection is too slow per-frame on CPU for that, same
constraint the Streamlit app's Live Stream tab documents), then runs
it through `/api/enhance/image` like the Image page.

`npm run build && npm run start` for a production build.

## Notes

- CORS is wide open (`allow_origins=["*"]`) for local dev against any
  `next dev` port. Tighten this in `web/backend/main.py` before
  deploying the backend somewhere public.
- The depth-based risk toggle falls back gracefully (like the
  Streamlit app) if the MiDaS model can't be downloaded (e.g. no
  outbound internet to torch.hub) — `/api/config` reports `has_depth`.
- Nothing here changes `requirements.txt`, `Dockerfile`, or the
  Hugging Face Space deployment — `web/backend/requirements.txt` is
  installed separately and on top of the existing project deps.
