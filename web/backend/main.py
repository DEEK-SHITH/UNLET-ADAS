"""
UNLET-ADAS API — FastAPI backend for the React/Next.js frontend.

Run from the repo root (so `src/` and `app/` resolve correctly):

    pip install -r requirements.txt -r web/backend/requirements.txt
    uvicorn web.backend.main:app --reload --port 8000

This does not touch app/streamlit_app.py or anything it depends on —
it is a second, independent way to reach the same enhance/detect
pipeline in src/enhance.py, for a custom frontend. The Streamlit app
keeps working exactly as before, deployed exactly as before.
"""
import base64
import io
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from . import pipeline as pl

TMP_DIR = tempfile.mkdtemp(prefix='unlet_adas_api_')


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading models...')
    pl.load_all_models()
    print(f'Models ready. has_yolo={pl.MODELS.has_yolo} '
          f'has_pothole={pl.MODELS.has_pothole} has_signs={pl.MODELS.has_signs}')
    yield
    shutil.rmtree(TMP_DIR, ignore_errors=True)


app = FastAPI(title='UNLET-ADAS API', version='1.0.0', lifespan=lifespan)

# Wide open by default for local dev against `next dev` on any port;
# tighten allow_origins to your deployed frontend's origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _rgb_to_data_url(rgb_array) -> str:
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'device': str(pl.MODELS.device) if pl.MODELS.device else None,
    }


@app.get('/api/config')
def config():
    return {
        'has_detect': pl.MODELS.has_yolo,
        'has_pothole': pl.MODELS.has_pothole,
        'has_signs': pl.MODELS.has_signs,
        'has_depth': pl.ensure_depth_model() if pl.MODELS.device else False,
        'classes': {str(k): v[0] for k, v in pl.ADAS_CLASSES.items()},
    }


@app.post('/api/enhance/image')
async def enhance_image(
    file: UploadFile = File(...),
    adaptive: bool = Form(True),
    det_conf: float = Form(0.25),
    det_imgsz: int = Form(640),
    enable_detect: bool = Form(True),
    enable_pothole: bool = Form(False),
    enable_signs: bool = Form(False),
    enable_lanes: bool = Form(False),
    use_depth_risk: bool = Form(False),
):
    if not pl.MODELS.enhancer:
        raise HTTPException(503, 'Enhancer model not loaded')

    raw = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception:
        raise HTTPException(400, 'Could not read image file')

    opts = pl.ImageOptions(
        adaptive=adaptive, det_conf=det_conf, det_imgsz=det_imgsz,
        enable_detect=enable_detect, enable_pothole=enable_pothole,
        enable_signs=enable_signs, enable_lanes=enable_lanes,
        use_depth_risk=use_depth_risk,
    )
    enh_rgb, result = pl.run_image_pipeline(pil_image, opts)
    result['original_image'] = _rgb_to_data_url(
        __import__('numpy').array(pil_image))
    result['enhanced_image'] = _rgb_to_data_url(enh_rgb)
    return result


@app.post('/api/enhance/video')
async def enhance_video(
    file: UploadFile = File(...),
    adaptive: bool = Form(True),
    det_conf: float = Form(0.25),
    det_imgsz: int = Form(640),
    enable_detect: bool = Form(True),
    enable_pothole: bool = Form(False),
    enable_lanes: bool = Form(False),
    fast_mode: bool = Form(True),
):
    if not pl.MODELS.enhancer:
        raise HTTPException(503, 'Enhancer model not loaded')

    job = pl.new_job()
    ext = os.path.splitext(file.filename or 'input.mp4')[1] or '.mp4'
    input_path = os.path.join(TMP_DIR, f'{job.id}_input{ext}')
    with open(input_path, 'wb') as f:
        f.write(await file.read())

    opts = pl.VideoOptions(
        adaptive=adaptive, det_conf=det_conf, det_imgsz=det_imgsz,
        enable_detect=enable_detect, enable_pothole=enable_pothole,
        enable_lanes=enable_lanes, fast_mode=fast_mode,
    )

    import threading
    thread = threading.Thread(
        target=pl.run_video_job, args=(job.id, input_path, TMP_DIR, opts), daemon=True)
    thread.start()

    return {'job_id': job.id}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    job = pl.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'Unknown job id')
    return {
        'id': job.id, 'status': job.status, 'progress': job.progress,
        'frames_done': job.frames_done, 'frames_total': job.frames_total,
        'error': job.error, 'counts': job.counts, 'high_risk': job.high_risk,
        'ready': job.status == 'done',
    }


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    job = pl.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, 'Unknown job id')
    job.cancel_requested = True
    return {'ok': True}


@app.get('/api/jobs/{job_id}/video')
def job_video(job_id: str):
    job = pl.JOBS.get(job_id)
    if not job or job.status != 'done' or not job.output_path:
        raise HTTPException(404, 'Video not ready')
    return FileResponse(job.output_path, media_type='video/mp4',
                         filename=f'unlet_adas_enhanced_{job_id}.mp4')
