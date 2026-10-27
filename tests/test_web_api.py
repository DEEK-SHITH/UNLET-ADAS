import pytest
import numpy as np
import cv2
import tempfile
import os
from fastapi.testclient import TestClient
from web.backend.main import app
import web.backend.pipeline as pl


@pytest.fixture(scope="module")
def client():
    # Ensure models are loaded
    if not pl.MODELS.enhancer:
        pl.load_all_models()
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_api_config(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "has_detect" in data
    assert "classes" in data


def test_api_enhance_live_frame(client):
    # Dummy dark frame 384x288
    img = np.full((288, 384, 3), 35, dtype=np.uint8)
    cv2.circle(img, (192, 144), 40, (150, 150, 150), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    res = client.post(
        "/api/enhance/live-frame",
        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        data={"quality_mode": "turbo", "interleaved": "true", "enable_detect": "true"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "enhanced_image" in data
    assert data["enhanced_image"].startswith("data:image/jpeg;base64,")
    assert "processing_ms" in data
    assert data["processing_ms"] > 0
    assert "fps" in data
    assert "detections" in data
    assert "counts" in data
    assert "high_risk" in data


def test_api_enhance_video_turbo(client):
    # Generate a brief 10-frame video
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(tmp.name, fourcc, 30.0, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        vw.write(frame)
    vw.release()

    try:
        with open(tmp.name, "rb") as f:
            res = client.post(
                "/api/enhance/video",
                files={"file": ("clip.mp4", f, "video/mp4")},
                data={"quality_preset": "turbo"},
            )
        assert res.status_code == 200
        job_id = res.json().get("job_id")
        assert job_id is not None

        # Check job status endpoint
        s_res = client.get(f"/api/jobs/{job_id}")
        assert s_res.status_code == 200
        assert s_res.json()["id"] == job_id
    finally:
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except OSError:
                pass
