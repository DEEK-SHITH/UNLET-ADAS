"""
Shared pytest fixtures for the end-to-end Streamlit UI tests.

Starts the real app (`streamlit run app/streamlit_app.py`) once per test
session, exactly as it runs in production, and drives it with Playwright.
No mocking of the model/detector — this exercises the real enhancement
and detection pipeline on CPU, the same way Streamlit Cloud does.
"""
import os
import subprocess
import sys
import time

import cv2
import pytest
import requests
from playwright.sync_api import sync_playwright

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_URL = 'http://localhost:8501'


@pytest.fixture(scope='session')
def streamlit_server():
    """Launch the Streamlit app as a subprocess and wait until it's ready."""
    proc = subprocess.Popen(
        [
            sys.executable, '-m', 'streamlit', 'run',
            os.path.join('app', 'streamlit_app.py'),
            '--server.headless', 'true',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.fileWatcherType', 'none',
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 120
    healthy = False
    while time.time() < deadline:
        try:
            r = requests.get(f'{APP_URL}/_stcore/health', timeout=2)
            if r.status_code == 200:
                healthy = True
                break
        except requests.exceptions.RequestException:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1)

    if not healthy:
        proc.terminate()
        out, _ = proc.communicate(timeout=10)
        pytest.fail(
            'Streamlit app did not become healthy within 120s. '
            f'Process log:\n{out}')

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope='session')
def browser():
    # PLAYWRIGHT_CHROMIUM_PATH is only ever set for local/dev environments
    # where the installed Playwright browser build doesn't match the pip
    # package's expected revision; CI installs a matching pair via
    # `playwright install chromium` and leaves this unset.
    launch_kwargs = {'headless': True}
    override = os.environ.get('PLAYWRIGHT_CHROMIUM_PATH')
    if override:
        launch_kwargs['executable_path'] = override

    with sync_playwright() as p:
        b = p.chromium.launch(**launch_kwargs)
        yield b
        b.close()


@pytest.fixture
def page(streamlit_server, browser, request):
    ctx = browser.new_context(viewport={'width': 1440, 'height': 1000})
    pg = ctx.new_page()
    pg.goto(APP_URL, wait_until='load', timeout=30000)
    # Streamlit renders an initial shell over the websocket connection —
    # give it time to finish hydrating and to finish loading the models
    # (the spinner in load_enhancer/load_detector) before tests interact.
    pg.wait_for_timeout(8000)

    yield pg

    os.makedirs(os.path.join(REPO_ROOT, 'test-artifacts'), exist_ok=True)
    safe_name = request.node.name.replace('/', '_')
    pg.screenshot(
        path=os.path.join(REPO_ROOT, 'test-artifacts', f'{safe_name}.png'),
        full_page=True)
    ctx.close()


@pytest.fixture(scope='session')
def dark_test_image(tmp_path_factory):
    """A small, dark synthetic image — enough to exercise the enhancer
    without depending on any external test-data file."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 40, size=(240, 320, 3)).astype('uint8')
    path = tmp_path_factory.mktemp('data') / 'dark_test.jpg'
    Image.fromarray(arr).save(path)
    return str(path)


def _trim_sample_video(out_path, seconds):
    """Trim the repo's own sample night-drive footage
    (results/original_night_drive.mp4) to ~seconds long, so tests run
    against real footage without a slow full-length run in CI."""
    src = os.path.join(REPO_ROOT, 'results', 'original_night_drive.mp4')
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    max_frames = int(fps * seconds)
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1
    writer.release()
    cap.release()

    if count == 0:
        pytest.fail(f'Could not read any frames from {src}')
    return count


@pytest.fixture(scope='session')
def short_test_video(tmp_path_factory):
    """A ~1s clip — enough to exercise the video pipeline end to end
    without a slow full-length run in CI."""
    out_path = tmp_path_factory.mktemp('data') / 'short_test.mp4'
    _trim_sample_video(out_path, seconds=1)
    return str(out_path)


@pytest.fixture(scope='session')
def medium_test_video(tmp_path_factory):
    """A ~8s clip — long enough to span several processing chunks, so
    a mid-run Cancel click actually has something to interrupt."""
    out_path = tmp_path_factory.mktemp('data') / 'medium_test.mp4'
    _trim_sample_video(out_path, seconds=8)
    return str(out_path)


def assert_no_app_error(page):
    """Streamlit renders uncaught exceptions from app code as a
    traceback block — fail loudly if one shows up mid-test instead of
    letting a later assertion fail with a confusing message."""
    body_text = page.locator('body').inner_text()
    assert 'Traceback (most recent call last)' not in body_text, (
        'Streamlit app raised an uncaught exception during this test')
