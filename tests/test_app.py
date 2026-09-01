"""
End-to-end smoke tests for the UNLET-ADAS Streamlit app.

Drives the real app in a headless browser and checks that each of the
three main tabs (Image, Video, Live Camera) loads and, where it doesn't
need real camera hardware, actually processes input end to end — not
just that the page renders.
"""
import time

from conftest import assert_no_app_error


def test_image_tab_enhances(page, dark_test_image):
    # Image Enhancement is the default active tab.
    file_input = page.locator('input[type="file"][accept*=".jpg"]')
    file_input.set_input_files(dark_test_image)

    enhance_btn = page.get_by_role('button', name='Enhance + Detect')
    enhance_btn.wait_for(timeout=15000)
    enhance_btn.click()

    page.get_by_text('UNLET Enhanced').wait_for(timeout=90000)
    page.get_by_role(
        'button', name='Download Enhanced Image').wait_for(timeout=15000)

    assert_no_app_error(page)


def test_video_tab_processes(page, short_test_video):
    page.get_by_role('tab', name='Video Enhancement').click()

    # Streamlit renders the real <input type=file> hidden and shows a
    # styled dropzone over it, so wait for it to be attached rather than
    # visible before uploading into it directly.
    file_input = page.locator('input[type="file"][accept*=".mp4"]')
    file_input.wait_for(state='attached', timeout=15000)
    file_input.set_input_files(short_test_video)

    enhance_btn = page.get_by_role('button', name='Enhance Video')
    enhance_btn.wait_for(timeout=15000)
    enhance_btn.click()

    # Chunked processing reruns the app repeatedly until it finishes —
    # give it a generous ceiling for a CPU-only CI runner.
    page.get_by_text('Processed', exact=False).wait_for(timeout=180000)
    page.get_by_role('button', name='Download Enhanced').wait_for(timeout=15000)
    page.get_by_role('button', name='Download Comparison').wait_for(timeout=15000)

    assert_no_app_error(page)


def test_video_cancel_stops_promptly(page, medium_test_video):
    # Regression test for a real bug: Cancel used to set the cancel
    # flag but then still run one more full processing chunk before
    # honoring it, so clicking Cancel didn't visibly do anything for
    # a long stretch. It must now stop within one chunk, quickly.
    page.get_by_role('tab', name='Video Enhancement').click()

    file_input = page.locator('input[type="file"][accept*=".mp4"]')
    file_input.wait_for(state='attached', timeout=15000)
    file_input.set_input_files(medium_test_video)

    enhance_btn = page.get_by_role('button', name='Enhance Video')
    enhance_btn.wait_for(timeout=15000)
    enhance_btn.click()

    cancel_btn = page.get_by_role('button', name='Cancel Enhancement')
    cancel_btn.wait_for(timeout=20000)
    page.wait_for_timeout(3000)  # let a bit of real progress happen

    t0 = time.time()
    cancel_btn.click()
    page.get_by_text('Cancelled', exact=False).first.wait_for(timeout=20000)
    elapsed = time.time() - t0

    assert elapsed < 20, (
        f'Cancel took {elapsed:.1f}s to take effect — should stop '
        'within roughly one processing chunk, not grind through the '
        'whole remaining video first.')

    body_text = page.locator('body').inner_text()
    assert 'kept the' in body_text and 'frames processed before' in body_text

    assert_no_app_error(page)


def test_live_camera_tab_loads(page):
    page.get_by_role('tab', name='Live Camera').click()

    # No real camera in CI, so this checks the tab renders correctly
    # (mode switch, snapshot control) rather than an actual capture.
    # The "Snapshot" text also appears in the About tab's feature list
    # elsewhere in the (fully rendered, mostly hidden) DOM, so scope to
    # the first/visible match rather than requiring strict uniqueness.
    page.get_by_text('Snapshot', exact=False).first.wait_for(timeout=15000)
    page.get_by_role('button', name='Turn Camera On').wait_for(timeout=15000)

    assert_no_app_error(page)


def test_about_tab_loads(page):
    page.get_by_role('tab', name='About Project').click()

    page.get_by_text('About UNLET-ADAS').wait_for(timeout=15000)
    page.get_by_text('System Architecture').wait_for(timeout=15000)

    assert_no_app_error(page)
