"""
Unit tests for download_dataset() in src/train_signs.py and
src/train_pothole.py (identical, independently-maintained functions).

No prior coverage existed for this function at all -- every real bug
it had (stale target directory, an unreliable location= kwarg, and a
"succeeded with no exception but produced an empty export" case) was
only ever found by a user hitting it live in Colab, never by CI. These
tests fake out the Roboflow SDK so the retry/verification logic can be
exercised offline, deterministically, for all three failure modes.

`roboflow` itself is a training-only optional dependency (see
requirements-dev.txt's comment / README's Train the .../Detector
sections -- it's not installed for the app or CI by design), so these
tests inject a fake module into sys.modules rather than importing the
real package -- they work identically whether or not `roboflow`
happens to be installed on the machine running them.
"""
import os
import sys
import time
import types

import pytest

import src.train_pothole as train_pothole
import src.train_signs as train_signs


class _FakeVersion:
    """Replays a scripted sequence of download() outcomes: 'raise' (an
    exception, like a dropped connection), 'empty' (no exception, but
    an empty directory with no data.yaml -- the silent-failure mode
    a real user hit), or a path to a directory to return as-is."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.call_count = 0

    def download(self, fmt):
        self.call_count += 1
        outcome = self._outcomes.pop(0)
        if outcome == 'raise':
            raise RuntimeError('simulated network error')
        if outcome == 'empty':
            import tempfile
            return types.SimpleNamespace(location=tempfile.mkdtemp())
        return types.SimpleNamespace(location=outcome)


class _FakeProject:
    def __init__(self, version):
        self._version = version

    def version(self, n):
        return self._version


class _FakeRoboflow:
    """Stands in for roboflow.Roboflow: Roboflow(api_key=...) then
    .workspace(...).project(...).version(...) -- this fake ignores the
    workspace/project names and just hands back the same fake project
    either way, since download_dataset()'s own project/workspace names
    aren't what's under test here."""

    def __init__(self, project):
        self._project = project

    def __call__(self, api_key):
        return self

    def workspace(self, name):
        return self

    def project(self, name):
        return self._project


@pytest.fixture(params=[train_signs, train_pothole], ids=['signs', 'pothole'])
def module(request):
    return request.param


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    # The retry loop sleeps between attempts; don't actually wait in tests.
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)


def _dataset_dir(tmp_path, name, names_yaml='names: [a, b]\n'):
    d = tmp_path / name
    d.mkdir()
    (d / 'data.yaml').write_text(names_yaml)
    return str(d)


def _patch_download(module, monkeypatch, outcomes):
    version = _FakeVersion(outcomes)
    fake_roboflow_module = types.ModuleType('roboflow')
    fake_roboflow_module.Roboflow = _FakeRoboflow(_FakeProject(version))
    monkeypatch.setitem(sys.modules, 'roboflow', fake_roboflow_module)
    return version


def test_succeeds_on_first_try(module, tmp_path, monkeypatch):
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    version = _patch_download(module, monkeypatch, [src_dir])

    result = module.download_dataset('fake-key', dest_dir)

    assert result == dest_dir
    assert os.path.exists(os.path.join(dest_dir, 'data.yaml'))
    assert version.call_count == 1


def test_retries_past_an_exception(module, tmp_path, monkeypatch):
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    version = _patch_download(module, monkeypatch, ['raise', src_dir])

    result = module.download_dataset('fake-key', dest_dir, retries=3)

    assert result == dest_dir
    assert os.path.exists(os.path.join(dest_dir, 'data.yaml'))
    assert version.call_count == 2


def test_retries_past_a_silent_empty_download(module, tmp_path, monkeypatch):
    # Regression test for the real bug: a "successful" (no exception)
    # download attempt that produced no data.yaml anywhere must be
    # retried, not treated as a final failure.
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    version = _patch_download(module, monkeypatch, ['empty', src_dir])

    result = module.download_dataset('fake-key', dest_dir, retries=3)

    assert result == dest_dir
    assert os.path.exists(os.path.join(dest_dir, 'data.yaml'))
    assert version.call_count == 2


def test_finds_data_yaml_nested_one_level_deeper(module, tmp_path, monkeypatch):
    src_root = tmp_path / 'nested_src'
    nested = src_root / 'road-signs-2'
    nested.mkdir(parents=True)
    (nested / 'data.yaml').write_text('names: [a]\n')
    dest_dir = str(tmp_path / 'dest')
    _patch_download(module, monkeypatch, [str(src_root)])

    result = module.download_dataset('fake-key', dest_dir)

    assert result == dest_dir
    assert os.path.exists(os.path.join(dest_dir, 'data.yaml'))


def test_raises_after_exhausting_all_retries(module, tmp_path, monkeypatch):
    dest_dir = str(tmp_path / 'dest')
    _patch_download(module, monkeypatch, ['raise', 'raise'])

    with pytest.raises(RuntimeError, match='2 attempts'):
        module.download_dataset('fake-key', dest_dir, retries=2)


def test_raises_after_exhausting_retries_on_repeated_empty_downloads(
        module, tmp_path, monkeypatch):
    dest_dir = str(tmp_path / 'dest')
    _patch_download(module, monkeypatch, ['empty', 'empty'])

    with pytest.raises(RuntimeError, match='no data.yaml'):
        module.download_dataset('fake-key', dest_dir, retries=2)


def test_does_not_leave_a_stale_dest_dir_from_a_previous_run(
        module, tmp_path, monkeypatch):
    dest_dir = tmp_path / 'dest'
    dest_dir.mkdir()
    (dest_dir / 'leftover_from_failed_run.txt').write_text('stale')
    src_dir = _dataset_dir(tmp_path, 'src')
    _patch_download(module, monkeypatch, [src_dir])

    result = module.download_dataset('fake-key', str(dest_dir))

    assert result == str(dest_dir)
    assert os.path.exists(os.path.join(str(dest_dir), 'data.yaml'))
    assert not os.path.exists(
        os.path.join(str(dest_dir), 'leftover_from_failed_run.txt'))
