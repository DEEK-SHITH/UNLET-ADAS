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
        self.requested_versions = []

    def version(self, n):
        self.requested_versions.append(n)
        return self._version


class _FakeRoboflow:
    """Stands in for roboflow.Roboflow: Roboflow(api_key=...) then
    .workspace(...).project(...).version(...). Records the
    workspace/project names it was called with, so tests can confirm
    a custom --roboflow_workspace/--roboflow_project/--roboflow_version
    actually reaches the SDK call, not just the default -- otherwise
    this always hands back the same fake project regardless of name."""

    def __init__(self, project):
        self._project = project
        self.requested_workspaces = []
        self.requested_projects = []

    def __call__(self, api_key):
        return self

    def workspace(self, name):
        self.requested_workspaces.append(name)
        return self

    def project(self, name):
        self.requested_projects.append(name)
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
    project = _FakeProject(version)
    fake_rf = _FakeRoboflow(project)
    fake_roboflow_module = types.ModuleType('roboflow')
    fake_roboflow_module.Roboflow = fake_rf
    monkeypatch.setitem(sys.modules, 'roboflow', fake_roboflow_module)
    return version, fake_rf, project


def test_succeeds_on_first_try(module, tmp_path, monkeypatch):
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    version, _, _ = _patch_download(module, monkeypatch, [src_dir])

    result = module.download_dataset('fake-key', dest_dir)

    assert result == dest_dir
    assert os.path.exists(os.path.join(dest_dir, 'data.yaml'))
    assert version.call_count == 1


def test_retries_past_an_exception(module, tmp_path, monkeypatch):
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    version, _, _ = _patch_download(module, monkeypatch, ['raise', src_dir])

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
    version, _, _ = _patch_download(module, monkeypatch, ['empty', src_dir])

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


def test_signs_download_defaults_to_indian_traffic_signs(tmp_path, monkeypatch):
    # The default dataset should keep working unchanged for anyone not
    # overriding it.
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    _, fake_rf, project = _patch_download(train_signs, monkeypatch, [src_dir])

    train_signs.download_dataset('fake-key', dest_dir)

    assert fake_rf.requested_workspaces == ['indiantrafficsigns']
    assert fake_rf.requested_projects == ['indian-traffic-signs1']
    assert project.requested_versions == [4]


def test_signs_download_honors_a_custom_dataset(tmp_path, monkeypatch):
    # Regression test for a real limitation found in practice: an
    # earlier default dataset's class list (Indonesian road signs)
    # produced zero detections on a Vienna-Convention-style sign, even
    # at near-zero confidence -- download_dataset() must be able to
    # point at a different, region-specific Roboflow project instead
    # of being hardcoded to one default forever.
    src_dir = _dataset_dir(tmp_path, 'src')
    dest_dir = str(tmp_path / 'dest')
    _, fake_rf, project = _patch_download(train_signs, monkeypatch, [src_dir])

    train_signs.download_dataset(
        'fake-key', dest_dir,
        workspace='roboflow-100', project_slug='road-signs-6ih4y',
        version_num=1)

    assert fake_rf.requested_workspaces == ['roboflow-100']
    assert fake_rf.requested_projects == ['road-signs-6ih4y']
    assert project.requested_versions == [1]
