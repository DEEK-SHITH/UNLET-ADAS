"""
Regression test for a real deployment-blocking bug: the Colab
notebook's inline model class and src/model.py's ZeroDCECBAM (which
the deployed app loads checkpoints into with strict=True) must define
the exact same set of parameter/buffer names, or a checkpoint trained
via the notebook fails to load into the app at all.

This bit for real: the notebook named its final curve-output layer
`self.out`, src/model.py names it `self.curve_out` -- architecturally
identical, but load_state_dict(strict=True) rejected the mismatch.
This test extracts the notebook's actual class definition (not a
hand-copied duplicate, which could drift independently) and diffs its
state_dict() keys against the real ZeroDCECBAM's.
"""
import os

import nbformat
import torch
import torch.nn as nn

from src.model import build_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(
    REPO_ROOT, 'notebooks', 'UNLET_ADAS_Colab.ipynb')


def _load_notebook_model_class():
    """Exec the notebook's own model-building code (Cell 4, from its
    first helper class through ZeroDCECBAM itself) and return the
    ZeroDCECBAM class object, so this test tracks whatever the
    notebook actually contains rather than a copy that could drift.
    Stops before the model-instantiation/verification lines, which
    need a real DEVICE and aren't needed here."""
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell4_source = nb.cells[4].source

    end = cell4_source.index('model = ZeroDCECBAM')
    class_source = cell4_source[:end]

    namespace = {'torch': torch, 'nn': nn}
    exec(class_source, namespace)
    return namespace['ZeroDCECBAM']


def test_notebook_model_state_dict_keys_match_src_model():
    NotebookZeroDCECBAM = _load_notebook_model_class()
    notebook_model = NotebookZeroDCECBAM(iters=8, ch=32)

    real_model = build_model(num_iters=8, channels=32)

    notebook_keys = set(notebook_model.state_dict().keys())
    real_keys = set(real_model.state_dict().keys())

    missing_from_notebook = real_keys - notebook_keys
    extra_in_notebook = notebook_keys - real_keys

    assert not missing_from_notebook and not extra_in_notebook, (
        f'Notebook model and src/model.py ZeroDCECBAM have diverged '
        f'parameter names -- a checkpoint trained via the notebook '
        f'would fail model.load_state_dict(strict=True) in the app.\n'
        f'In src/model.py but not the notebook: {missing_from_notebook}\n'
        f'In the notebook but not src/model.py: {extra_in_notebook}')


def test_notebook_checkpoint_loads_into_real_model():
    """End-to-end: train the notebook's actual model class for one
    forward pass, save its state_dict, and confirm it loads directly
    (strict=True, no key remapping) into the real deployed model --
    exactly the path a real Colab-trained checkpoint takes."""
    NotebookZeroDCECBAM = _load_notebook_model_class()
    notebook_model = NotebookZeroDCECBAM(iters=8, ch=32)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(notebook_model.state_dict(), f.name)
        ckpt_path = f.name

    try:
        real_model = build_model(num_iters=8, channels=32)
        real_model.load_state_dict(
            torch.load(ckpt_path, map_location='cpu'), strict=True)
    finally:
        os.unlink(ckpt_path)
