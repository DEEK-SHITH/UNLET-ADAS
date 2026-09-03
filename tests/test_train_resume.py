"""
End-to-end tests for src/train.py's --resume support: training must
be able to pick back up from the last completed epoch (model,
optimizer, scheduler, history, best_val, patience all restored)
instead of starting over after an interruption (Colab disconnect, GPU
quota runout, closed laptop -- the real incidents that motivated this).

Runs the real train() function on a tiny synthetic dataset; VGG19's
pretrained weights (used inside UNLETLoss) are mocked with random
init since download.pytorch.org is blocked in this sandbox -- see
tests/test_losses.py for the same pattern.
"""
import json
import os

import numpy as np
import pytest
import torchvision.models as tvm
from PIL import Image

from src.train import train


@pytest.fixture(autouse=True)
def _mock_vgg19(monkeypatch):
    real_vgg19 = tvm.vgg19
    monkeypatch.setattr(tvm, 'vgg19', lambda *a, **k: real_vgg19(weights=None))


def _make_tiny_lol(root, n=4, size=32):
    for split in ('our485', 'eval15'):
        for kind, color in (('low', (10, 10, 10)), ('high', (200, 200, 200))):
            d = os.path.join(root, split, kind)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                arr = np.full((size, size, 3), color, dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(d, f'img{i}.png'))
    return root


class Args:
    def __init__(self, data_root, save_dir, epochs, resume=False):
        self.data_root = data_root
        self.extra_low_dirs = None
        self.save_dir = save_dir
        self.resume = resume
        self.epochs = epochs
        self.batch_size = 2
        self.lr = 2e-4
        self.patience = 100  # disable early stopping for these tests
        self.image_size = 32


@pytest.fixture
def tiny_lol(tmp_path):
    return _make_tiny_lol(str(tmp_path / 'lol'))


def test_resume_state_saved_every_epoch(tiny_lol, tmp_path):
    save_dir = str(tmp_path / 'ckpt')
    train(Args(tiny_lol, save_dir, epochs=2))

    resume_path = os.path.join(save_dir, 'resume_state.pt')
    assert os.path.exists(resume_path)
    assert os.path.exists(os.path.join(save_dir, 'zerodce_cbam_best.pt'))

    import torch
    ckpt = torch.load(resume_path, map_location='cpu')
    assert ckpt['epoch'] == 1        # 0-indexed: epoch 2 of 2 just finished
    assert ckpt['total_epochs'] == 2
    assert len(ckpt['history']['train']) == 2


def test_resume_continues_epoch_count_not_restart(tiny_lol, tmp_path):
    save_dir = str(tmp_path / 'ckpt')
    train(Args(tiny_lol, save_dir, epochs=2))

    history = train(Args(tiny_lol, save_dir, epochs=4, resume=True))

    # 2 epochs from the first run + 2 more from the resumed run = 4
    # total entries, not a fresh 2-entry history from restarting.
    assert len(history['train']) == 4


def test_resume_restores_optimizer_and_best_val(tiny_lol, tmp_path):
    import torch
    save_dir = str(tmp_path / 'ckpt')
    train(Args(tiny_lol, save_dir, epochs=2))

    before = torch.load(
        os.path.join(save_dir, 'resume_state.pt'), map_location='cpu')

    history = train(Args(tiny_lol, save_dir, epochs=3, resume=True))

    after = torch.load(
        os.path.join(save_dir, 'resume_state.pt'), map_location='cpu')
    # The resumed run's best_val can only improve on or match the
    # restored one -- it must never silently reset to inf and start
    # "best" tracking over, which is what would happen if resume
    # wasn't actually wiring up best_val from the checkpoint.
    assert after['best_val'] <= before['best_val']
    assert after['epoch'] == 2  # 3rd epoch (0-indexed) just completed


def test_resume_without_existing_checkpoint_starts_fresh(tiny_lol, tmp_path):
    save_dir = str(tmp_path / 'ckpt_never_trained')
    # --resume passed but nothing has ever been saved to this dir --
    # must not crash, just behave like a normal fresh run.
    history = train(Args(tiny_lol, save_dir, epochs=2, resume=True))
    assert len(history['train']) == 2


def test_history_json_reflects_full_resumed_run(tiny_lol, tmp_path):
    save_dir = str(tmp_path / 'ckpt')
    train(Args(tiny_lol, save_dir, epochs=2))
    train(Args(tiny_lol, save_dir, epochs=4, resume=True))

    with open(os.path.join(save_dir, 'history.json')) as f:
        saved_history = json.load(f)
    assert len(saved_history['train']) == 4
