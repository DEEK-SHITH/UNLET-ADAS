"""
Unit tests for src/losses.py's UNLETLoss, focused on the per-sample
ground-truth masking needed for mixed batches (LOL pairs alongside
extra unpaired low-light images, src/train.py --extra_low_dirs).

Regression coverage for a real bug: the original implementation
checked target.sum() > 0 over the WHOLE batch, so a batch containing
even one paired sample would apply the supervised losses (L1,
perceptual, SSIM, frequency) to every row, including unpaired ones --
incorrectly pushing those toward an all-black image. It must instead
be applied only to the rows that actually have a non-zero target.
"""
import pytest
import torch
import torch.nn.functional as F
import torchvision.models as tvm

from src.losses import UNLETLoss


@pytest.fixture
def criterion(monkeypatch):
    # VGGPerceptualLoss normally downloads ImageNet-pretrained VGG19
    # weights, which needs internet access this sandbox blocks. The
    # masking logic under test only depends on VGGPerceptualLoss's
    # module structure, not on real pretrained features, so swap in
    # randomly-initialized weights instead of skipping this test.
    real_vgg19 = tvm.vgg19
    monkeypatch.setattr(tvm, 'vgg19', lambda *a, **k: real_vgg19(weights=None))
    return UNLETLoss(device='cpu')


def test_mixed_batch_matches_paired_only_subset_loss(criterion):
    torch.manual_seed(0)
    b, c, h, w = 2, 3, 32, 32
    enhanced = torch.rand(b, c, h, w)
    original = torch.rand(b, c, h, w)
    curves = torch.rand(b, 3, h, w)

    # Row 0 is paired (real target), row 1 is unpaired (all-zero).
    target = torch.zeros(b, c, h, w)
    target[0] = torch.rand(c, h, w)

    mixed_loss = criterion(enhanced, curves, original, target)

    # Ground truth: unsupervised loss over the FULL batch, plus the
    # supervised loss computed ONLY on the paired row (index 0).
    from src.losses import (color_constancy_loss, exposure_loss,
                             spatial_consistency_loss, smoothness_loss,
                             frequency_loss)
    unsupervised = (
        50.0 * color_constancy_loss(enhanced) +
        10.0 * exposure_loss(enhanced) +
        1.0 * spatial_consistency_loss(enhanced, original) +
        200.0 * smoothness_loss(curves)
    )
    e0, t0 = enhanced[0:1], target[0:1]
    supervised_paired_only = (
        1.0 * F.l1_loss(e0, t0) +
        0.1 * criterion.perceptual(e0, t0) +
        2.0 * criterion.ssim(e0, t0) +
        0.1 * frequency_loss(e0, t0)
    )
    expected = unsupervised + supervised_paired_only

    assert torch.allclose(mixed_loss, expected, atol=1e-5)


def test_all_unpaired_batch_uses_unsupervised_only(criterion):
    torch.manual_seed(1)
    b, c, h, w = 2, 3, 32, 32
    enhanced = torch.rand(b, c, h, w)
    original = torch.rand(b, c, h, w)
    curves = torch.rand(b, 3, h, w)
    target = torch.zeros(b, c, h, w)

    loss_with_zero_target = criterion(enhanced, curves, original, target)
    loss_with_none = criterion(enhanced, curves, original, None)

    assert torch.allclose(loss_with_zero_target, loss_with_none, atol=1e-6)


def test_all_paired_batch_matches_old_whole_batch_behavior(criterion):
    """When every row in the batch has a real target, per-sample
    masking must produce the same result as computing the supervised
    losses over the whole batch (the pre-existing, still-correct
    LOL-only case)."""
    torch.manual_seed(2)
    b, c, h, w = 3, 3, 32, 32
    enhanced = torch.rand(b, c, h, w)
    original = torch.rand(b, c, h, w)
    curves = torch.rand(b, 3, h, w)
    target = torch.rand(b, c, h, w)  # all rows paired

    from src.losses import (color_constancy_loss, exposure_loss,
                             spatial_consistency_loss, smoothness_loss,
                             frequency_loss)
    unsupervised = (
        50.0 * color_constancy_loss(enhanced) +
        10.0 * exposure_loss(enhanced) +
        1.0 * spatial_consistency_loss(enhanced, original) +
        200.0 * smoothness_loss(curves)
    )
    supervised_whole_batch = (
        1.0 * F.l1_loss(enhanced, target) +
        0.1 * criterion.perceptual(enhanced, target) +
        2.0 * criterion.ssim(enhanced, target) +
        0.1 * frequency_loss(enhanced, target)
    )
    expected = unsupervised + supervised_whole_batch

    actual = criterion(enhanced, curves, original, target)
    assert torch.allclose(actual, expected, atol=1e-5)
