"""
UNLET-ADAS: Loss Functions
===========================
Composite loss for Zero-DCE++ training:
- Color Constancy Loss  (weight=50) — prevents color cast
- Exposure Loss         (weight=10) — targets correct brightness
- Spatial Loss          (weight=1)  — preserves structure
- Smoothness Loss       (weight=200)— smooth curves
- L1 Reconstruction     (weight=1)  — pixel fidelity
- Perceptual Loss       (weight=0.1)— texture preservation
- SSIM Loss             (weight=2)  — structural similarity
- Frequency Loss        (weight=0.1)— edge preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ─────────────────────────────────────────────
# Unsupervised losses (no ground truth needed)
# ─────────────────────────────────────────────

def color_constancy_loss(x):
    """
    Prevents color cast by forcing R, G, B channel
    means to be equal. Weight=50 prevents green tint.
    """
    mean = x.mean(dim=[2, 3])
    r, g, b = mean[:, 0], mean[:, 1], mean[:, 2]
    return ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()


def exposure_loss(x, target=0.6, patch_size=16):
    """
    Pushes average patch brightness toward target value.
    target=0.6 works well for dark road scenes.
    """
    gray = (0.299 * x[:, 0] +
            0.587 * x[:, 1] +
            0.114 * x[:, 2]).unsqueeze(1)
    pool = F.avg_pool2d(gray, patch_size, stride=patch_size)
    return F.mse_loss(pool, torch.ones_like(pool) * target)


def spatial_consistency_loss(enhanced, original, patch=4):
    """Preserves spatial structure of the original image."""
    ep = F.avg_pool2d(enhanced, patch, stride=patch)
    op = F.avg_pool2d(original, patch, stride=patch)
    return F.mse_loss(ep, op)


def smoothness_loss(curves):
    """Ensures smooth curve transitions across pixels."""
    dx = curves[:, :, :, 1:] - curves[:, :, :, :-1]
    dy = curves[:, :, 1:, :] - curves[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


# ─────────────────────────────────────────────
# Supervised losses (need ground truth)
# ─────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps.
    Preserves textures and fine details.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        vgg    = tvm.vgg19(
            weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        self.s1 = nn.Sequential(*list(vgg)[:4]).eval()
        self.s2 = nn.Sequential(*list(vgg)[4:9]).eval()
        self.s3 = nn.Sequential(*list(vgg)[9:18]).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        mean = torch.tensor(
            [0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        std  = torch.tensor(
            [0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0
        for sl in [self.s1, self.s2, self.s3]:
            x = sl(x); y = sl(y)
            loss += F.l1_loss(x, y)
        return loss


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for structural similarity."""
    def __init__(self, window_size=11):
        super().__init__()
        self.ws = window_size
        sigma   = 1.5
        coords  = torch.arange(
            window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        k = (g.unsqueeze(0) * g.unsqueeze(1)
             ).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer('kernel', k)

    def forward(self, x, y):
        p   = self.ws // 2
        mx  = F.conv2d(x,   self.kernel, padding=p, groups=3)
        my  = F.conv2d(y,   self.kernel, padding=p, groups=3)
        sx  = F.conv2d(x*x, self.kernel, padding=p, groups=3) - mx**2
        sy  = F.conv2d(y*y, self.kernel, padding=p, groups=3) - my**2
        sxy = F.conv2d(x*y, self.kernel, padding=p, groups=3) - mx*my
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mx*my+C1) * (2*sxy+C2)) / \
               ((mx**2+my**2+C1) * (sx+sy+C2))
        return 1 - ssim.mean()


def frequency_loss(x, y):
    """Frequency domain loss — preserves high-freq edges."""
    xf = torch.fft.fft2(x, norm='ortho')
    yf = torch.fft.fft2(y, norm='ortho')
    return F.l1_loss(torch.abs(xf), torch.abs(yf))


# ─────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────

class UNLETLoss(nn.Module):
    """
    Complete composite loss for UNLET-ADAS training.

    Weights tuned to prevent green color cast:
    - color_constancy weight = 50 (key fix for green tint)
    - ssim weight = 2 (better structural preservation)
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.perceptual = VGGPerceptualLoss(device)
        self.ssim       = SSIMLoss()

    def forward(self, enhanced, curves, original, target=None):
        # Unsupervised losses (always computed)
        loss = (
            50.0  * color_constancy_loss(enhanced) +
            10.0  * exposure_loss(enhanced) +
            1.0   * spatial_consistency_loss(enhanced, original) +
            200.0 * smoothness_loss(curves)
        )

        # Supervised losses (only for samples with ground truth). A
        # batch can mix paired and unpaired (all-zero target) rows --
        # e.g. LOL images alongside extra unpaired low-light data
        # (src/train.py --extra_low_dirs) -- so this must be a
        # per-sample mask, not a whole-batch target.sum() check: that
        # would apply the supervised loss to unpaired rows too,
        # pushing them toward an all-black image.
        if target is not None:
            has_gt = target.reshape(target.size(0), -1).sum(dim=1) > 0
            if has_gt.any():
                e, t = enhanced[has_gt], target[has_gt]
                loss = loss + (
                    1.0 * F.l1_loss(e, t) +
                    0.1 * self.perceptual(e, t) +
                    2.0 * self.ssim(e, t) +
                    0.1 * frequency_loss(e, t)
                )

        return loss