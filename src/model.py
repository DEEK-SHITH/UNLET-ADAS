"""
UNLET-ADAS: Zero-DCE++ with CBAM Attention
==========================================
B.E. Major Project | SJBIT Bengaluru | 2025-26

Architecture:
- Zero-DCE++ curve estimation backbone
- CBAM (Channel + Spatial) attention at every layer
- Depthwise separable convolutions for efficiency
- 8-iteration curve enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, max(channels // ratio, 1), bias=False),
            nn.ReLU(),
            nn.Linear(max(channels // ratio, 1), channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        a = self.fc(self.avg(x).view(B, C))
        m = self.fc(self.mx(x).view(B, C))
        return self.sigmoid(a + m).view(B, C, 1, 1) * x


class SpatialAttention(nn.Module):
    """Spatial attention using avg + max pooling."""
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return self.sigmoid(self.conv(
            torch.cat([avg, mx], 1))) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


def dw_block(in_ch, out_ch):
    """Depthwise separable conv block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3,
                  padding=1, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


class ZeroDCECBAM(nn.Module):
    """
    Zero-DCE++ with CBAM Attention.

    Estimates per-pixel enhancement curves A such that:
        I_enhanced = I + A * I * (1 - I)   [repeated N times]

    This is applied iteratively for progressive enhancement.
    """
    def __init__(self, num_iters=8, channels=32):
        super().__init__()
        self.num_iters = num_iters

        # Encoder
        self.e1  = dw_block(3,           channels)
        self.cb1 = CBAM(channels)
        self.e2  = dw_block(channels,    channels)
        self.cb2 = CBAM(channels)
        self.e3  = dw_block(channels,    channels)
        self.cb3 = CBAM(channels)
        self.e4  = dw_block(channels,    channels)
        self.cb4 = CBAM(channels)

        # Decoder with skip connections
        self.d3  = dw_block(channels * 2, channels)
        self.cb5 = CBAM(channels)
        self.d2  = dw_block(channels * 2, channels)
        self.cb6 = CBAM(channels)
        self.d1  = dw_block(channels * 2, channels)
        self.cb7 = CBAM(channels)

        # Curve parameter output
        self.curve_out = nn.Sequential(
            nn.Conv2d(channels, 3 * num_iters,
                      3, padding=1, bias=False),
            nn.Tanh())

    def estimate_curves(self, x):
        """Predict per-pixel enhancement curves for input x (any resolution)."""
        # Encoder
        e1 = self.cb1(self.e1(x))
        e2 = self.cb2(self.e2(e1))
        e3 = self.cb3(self.e3(e2))
        e4 = self.cb4(self.e4(e3))

        # Decoder
        d3 = self.cb5(self.d3(torch.cat([e4, e3], 1)))
        d2 = self.cb6(self.d2(torch.cat([d3, e2], 1)))
        d1 = self.cb7(self.d1(torch.cat([d2, e1], 1)))

        return self.curve_out(d1)

    def apply_curves(self, x, curves):
        """Apply the iterative curve formula to x using given curve maps."""
        enhanced = x
        for i in range(self.num_iters):
            A        = curves[:, i*3:(i+1)*3]
            enhanced = torch.clamp(
                enhanced + A * enhanced * (1 - enhanced),
                0, 1)
        return enhanced

    def forward(self, x):
        curves   = self.estimate_curves(x)
        enhanced = self.apply_curves(x, curves)
        return enhanced, curves

    @torch.no_grad()
    def enhance_full_res(self, x_full, proxy_size=256):
        """
        Estimate curves on a small downsized proxy (cheap, robust to
        any input size) then apply them directly to the full-resolution
        input. Curve maps are smooth by construction (smoothness_loss
        during training penalizes high-frequency curves), so upsampling
        them introduces no blur — unlike resizing actual pixel content
        down and back up, which is what causes soft/blurry output.
        """
        proxy       = F.interpolate(
            x_full, size=(proxy_size, proxy_size),
            mode='bilinear', align_corners=False)
        curves      = self.estimate_curves(proxy)
        curves_full = F.interpolate(
            curves, size=x_full.shape[2:],
            mode='bilinear', align_corners=False)
        return self.apply_curves(x_full, curves_full), curves_full


def build_model(num_iters=8, channels=32):
    """Build and return the UNLET-ADAS enhancement model."""
    model  = ZeroDCECBAM(num_iters=num_iters, channels=channels)
    params = sum(p.numel() for p in model.parameters())
    print(f'UNLET-ADAS model built')
    print(f'Parameters : {params:,}')
    print(f'Iterations : {num_iters}')
    print(f'Channels   : {channels}')
    return model


if __name__ == '__main__':
    import torch
    model = build_model()
    dummy = torch.rand(2, 3, 256, 256)
    enh, curves = model(dummy)
    print(f'Input  : {dummy.shape}')
    print(f'Output : {enh.shape}')
    print(f'Curves : {curves.shape}')