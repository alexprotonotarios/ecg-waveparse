"""Clean-room MIT-licensed PyTorch architectures for WaveParse v1.

Both models use standard torchvision ResNet-34 encoders (BSD-3-Clause ImageNet-1K
initialization) with lightweight multi-scale FPN/UNet decoders, requiring zero
third-party model repositories (no timm, no smp, no Kaggle checkpoints).

  1. WaveParseGridRectifierNet (~24M params, ~48 MB fp16):
     Layout-agnostic 2D ECG paper-grid & orientation network.
     Predicts 4-way orientation (0/90/180/270 deg), foreground paper mask,
     5-mm horizontal/vertical gridlines, 5-mm grid intersections, and local
     5-mm paper axis step vectors (dx, dy) for Thin-Plate Spline unwarping
     of curved/wrinkled phone photos in ANY layout (3x4, 6x2, 12x1, Cabrera).

  2. WaveParseLeadDecoderNet (~24M params, ~48 MB fp16):
     Universal single-lead column-heatmap & visibility decoder.
     Input: 4-channel (RGB + baseline Gaussian prior) 512x1024 crop covering
            ±4.0 mV (80 mm) vertical x 2.5 s (62.5 mm + 1.5 mm pad) horizontal.
     Output: column-wise vertical heatmap logits (B, 1, 512, 1024) +
             per-column trace visibility logits (B, 1, 1024) for honest gaps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _ConvBnSiLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = _ConvBnSiLU(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvBnSiLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(x))


def _make_resnet34_encoder(in_chans: int = 3, pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet34_Weights.DEFAULT if pretrained else None
    net = tvm.resnet34(weights=weights)
    if in_chans != 3:
        old = net.conv1
        new_conv = nn.Conv2d(
            in_chans,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3].copy_(old.weight)
            if in_chans > 3:
                mean_w = old.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:].copy_(mean_w.repeat(1, in_chans - 3, 1, 1) * 0.25)
        net.conv1 = new_conv
    return net


class WaveParseGridRectifierNet(nn.Module):
    """Layout-agnostic 2D paper-grid & orientation rectifier network.

    Input:  (B, 3, H, W) normalized RGB (default H=W=768)
    Output dict:
      - orientation_logits: (B, 4) -> [0, 90_cw, 180, 270_cw]
      - maps: (B, 8, H//2, W//2):
          0: paper_mask logit
          1: grid_h (5 mm horizontal lines) logit
          2: grid_v (5 mm vertical lines) logit
          3: grid_intersections (5 mm x 5 mm crossings) logit
          4: step_x_dx (local +x 5mm vector dx / W_map, tanh-free linear)
          5: step_x_dy (local +x 5mm vector dy / H_map)
          6: step_y_dx (local +y 5mm vector dx / W_map)
          7: step_y_dy (local +y 5mm vector dy / H_map)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        enc = _make_resnet34_encoder(in_chans=3, pretrained=pretrained)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # stride 2, 64 ch
        self.maxpool = enc.maxpool
        self.layer1 = enc.layer1  # stride 4, 64 ch
        self.layer2 = enc.layer2  # stride 8, 128 ch
        self.layer3 = enc.layer3  # stride 16, 256 ch
        self.layer4 = enc.layer4  # stride 32, 512 ch

        self.orient_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, 4),
        )

        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 64, 48)  # stride 2 (384x384 for 768 input)
        self.head = nn.Sequential(
            _ConvBnSiLU(48, 48),
            nn.Conv2d(48, 8, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s2 = self.stem(x)
        s4 = self.layer1(self.maxpool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        orient = self.orient_head(s32)
        d = self.up4(s32, s16)
        d = self.up3(d, s8)
        d = self.up2(d, s4)
        d = self.up1(d, s2)
        maps = self.head(d)
        return {"orientation_logits": orient, "maps": maps}


class WaveParseLeadDecoderNet(nn.Module):
    """Universal single-lead column-heatmap & visibility decoder.

    Input:  (B, 4, H, W) where ch 0..2 = RGB [0,1] and ch 3 = baseline prior [0,1]
            Default H=512 (±4.0 mV at 6.4 px/mm), W=1024 (2.5 s = 62.5 mm active + pad at 16 px/mm)
    Output dict:
      - heatmap_logits: (B, 1, H, W) — column-wise vertical trace probability logits
      - visibility_logits: (B, 1, W) — per-column trace visibility probability logits (0 = occluded/gap)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        enc = _make_resnet34_encoder(in_chans=4, pretrained=pretrained)
        # Keep a lightweight full-resolution stem branch so thin 1-px QRS spikes never blur out!
        self.fine_stem = nn.Sequential(
            _ConvBnSiLU(4, 32, k=3, p=1),
            _ConvBnSiLU(32, 32, k=3, p=1),
        )
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # stride 2, 64 ch
        self.maxpool = enc.maxpool
        self.layer1 = enc.layer1  # stride 4, 64 ch
        self.layer2 = enc.layer2  # stride 8, 128 ch
        self.layer3 = enc.layer3  # stride 16, 256 ch
        self.layer4 = enc.layer4  # stride 32, 512 ch

        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 64, 48)   # stride 2
        self.up0 = _UpBlock(48, 32, 32)   # stride 1 (full H x W!)

        self.heatmap_head = nn.Sequential(
            _ConvBnSiLU(32, 32, k=3, p=1),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # Column-wise visibility head pools vertically across decoder features
        self.vis_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.fine_stem(x)
        s2 = self.stem(x)
        s4 = self.layer1(self.maxpool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)

        d = self.up4(s32, s16)
        d = self.up3(d, s8)
        d = self.up2(d, s4)
        d = self.up1(d, s2)
        d0 = self.up0(d, s1)  # (B, 32, H, W)

        heatmap_logits = self.heatmap_head(d0)  # (B, 1, H, W)
        # Pool vertical mean + max per column for visibility prediction
        col_feat = torch.cat([d0.mean(dim=2), d0.amax(dim=2)], dim=1)  # (B, 64, W)
        vis_logits = self.vis_head(col_feat)  # (B, 1, W)
        return {"heatmap_logits": heatmap_logits, "visibility_logits": vis_logits}
