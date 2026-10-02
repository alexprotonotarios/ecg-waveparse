"""Release consumed CPU U-Net tensors while preserving upstream operations.

The forward operator sequence derives from Open-ECG-Digitizer's UNet (CC BY-SA 4.0).
Normalization continues to receive complete spatial tensors.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import Tensor, nn
from src.model.unet import UNet


def _release_consumed_tensors(self: UNet, x: Tensor) -> Tensor:
    if (
        x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 4
        or self.training or torch.is_grad_enabled() or torch.is_autocast_enabled("cpu")
    ):
        return UNet.forward(self, x)
    skips = []
    for encoder, skip, down in zip(self.encoders, self.encoder_skips, self.encoder_downscaling):
        x = encoder(x) + skip(x)
        skips.append(x)
        x = down(x)
    skips.reverse()
    x = skips.pop(0)
    for decoder, skip in zip(self.decoders, self.decoder_skips):
        target = skips.pop(0)
        x = self._upsample(x, target)
        combined = torch.cat([x, target], dim=1)
        del x, target
        x = decoder(combined)
        del combined
    assert not skips
    return self.final_conv(x)


def release_cpu_unet_tensors(model: nn.Module) -> bool:
    """Install once on the pinned upstream U-Net, preserving custom models."""
    if type(model) is not UNet or "forward" in model.__dict__:
        return False
    if not (
        len(model.encoders) == len(model.encoder_skips) == len(model.encoder_downscaling)
        and len(model.decoders) == len(model.decoder_skips) == len(model.encoders) - 1
    ):
        return False
    model.forward = MethodType(_release_consumed_tensors, model)
    return True
