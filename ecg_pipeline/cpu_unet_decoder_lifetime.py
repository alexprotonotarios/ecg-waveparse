"""Release consumed Mac CPU encoder/decoder inputs without changing leaf operations.

The U-Net operator sequence derives from Open-ECG-Digitizer (CC BY-SA 4.0).
Normalization continues to receive complete spatial tensors. Bare Sequential
containers are traversed directly so their call frames do not retain a large
input after its consuming convolution has returned. Encoder input needed by
the actual skip operator remains live until that operator completes.
"""
from __future__ import annotations

from types import MethodType
import torch
from torch import Tensor, nn
from torch.nn.modules import module as module_hooks
from src.model.unet import UNet

_SEQUENTIAL_FORWARD = nn.Sequential.forward


def _decoder_leaves(container: nn.Sequential) -> list[nn.Module] | None:
    if (
        type(container) is not nn.Sequential or 'forward' in container.__dict__
        or getattr(container.forward, '__func__', None) is not _SEQUENTIAL_FORWARD
    ):
        return None
    if any(getattr(container, name) for name in (
        '_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks',
    )):
        return None
    leaves = []
    for child in container:
        if isinstance(child, nn.Sequential):
            nested = _decoder_leaves(child)
            if nested is None:
                return None
            leaves.extend(nested)
        else:
            leaves.append(child)
    return leaves


def _release_decoder_inputs(self: UNet, x: Tensor) -> Tensor:
    if (
        type(x) is not Tensor or x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 4
        or self.training or torch.is_grad_enabled() or torch.is_autocast_enabled('cpu')
        or any(getattr(module_hooks, name) for name in (
            '_global_forward_pre_hooks', '_global_forward_hooks',
            '_global_backward_pre_hooks', '_global_backward_hooks',
        ))
    ):
        return UNet.forward(self, x)
    encoders = [_decoder_leaves(encoder) for encoder in self.encoders]
    decoders = [_decoder_leaves(decoder) for decoder in self.decoders]
    if any(leaves is None for leaves in encoders + decoders):
        return UNet.forward(self, x)
    skips = []
    for leaves, skip, down in zip(encoders, self.encoder_skips, self.encoder_downscaling):
        encoded = x
        for layer in leaves:
            encoded = layer(encoded)
        # Preserve the exact original addition/skip order and original input.
        x = encoded + skip(x)
        del encoded
        skips.append(x)
        x = down(x)
    skips.reverse()
    x = skips.pop(0)
    for leaves in decoders:
        target = skips.pop(0)
        x = self._upsample(x, target)
        combined = torch.cat([x, target], dim=1)
        del x, target
        # No parent Sequential call retains the original concatenation. Each
        # actual leaf still uses Module.__call__, including its own hooks.
        for layer in leaves:
            combined = layer(combined)
        x = combined
        del combined
    assert not skips
    return self.final_conv(x)


def release_cpu_unet_decoder_inputs(model: nn.Module) -> bool:
    """Install once only on an unmodified pinned U-Net; keep custom forwards."""
    if type(model) is not UNet or 'forward' in model.__dict__:
        return False
    if not (
        len(model.encoders) == len(model.encoder_skips) == len(model.encoder_downscaling)
        and len(model.decoders) == len(model.decoder_skips) == len(model.encoders) - 1
    ):
        return False
    model.forward = MethodType(_release_decoder_inputs, model)
    return True
