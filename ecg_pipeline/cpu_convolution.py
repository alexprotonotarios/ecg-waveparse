"""Limit temporary convolution matrices for the pinned macOS CPU U-Nets.

Only individual convolutions are evaluated in row stripes. Every normalization
layer still sees the complete spatial tensor. The output and model weights keep
their original resolution and precision; the workspace target is not a cap on
the whole model's live tensors or RSS.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


CONVOLUTION_WORKSPACE_BYTES = 16 * 1024**2


def _geometry(convolution: nn.Conv2d) -> tuple[int, int, int] | None:
    signature = (
        convolution.kernel_size, convolution.stride, convolution.padding,
        convolution.dilation, convolution.groups, convolution.padding_mode,
    )
    if signature == ((3, 3), (1, 1), (1, 1), (1, 1), 1, "replicate"):
        return 1, 1, 3
    if signature == ((2, 2), (2, 2), (0, 0), (1, 1), 1, "zeros"):
        return 2, 0, 2
    return None


class _RowBoundedConv2d(nn.Module):
    def __init__(self, convolution: nn.Conv2d, workspace_bytes: int) -> None:
        super().__init__()
        if type(workspace_bytes) is not int or workspace_bytes < 1:
            raise ValueError("Convolution workspace must be a positive integer.")
        self.convolution = convolution
        self.workspace_bytes = workspace_bytes
        self.train(convolution.training)

    def forward(self, image: Tensor) -> Tensor:
        convolution = self.convolution
        geometry = _geometry(convolution)
        if (
            image.device.type != "cpu" or image.ndim != 4
            or image.dtype != torch.float32 or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cpu") or geometry is None
            or image.shape[0] == 0
        ):
            return convolution(image)
        stride, padding, kernel = geometry
        height, width = image.shape[-2:]
        output_height, output_width = height // stride, width // stride
        if output_height == 0 or output_width == 0:
            return convolution(image)
        # The CPU im2col matrix contains N * Cin * K^2 values per output pixel.
        row_bytes = image.shape[0] * image.shape[1] * kernel**2 * output_width * image.element_size()
        rows = max(1, self.workspace_bytes // row_bytes)
        if rows >= output_height:
            return convolution(image)
        output = image.new_empty((image.shape[0], convolution.out_channels, output_height, output_width))
        for start in range(0, output_height, rows):
            end = min(start + rows, output_height)
            stripe = image[:, :, max(0, start * stride - padding):min(height, end * stride + padding), :]
            if padding:
                # Interior rows use their real neighbours. Replicate only at
                # the original image edges, never at a stripe boundary.
                stripe = functional.pad(
                    stripe, (1, 1, int(start == 0), int(end == height)), mode="replicate",
                )
            value = functional.conv2d(stripe, convolution.weight, convolution.bias, stride=stride)
            output[:, :, start:end, :] = value
            del stripe, value
        return output


def bound_cpu_convolutions(
    module: nn.Module, *, workspace_bytes: int = CONVOLUTION_WORKSPACE_BYTES,
) -> int:
    """Wrap the pinned convolution geometries after loading the original weights.

    The caller enables this only for macOS CPU inference. Unsupported operators,
    autograd, autocast and other devices retain their original execution path.
    Repeated installation leaves existing wrappers unchanged.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, _RowBoundedConv2d):
            continue
        if isinstance(child, nn.Conv2d) and _geometry(child) is not None:
            setattr(module, name, _RowBoundedConv2d(child, workspace_bytes))
            replaced += 1
        else:
            replaced += bound_cpu_convolutions(child, workspace_bytes=workspace_bytes)
    return replaced
