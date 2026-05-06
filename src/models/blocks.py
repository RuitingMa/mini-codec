"""Building blocks shared between the encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm as wn


def asymmetric_pad1d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Pad ``x`` so that ``Conv1d(kernel=k, stride=s, padding=0)`` produces an
    output of length ``ceil(T / s)``. Splits the required (k - s) padding
    asymmetrically so the formula works for any stride, including odd ones.
    """
    pad_total = kernel_size - stride
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(x, (pad_left, pad_right))


class ResidualBlock(nn.Module):
    """Two-conv residual unit with dilations (1, 3) and ELU activations.

    Receptive field grows from 3 (first conv) to 9 (combined), giving each
    encoder/decoder stage useful temporal context without inflating channel
    count or layer depth.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = wn(nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1))
        self.conv2 = wn(nn.Conv1d(channels, channels, kernel_size=3, padding=3, dilation=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.elu(x)
        y = self.conv1(y)
        y = F.elu(y)
        y = self.conv2(y)
        return x + y
