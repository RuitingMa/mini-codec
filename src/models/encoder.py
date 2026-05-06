"""Convolutional encoder for the mini-codec.

Takes a raw mono waveform [B, 1, T] and produces a latent sequence
[B, latent_dim, T // prod(strides)]. The default config — strides (2, 4, 5, 5),
base channels 32, latent dim 128 — gives a 200x temporal compression, so
1 second of 16 kHz audio (16 000 samples) becomes 80 latent frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, asymmetric_pad1d, wn


class EncoderBlock(nn.Module):
    """One encoder stage: residual unit, then strided downsample that doubles channels.

    The downsampling conv uses kernel = 2 * stride (matching Encodec) with
    asymmetric pre-padding so the output length is exactly
    ``input_length // stride`` for inputs that divide evenly by ``stride``.
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.kernel = 2 * stride
        self.residual = ResidualBlock(channels)
        self.downsample = wn(nn.Conv1d(
            channels, 2 * channels, kernel_size=self.kernel, stride=stride, padding=0
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual(x)
        x = F.elu(x)
        x = asymmetric_pad1d(x, self.kernel, self.stride)
        return self.downsample(x)


class Encoder(nn.Module):
    """1D convolutional encoder.

    Layout: ``stem -> N x EncoderBlock -> ELU -> head``. Each block doubles the
    channel count, so after ``len(strides)`` stages the channel dim is
    ``base_channels * 2 ** len(strides)`` before being projected to
    ``latent_dim`` by the final 1x1-style head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        strides: tuple[int, ...] = (2, 4, 5, 5),
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.strides = tuple(strides)
        self.total_downsample = 1
        for s in self.strides:
            self.total_downsample *= s

        self.stem = wn(nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3))

        c = base_channels
        blocks: list[nn.Module] = []
        for s in self.strides:
            blocks.append(EncoderBlock(c, s))
            c *= 2
        self.blocks = nn.Sequential(*blocks)
        self.head = wn(nn.Conv1d(c, latent_dim, kernel_size=7, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = F.elu(x)
        return self.head(x)
