"""Convolutional decoder for the mini-codec — mirror image of the encoder.

Takes a latent sequence [B, latent_dim, T'] and upsamples it back to a raw
mono waveform [B, 1, T' * prod(strides)]. With the default config this means
80 latent frames are reconstructed into 16 000 audio samples (1 s @ 16 kHz).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualBlock, wn


def _transpose_padding(stride: int) -> tuple[int, int]:
    """Return ``(padding, output_padding)`` such that
    ``ConvTranspose1d(kernel=2*stride, stride=stride, padding=p, output_padding=op)``
    produces output of length ``stride * input_length``. Even strides need no
    output padding; odd strides need ``output_padding=1`` to close a 1-sample gap.
    """
    if stride % 2 == 0:
        return stride // 2, 0
    return (stride + 1) // 2, 1


class DecoderBlock(nn.Module):
    """One decoder stage: upsampling transposed conv (channels halve) + residual unit."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        kernel = 2 * stride
        padding, output_padding = _transpose_padding(stride)
        self.upsample = wn(nn.ConvTranspose1d(
            channels,
            channels // 2,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ))
        self.residual = ResidualBlock(channels // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(x)
        x = self.upsample(x)
        return self.residual(x)


class Decoder(nn.Module):
    """1D transposed-convolutional decoder.

    Layout: ``head -> N x DecoderBlock (in reversed stride order) -> ELU -> tail``.
    Channels start at ``base_channels * 2 ** len(strides)`` and halve at each
    stage to land back at ``base_channels`` before the output projection.
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        strides: tuple[int, ...] = (2, 4, 5, 5),
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.strides = tuple(strides)

        c = base_channels * (2 ** len(self.strides))
        self.head = wn(nn.Conv1d(latent_dim, c, kernel_size=7, padding=3))

        blocks: list[nn.Module] = []
        for s in reversed(self.strides):
            blocks.append(DecoderBlock(c, s))
            c //= 2
        self.blocks = nn.Sequential(*blocks)

        self.tail = wn(nn.Conv1d(c, out_channels, kernel_size=7, padding=3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.head(z)
        x = self.blocks(x)
        x = F.elu(x)
        return self.tail(x)
