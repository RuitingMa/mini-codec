"""Shape and gradient-flow tests for the encoder/decoder pair.

These cover the contract that the encoder and decoder are exact inverses in
length: ``decoder(encoder(x)).shape == x.shape``. They do not test that the
network has converged or that audio is reconstructed faithfully — those
require training and live in the smoke-test scripts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.decoder import Decoder
from src.models.encoder import Encoder


def test_encoder_output_shape_default() -> None:
    enc = Encoder()  # strides (2, 4, 5, 5) -> 200x downsampling
    x = torch.randn(2, 1, 16000)
    z = enc(x)
    # 16000 / 200 = 80 latent frames; latent_dim = 128.
    assert z.shape == (2, 128, 80)


def test_decoder_output_shape_default() -> None:
    dec = Decoder()
    z = torch.randn(2, 128, 80)
    x = dec(z)
    assert x.shape == (2, 1, 16000)


def test_round_trip_shape_default() -> None:
    enc, dec = Encoder(), Decoder()
    x = torch.randn(3, 1, 16000)
    x_hat = dec(enc(x))
    assert x_hat.shape == x.shape


def test_round_trip_shape_alt_strides() -> None:
    """Same contract should hold for non-default stride configs."""
    strides = (2, 4, 5, 8)  # Encodec 24 kHz config (320x), still all-positive integers
    enc = Encoder(strides=strides)
    dec = Decoder(strides=strides)
    # Input length must be a multiple of prod(strides) = 320 for the contract.
    x = torch.randn(1, 1, 320 * 30)  # 30 latent frames
    x_hat = dec(enc(x))
    assert x_hat.shape == x.shape


def test_grad_flow() -> None:
    """One forward + backward must produce finite gradients for every parameter."""
    enc, dec = Encoder(), Decoder()
    x = torch.randn(1, 1, 800)  # 4 latent frames; small to keep the test fast on CPU
    x_hat = dec(enc(x))
    loss = F.l1_loss(x_hat, x)
    loss.backward()
    for name, p in list(enc.named_parameters()) + list(dec.named_parameters()):
        assert p.grad is not None, f"{name} received no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad contains non-finite values"


def test_param_count_in_range() -> None:
    """Sanity bound on the default model size — guards against accidental blow-up."""
    enc, dec = Encoder(), Decoder()
    n = sum(p.numel() for p in enc.parameters()) + sum(
        p.numel() for p in dec.parameters()
    )
    # Empirically ~5.4M for the default config; assert a generous upper bound.
    assert 1_000_000 < n < 20_000_000, f"unexpected total param count: {n:,}"
