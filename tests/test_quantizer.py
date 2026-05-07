"""Unit tests for VectorQuantizer and ResidualVectorQuantizer.

These verify the contract used by the rest of the pipeline (shapes,
gradient flow back to the encoder via straight-through, EMA codebook
updates only during training, commitment loss is non-zero) without
asserting any actual quantization quality — that requires real training.
"""

from __future__ import annotations

import torch

from src.models.quantizer import ResidualVectorQuantizer, VectorQuantizer


def test_vq_output_shape() -> None:
    vq = VectorQuantizer(codebook_size=64, dim=16)
    x = torch.randn(2, 16, 10)
    q, idx, loss = vq(x)
    assert q.shape == x.shape
    assert idx.shape == (2, 10)
    assert idx.dtype == torch.int64
    assert loss.dim() == 0  # scalar


def test_rvq_output_shape() -> None:
    rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=64, dim=16)
    x = torch.randn(2, 16, 10)
    q, idx, loss = rvq(x)
    assert q.shape == x.shape
    assert idx.shape == (2, 4, 10)
    assert loss.dim() == 0


def test_indices_in_codebook_range() -> None:
    rvq = ResidualVectorQuantizer(num_quantizers=3, codebook_size=32, dim=8)
    x = torch.randn(2, 8, 5)
    _, idx, _ = rvq(x)
    assert idx.min() >= 0
    assert idx.max() < 32


def test_straight_through_gradient() -> None:
    """Gradient on the quantized output must reach the encoder input as identity."""
    vq = VectorQuantizer(codebook_size=16, dim=8)
    x = torch.randn(1, 8, 4, requires_grad=True)
    q, _, _ = vq(x)
    # If we sum q and backprop, x.grad should equal d(sum)/dx = ones-like.
    q.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_commitment_loss_is_positive() -> None:
    # Disable k-means init: with it on, the codebook is seeded directly
    # from x and the commitment loss is identically zero on the first
    # forward pass. We're testing the loss formula itself here, not the
    # init path.
    vq = VectorQuantizer(
        codebook_size=64, dim=16, commitment_weight=1.0, kmeans_init=False
    )
    x = torch.randn(2, 16, 10)
    _, _, loss = vq(x)
    # Random encoder output vs near-zero-init codebook will have a clearly
    # positive distance.
    assert loss.item() > 0


def test_codebook_updates_only_in_training_mode() -> None:
    """Eval-mode forward must leave the codebook untouched."""
    vq = VectorQuantizer(codebook_size=16, dim=4)
    vq.eval()
    book_before = vq.codebook.clone()
    x = torch.randn(4, 4, 8)
    for _ in range(5):
        vq(x)
    assert torch.equal(vq.codebook, book_before)


def test_codebook_updates_in_training_mode() -> None:
    """Training-mode forward must move the codebook (EMA update)."""
    vq = VectorQuantizer(codebook_size=16, dim=4)
    vq.train()
    book_before = vq.codebook.clone()
    x = torch.randn(4, 4, 8)
    for _ in range(5):
        vq(x)
    # After 5 EMA steps the codebook should have moved meaningfully.
    assert not torch.equal(vq.codebook, book_before)


def test_rvq_commitment_loss_aggregates_layers() -> None:
    """The RVQ commitment loss should be the mean across layers, so adding
    more layers (which all see non-trivial residuals) shouldn't blow up the
    scale of the loss."""
    rvq2 = ResidualVectorQuantizer(num_quantizers=2, codebook_size=32, dim=8)
    rvq8 = ResidualVectorQuantizer(num_quantizers=8, codebook_size=32, dim=8)
    x = torch.randn(2, 8, 16)
    _, _, l2 = rvq2(x)
    _, _, l8 = rvq8(x)
    # Same order of magnitude (within 5x); not a tight bound, just a sanity check.
    assert 0.2 < l2.item() / l8.item() < 5.0


def test_rvq_residual_decreases() -> None:
    """Each successive RVQ layer should be quantizing a smaller residual."""
    rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=128, dim=16)
    rvq.eval()  # don't drift the codebook during this measurement
    x = torch.randn(1, 16, 32)
    residual = x
    norms = []
    for layer in rvq.layers:
        q, _, _ = layer(residual)
        residual = residual - q
        norms.append(residual.norm().item())
    # With a randomly initialised codebook the residual norm trajectory is
    # noisy but should not strictly increase across all four layers.
    assert norms[-1] < norms[0]
