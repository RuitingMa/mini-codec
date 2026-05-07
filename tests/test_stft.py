"""Tests for the multi-scale mel reconstruction loss."""

from __future__ import annotations

import pytest
import torch

from src.losses.stft import MultiScaleMelLoss


def test_loss_is_scalar() -> None:
    loss_fn = MultiScaleMelLoss(sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    x_hat = torch.randn(2, 1, 16000)
    out = loss_fn(x, x_hat)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_zero_loss_on_identical_inputs() -> None:
    loss_fn = MultiScaleMelLoss(sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    out = loss_fn(x, x)
    # log(S) - log(S) = 0 elementwise, so the loss must be exactly 0.
    assert out.item() == 0.0


def test_positive_loss_on_different_inputs() -> None:
    torch.manual_seed(0)
    loss_fn = MultiScaleMelLoss(sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    x_hat = torch.randn(2, 1, 16000)
    out = loss_fn(x, x_hat)
    assert out.item() > 0


def test_gradient_flows_to_x_hat() -> None:
    loss_fn = MultiScaleMelLoss(sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    x_hat = torch.randn(2, 1, 16000, requires_grad=True)
    loss = loss_fn(x, x_hat)
    loss.backward()
    assert x_hat.grad is not None
    assert torch.isfinite(x_hat.grad).all()
    # Some gradient must be non-trivially non-zero.
    assert x_hat.grad.abs().mean().item() > 0


def test_shape_mismatch_raises() -> None:
    loss_fn = MultiScaleMelLoss(sample_rate=16000)
    x = torch.randn(2, 1, 16000)
    x_hat = torch.randn(2, 1, 8000)
    with pytest.raises(ValueError):
        loss_fn(x, x_hat)


def test_l1_only_is_smaller_than_l1_plus_l2() -> None:
    """Setting l2_weight=0 must give a strictly smaller loss on the same input."""
    torch.manual_seed(0)
    full = MultiScaleMelLoss(sample_rate=16000, l2_weight=1.0)
    l1 = MultiScaleMelLoss(sample_rate=16000, l2_weight=0.0)
    x = torch.randn(1, 1, 16000)
    x_hat = torch.randn(1, 1, 16000)
    assert l1(x, x_hat).item() < full(x, x_hat).item()


def test_smallest_window_is_feasible() -> None:
    """The 64-sample window is the trickiest scale (only ~33 freq bins) —
    make sure we configure mel bins so it doesn't blow up at construction."""
    loss_fn = MultiScaleMelLoss(sample_rate=16000, window_sizes=(64,))
    x = torch.randn(1, 1, 16000)
    x_hat = torch.randn(1, 1, 16000)
    out = loss_fn(x, x_hat)
    assert torch.isfinite(out)
    assert out.item() > 0


def test_loss_decreases_with_less_noise() -> None:
    """Adding less noise to x must give a strictly smaller loss than adding
    a lot of noise. Pure additive-noise perturbations are the cleanest way
    to express monotonic 'closer in time domain' for log-mel."""
    torch.manual_seed(0)
    loss_fn = MultiScaleMelLoss(sample_rate=16000)
    x = torch.randn(1, 1, 16000)
    noise = torch.randn(1, 1, 16000)
    near = x + 0.05 * noise
    far = x + 1.0 * noise
    assert loss_fn(x, near).item() < loss_fn(x, far).item()
