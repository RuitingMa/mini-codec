"""Tests for ``PerceptualLoss``.

Tests inject a small mock feature extractor so they don't have to download
the ~360 MB HuBERT weights. The mock matches the
``extract_features(x, num_layers=N) -> (list[Tensor], Optional[Tensor])``
contract used by ``torchaudio.models.HuBERTModel`` and friends.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.losses.perceptual import PerceptualLoss


class _MockFeatureModel(nn.Module):
    """Stand-in for HuBERT base: a tiny conv stack that returns intermediate
    "layer" features so we can test the loss wrapper without network IO."""

    def __init__(self, num_layers: int = 6, hidden: int = 16) -> None:
        super().__init__()
        self.num_layers = num_layers
        # First layer downsamples like HuBERT (its CNN feature extractor takes
        # ~16 kHz waveform to ~50 Hz frames); we just use stride 320 to mimic.
        self.stem = nn.Conv1d(1, hidden, kernel_size=10, stride=320, padding=5)
        self.layers = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel_size=3, padding=1) for _ in range(num_layers)]
        )

    def extract_features(
        self, waveforms: torch.Tensor, num_layers: int | None = None
    ) -> tuple[list[torch.Tensor], None]:
        # waveforms: [B, T]
        x = waveforms.unsqueeze(1)  # [B, 1, T]
        x = self.stem(x)
        n = num_layers if num_layers is not None else self.num_layers
        out = []
        for layer in self.layers[:n]:
            x = layer(x)
            # Return as [B, T', H] (HuBERT layout).
            out.append(x.transpose(1, 2))
        return out, None


def _make_loss(layer: int = 3) -> PerceptualLoss:
    return PerceptualLoss(layer=layer, feature_model=_MockFeatureModel(num_layers=6))


def test_loss_is_scalar() -> None:
    loss_fn = _make_loss()
    x = torch.randn(2, 1, 16000)
    x_hat = torch.randn(2, 1, 16000)
    out = loss_fn(x, x_hat)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_zero_loss_on_identical_inputs() -> None:
    loss_fn = _make_loss()
    x = torch.randn(2, 1, 16000)
    out = loss_fn(x, x)
    assert out.item() == 0.0


def test_positive_loss_on_different_inputs() -> None:
    torch.manual_seed(0)
    loss_fn = _make_loss()
    out = loss_fn(torch.randn(2, 1, 16000), torch.randn(2, 1, 16000))
    assert out.item() > 0


def test_gradient_flows_to_x_hat_but_not_feature_model() -> None:
    """The codec should receive gradient via the loss; HuBERT must not."""
    loss_fn = _make_loss()
    x = torch.randn(1, 1, 16000)
    x_hat = torch.randn(1, 1, 16000, requires_grad=True)
    loss = loss_fn(x, x_hat)
    loss.backward()
    # x_hat got gradient.
    assert x_hat.grad is not None
    assert torch.isfinite(x_hat.grad).all()
    assert x_hat.grad.abs().mean().item() > 0
    # Feature model parameters are frozen — no grads accumulated.
    for name, p in loss_fn.feature_model.named_parameters():
        assert p.requires_grad is False, f"{name} should be frozen"
        assert p.grad is None or p.grad.abs().sum().item() == 0.0


def test_feature_model_stays_in_eval_mode_when_loss_is_trained() -> None:
    """We don't want HuBERT's dropout/BN to come on when the codec calls .train()."""
    loss_fn = _make_loss()
    loss_fn.train()  # parent goes to train mode
    assert loss_fn.feature_model.training is False


def test_shape_mismatch_raises() -> None:
    loss_fn = _make_loss()
    with pytest.raises(ValueError):
        loss_fn(torch.randn(1, 1, 16000), torch.randn(1, 1, 8000))


def test_invalid_layer_raises() -> None:
    with pytest.raises(ValueError):
        PerceptualLoss(layer=0, feature_model=_MockFeatureModel())


def test_resample_path_for_non_16k_input() -> None:
    """If the codec runs at e.g. 24 kHz, the loss must resample to HuBERT's 16 kHz."""
    loss_fn = PerceptualLoss(
        layer=3, sample_rate=24000, feature_model=_MockFeatureModel(num_layers=6)
    )
    x = torch.randn(1, 1, 24000)
    x_hat = torch.randn(1, 1, 24000)
    out = loss_fn(x, x_hat)
    assert torch.isfinite(out)


def test_invalid_backend_raises() -> None:
    from src.losses.perceptual import _build_default_feature_model

    with pytest.raises(ValueError, match="unknown perceptual loss backend"):
        _build_default_feature_model(backend="bogus")


def test_huggingface_backend_helpful_error_when_transformers_missing() -> None:
    """If `transformers` isn't installed and the user picks backend='huggingface',
    we surface an actionable install hint rather than a bare ImportError."""
    try:
        import transformers  # noqa: F401

        pytest.skip("transformers IS installed; this test only validates the missing-deps path")
    except ImportError:
        pass
    from src.losses.perceptual import _HuggingFaceHuBERTAdapter

    with pytest.raises(ImportError, match="transformers"):
        _HuggingFaceHuBERTAdapter()
