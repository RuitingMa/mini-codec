"""Perceptual feature-matching loss against a frozen pretrained audio encoder.

The loss is the L1 distance between intermediate features extracted from
``x`` and ``x_hat`` by a pretrained self-supervised audio encoder
(``torchaudio.pipelines.HUBERT_BASE`` by default; switchable to a
HuggingFace ``transformers`` HuBERT for environments where the
torchaudio weight URL is unreachable, e.g. AutoDL CN regions where the
Meta CDN is gated). The encoder is frozen on construction — gradients
flow through it back to ``x_hat`` so the codec can learn to satisfy
the perceptual constraint, but no gradient ever accumulates on the
encoder's own parameters.

Why this term exists for the mini-codec specifically: the W4 baseline (16 kHz
LibriSpeech, 3.2 kbps, no GAN) recovered the spectral *envelope* well — Mel
L1 std across 256 utterances was 0.009 — but produced large per-sample
SI-SDR variance (~10 dB) that traced cleanly to phase / fine-time-structure
errors. Penalising the L1 distance between HuBERT's mid-layer features
(typically layer 6 for phonetic / acoustic detail) gives the codec gradient
signal that depends on *both* spectral content and temporal alignment, which
should reduce phase-randomisation artefacts without resorting to adversarial
training.

Reference: Kumar et al., *High-Fidelity Audio Compression with Improved
RVQGAN* (DAC), 2023, §3.2 — uses an analogous discriminator-feature
matching term. Our version is purely on a *pretrained* encoder so we don't
have to train an adversary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class _HuggingFaceHuBERTAdapter(nn.Module):
    """Wraps ``transformers.HubertModel`` to expose torchaudio's
    ``extract_features(waveforms, num_layers=N) -> (list[Tensor], None)``
    contract, so ``PerceptualLoss`` can stay backend-agnostic.

    Used when the torchaudio bundle's Meta CDN URL isn't reachable (e.g.
    AutoDL CN regions); pair with ``HF_ENDPOINT=https://hf-mirror.com``
    for the actual weight download.
    """

    def __init__(self, repo: str = "facebook/hubert-base-ls960") -> None:
        super().__init__()
        try:
            from transformers import HubertModel
        except ImportError as e:
            raise ImportError(
                "backend='huggingface' requires the transformers package. "
                "Install with `pip install transformers` (or "
                "`pip install -e \".[hf]\"` from the repo root)."
            ) from e
        self.model = HubertModel.from_pretrained(repo)

    def extract_features(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        num_layers: int | None = None,
    ) -> tuple[list[torch.Tensor], None]:
        out = self.model(waveforms, output_hidden_states=True)
        # hidden_states: tuple of (n_transformer_layers + 1) tensors, where
        # index 0 is the post-CNN embedding fed into layer 1. We follow the
        # torchaudio convention: layer N output is index N (1-based).
        layers = list(out.hidden_states[1:])
        if num_layers is not None:
            layers = layers[:num_layers]
        return layers, None


def _build_default_feature_model(
    backend: str = "torchaudio",
    hf_repo: str = "facebook/hubert-base-ls960",
) -> nn.Module:
    """Construct a frozen HuBERT-base feature model from the chosen backend."""
    if backend == "torchaudio":
        return torchaudio.pipelines.HUBERT_BASE.get_model()
    if backend == "huggingface":
        return _HuggingFaceHuBERTAdapter(hf_repo)
    raise ValueError(
        f"unknown perceptual loss backend: {backend!r} "
        f"(expected 'torchaudio' or 'huggingface')"
    )


class PerceptualLoss(nn.Module):
    """L1 distance between hidden features of a frozen pretrained encoder.

    Args:
        layer: 1-based index of the transformer layer whose output is used
            as the perceptual feature. Range ``1..12`` for HuBERT base. ``6``
            is the default — empirically a phonetic/acoustic mid-layer.
        sample_rate: the target sample rate of the *input* tensors. If it
            differs from HuBERT's native 16 kHz, a resampler is applied
            inside the loss before the feature extraction.
        feature_model: optionally inject a custom feature model that
            implements ``extract_features(waveforms, num_layers=...)
            -> (list[Tensor], Optional[Tensor])``. Tests use a mock here so
            unit tests don't have to download the ~360 MB HuBERT weights.
        backend: ``"torchaudio"`` (default) loads HuBERT base via
            ``torchaudio.pipelines.HUBERT_BASE``, which downloads from
            Meta's CDN. ``"huggingface"`` loads the same architecture from
            ``transformers.HubertModel.from_pretrained(hf_repo)``, which
            respects ``HF_ENDPOINT`` for mirror downloads.
        hf_repo: HuggingFace model id used when ``backend='huggingface'``.
    """

    def __init__(
        self,
        layer: int = 6,
        sample_rate: int = 16000,
        feature_model: nn.Module | None = None,
        backend: str = "torchaudio",
        hf_repo: str = "facebook/hubert-base-ls960",
    ) -> None:
        super().__init__()
        if layer < 1:
            raise ValueError(f"layer must be >= 1, got {layer}")
        self.layer = layer
        self.sample_rate = sample_rate

        if sample_rate != 16000:
            # HuBERT base is trained at 16 kHz; resample inside the loss so
            # callers can use this term even when the codec runs at 24 kHz.
            self.resample: torchaudio.transforms.Resample | None = (
                torchaudio.transforms.Resample(sample_rate, 16000)
            )
        else:
            self.resample = None

        if feature_model is None:
            feature_model = _build_default_feature_model(backend, hf_repo)
        self.feature_model = feature_model

        # Freeze every parameter — we only need the forward pass.
        for p in self.feature_model.parameters():
            p.requires_grad_(False)
        self.feature_model.eval()

    def train(self, mode: bool = True) -> "PerceptualLoss":
        # Override: regardless of the parent's training mode, the feature
        # extractor stays in eval mode (no dropout, no BN updates).
        super().train(mode)
        self.feature_model.eval()
        return self

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, x_hat: ``[B, 1, T]`` mono waveforms; must match in shape.

        Returns:
            scalar L1 distance between the chosen layer's features.
        """
        if x.shape != x_hat.shape:
            raise ValueError(
                f"x {tuple(x.shape)} and x_hat {tuple(x_hat.shape)} must match"
            )
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        if self.resample is not None:
            x = self.resample(x)
            x_hat = self.resample(x_hat)

        # extract_features returns features for layers 1..num_layers; we
        # want the last one (= self.layer).
        feats_x, _ = self.feature_model.extract_features(
            x, num_layers=self.layer
        )
        feats_hat, _ = self.feature_model.extract_features(
            x_hat, num_layers=self.layer
        )
        return (feats_x[-1] - feats_hat[-1]).abs().mean()
