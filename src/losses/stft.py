"""Multi-scale mel-spectrogram reconstruction loss.

This is the spectral term Encodec calls ``L_f`` in §3.4 of the paper. For
each STFT window size ``w`` in ``window_sizes``, we compute the log-mel
spectrogram of both ``x`` and ``x_hat`` and accumulate their L1 + L2
distances; the final loss averages those distances over scales.

Why this matters for the mini-codec specifically: a time-domain L1 loss on
its own gives a constant-magnitude gradient (±1) that is too weak to lift
the encoder out of the "predict zero" basin when the target signal has low
amplitude (see ``project_overfit_plateau`` in the project memory). A
spectral loss, evaluated in log-magnitude, has gradient that scales with
the *ratio* of predicted to true amplitude — that is exactly the signal
the encoder needs to learn to produce non-trivial output.

Reference: Défossez et al., *High Fidelity Neural Audio Compression*, §3.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class MultiScaleMelLoss(nn.Module):
    """L1 + L2 distance between log-mel spectrograms of ``x`` and ``x_hat``,
    averaged over multiple STFT window sizes.

    Args:
        sample_rate: target sample rate of the input waveforms (used to
            build the mel filterbanks).
        window_sizes: one ``n_fft`` per scale. Default is powers of two from
            64 to 2048 — at 16 kHz this spans ~4 ms (capturing fine timbre
            and onsets) to ~128 ms (capturing prosody) windows.
        n_mels: requested number of mel bins per scale. Capped per-scale so
            the filterbank stays feasible for short windows.
        eps: epsilon added before ``log`` to keep silent frames finite.
        l2_weight: weight on the L2 distance relative to L1. Setting this
            to zero recovers pure-L1 spectral loss; Encodec uses 1.0.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_sizes: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
        n_mels: int = 64,
        eps: float = 1e-5,
        l2_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.l2_weight = l2_weight
        self.window_sizes = tuple(window_sizes)
        # n_fft // 2 + 1 frequency bins are available per scale; capping
        # n_mels at n_fft // 8 keeps roughly a 4:1 ratio of freq bins to
        # mel bins, which empirically avoids empty mel filters at the
        # short-window end of the scale list.
        self.mels = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=w,
                    hop_length=w // 4,
                    n_mels=min(n_mels, max(8, w // 8)),
                    power=1.0,  # magnitude, not power
                    center=True,
                )
                for w in self.window_sizes
            ]
        )

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, x_hat: ``[B, 1, T]`` mono waveforms; must match in shape.

        Returns:
            Scalar loss = mean over scales of ``L1 + l2_weight * L2`` on
            log-mel spectrograms.
        """
        if x.shape != x_hat.shape:
            raise ValueError(
                f"x {tuple(x.shape)} and x_hat {tuple(x_hat.shape)} must match"
            )
        # MelSpectrogram expects [..., T]; strip the channel dim.
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)

        loss = x.new_zeros(())
        for mel in self.mels:
            log_x = mel(x).clamp_min(self.eps).log()
            log_h = mel(x_hat).clamp_min(self.eps).log()
            diff = log_x - log_h
            l1 = diff.abs().mean()
            l2 = diff.pow(2).mean()
            loss = loss + l1 + self.l2_weight * l2
        return loss / len(self.mels)
