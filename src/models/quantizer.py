"""Residual Vector Quantizer for the mini-codec.

A stack of ``num_quantizers`` independent vector quantizers, each one
operating on the residual left by all previous layers. Codebooks are not
trained by gradient descent — they are updated by an exponential moving
average over the encoder outputs that get assigned to each code, which is
the standard recipe in SoundStream / Encodec and avoids most of the
codebook-collapse pathologies of pure-gradient VQ-VAE.

The encoder receives gradient via the straight-through estimator:
``q = x + (q_nearest - x).detach()``, so the encoder learns to produce
latents that the codebook can quantize cleanly. A small commitment loss
(MSE between encoder output and its nearest code, codebook side
detached) is returned so the trainer can fold it into the total loss.

Reference: SoundStream §3.2, Encodec §3.2 / appendix A. K-means-style
init (sampling K vectors from the first training batch) and dead-code
restart (re-seeding codes with near-zero EMA usage) are included; both
are needed in practice — the first integration test of this module on
LibriSpeech without them showed all four RVQ layers collapsing to a
single active code within ~30 steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Single vector quantizer with EMA codebook updates.

    Args:
        codebook_size: number of code vectors (K).
        dim: code dimensionality (D); must equal the per-frame latent dim.
        decay: EMA decay applied to codebook statistics every step.
        eps: Laplace-smoothing epsilon for cluster sizes (avoids div-by-zero
            on codes that received no assignments in a given batch).
        commitment_weight: scalar multiplier on the commitment loss before
            it's returned. Folded here so the caller can sum quantizer
            losses without having to remember the weight.
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_weight: float = 1.0,
        kmeans_init: bool = True,
        dead_code_threshold: float = 0.01,
        dead_code_steps: int = 20,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.commitment_weight = commitment_weight
        self.kmeans_init = kmeans_init
        # A code is restarted if either of these triggers fire:
        #   - it has gone `dead_code_steps` consecutive steps without being
        #     assigned a single frame (fast trigger; catches collapse early);
        #   - its EMA cluster size has decayed below `dead_code_threshold`
        #     (slow trigger; catches codes that flicker and then fade).
        # Both are needed at our scale: with B*T ≈ 320 frames per batch and
        # K = 1024 codes, the EMA-only trigger only fires after several
        # hundred steps under the default 0.99 decay, by which time layer-1
        # collapse has long since hardened.
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_steps = dead_code_steps

        # The codebook is a buffer, not a parameter — its updates come from
        # the EMA path below, not from autograd. Initial values are tiny
        # noise; if `kmeans_init` is on they will be overwritten on the
        # first training-mode forward call from the encoder output itself.
        codebook = torch.randn(codebook_size, dim) * 0.01
        self.register_buffer("codebook", codebook)
        # EMA accumulators.
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", codebook.clone())
        # Per-code counter of consecutive steps with zero assignments —
        # reset to 0 the moment a code is picked again.
        self.register_buffer("zero_steps", torch.zeros(codebook_size))
        # Whether the codebook has been seeded from data yet.
        self.register_buffer(
            "_initialized", torch.tensor(not kmeans_init, dtype=torch.bool)
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, D, T]`` continuous latent.

        Returns:
            quantized: ``[B, D, T]``. Identical shape to ``x``; gradient flows
                back to ``x`` as identity (straight-through estimator).
            indices: ``[B, T]`` int64 codebook indices (one per frame).
            commitment_loss: scalar tensor, already weighted.
        """
        B, D, T = x.shape
        if D != self.dim:
            raise ValueError(f"VectorQuantizer expected dim={self.dim}, got {D}")

        # [B*T, D] for nearest-neighbour lookup.
        x_flat = x.transpose(1, 2).reshape(-1, D)

        # First-step data-driven init: replace the random codebook with K
        # samples from the current batch (with replacement if the batch is
        # smaller than K). One pass of "k-means" with fully random reseat.
        if self.training and not bool(self._initialized.item()):
            with torch.no_grad():
                n = x_flat.shape[0]
                idx = torch.randint(0, n, (self.codebook_size,), device=x_flat.device)
                seed = x_flat[idx].detach()
                self.codebook.copy_(seed)
                self.ema_embed_sum.copy_(seed)
                # Give every code one "expected assignment" worth of mass so
                # the dead-code restart below doesn't immediately wipe them.
                self.ema_cluster_size.fill_(1.0)
                self.zero_steps.zero_()
                self._initialized.fill_(True)

        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x . c.
        dist = (
            x_flat.pow(2).sum(-1, keepdim=True)
            - 2.0 * x_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(-1)
        )  # [B*T, K]
        indices_flat = dist.argmin(dim=-1)  # [B*T]
        quantized_flat = self.codebook[indices_flat]  # [B*T, D]

        # EMA codebook update — only when training.
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices_flat, self.codebook_size).type_as(x_flat)
                cluster_size_step = onehot.sum(0)  # [K]
                embed_sum_step = onehot.t() @ x_flat  # [K, D]

                self.ema_cluster_size.mul_(self.decay).add_(
                    cluster_size_step, alpha=1.0 - self.decay
                )
                self.ema_embed_sum.mul_(self.decay).add_(
                    embed_sum_step, alpha=1.0 - self.decay
                )

                # Update the zero-step streak: increment for codes that got
                # no assignment this step, reset for codes that did.
                used_this_step = (cluster_size_step > 0).float()
                self.zero_steps.add_(1.0).mul_(1.0 - used_this_step)

                # Laplace smoothing so unused codes don't divide by zero.
                n = self.ema_cluster_size.sum()
                cluster_size_smoothed = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                self.codebook.copy_(
                    self.ema_embed_sum / cluster_size_smoothed.unsqueeze(-1)
                )

                # Dead-code restart: a code is dead if it has gone too many
                # consecutive steps without a single assignment OR its EMA
                # usage has decayed below the slow threshold.
                dead = (self.zero_steps >= self.dead_code_steps) | (
                    self.ema_cluster_size < self.dead_code_threshold
                )
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    rand_idx = torch.randint(
                        0, x_flat.shape[0], (n_dead,), device=x_flat.device
                    )
                    new_codes = x_flat[rand_idx].detach()
                    self.codebook[dead] = new_codes
                    self.ema_embed_sum[dead] = new_codes
                    # Bump cluster size and zero out the streak so we don't
                    # wipe these again next step.
                    self.ema_cluster_size[dead] = 1.0
                    self.zero_steps[dead] = 0.0

        # Reshape back to [B, D, T].
        quantized = quantized_flat.view(B, T, D).transpose(1, 2).contiguous()
        indices = indices_flat.view(B, T)

        # Commitment loss: pull encoder output toward its chosen code.
        # Codebook side is detached — its update is the EMA path above.
        commitment_loss = F.mse_loss(x, quantized.detach())

        # Straight-through estimator: forward sees quantized, backward sees x.
        quantized = x + (quantized - x).detach()

        return quantized, indices, self.commitment_weight * commitment_loss


class ResidualVectorQuantizer(nn.Module):
    """Stack of ``num_quantizers`` ``VectorQuantizer``s applied residually.

    Bitrate at inference (assuming all layers active) is
    ``num_quantizers * log2(codebook_size) * frame_rate`` bits per second.
    With the project default of 4 layers, 1024 codes per layer, and a
    80 Hz frame rate at 16 kHz, that's 3.2 kbps.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        dim: int = 128,
        decay: float = 0.99,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.dim = dim
        self.layers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=codebook_size,
                    dim=dim,
                    decay=decay,
                    commitment_weight=commitment_weight,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, D, T]`` continuous latent.

        Returns:
            quantized: ``[B, D, T]`` sum of all layer quantizations.
            indices: ``[B, num_quantizers, T]`` int64 code indices, one slice per layer.
            commitment_loss: scalar (mean over layers).
        """
        residual = x
        quantized = torch.zeros_like(x)
        all_indices: list[torch.Tensor] = []
        commitment_losses: list[torch.Tensor] = []

        for layer in self.layers:
            q_layer, idx_layer, c_loss = layer(residual)
            quantized = quantized + q_layer
            # The straight-through trick already detaches the residual update
            # from autograd, so subtracting q_layer here is gradient-safe.
            residual = residual - q_layer
            all_indices.append(idx_layer)
            commitment_losses.append(c_loss)

        indices = torch.stack(all_indices, dim=1)  # [B, N, T]
        commitment_loss = torch.stack(commitment_losses).mean()

        return quantized, indices, commitment_loss
