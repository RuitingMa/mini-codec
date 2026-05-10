# mini-codec

A from-scratch PyTorch implementation of a neural audio codec
(Encodec / SoundStream-style), trained on LibriSpeech and evaluated
under deterministic test-clean splits.

> **Status**: baseline trained and evaluated; perceptual-loss ablation
> closed as a clean negative result; a four-point bitrate scan from
> 1.6 kbps to 12.8 kbps gives a clean rate-distortion curve. Report
> writeup is the remaining work.

## What's implemented

- **Encoder**: 1D conv stack with `weight_norm` on every convolution.
  Downsampling factors `(2, 4, 5, 5)` give a 200× temporal compression,
  i.e. 80 Hz frame rate at the project's 16 kHz native sample rate.
- **Quantizer**: 4-layer Residual Vector Quantization, 1024 codes per
  layer. Codebooks are updated by EMA (no autograd through them), seeded
  k-means style from the first training batch, and protected from
  collapse by a dual-trigger dead-code restart (consecutive-zero
  streak ≥ 20 *or* EMA cluster size < 0.01). All non-obvious at our
  scale — see `src/models/quantizer.py` for the long version.
- **Decoder**: mirror image of the encoder using `ConvTranspose1d`,
  with explicit handling of odd strides so the round-trip preserves
  length to the sample.
- **Losses**: time-domain L1 + multi-scale log-mel STFT (windows
  `[64, 128, 256, 512, 1024, 2048]`) + RVQ commitment. An optional
  HuBERT-base-layer-6 feature-matching perceptual loss is wired in
  for the comparison experiment.
- **Training**: `src/train.py` is yaml-driven; logger is one of
  `none / tensorboard / wandb`; periodic checkpoints; HuBERT can be
  loaded from torchaudio's bundle or HuggingFace (mirror-friendly).
- **Evaluation**: deterministic test-clean scoring with SI-SDR +
  multi-scale Mel L1, persisted as both `metrics.json` (aggregates)
  and `per_sample.csv` (every utterance), plus dumped input/recon
  wav pairs. Cross-experiment side-by-side via `scripts/compare_evals.py`.
- **Tests**: 40 unit tests covering shape contracts, gradient flow,
  EMA train/eval split, codebook restart, and the perceptual-loss
  feature-model interface.

Total trainable parameters: **5.4 M** (encoder + RVQ + decoder). Target
bitrate at the default config: **3.2 kbps** (4 layers × log₂(1024) bits ×
80 Hz frame rate).

## Quickstart

```bash
# Environment
conda create -n mini-codec python=3.11 -y
conda activate mini-codec
pip install -e ".[dev]"

# Tests run on CPU in a few seconds
pytest

# LibriSpeech splits
python scripts/download_librispeech.py --split dev-clean        # ~5h, monitor / sanity
python scripts/download_librispeech.py --split train-clean-100  # ~100h, for real training
python scripts/download_librispeech.py --split test-clean       # ~5h, held-out for final eval

# Train (needs a GPU; ~35 min on a single RTX 4090)
python -m src.train --config configs/baseline_train100.yaml --logger tensorboard

# Eval on test-clean
python scripts/eval.py --ckpt outputs/baseline_train100/ckpt_00050000.pt \
    --split test-clean --num-samples 256 --num-dump 32

# Cross-experiment compare (once multiple variants have been trained + evaluated)
python scripts/compare_evals.py \
    --eval-dirs outputs/baseline_train100/eval_ckpt_00050000 \
                outputs/exp_b_perceptual/eval_ckpt_00050000 \
    --names baseline +perceptual \
    --out outputs/compare
```

CPU is sufficient for everything except training on `train-clean-100`.
The default `pip install` pulls PyTorch CPU wheels; for GPU, install
the appropriate `torch` / `torchaudio` wheel separately (e.g. cu128 on
recent CUDA drivers) before — or with `--force-reinstall` after —
`pip install -e .`.

## Project layout

```
mini-codec/
├── configs/
│   ├── baseline.yaml              # dev-clean smoke / pipeline check
│   ├── baseline_train100.yaml     # production baseline
│   ├── exp_b_perceptual.yaml      # baseline + HuBERT perceptual loss (additive)
│   └── exp_b_swap_stft.yaml       # STFT replaced by perceptual (anti-gaming control)
├── src/
│   ├── data/librispeech.py
│   ├── models/{encoder,decoder,quantizer,blocks}.py
│   ├── losses/{stft,perceptual}.py
│   └── train.py
├── scripts/
│   ├── download_librispeech.py    # torchaudio standard layout
│   ├── parquet_to_librispeech.py  # HuggingFace mirror → standard layout (CN-friendly)
│   ├── smoke_overfit.py           # single-sample architecture sanity
│   ├── eval.py                    # SI-SDR + Mel L1 + per-sample CSV + wav dump
│   └── compare_evals.py           # cross-experiment side-by-side
└── tests/                         # 40 unit tests, pytest
```

## Current results

All numbers below come from the `test-clean` split (256 utterances,
deterministic crops, seed 0; each checkpoint trained for 50 000 steps
on `train-clean-100`).

### Rate-distortion sweep

| bitrate | num quantizers | SI-SDR median (dB) | IQR (dB) | Mel L1 median |
|---------|----------------|--------------------|----------|---------------|
| 1.6 kbps  | 2  | -20.03 | 12.5 | 0.351 |
| 3.2 kbps  | 4  | -14.10 | 12.1 | 0.309 |
| 6.4 kbps  | 8  |  -6.63 |  9.7 | 0.256 |
| 12.8 kbps | 16 |  -0.95 |  8.2 | 0.220 |

The curve is approximately linear in log-bitrate at roughly **6 dB of
SI-SDR per octave of bitrate**, and the inter-quartile range narrows
from 12.5 dB at the low-bitrate end to 8.2 dB at the high-bitrate end —
higher bitrates are not just more accurate but also more consistent
across utterances. Plot: [`scripts/plot_rd_curve.py`](scripts/plot_rd_curve.py)
output (`rd_curve_sdr.png`, `per_sample_delta.png`).

The gap to Encodec's published with-GAN numbers (Table 4 of the paper,
24 kHz mixed-data) narrows from ~21 dB at 1.6 kbps to ~13 dB at
12.8 kbps. That's consistent with a story where adversarial loss
contributes most where the information bottleneck is tightest — at low
bitrate the discriminator's perceptual prior fills in detail the
encoder physically cannot store; at high bitrate the encoder has
enough bits that GAN provides relatively less.

### Diagnostic finding from the baseline

At 3.2 kbps, the per-sample multi-scale Mel L1 is **tightly clustered
(std ≈ 0.009)** but per-sample SI-SDR ranges over **~30 dB** and is
heavily right-tailed by silence-dominated outliers. Spectrograms of
the input and reconstruction match in envelope structure but diverge
in fine-time-structure — the textbook signature of phase / fine-time
errors in a GAN-less codec.

### Perceptual-loss ablation (closed as informative negative)

To test whether features from a frozen self-supervised audio encoder
could substitute for adversarial training in supplying phase-aware
gradient, two variants of the 3.2 kbps baseline were trained: an
**additive** version (`L1 + STFT + commit + perceptual`) and a
**replacement** version (`L1 + commit + perceptual`, STFT off) using
HuBERT-base layer-6 feature L1.

- **Additive**: SI-SDR median -14.92 dB vs baseline -14.10 dB —
  statistically indistinguishable change; Mel L1 marginally improved
  (0.301 vs 0.309).
- **Replacement**: SI-SDR median -30.12 dB — model collapses without
  STFT, so the perceptual loss is independently too weak to support
  codec training.

Read together: HuBERT layer-6 features are **redundant with multi-scale
STFT in the spectral envelope dimension** (the small Mel L1 gain) and
do **not** add phase-aware gradient (no SI-SDR change). HuBERT-style
self-supervised audio encoders, trained on ASR-leaning prediction
objectives, encode envelope/phonetic content but not phase — so they
cannot replace adversarial training for this failure mode.

## Limitations and scope

- **No adversarial discriminator.** Adding a multi-STFT discriminator
  (Encodec §3.3) is the standard way to lift phase quality, but
  including it would broaden scope past the encoder / quantizer /
  loss design questions this project is built around.
- **Strict split discipline.** `train-clean-100` is the only training
  split; `dev-clean` is for monitoring during training; `test-clean`
  is reserved for the final reported numbers and was scored once.
- **Single seed.** Bitrate sweep covers four rates; each is one seed
  and one training run. Variance across seeds is unmeasured.
- **Speech only.** Training is on LibriSpeech (clean read speech);
  generalisation to music or noisy speech is not characterised here.

## References

- Défossez et al., *High Fidelity Neural Audio Compression* (Encodec),
  2022. [arXiv:2210.13438](https://arxiv.org/abs/2210.13438).
- Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec*,
  2021. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312).

## License

MIT — see [LICENSE](LICENSE).
