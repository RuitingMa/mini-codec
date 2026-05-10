# mini-codec

A from-scratch PyTorch implementation of a neural audio codec
(Encodec / SoundStream-style), trained on LibriSpeech and evaluated
under deterministic test-clean splits.

> **Status**: baseline pipeline trained and evaluated; a perceptual-loss
> ablation against the baseline is being analysed.

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

Baseline on `train-clean-100`, evaluated on `test-clean` (256 utterances,
deterministic crops, seed 0):

| metric | value |
|--------|-------|
| SI-SDR | **-15.70 ± 10.37 dB** |
| multi-scale Mel L1 | 0.40 ± 0.22 |
| bitrate | 3.2 kbps |

These numbers are in the expected range for a GAN-less codec at this
bitrate. Encodec's own ablation (§3.5 of the paper) reports roughly a
6 dB SI-SDR loss when adversarial training is removed; this baseline
is intentionally GAN-free, so the gap to published Encodec numbers
absorbs that ~6 dB and an additional bitrate gap (Encodec's lowest
published rate is 1.5–6 kbps with mixed train data).

A diagnostic note from the baseline analysis: per-sample multi-scale
Mel L1 is tightly clustered (std = 0.009 across the dumped subset)
while per-sample SI-SDR varies over a ~27 dB range. The spectral
envelope is being reproduced consistently across utterances; phase
and fine-time-structure are not. That gap is the textbook failure
mode of GAN-less audio codecs at low bitrate, and it motivates the
ongoing perceptual-loss ablation.

## Limitations and scope

- **No adversarial discriminator.** Adding a multi-STFT discriminator
  (Encodec §3.3) is the standard way to lift phase quality, but
  including it would broaden scope past the encoder / quantizer /
  loss design questions this project is built around.
- **Strict split discipline.** `train-clean-100` is the only training
  split; `dev-clean` is for monitoring during training; `test-clean`
  is reserved for the final reported numbers and was scored once.
- **Single-condition baseline.** All numbers above come from one seed
  and one bitrate. A bitrate sweep is on the followup list.

## References

- Défossez et al., *High Fidelity Neural Audio Compression* (Encodec),
  2022. [arXiv:2210.13438](https://arxiv.org/abs/2210.13438).
- Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec*,
  2021. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312).

## License

MIT — see [LICENSE](LICENSE).
