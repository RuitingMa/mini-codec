# mini-codec

A from-scratch implementation of a neural audio codec (Encodec / SoundStream-style),
trained on a small speech subset, with one focused improvement experiment on top of the baseline.

> **Status**: 🚧 in progress — week 1 of an 11-week project.

## What this is

A compact, readable PyTorch implementation of:

- **Encoder**: 1D convolutional stack that downsamples a raw waveform into a latent sequence.
- **Quantizer**: Residual Vector Quantization (RVQ) with multiple codebooks for tunable bitrate.
- **Decoder**: a (near-)mirror image of the encoder using transposed convolutions.
- **Training**: time-domain L1 + multi-scale STFT reconstruction losses + commitment loss.

After the baseline is working, one targeted research question is investigated end-to-end
(candidate directions are tracked in [docs/plan.md](docs/plan.md)).

## Why

This repo is a research / portfolio project, not a production codec.
Goals, in order:

1. Demonstrate that I can implement a non-trivial generative audio model from first principles.
2. Run a clean, well-controlled comparison experiment around one design choice.
3. Produce a short technical report and a listenable demo.

## Project layout

```
mini-codec/
├── configs/        # YAML experiment configs (kept out of code)
├── src/
│   ├── models/     # encoder, decoder, quantizer (one file each)
│   ├── losses/     # reconstruction + commitment losses
│   ├── data/       # dataset / dataloader utilities
│   └── train.py    # training entry point
├── scripts/        # thin wrappers that launch experiments
├── notebooks/      # exploratory analysis (not for shipping logic)
└── tests/          # unit tests for the core modules
```

## Quickstart (placeholder)

```bash
# install (uv-based, see pyproject.toml)
uv sync

# run unit tests
uv run pytest

# train baseline (placeholder — config will land in W2-W3)
uv run python -m src.train --config configs/baseline.yaml
```

## References

- Défossez et al., *High Fidelity Neural Audio Compression* (Encodec), 2022.
- Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec*, 2021.
