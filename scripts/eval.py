"""Offline evaluation of a mini-codec checkpoint.

Loads a checkpoint, runs the encoder + RVQ + decoder on a fixed-seed
subset of a LibriSpeech split, reports objective reconstruction metrics
and dumps a handful of input/recon wav pairs for ear-checking.

Metrics:
    SI-SDR  scale-invariant signal-to-distortion ratio (dB; higher is better)
    Mel L1  multi-scale log-mel L1 distance (lower is better)

Run:
    python scripts/eval.py --ckpt outputs/baseline/ckpt_<step>.pt \\
        --root datasets --split dev-clean --num-samples 32 --num-dump 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Make `src.*` importable when this script is run as `python scripts/foo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.data.librispeech import LibriSpeechSegments  # noqa: E402
from src.losses.stft import MultiScaleMelLoss  # noqa: E402
from src.models.decoder import Decoder  # noqa: E402
from src.models.encoder import Encoder  # noqa: E402
from src.models.quantizer import ResidualVectorQuantizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="Path to a checkpoint .pt.")
    p.add_argument("--root", type=Path, default=Path("datasets"))
    p.add_argument("--split", type=str, default="dev-clean")
    p.add_argument("--num-samples", type=int, default=32,
                   help="How many utterances to score.")
    p.add_argument("--num-dump", type=int, default=8,
                   help="How many input/recon wav pairs to write to disk.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir; defaults to <ckpt parent>/eval_<ckpt stem>/.")
    return p.parse_args()


def si_sdr_db(s: torch.Tensor, s_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB. Accepts ``[B, 1, T]`` or ``[B, T]`` and
    returns ``[B]``.

    Defined as ``10 log10 (||s_target||^2 / ||e_noise||^2)`` where
    ``s_target`` is the orthogonal projection of ``s_hat`` onto ``s`` and
    ``e_noise = s_hat - s_target``. Insensitive to global scaling of either
    signal, which makes it the standard objective metric for speech
    reconstruction.
    """
    if s.dim() == 3:
        s = s.squeeze(1)
        s_hat = s_hat.squeeze(1)
    s = s - s.mean(dim=-1, keepdim=True)
    s_hat = s_hat - s_hat.mean(dim=-1, keepdim=True)
    dot = (s_hat * s).sum(dim=-1, keepdim=True)
    s_norm_sq = (s * s).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / s_norm_sq) * s
    e_noise = s_hat - s_target
    target_p = s_target.pow(2).sum(dim=-1)
    noise_p = e_noise.pow(2).sum(dim=-1)
    return 10 * torch.log10((target_p + eps) / (noise_p + eps))


def build_models_from_config(cfg: dict) -> tuple[Encoder, ResidualVectorQuantizer, Decoder]:
    m = cfg["model"]
    encoder = Encoder(
        base_channels=m["encoder"]["base_channels"],
        strides=tuple(m["encoder"]["strides"]),
        latent_dim=m["encoder"]["latent_dim"],
    )
    decoder = Decoder(
        base_channels=m["decoder"]["base_channels"],
        strides=tuple(m["decoder"]["strides"]),
        latent_dim=m["decoder"]["latent_dim"],
    )
    rvq = ResidualVectorQuantizer(
        num_quantizers=m["quantizer"]["num_quantizers"],
        codebook_size=m["quantizer"]["codebook_size"],
        dim=m["encoder"]["latent_dim"],
        decay=m["quantizer"]["decay"],
        dead_code_steps=m["quantizer"]["dead_code_steps"],
        dead_code_threshold=m["quantizer"]["dead_code_threshold"],
    )
    return encoder, rvq, decoder


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    out_dir = args.out or args.ckpt.parent / f"eval_{args.ckpt.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    enc, rvq, dec = build_models_from_config(cfg)
    enc.load_state_dict(ckpt["encoder"])
    rvq.load_state_dict(ckpt["rvq"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval()
    rvq.eval()
    dec.eval()

    sample_rate = cfg["data"]["sample_rate"]
    ds = LibriSpeechSegments(
        root=args.root,
        split=args.split,
        sample_rate=sample_rate,
        segment_seconds=cfg["data"]["segment_seconds"],
        random_crop=False,  # deterministic for reproducible eval
        seed=0,
    )
    n_eval = min(args.num_samples, len(ds))

    mel_l1 = MultiScaleMelLoss(
        sample_rate=sample_rate,
        window_sizes=tuple(cfg["loss"]["stft"]["window_sizes"]),
        n_mels=cfg["loss"]["stft"]["n_mels"],
        l2_weight=0.0,  # pure L1 for eval
    )

    si_sdrs: list[float] = []
    mels: list[float] = []

    print(f"loaded {args.ckpt} (step {ckpt['step']})")
    print(f"evaluating on {n_eval} {args.split} utterances...")
    with torch.no_grad():
        for i in range(n_eval):
            x = ds[i].unsqueeze(0)  # [1, 1, T]
            z = enc(x)
            q, _, _ = rvq(z)
            x_hat = dec(q)

            si_sdrs.append(si_sdr_db(x, x_hat).item())
            mels.append(mel_l1(x, x_hat).item())

            if i < args.num_dump:
                sf.write(
                    str(samples_dir / f"{i:03d}_input.wav"),
                    x.squeeze().numpy(),
                    sample_rate,
                )
                sf.write(
                    str(samples_dir / f"{i:03d}_recon.wav"),
                    x_hat.squeeze().numpy().clip(-1.0, 1.0),
                    sample_rate,
                )

    si_sdrs_t = torch.tensor(si_sdrs)
    mels_t = torch.tensor(mels)
    bitrate = (
        cfg["model"]["quantizer"]["num_quantizers"]
        * math.log2(cfg["model"]["quantizer"]["codebook_size"])
        * (sample_rate / math.prod(cfg["model"]["encoder"]["strides"]))
    )

    metrics = {
        "ckpt": str(args.ckpt),
        "step": int(ckpt["step"]),
        "split": args.split,
        "num_samples": n_eval,
        "si_sdr_mean_db": float(si_sdrs_t.mean()),
        "si_sdr_median_db": float(si_sdrs_t.median()),
        "si_sdr_std_db": float(si_sdrs_t.std()),
        "si_sdr_min_db": float(si_sdrs_t.min()),
        "si_sdr_max_db": float(si_sdrs_t.max()),
        "mel_l1_mean": float(mels_t.mean()),
        "mel_l1_median": float(mels_t.median()),
        "mel_l1_std": float(mels_t.std()),
        "bitrate_bps": float(bitrate),
    }

    # Persist per-sample numbers so downstream analyses (compare_evals,
    # paired statistical tests) don't have to re-decode the dumped wavs
    # (which is also the only way to get scores for the >num_dump tail
    # of utterances that are *evaluated* but not dumped).
    per_sample_path = out_dir / "per_sample.csv"
    with per_sample_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "si_sdr_db", "mel_l1"])
        for i, (sdr, mel) in enumerate(zip(si_sdrs, mels)):
            w.writerow([i, f"{sdr:.4f}", f"{mel:.4f}"])

    print()
    print(
        f"  SI-SDR  : mean {metrics['si_sdr_mean_db']:>7.2f}  "
        f"median {metrics['si_sdr_median_db']:>7.2f}  "
        f"std {metrics['si_sdr_std_db']:>5.2f} dB"
    )
    print(
        f"  Mel L1  : mean {metrics['mel_l1_mean']:>7.4f}  "
        f"median {metrics['mel_l1_median']:>7.4f}  "
        f"std {metrics['mel_l1_std']:>6.4f}"
    )
    print(f"  bitrate : {metrics['bitrate_bps']:>7.0f} bps  ({metrics['bitrate_bps']/1000:.1f} kbps)")
    print(f"  dumped  : {min(args.num_dump, n_eval)} input/recon wav pairs to {samples_dir}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  metrics : {out_dir / 'metrics.json'}")
    print(f"  per-sample CSV : {per_sample_path}")


if __name__ == "__main__":
    main()
