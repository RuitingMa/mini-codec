"""Side-by-side comparison of multiple mini-codec eval-result directories.

For each named experiment, reads the dumped ``*_input.wav`` / ``*_recon.wav``
pairs, recomputes per-sample SI-SDR and multi-scale Mel L1, and emits:

    summary.json            mean / median / std for each experiment
    per_sample.csv          long-form CSV (experiment, idx, si_sdr, mel_l1)
    sdr_hist.png            overlaid SI-SDR histograms
    mel_l1_box.png          box plot of Mel L1 per experiment
    spectrogram_*.png       3-panel spectrograms (input + each recon) for
                            the worst / median / best samples picked from
                            the FIRST experiment passed (typically baseline)

Usage:
    python scripts/compare_evals.py \\
        --eval-dirs outputs/baseline_train100/eval_ckpt_00050000 \\
                    outputs/exp_b_perceptual/eval_ckpt_00050000 \\
                    outputs/exp_b_swap_stft/eval_ckpt_00050000 \\
        --names baseline +perceptual swap_stft \\
        --out outputs/compare_b

Each --eval-dirs entry can point to either ``<eval_dir>`` (the script
finds ``samples/`` underneath) or directly at the samples directory.
The script does not need any checkpoint or model — only the wav pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make `src.*` importable when this script is run as `python scripts/foo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from scripts.eval import si_sdr_db  # noqa: E402
from src.losses.stft import MultiScaleMelLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more eval directories (or their samples/ subdirs).",
    )
    p.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Display names, one per --eval-dirs entry.",
    )
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def find_eval_root(p: Path) -> Path:
    """Resolve the eval root from either ``<eval_dir>`` or ``<eval_dir>/samples``."""
    if (p.parent / "per_sample.csv").exists() and p.name == "samples":
        return p.parent
    return p


def find_samples_dir(p: Path) -> Path:
    """Accept both ``eval_dir`` and ``eval_dir/samples`` as input."""
    if (p / "samples").is_dir():
        return p / "samples"
    return p


def collect_metrics(eval_dir: Path, mel_l1: MultiScaleMelLoss) -> list[dict]:
    """Return per-sample (idx, sdr, mel) rows, preferring ``per_sample.csv``
    written by ``scripts/eval.py`` (covers the full --num-samples set, not
    just the dumped wav pairs). Falls back to recomputing from the dumped
    wavs when the CSV is absent (older eval directories)."""
    root = find_eval_root(eval_dir)
    csv_path = root / "per_sample.csv"
    if csv_path.exists():
        rows = []
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "idx": int(r["idx"]),
                        "sdr": float(r["si_sdr_db"]),
                        "mel": float(r["mel_l1"]),
                    }
                )
        return sorted(rows, key=lambda r: r["idx"])

    # Fallback: recompute from whatever wav pairs were dumped.
    samples_dir = find_samples_dir(eval_dir)
    rows = []
    for p_in in sorted(samples_dir.glob("*_input.wav")):
        idx = int(p_in.stem.split("_")[0])
        p_re = p_in.parent / p_in.name.replace("_input", "_recon")
        if not p_re.exists():
            continue
        x_in, _ = sf.read(p_in, dtype="float32")
        x_re, _ = sf.read(p_re, dtype="float32")
        n = min(len(x_in), len(x_re))
        x_in, x_re = x_in[:n], x_re[:n]
        xi = torch.from_numpy(x_in).unsqueeze(0).unsqueeze(0)
        xr = torch.from_numpy(x_re).unsqueeze(0).unsqueeze(0)
        rows.append(
            {
                "idx": idx,
                "sdr": si_sdr_db(xi, xr).item(),
                "mel": mel_l1(xi, xr).item(),
            }
        )
    return sorted(rows, key=lambda r: r["idx"])


def load_wav_pair(eval_dir: Path, idx: int) -> tuple | None:
    """Load one ``XXX_input.wav`` / ``XXX_recon.wav`` pair for spectrogram
    plotting; returns ``(x_in, x_re)`` numpy arrays or ``None`` if either
    file is missing (likely the case for indices beyond ``--num-dump``)."""
    samples_dir = find_samples_dir(eval_dir)
    p_in = samples_dir / f"{idx:03d}_input.wav"
    p_re = samples_dir / f"{idx:03d}_recon.wav"
    if not (p_in.exists() and p_re.exists()):
        return None
    x_in, _ = sf.read(p_in, dtype="float32")
    x_re, _ = sf.read(p_re, dtype="float32")
    n = min(len(x_in), len(x_re))
    return x_in[:n], x_re[:n]


def stft_db(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    S = np.abs(np.fft.rfft(frames, axis=-1)).T
    return 20 * np.log10(S + 1e-6)


def write_summary(data: dict[str, list[dict]], out_dir: Path) -> dict:
    summary: dict = {}
    print(
        f"\n{'experiment':<14} | {'n':>3} | "
        f"{'SI-SDR mean':>11} | {'median':>7} | {'std':>5} | "
        f"{'Mel L1 mean':>11} | {'std':>6}"
    )
    print("-" * 80)
    for name, rows in data.items():
        sdrs = np.array([r["sdr"] for r in rows])
        mels = np.array([r["mel"] for r in rows])
        s = {
            "n": int(len(rows)),
            "sdr_mean_db": float(sdrs.mean()),
            "sdr_median_db": float(np.median(sdrs)),
            "sdr_std_db": float(sdrs.std()),
            "sdr_min_db": float(sdrs.min()),
            "sdr_max_db": float(sdrs.max()),
            "mel_l1_mean": float(mels.mean()),
            "mel_l1_std": float(mels.std()),
        }
        summary[name] = s
        print(
            f"{name:<14} | {s['n']:>3} | "
            f"{s['sdr_mean_db']:>11.2f} | {s['sdr_median_db']:>7.2f} | "
            f"{s['sdr_std_db']:>5.2f} | {s['mel_l1_mean']:>11.4f} | "
            f"{s['mel_l1_std']:>6.4f}"
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_per_sample_csv(data: dict[str, list[dict]], out_dir: Path) -> Path:
    csv_path = out_dir / "per_sample.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "idx", "si_sdr_db", "mel_l1"])
        for name, rows in data.items():
            for r in rows:
                w.writerow([name, r["idx"], f"{r['sdr']:.4f}", f"{r['mel']:.4f}"])
    return csv_path


def plot_sdr_histogram(data: dict[str, list[dict]], out_path: Path) -> None:
    all_sdrs = np.concatenate(
        [np.array([r["sdr"] for r in rows]) for rows in data.values()]
    )
    bins = np.linspace(all_sdrs.min(), all_sdrs.max(), 18)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, rows in data.items():
        sdrs = np.array([r["sdr"] for r in rows])
        label = f"{name} (median {np.median(sdrs):.1f} dB)"
        ax.hist(sdrs, bins=bins, alpha=0.45, label=label)
    ax.set_xlabel("SI-SDR (dB)")
    ax.set_ylabel("count")
    ax.set_title("Per-sample SI-SDR distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mel_box(data: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    boxes = [np.array([r["mel"] for r in rows]) for rows in data.values()]
    ax.boxplot(boxes, tick_labels=list(data.keys()), showmeans=True)
    ax.set_ylabel("Mel L1 (lower = better)")
    ax.set_title("Mel L1 distribution per experiment")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spectrogram_triplet(
    label: str,
    ref_idx: int,
    data: dict[str, list[dict]],
    eval_dirs_by_name: dict[str, Path],
    names: list[str],
    out_path: Path,
) -> bool:
    """Plot ``input`` + one ``recon`` per experiment, all sharing the same
    colour scale. ``ref_idx`` selects which utterance to compare across.

    Wav data is loaded on demand (so this works even when per-sample
    metrics came from per_sample.csv covering more samples than were
    dumped). Returns False if the required input wav isn't dumped, so
    the caller can pick a different ref idx."""
    first_name = names[0]
    in_pair = load_wav_pair(eval_dirs_by_name[first_name], ref_idx)
    if in_pair is None:
        return False
    x_in, _ = in_pair
    S_in = stft_db(x_in)
    vmax = S_in.max()
    vmin = vmax - 80
    n_panels = 1 + len(names)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5), sharey=True)
    axes[0].imshow(
        S_in,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[0, len(x_in) / 16000, 0, 8000],
        cmap="magma",
    )
    axes[0].set_title(f"input (idx {ref_idx}, {label})")
    axes[0].set_ylabel("Hz")
    axes[0].set_xlabel("time (s)")

    for i, name in enumerate(names):
        sdr = next(
            (r["sdr"] for r in data[name] if r["idx"] == ref_idx), None
        )
        pair = load_wav_pair(eval_dirs_by_name[name], ref_idx)
        if pair is None:
            axes[i + 1].set_title(f"{name} (idx {ref_idx} not dumped)")
            continue
        _, x_re = pair
        sdr_str = f"{sdr:.1f} dB" if sdr is not None else "?? dB"
        axes[i + 1].imshow(
            stft_db(x_re),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=[0, len(x_re) / 16000, 0, 8000],
            cmap="magma",
        )
        axes[i + 1].set_title(f"{name} (SI-SDR {sdr_str})")
        axes[i + 1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    if len(args.eval_dirs) != len(args.names):
        raise ValueError("--eval-dirs and --names must have the same length")
    args.out.mkdir(parents=True, exist_ok=True)

    mel_l1 = MultiScaleMelLoss(sample_rate=16000, l2_weight=0.0)

    data: dict[str, list[dict]] = {}
    eval_dirs_by_name: dict[str, Path] = {}
    for name, eval_dir in zip(args.names, args.eval_dirs):
        rows = collect_metrics(eval_dir, mel_l1)
        if not rows:
            raise RuntimeError(f"no per-sample data found under {eval_dir}")
        data[name] = rows
        eval_dirs_by_name[name] = eval_dir
        src = (
            "per_sample.csv"
            if (find_eval_root(eval_dir) / "per_sample.csv").exists()
            else "wav fallback"
        )
        print(f"  {name:<14} {len(rows):>4} rows from {src}  ({eval_dir})")

    write_summary(data, args.out)
    csv_path = write_per_sample_csv(data, args.out)
    print(f"\nwrote per_sample.csv -> {csv_path}")
    plot_sdr_histogram(data, args.out / "sdr_hist.png")
    plot_mel_box(data, args.out / "mel_l1_box.png")
    print(f"wrote sdr_hist.png + mel_l1_box.png under {args.out}")

    # Pick worst / median / best from the FIRST named experiment, restricted
    # to indices whose wav was dumped (so the spectrogram panels actually
    # have audio to render). For each target rank we try the exact rank
    # first and then walk outward, so e.g. when num_dump=32 but we have CSV
    # scores for 256, the picks still resolve to the closest dumped sample
    # to the target percentile.
    first_name = args.names[0]
    first_eval_dir = eval_dirs_by_name[first_name]
    sorted_first = sorted(data[first_name], key=lambda r: r["sdr"])
    n = len(sorted_first)

    def _pick_near(target_rank: int) -> int | None:
        for offset in range(n):
            for direction in (0, +1, -1) if offset == 0 else (+1, -1):
                i = target_rank + direction * offset
                if 0 <= i < n:
                    idx = sorted_first[i]["idx"]
                    if load_wav_pair(first_eval_dir, idx) is not None:
                        return idx
        return None

    picks = [
        (label, idx)
        for label, idx in [
            ("worst", _pick_near(0)),
            ("median", _pick_near(n // 2)),
            ("best", _pick_near(n - 1)),
        ]
        if idx is not None
    ]
    for label, idx in picks:
        path = args.out / f"spectrogram_{label}_{idx:03d}.png"
        if plot_spectrogram_triplet(label, idx, data, eval_dirs_by_name, args.names, path):
            print(f"wrote {path}")
        else:
            print(f"skipped {label} (idx {idx} wav not dumped)")


if __name__ == "__main__":
    main()
