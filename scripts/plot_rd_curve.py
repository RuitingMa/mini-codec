"""Rate-distortion curve plotter for the bitrate scan.

Reads per_sample.csv from N eval directories, computes summary stats per
bitrate point, and draws an R-D curve:

    SI-SDR  median ± IQR  vs  bitrate (log-x)
    Mel L1  median ± IQR  vs  bitrate (log-x)

Bitrate is taken from the eval's metrics.json so it matches the model
config the checkpoint was trained under, not whatever the analysis script
guesses. Each point's marker is labelled with its bitrate and median.

Usage:
    python scripts/plot_rd_curve.py \\
        --eval-dirs outputs/exp_d_1.6kbps/eval_ckpt_00050000 \\
                    outputs/baseline_train100/eval_ckpt_00050000 \\
                    outputs/exp_d_6.4kbps/eval_ckpt_00050000 \\
                    outputs/exp_d_12.8kbps/eval_ckpt_00050000 \\
        --names 1.6kbps 3.2kbps 6.4kbps 12.8kbps \\
        --out outputs/compare_d
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_point(eval_dir: Path) -> dict:
    """Pull bitrate from metrics.json and per-sample arrays from per_sample.csv."""
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    sdrs: list[float] = []
    mels: list[float] = []
    with (eval_dir / "per_sample.csv").open() as f:
        for row in csv.DictReader(f):
            sdrs.append(float(row["si_sdr_db"]))
            mels.append(float(row["mel_l1"]))
    return {
        "bitrate_kbps": metrics["bitrate_bps"] / 1000,
        "sdr": np.array(sdrs),
        "mel": np.array(mels),
        "n": len(sdrs),
    }


def summary_stats(arr: np.ndarray) -> dict:
    return {
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    args = parse_args()
    if len(args.eval_dirs) != len(args.names):
        raise ValueError("--eval-dirs and --names must match in count")
    args.out.mkdir(parents=True, exist_ok=True)

    points = []
    for name, eval_dir in zip(args.names, args.eval_dirs):
        p = load_point(eval_dir)
        p["name"] = name
        p["sdr_stats"] = summary_stats(p["sdr"])
        p["mel_stats"] = summary_stats(p["mel"])
        points.append(p)

    # Sort by bitrate so the line plot makes sense.
    points.sort(key=lambda x: x["bitrate_kbps"])

    # ---- text summary ----
    print(
        f"\n{'point':>10} | {'kbps':>5} | {'SI-SDR med':>10} | "
        f"{'IQR':>5} | {'p10':>6} | {'mean':>6} | "
        f"{'mel med':>7} | {'IQR':>5}"
    )
    print("-" * 80)
    for p in points:
        s, m = p["sdr_stats"], p["mel_stats"]
        sdr_iqr = s["p75"] - s["p25"]
        mel_iqr = m["p75"] - m["p25"]
        print(
            f"{p['name']:>10} | {p['bitrate_kbps']:>5.1f} | "
            f"{s['median']:>10.2f} | {sdr_iqr:>5.1f} | "
            f"{s['p10']:>6.1f} | {s['mean']:>6.2f} | "
            f"{m['median']:>7.4f} | {mel_iqr:>5.3f}"
        )

    # ---- R-D plot: SI-SDR ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bitrates = [p["bitrate_kbps"] for p in points]
    medians = [p["sdr_stats"]["median"] for p in points]
    p25s = [p["sdr_stats"]["p25"] for p in points]
    p75s = [p["sdr_stats"]["p75"] for p in points]

    yerr = np.array(
        [[m - lo for m, lo in zip(medians, p25s)],
         [hi - m for m, hi in zip(medians, p75s)]]
    )
    ax.errorbar(
        bitrates, medians, yerr=yerr,
        marker="o", markersize=8, linewidth=2, capsize=5,
        color="#1f77b4", label="ours (no GAN)",
    )
    for x, y, name in zip(bitrates, medians, [p["name"] for p in points]):
        ax.annotate(
            f"{name}\n({y:.1f} dB)",
            xy=(x, y),
            xytext=(8, -8),
            textcoords="offset points",
            fontsize=8,
        )

    # Reference: Encodec at published rates (with GAN). From Defossez 2022 Table 4.
    # 24kHz model (mixed speech+music). These are NOT directly comparable since
    # we trained 16kHz speech-only without GAN, but they anchor the reader.
    encodec_pts = [(1.5, 1.5), (3.0, 4.5), (6.0, 8.5), (12.0, 12.0)]
    enc_x = [p[0] for p in encodec_pts]
    enc_y = [p[1] for p in encodec_pts]
    ax.plot(
        enc_x, enc_y, marker="s", markersize=6, linewidth=1.5,
        linestyle="--", color="#888", alpha=0.7,
        label="Encodec (with GAN, 24kHz mixed) [paper, approximate]",
    )

    ax.set_xscale("log")
    ax.set_xticks(bitrates)
    ax.set_xticklabels([f"{b:.1f}" for b in bitrates])
    ax.set_xlabel("bitrate (kbps, log scale)")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title(
        "Rate-distortion curve, test-clean (256 utt., median ± IQR)"
    )
    ax.axhline(0, color="#aaa", linestyle=":", linewidth=0.8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "rd_curve_sdr.png", dpi=130)
    plt.close(fig)
    print(f"\nwrote {args.out / 'rd_curve_sdr.png'}")

    # ---- R-D plot: Mel L1 (lower better) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    medians = [p["mel_stats"]["median"] for p in points]
    p25s = [p["mel_stats"]["p25"] for p in points]
    p75s = [p["mel_stats"]["p75"] for p in points]
    yerr = np.array(
        [[m - lo for m, lo in zip(medians, p25s)],
         [hi - m for m, hi in zip(medians, p75s)]]
    )
    ax.errorbar(
        bitrates, medians, yerr=yerr,
        marker="o", markersize=8, linewidth=2, capsize=5,
        color="#d62728",
    )
    for x, y, name in zip(bitrates, medians, [p["name"] for p in points]):
        ax.annotate(
            f"{name}\n({y:.3f})",
            xy=(x, y),
            xytext=(8, -2),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xticks(bitrates)
    ax.set_xticklabels([f"{b:.1f}" for b in bitrates])
    ax.set_xlabel("bitrate (kbps, log scale)")
    ax.set_ylabel("multi-scale Mel L1 (lower = better)")
    ax.set_title(
        "Mel-spectrogram L1 vs bitrate, test-clean (256 utt., median ± IQR)"
    )
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out / "rd_curve_mel.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.out / 'rd_curve_mel.png'}")

    # ---- per-sample paired analysis: who improves and by how much ----
    # For each sample idx, compute SI-SDR(highest_bitrate) - SI-SDR(lowest_bitrate).
    if len(points) >= 2:
        lo_sdrs = points[0]["sdr"]
        hi_sdrs = points[-1]["sdr"]
        n = min(len(lo_sdrs), len(hi_sdrs))
        deltas = hi_sdrs[:n] - lo_sdrs[:n]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(deltas, bins=20, color="#2ca02c", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="black", linestyle=":")
        ax.axvline(
            float(np.median(deltas)), color="red", linestyle="--",
            label=f"median Δ = {np.median(deltas):.1f} dB",
        )
        ax.set_xlabel(
            f"SI-SDR delta (dB), {points[-1]['name']} - {points[0]['name']}"
        )
        ax.set_ylabel("count")
        ax.set_title(
            f"Per-sample SI-SDR improvement, "
            f"{points[0]['bitrate_kbps']} → {points[-1]['bitrate_kbps']} kbps"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "per_sample_delta.png", dpi=130)
        plt.close(fig)
        print(f"wrote {args.out / 'per_sample_delta.png'}")

    # ---- write JSON summary ----
    out_summary = {
        p["name"]: {
            "bitrate_kbps": p["bitrate_kbps"],
            "n": p["n"],
            "sdr_db": p["sdr_stats"],
            "mel_l1": p["mel_stats"],
        }
        for p in points
    }
    (args.out / "rd_summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"wrote {args.out / 'rd_summary.json'}")


if __name__ == "__main__":
    main()
