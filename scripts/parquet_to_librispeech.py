"""Convert HuggingFace ``openslr/librispeech_asr`` parquet shards to the
canonical LibriSpeech FLAC tree that ``LibriSpeechSegments`` expects.

Why this exists: openslr.org's tar.gz is unreliable from China (Magicdata
mirror returns 503, ELDA peaks at 100 KB/s). The HuggingFace mirror via
hf-mirror.com is fast (~8 MB/s) and reliable, but ships parquet shards
instead of a flac directory tree. This script bridges the gap.

Run after ``hf download --type dataset openslr/librispeech_asr ...``:

    python scripts/parquet_to_librispeech.py \\
        --src datasets/hf_librispeech/all/train.clean.100 \\
        --dst datasets/LibriSpeech/train-clean-100

The HF schema for this dataset (openslr/librispeech_asr) carries one row
per utterance with at least ``file`` (original .flac path under the
LibriSpeech tree) and ``audio`` (struct with ``bytes`` of FLAC-encoded
audio plus ``path``). The ``file`` column directly tells us the target
relative path, so this script is little more than a parquet -> file write.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src.*` importable when this script is run as `python scripts/foo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from tqdm import tqdm  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Directory containing the parquet shards "
        "(e.g. datasets/hf_librispeech/all/train.clean.100).",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output directory; will be populated with <spk>/<chap>/*.flac. "
        "Pass datasets/LibriSpeech/train-clean-100 (or test-clean, etc.) "
        "to match the layout LibriSpeechSegments looks for. Required unless "
        "--inspect-only is set.",
    )
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the schema of the first parquet shard and exit. Use this "
        "to verify the audio / file column names before running a real conversion.",
    )
    return p.parse_args()


def inspect(src: Path) -> None:
    parquets = sorted(src.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquet files in {src}")
    table = pq.read_table(parquets[0], columns=None)
    print(f"--- {parquets[0].name} ---")
    print(f"rows: {table.num_rows}")
    print(f"schema:\n{table.schema}")
    print()
    print("--- first row (audio bytes truncated) ---")
    row = table.slice(0, 1).to_pylist()[0]
    for k, v in row.items():
        if isinstance(v, dict):
            shown = {kk: (f"<{len(vv)} bytes>" if isinstance(vv, (bytes, bytearray)) else vv)
                     for kk, vv in v.items()}
            print(f"  {k}: {shown}")
        elif isinstance(v, (bytes, bytearray)):
            print(f"  {k}: <{len(v)} bytes>")
        else:
            print(f"  {k}: {v!r}")


def convert(src: Path, dst: Path) -> None:
    parquets = sorted(src.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquet files in {src}")
    dst.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for shard in parquets:
        # Pull only the columns we need. Loading all rows into memory is fine
        # here — each shard is ~450 MB and we only keep the audio bytes long
        # enough to write them to disk.
        table = pq.read_table(shard, columns=["file", "audio"])
        n_rows = table.num_rows
        print(f"\n{shard.name}: {n_rows:,} utterances")

        files_col = table.column("file").to_pylist()
        audio_col = table.column("audio").to_pylist()

        for rel_path, audio in tqdm(
            zip(files_col, audio_col), total=n_rows, unit="utt"
        ):
            # `rel_path` looks like 'train-clean-100/19/198/19-198-0000.flac'
            # — strip the split prefix so it lands directly under `dst`.
            parts = Path(rel_path).parts
            if parts and parts[0] in {
                "train-clean-100",
                "train-clean-360",
                "train-other-500",
                "dev-clean",
                "dev-other",
                "test-clean",
                "test-other",
            }:
                relative = Path(*parts[1:])
            else:
                relative = Path(rel_path)

            target = dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            audio_bytes = audio["bytes"] if isinstance(audio, dict) else audio
            if audio_bytes is None:
                raise RuntimeError(
                    f"row for {rel_path} has no audio bytes — schema may not "
                    "match what this script expects; rerun with --inspect-only "
                    "to see the actual columns."
                )
            target.write_bytes(audio_bytes)
            n_written += 1

    print(f"\nwrote {n_written:,} flac files under {dst}")


def main() -> None:
    args = parse_args()
    if args.inspect_only:
        inspect(args.src)
        return
    if args.dst is None:
        raise SystemExit("--dst is required for actual conversion (omit only with --inspect-only)")
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
