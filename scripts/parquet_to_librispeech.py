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
per utterance with:
  - ``audio`` struct: ``{bytes: <FLAC-encoded>, path: <just filename>}``
  - ``speaker_id`` int, ``chapter_id`` int, ``id`` string (e.g. ``374-180298-0000``)
  - ``text``, ``file`` (absolute path from the upload machine, not portable)

The target FLAC path is ``<dst>/<speaker_id>/<chapter_id>/<id>.flac``,
which matches exactly what ``LibriSpeechSegments`` discovers via rglob.
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
        # Pull only the columns we need. Each shard is ~450 MB on disk and
        # ~1 GB once decoded into Python lists; processing one at a time
        # keeps peak memory predictable.
        table = pq.read_table(
            shard, columns=["audio", "speaker_id", "chapter_id", "id"]
        )
        n_rows = table.num_rows
        print(f"\n{shard.name}: {n_rows:,} utterances")

        audio_col = table.column("audio").to_pylist()
        spk_col = table.column("speaker_id").to_pylist()
        chap_col = table.column("chapter_id").to_pylist()
        id_col = table.column("id").to_pylist()

        for audio, spk, chap, utt_id in tqdm(
            zip(audio_col, spk_col, chap_col, id_col), total=n_rows, unit="utt"
        ):
            target = dst / str(spk) / str(chap) / f"{utt_id}.flac"
            target.parent.mkdir(parents=True, exist_ok=True)
            audio_bytes = audio["bytes"] if isinstance(audio, dict) else audio
            if audio_bytes is None:
                raise RuntimeError(
                    f"row for id={utt_id} has no audio bytes — schema may not "
                    "match what this script expects; rerun with --inspect-only."
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
