"""One-shot LibriSpeech downloader.

Run from the repo root, e.g.

    python scripts/download_librispeech.py --root datasets --split dev-clean

The downloaded tree lives at ``<root>/LibriSpeech/<split>/`` which is the
layout that ``LibriSpeechSegments`` expects. The default ``datasets/`` folder
is gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.librispeech import download_librispeech


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="Directory to place the LibriSpeech tree under (default: datasets/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev-clean",
        help="LibriSpeech split, e.g. dev-clean, test-clean, train-clean-100.",
    )
    args = parser.parse_args()

    target = download_librispeech(args.root, args.split)
    print(f"LibriSpeech/{args.split} ready at {target}")


if __name__ == "__main__":
    main()
