"""Single-sample overfit smoke test for the encoder + decoder pair.

Picks one ~1-second clip from a LibriSpeech split and trains the encoder
and decoder (no quantizer in the middle) to reconstruct it via L1 loss.
The point is not faithful audio — it's to confirm the architecture has
the capacity to drive the loss substantially below the untrained baseline.
If this *fails to converge*, the architecture has a bug; if it converges,
we can move on to plumbing in RVQ in W3.

Outputs (under ``--out``, default ``outputs/smoke_overfit/``):
    loss.png   — log-scale L1 loss vs. step
    input.wav  — the source clip
    recon.wav  — final-step reconstruction
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src.*` importable when this script is run as `python scripts/foo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless rendering; safe on Windows without a display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F

from src.data.librispeech import LibriSpeechSegments  # noqa: E402
from src.models.decoder import Decoder  # noqa: E402
from src.models.encoder import Encoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("datasets"))
    p.add_argument("--split", type=str, default="dev-clean")
    p.add_argument("--utterance-idx", type=int, default=0,
                   help="Index into the dataset for the clip to overfit.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/smoke_overfit"))
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # Deterministic crop so the same clip is used every iteration.
    ds = LibriSpeechSegments(
        root=args.root,
        split=args.split,
        sample_rate=16000,
        segment_seconds=1.0,
        random_crop=False,
        seed=args.seed,
    )
    print(f"loaded {len(ds)} utterances; using index {args.utterance_idx}")
    x = ds[args.utterance_idx].unsqueeze(0)  # [1, 1, 16000]
    sf.write(str(args.out / "input.wav"), x.squeeze().numpy(), 16000)

    enc, dec = Encoder(), Decoder()
    enc.train()
    dec.train()
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=args.lr
    )

    losses: list[float] = []
    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x_hat = dec(enc(x))
        loss = F.l1_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - t_start
            print(f"step {step:4d}/{args.steps}  L1={loss.item():.5f}  ({elapsed:.1f}s)")

    # Final-step reconstruction with eval-mode (no dropout etc — currently a no-op
    # but kept for forward-compatibility once we add normalization layers).
    enc.eval()
    dec.eval()
    with torch.no_grad():
        x_hat = dec(enc(x))
    sf.write(
        str(args.out / "recon.wav"),
        x_hat.squeeze().numpy().clip(-1.0, 1.0),
        16000,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("L1 loss (log scale)")
    ax.set_title(f"Single-sample overfit ({args.steps} steps, lr={args.lr})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "loss.png", dpi=120)
    plt.close(fig)

    init, final = losses[0], losses[-1]
    drop_pct = (1.0 - final / init) * 100
    print()
    print(f"initial L1 = {init:.5f}")
    print(f"final L1   = {final:.5f}  (after {args.steps} steps)")
    print(f"reduction  = {drop_pct:.1f}%")
    print(f"artifacts written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
