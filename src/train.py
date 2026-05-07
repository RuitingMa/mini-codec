"""Training entrypoint for the mini-codec.

Loads a YAML config, builds the encoder + RVQ + decoder, and trains
against a weighted sum of time-domain L1, multi-scale mel STFT, and RVQ
commitment losses (no adversarial term — see project memo).

Run:
    python -m src.train --config configs/baseline.yaml [overrides]

Common overrides: --total-steps, --batch-size, --lr, --out-dir, --wandb.
Anything more specialised should go in the yaml directly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.librispeech import LibriSpeechSegments
from src.losses.stft import MultiScaleMelLoss
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.quantizer import ResidualVectorQuantizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging regardless of the yaml's wandb.enabled flag.",
    )
    return p.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if args.total_steps is not None:
        cfg["train"]["total_steps"] = args.total_steps
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.out_dir is not None:
        cfg["train"]["out_dir"] = str(args.out_dir)
    if args.wandb:
        cfg["wandb"]["enabled"] = True
    return cfg


def build_model(cfg: dict) -> tuple[Encoder, ResidualVectorQuantizer, Decoder]:
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


def cycle(loader: DataLoader):
    """Indefinite generator that recreates the loader's iterator each epoch
    (so shuffling actually re-shuffles)."""
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    torch.manual_seed(cfg["train"]["seed"])
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # --- data ---
    d = cfg["data"]
    ds = LibriSpeechSegments(
        root=d["root"],
        split=d["split"],
        sample_rate=d["sample_rate"],
        segment_seconds=d["segment_seconds"],
        seed=cfg["train"]["seed"],
    )
    dl = DataLoader(
        ds,
        batch_size=d["batch_size"],
        shuffle=True,
        num_workers=d["num_workers"],
        drop_last=True,
    )
    print(
        f"dataset: {len(ds)} utterances, {len(dl)} batches/epoch "
        f"(batch={d['batch_size']})"
    )

    # --- model + loss ---
    enc, rvq, dec = build_model(cfg)
    stft = MultiScaleMelLoss(
        sample_rate=d["sample_rate"],
        window_sizes=tuple(cfg["loss"]["stft"]["window_sizes"]),
        n_mels=cfg["loss"]["stft"]["n_mels"],
    )
    n_params = (
        sum(p.numel() for p in enc.parameters())
        + sum(p.numel() for p in rvq.parameters())
        + sum(p.numel() for p in dec.parameters())
    )
    print(f"model: {n_params / 1e6:.2f}M params (encoder + RVQ + decoder)")

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(rvq.parameters()) + list(dec.parameters()),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    # --- wandb (lazy import; only required when enabled) ---
    use_wandb = bool(cfg["wandb"]["enabled"])
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )

    # --- train loop ---
    train_cfg = cfg["train"]
    loss_w = cfg["loss"]
    enc.train()
    rvq.train()
    dec.train()
    data_iter = cycle(dl)
    t0 = time.perf_counter()

    print(f"\nstarting training: {train_cfg['total_steps']} steps")
    print(
        f"{'step':>6} | {'L1':>7} | {'STFT':>7} | {'commit':>7} | "
        f"{'total':>7} | active codes per layer"
    )
    print("-" * 80)

    for step in range(1, train_cfg["total_steps"] + 1):
        x = next(data_iter)
        z = enc(x)
        q, idx, c_loss = rvq(z)
        x_hat = dec(q)

        l1 = F.l1_loss(x_hat, x)
        s = stft(x, x_hat)
        loss = (
            loss_w["l1_weight"] * l1
            + loss_w["stft_weight"] * s
            + loss_w["commitment_weight"] * c_loss
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % train_cfg["log_every"] == 0 or step == 1:
            active = [
                int(idx[:, i].unique().numel()) for i in range(rvq.num_quantizers)
            ]
            print(
                f"{step:>6} | {l1.item():>7.4f} | {s.item():>7.4f} | "
                f"{c_loss.item():>7.4f} | {loss.item():>7.4f} | {active}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "loss/l1": l1.item(),
                        "loss/stft": s.item(),
                        "loss/commit": c_loss.item(),
                        "loss/total": loss.item(),
                        **{
                            f"codebook/active_layer_{i}": a
                            for i, a in enumerate(active)
                        },
                    },
                    step=step,
                )

        if (
            step % train_cfg["ckpt_every"] == 0
            or step == train_cfg["total_steps"]
        ):
            ckpt_path = out_dir / f"ckpt_{step:08d}.pt"
            torch.save(
                {
                    "step": step,
                    "encoder": enc.state_dict(),
                    "rvq": rvq.state_dict(),
                    "decoder": dec.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path}")

    elapsed = time.perf_counter() - t0
    print(
        f"\ndone in {elapsed:.1f}s "
        f"({elapsed / train_cfg['total_steps'] * 1000:.0f} ms/step)"
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
