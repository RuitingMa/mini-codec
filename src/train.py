"""Training entrypoint for the mini-codec.

Loads a YAML config, builds the encoder + RVQ + decoder, and trains
against a weighted sum of time-domain L1, multi-scale mel STFT, and RVQ
commitment losses (no adversarial term — see project memo).

Run:
    python -m src.train --config configs/baseline.yaml [overrides]

Common overrides: --total-steps, --batch-size, --lr, --out-dir,
--logger {none,tensorboard,wandb}. Anything more specialised should
go in the yaml directly.
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
from src.losses.perceptual import PerceptualLoss
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
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override data.split, e.g. 'train-clean-100' or 'test-clean'.",
    )
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=None,
        help="Steps between checkpoint dumps; useful to stop the smoke "
        "config's every-100-steps default from spamming on long runs.",
    )
    p.add_argument(
        "--perceptual-layer",
        type=int,
        default=None,
        help="Override loss.perceptual.layer for HuBERT layer sweeps.",
    )
    p.add_argument(
        "--perceptual-weight",
        type=float,
        default=None,
        help="Override loss.perceptual_weight (lambda) for the additive run.",
    )
    p.add_argument(
        "--perceptual-backend",
        choices=["torchaudio", "huggingface"],
        default=None,
        help="Override loss.perceptual.backend; useful when the torchaudio "
        "Meta CDN is unreachable and you want to fall back to HF mirror.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--logger",
        choices=["none", "tensorboard", "wandb"],
        default=None,
        help="Override logger.type. Defaults to whatever the yaml says.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Shortcut for --logger wandb.",
    )
    return p.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict:
    # Explicit UTF-8 so config files with non-ASCII comments (em-dashes,
    # CJK notes, etc.) are readable on Windows where the default text
    # encoding is GBK rather than UTF-8.
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.total_steps is not None:
        cfg["train"]["total_steps"] = args.total_steps
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.split is not None:
        cfg["data"]["split"] = args.split
    if args.ckpt_every is not None:
        cfg["train"]["ckpt_every"] = args.ckpt_every
    if args.out_dir is not None:
        cfg["train"]["out_dir"] = str(args.out_dir)
    if args.perceptual_layer is not None:
        cfg["loss"].setdefault("perceptual", {})["layer"] = args.perceptual_layer
    if args.perceptual_weight is not None:
        cfg["loss"]["perceptual_weight"] = args.perceptual_weight
    if args.perceptual_backend is not None:
        cfg["loss"].setdefault("perceptual", {})["backend"] = args.perceptual_backend
    if args.logger is not None:
        cfg["logger"]["type"] = args.logger
    if args.wandb:
        cfg["logger"]["type"] = "wandb"
    return cfg


class _NoOpLogger:
    def log(self, metrics: dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class _TensorboardLogger:
    """Thin wrapper over ``torch.utils.tensorboard.SummaryWriter``.

    Writes scalars under ``<out_dir>/tb/`` so AutoDL's built-in
    TensorBoard service (or a local ``tensorboard --logdir``) picks them
    up automatically. Avoids any external network dependency.
    """

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: dict, step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def finish(self) -> None:
        self.writer.close()


class _WandbLogger:
    def __init__(self, cfg: dict) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )

    def log(self, metrics: dict, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        self._wandb.finish()


def make_logger(cfg: dict, out_dir: Path):
    typ = cfg["logger"]["type"]
    if typ == "none":
        return _NoOpLogger()
    if typ == "tensorboard":
        return _TensorboardLogger(out_dir / "tb")
    if typ == "wandb":
        return _WandbLogger(cfg["logger"])
    raise ValueError(f"unknown logger.type: {typ!r}")


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
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

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
        pin_memory=device.type == "cuda",
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
    enc.to(device)
    rvq.to(device)
    dec.to(device)
    stft.to(device)

    # Optional W5 perceptual term — only built when its weight > 0 so
    # baseline runs don't pay the HuBERT load / forward cost.
    percep_weight = float(cfg["loss"].get("perceptual_weight", 0.0))
    percep: PerceptualLoss | None = None
    if percep_weight > 0.0:
        percep_cfg = cfg["loss"].get("perceptual", {})
        percep = PerceptualLoss(
            layer=int(percep_cfg.get("layer", 6)),
            sample_rate=d["sample_rate"],
            backend=str(percep_cfg.get("backend", "torchaudio")),
            hf_repo=str(percep_cfg.get("hf_repo", "facebook/hubert-base-ls960")),
        ).to(device)
        print(
            f"perceptual loss: HuBERT base layer {percep.layer}, "
            f"backend={percep_cfg.get('backend', 'torchaudio')}, "
            f"weight {percep_weight}"
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

    # --- logger (none / tensorboard / wandb; lazy import for wandb) ---
    logger = make_logger(cfg, out_dir)
    print(f"logger: {cfg['logger']['type']}")

    # --- train loop ---
    train_cfg = cfg["train"]
    loss_w = cfg["loss"]
    enc.train()
    rvq.train()
    dec.train()
    data_iter = cycle(dl)
    t0 = time.perf_counter()

    print(f"\nstarting training: {train_cfg['total_steps']} steps")
    header = f"{'step':>6} | {'L1':>7} | {'STFT':>7} | {'commit':>7}"
    if percep is not None:
        header += f" | {'percep':>7}"
    header += f" | {'total':>7} | active codes per layer"
    print(header)
    print("-" * len(header))

    for step in range(1, train_cfg["total_steps"] + 1):
        x = next(data_iter).to(device, non_blocking=True)
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
        if percep is not None:
            p_loss = percep(x, x_hat)
            loss = loss + percep_weight * p_loss
        else:
            p_loss = None
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % train_cfg["log_every"] == 0 or step == 1:
            active = [
                int(idx[:, i].unique().numel()) for i in range(rvq.num_quantizers)
            ]
            row = (
                f"{step:>6} | {l1.item():>7.4f} | {s.item():>7.4f} | "
                f"{c_loss.item():>7.4f}"
            )
            if p_loss is not None:
                row += f" | {p_loss.item():>7.4f}"
            row += f" | {loss.item():>7.4f} | {active}"
            print(row)
            log_dict = {
                "loss/l1": l1.item(),
                "loss/stft": s.item(),
                "loss/commit": c_loss.item(),
                "loss/total": loss.item(),
                **{
                    f"codebook/active_layer_{i}": a
                    for i, a in enumerate(active)
                },
            }
            if p_loss is not None:
                log_dict["loss/perceptual"] = p_loss.item()
            logger.log(log_dict, step=step)

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
    logger.finish()


if __name__ == "__main__":
    main()
