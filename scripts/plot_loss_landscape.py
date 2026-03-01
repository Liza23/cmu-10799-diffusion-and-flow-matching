#!/usr/bin/env python3
"""
Compute a 2D loss landscape plane and save 2D/3D plots.

Typical usage:
  python scripts/plot_loss_landscape.py \
    --checkpoint logs/ddpm_20260225_222457/checkpoints/ddpm_0100000.pt \
    --config logs/ddpm_20260225_222457/config.yaml \
    --method ddpm \
    --out-dir logs/ddpm_20260225_222457/loss_landscape
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from src.data import create_dataloader
from src.models import create_model_from_config
from src.methods.ddpm import DDPM
from src.methods.flow_matching import FlowMatching
from src.utils.ema import EMA


def load_config(config_path: Optional[str], ckpt_path: str) -> dict:
    if config_path:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "config" not in ckpt:
        raise ValueError("Checkpoint does not include config. Please pass --config.")
    return ckpt["config"]


def load_checkpoint_model(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model' key: {ckpt_path}")
    model.load_state_dict(ckpt["model"])
    return ckpt


def get_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    config: dict,
    split: str,
    batch_size_override: Optional[int],
    num_workers_override: Optional[int],
    no_augment: bool,
    shuffle: bool,
):
    data_config = config["data"]
    training_config = config["training"]

    batch_size = batch_size_override or training_config["batch_size"]
    num_workers = (
        num_workers_override if num_workers_override is not None else data_config["num_workers"]
    )

    augment = data_config.get("augment", True)
    if no_augment:
        augment = False
    elif split != "train":
        augment = False

    return create_dataloader(
        root=data_config.get("root", "./data/celeba-subset"),
        split=split,
        image_size=data_config["image_size"],
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=data_config["pin_memory"],
        augment=augment,
        shuffle=shuffle,
        drop_last=True,
        from_hub=data_config.get("from_hub", False),
        repo_name=data_config.get("repo_name", "electronickale/cmu-10799-celeba64-subset"),
        use_attributes=data_config.get("use_attributes", False),
        attribute_names=data_config.get("attribute_names"),
    )


def build_method(model, config: dict, method_name: str, device: torch.device):
    if method_name == "ddpm":
        return DDPM.from_config(model, config, device)
    if method_name == "flow_matching":
        return FlowMatching.from_config(model, config, device)
    raise ValueError(f"Unknown method: {method_name}")


def make_direction(params: List[torch.nn.Parameter], seed: int, norm: str) -> List[torch.Tensor]:
    seed_all(seed)
    dirs = []
    for p in params:
        d = torch.randn_like(p)
        if norm == "layer":
            p_norm = torch.norm(p.detach())
            d_norm = torch.norm(d)
            if d_norm > 0:
                d = d * (p_norm / d_norm)
        dirs.append(d)
    return dirs


def apply_directions(
    params: List[torch.nn.Parameter],
    base: List[torch.Tensor],
    d1: List[torch.Tensor],
    d2: List[torch.Tensor],
    alpha: float,
    beta: float,
):
    with torch.no_grad():
        for p, w, v1, v2 in zip(params, base, d1, d2):
            p.data.copy_(w + alpha * v1 + beta * v2)


def collect_batches(
    dataloader,
    num_batches: int,
    device: torch.device,
):
    batches = []
    it = iter(dataloader)
    for _ in range(num_batches):
        batch = next(it)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x0, cond = batch
        else:
            x0, cond = batch, None
        x0 = x0.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True) if cond is not None else None
        batches.append((x0, cond))
    return batches


def evaluate_loss_plane(
    method,
    params: List[torch.nn.Parameter],
    base: List[torch.Tensor],
    d1: List[torch.Tensor],
    d2: List[torch.Tensor],
    batches: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    alphas: np.ndarray,
    betas: np.ndarray,
    seed: int,
    cond_drop_prob: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    z = np.zeros((len(betas), len(alphas)), dtype=np.float64)
    z_attr = None

    has_attr = False
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            apply_directions(params, base, d1, d2, float(alpha), float(beta))
            seed_all(seed)
            total_loss = 0.0
            total_attr = 0.0
            attr_count = 0

            with torch.no_grad():
                for x0, cond in batches:
                    loss, metrics = method.compute_loss(
                        x0, cond=cond, cond_drop_prob=cond_drop_prob
                    )
                    total_loss += float(loss.item())
                    if "loss_attr" in metrics:
                        total_attr += float(metrics["loss_attr"].item())
                        attr_count += 1

            z[i, j] = total_loss / max(1, len(batches))
            if attr_count > 0:
                if z_attr is None:
                    z_attr = np.zeros_like(z)
                z_attr[i, j] = total_attr / attr_count
                has_attr = True

    return z, z_attr if has_attr else None


def save_csv(
    out_path: Path,
    alphas: np.ndarray,
    betas: np.ndarray,
    z: np.ndarray,
    z_attr: Optional[np.ndarray],
):
    import csv

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        headers = ["alpha", "beta", "loss"]
        if z_attr is not None:
            headers.append("loss_attr")
        writer.writerow(headers)
        for i, beta in enumerate(betas):
            for j, alpha in enumerate(alphas):
                row = [f"{alpha:.6f}", f"{beta:.6f}", f"{z[i, j]:.8f}"]
                if z_attr is not None:
                    row.append(f"{z_attr[i, j]:.8f}")
                writer.writerow(row)


def plot_surface(
    out_dir: Path,
    alphas: np.ndarray,
    betas: np.ndarray,
    z: np.ndarray,
    title: str,
    prefix: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f"[warn] matplotlib not available, skipping plots: {e}")
        return

    A, B = np.meshgrid(alphas, betas)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_surface.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    c = ax.contourf(A, B, z, levels=40, cmap="viridis")
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(f"{title} (contour)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_contour.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 2D/3D loss landscape for a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    parser.add_argument("--method", default="ddpm", choices=["ddpm", "flow_matching"])
    parser.add_argument("--out-dir", required=True, help="Output directory for plots and data")
    parser.add_argument("--grid-size", type=int, default=21, help="Grid size per axis")
    parser.add_argument("--range", dest="grid_range", type=float, default=0.5, help="Alpha/Beta range")
    parser.add_argument("--num-batches", type=int, default=2, help="Number of batches per eval point")
    parser.add_argument("--split", default="train", help="Dataset split: train|validation|all")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for eval")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers for eval")
    parser.add_argument("--device", default=None, help="Device string (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for directions and loss eval")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data loader")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available")
    parser.add_argument("--cond-drop-prob", type=float, default=None, help="Override cond_drop_prob")
    parser.add_argument("--direction-norm", default="layer", choices=["layer"], help="Direction normalization")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    config = load_config(args.config, args.checkpoint)

    # Build model and method
    model = create_model_from_config(config).to(device)
    ckpt = load_checkpoint_model(model, args.checkpoint, device)
    method = build_method(model, config, args.method, device)
    method.eval_mode()

    if args.use_ema and "ema" in ckpt:
        ema = EMA(model, decay=config["training"].get("ema_decay", 0.9999))
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        ema.apply_shadow()

    # Build dataloader and preload batches
    dataloader = build_dataloader(
        config=config,
        split=args.split,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        no_augment=args.no_augment,
        shuffle=args.shuffle,
    )
    batches = collect_batches(dataloader, args.num_batches, device=device)

    # Prepare directions and base weights
    params = [p for p in model.parameters() if p.requires_grad]
    base = [p.detach().clone() for p in params]
    d1 = make_direction(params, seed=args.seed, norm=args.direction_norm)
    d2 = make_direction(params, seed=args.seed + 1, norm=args.direction_norm)

    alphas = np.linspace(-args.grid_range, args.grid_range, args.grid_size)
    betas = np.linspace(-args.grid_range, args.grid_range, args.grid_size)

    cond_drop_prob = (
        args.cond_drop_prob
        if args.cond_drop_prob is not None
        else config["training"].get("cond_drop_prob", 0.0)
    )

    z, z_attr = evaluate_loss_plane(
        method=method,
        params=params,
        base=base,
        d1=d1,
        d2=d2,
        batches=batches,
        alphas=alphas,
        betas=betas,
        seed=args.seed,
        cond_drop_prob=cond_drop_prob,
    )

    # Save numeric outputs
    np.save(out_dir / "loss_surface.npy", z)
    if z_attr is not None:
        np.save(out_dir / "loss_attr_surface.npy", z_attr)
    save_csv(out_dir / "loss_surface.csv", alphas, betas, z, z_attr)

    meta = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "method": args.method,
        "grid_size": args.grid_size,
        "range": args.grid_range,
        "num_batches": args.num_batches,
        "split": args.split,
        "batch_size": args.batch_size or config["training"]["batch_size"],
        "seed": args.seed,
        "cond_drop_prob": cond_drop_prob,
        "use_ema": args.use_ema,
    }
    (out_dir / "loss_surface_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    plot_surface(
        out_dir=out_dir,
        alphas=alphas,
        betas=betas,
        z=z,
        title="Loss Landscape",
        prefix="loss",
    )

    if z_attr is not None:
        plot_surface(
            out_dir=out_dir,
            alphas=alphas,
            betas=betas,
            z=z_attr,
            title="Loss_attr Landscape",
            prefix="loss_attr",
        )

    print(f"[ok] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
