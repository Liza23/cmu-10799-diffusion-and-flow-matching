#!/usr/bin/env python3
"""
Check that attribute conditioning is working by comparing samples under different conditions.

Generates three grids with the SAME seed:
  1. Unconditional (no attributes)
  2. Condition A (e.g. Eyeglasses)
  3. Condition B (e.g. Blond_Hair, Male)

If conditioning works, the three grids should look different and match the requested attributes.

Usage:
  python scripts/check_conditioning.py --checkpoint path/to/conditional_ddpm_final.pt --method ddpm
  python scripts/check_conditioning.py --checkpoint path/to/conditional_ddpm_final.pt --method ddpm --seed 42 --attr_a "Eyeglasses" --attr_b "Blond_Hair,Male"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models import create_model_from_config
from src.data import save_image, CELEBA_40_ATTRIBUTES
from src.methods import DDPM, FlowMatching
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = create_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(ckpt["ema"])
    return model, config, ema


def build_cond(attr_str: str | None, num_attributes: int, batch_size: int, device: torch.device) -> torch.Tensor | None:
    if not attr_str or not attr_str.strip():
        return None
    attr_to_idx = {n: i for i, n in enumerate(CELEBA_40_ATTRIBUTES)}
    cond = torch.zeros(batch_size, num_attributes, device=device, dtype=torch.float32)
    for name in (a.strip() for a in attr_str.split(",") if a.strip()):
        if name in attr_to_idx and attr_to_idx[name] < num_attributes:
            cond[:, attr_to_idx[name]] = 1.0
    return cond


def main():
    ap = argparse.ArgumentParser(description="Check attribute conditioning")
    ap.add_argument("--checkpoint", required=True, help="Path to conditional checkpoint")
    ap.add_argument("--method", required=True, choices=["ddpm", "flow_matching"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=16, help="Samples per grid")
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--attr_a", type=str, default="", help="Condition A (comma-separated)")
    ap.add_argument("--attr_b", type=str, default="Gray_Hair,Female", help="Condition B (comma-separated)")
    ap.add_argument("--output_dir", type=str, default="conditioning_check")
    ap.add_argument("--no_ema", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Loading {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    if not args.no_ema:
        ema.apply_shadow()

    if args.method == "ddpm":
        method = DDPM.from_config(model, config, device)
    else:
        method = FlowMatching.from_config(model, config, device)
    method.eval_mode()

    use_cond = config.get("model", {}).get("use_conditioning", False)
    if not use_cond:
        print("Warning: checkpoint has use_conditioning=False. Conditioning check may show no difference.")
    num_attrs = config.get("model", {}).get("num_attributes") or config.get("data", {}).get("num_attributes", 40)
    data_config = config["data"]
    image_shape = (data_config["channels"], data_config["image_size"], data_config["image_size"])
    num_steps = args.num_steps or config.get("sampling", {}).get("num_steps", 1000 if args.method == "ddpm" else 100)
    sampler = config.get("sampling", {}).get("sampler", "ddpm")

    os.makedirs(args.output_dir, exist_ok=True)

    conditions = [
        ("unconditional", None),
        ("cond_A_" + args.attr_a.replace(",", "_"), args.attr_a),
        ("cond_B_" + args.attr_b.replace(",", "_"), args.attr_b),
    ]

    for label, attr_str in conditions:
        print(f"Generating {args.num_samples} samples: {label}...")
        cond = build_cond(attr_str, num_attrs, args.num_samples, device) if attr_str else None
        with torch.no_grad():
            samples = method.sample(
                batch_size=args.num_samples,
                image_shape=image_shape,
                num_steps=num_steps,
                sampler=sampler,
                eta=config.get("sampling", {}).get("eta", 0.0),
                cond=cond,
                guidance_scale=args.guidance_scale if cond is not None else 1.0,
            )
        # [0,1] for saving
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0.0, 1.0)
        nrow = max(1, int(args.num_samples ** 0.5))
        path = os.path.join(args.output_dir, f"{label}.png")
        save_image(samples, path, nrow=nrow)
        print(f"  Saved {path}")

    if not args.no_ema:
        ema.restore()

    print(f"\nDone. Check images in {args.output_dir}/")
    print("  - unconditional.png  : no attributes")
    print(f"  - cond_A_*.png       : {args.attr_a}")
    print(f"  - cond_B_*.png      : {args.attr_b}")
    print("If conditioning works, the three grids should look different and match the attributes.")


if __name__ == "__main__":
    main()
