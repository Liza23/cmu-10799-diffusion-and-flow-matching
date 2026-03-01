"""
KID Evaluation Script

Generate 1k samples from a trained model, compute KID (Kernel Inception Distance)
vs. the reference dataset using torch-fidelity, and report mean ± std.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate model with KID (1k samples by default). Reports KID mean ± std."
    )
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["ddpm", "flow_matching"],
        help="Method (ddpm or flow_matching)",
    )
    ap.add_argument(
        "--dataset-path",
        type=str,
        default="data/celeba-subset/train/images",
        help="Directory of reference images for KID (default: data/celeba-subset/train/images)",
    )
    ap.add_argument("--num-samples", type=int, default=1000, help="Number of samples (default: 1000)")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for generation (default: 256)")
    ap.add_argument("--num-steps", type=int, default=None, help="Sampling steps (default: from config)")
    ap.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated CelebA attributes for conditional generation (passed to sample.py)",
    )
    ap.add_argument(
        "--neg-attributes",
        "--neg_attributes",
        dest="neg_attributes",
        type=str,
        default=None,
        help="Comma-separated attributes to set to 0 explicitly (passed to sample.py)",
    )
    ap.add_argument(
        "--guidance-scale",
        "--guidance",
        dest="guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (passed to sample.py; --guidance is an alias)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save generated images (default: <checkpoint_dir>/samples/generated)",
    )
    ap.add_argument(
        "--regenerate",
        action="store_true",
        help="Always regenerate samples; default is to reuse if exactly num_samples exist",
    )
    ap.add_argument(
        "--no-ema",
        action="store_true",
        help="Use training weights instead of EMA weights when sampling",
    )
    ap.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.is_dir():
        print(
            f"Error: dataset path must be a directory of images: {dataset_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_images = list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.jpeg"))
    if not ref_images:
        print(
            f"Error: no .png/.jpg/.jpeg images found in {dataset_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    checkpoint_dir = checkpoint.parent
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = str(checkpoint_dir / "samples" / "generated")
    else:
        out_dir = str(Path(out_dir).resolve())
    cache_dir = str(checkpoint_dir / "samples" / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # --- Step 1: Generate samples ---
    need_gen = True
    if not args.regenerate:
        existing = (
            glob.glob(os.path.join(out_dir, "*.png"))
            + glob.glob(os.path.join(out_dir, "*.jpg"))
            + glob.glob(os.path.join(out_dir, "*.jpeg"))
        )
        if len(existing) == args.num_samples:
            print(f"Using {len(existing)} existing samples in {out_dir} (use --regenerate to force)")
            need_gen = False
        elif len(existing) > 0:
            print(f"Found {len(existing)} existing samples but need exactly {args.num_samples}; regenerating.")

    if need_gen:
        print(f"Generating {args.num_samples} samples...")
        if args.guidance_scale is not None and not (args.attributes or args.neg_attributes):
            print(
                "Warning: guidance was set without --attributes/--neg-attributes; generation may still be unconditional.",
                file=sys.stderr,
            )
        os.makedirs(out_dir, exist_ok=True)
        # Clear existing to avoid mixing old/new
        for f in glob.glob(os.path.join(out_dir, "*.png")) + glob.glob(os.path.join(out_dir, "*.jpg")) + glob.glob(os.path.join(out_dir, "*.jpeg")):
            try:
                os.remove(f)
            except OSError:
                pass
        project_root = Path(__file__).resolve().parent
        sample_py = project_root / "sample.py"
        if not sample_py.exists():
            print("Error: sample.py not found next to evaluate.py", file=sys.stderr)
            sys.exit(1)
        cmd = [
            sys.executable,
            str(sample_py),
            "--checkpoint", str(checkpoint),
            "--method", args.method,
            "--output_dir", out_dir,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
        ]
        if args.no_ema:
            cmd += ["--no_ema"]
        if args.num_steps is not None:
            cmd += ["--num_steps", str(args.num_steps)]
        if args.attributes:
            cmd += ["--attributes", args.attributes]
        if args.neg_attributes:
            cmd += ["--neg-attributes", args.neg_attributes]
        if args.guidance_scale is not None:
            cmd += ["--guidance_scale", str(args.guidance_scale)]
        subprocess.run(cmd, check=True, cwd=project_root)
        print(f"Saved samples to {out_dir}")

    # --- Step 2: KID via torch-fidelity ---
    try:
        import torch_fidelity
    except ImportError:
        print("Error: torch-fidelity not found. Install with: pip install torch-fidelity", file=sys.stderr)
        sys.exit(1)

    print("Computing KID...")
    metrics = torch_fidelity.calculate_metrics(
        input1=out_dir,
        input2=str(dataset_path),
        cuda=(args.device == "cuda"),
        kid=True,
        fid=False,
        isc=False,
        verbose=True,
        cache_root=cache_dir,
        batch_size=args.batch_size,
    )

    kid_mean = metrics["kernel_inception_distance_mean"]
    kid_std = metrics["kernel_inception_distance_std"]

    print()
    print("========================================")
    print(f"KID Evaluation ({args.num_samples} samples)")
    print("========================================")
    print(f"  KID mean: {kid_mean:.6f}")
    print(f"  KID std:  {kid_std:.6f}")
    print(f"  KID:      {kid_mean:.6f} ± {kid_std:.6f}")
    print("========================================")


if __name__ == "__main__":
    main()
