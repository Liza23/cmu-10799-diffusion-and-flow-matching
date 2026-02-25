#!/usr/bin/env python3
"""
Compute KID (Kernel Inception Distance) per attribute.

For each attribute, generates N samples (default 100) conditioned on that attribute,
saves them to a subdir, then runs torch-fidelity KID vs. a reference set of *real*
images that have that attribute (not the full dataset). Reference is built from
attributes.csv next to the image dir (train/attributes.csv when dataset-path is train/images).
Output: CSV, summary table, and bar chart (KID mean ± std per attribute).

Resumable: if an attribute's folder already has >= num_samples images, generation is
skipped and KID is computed on existing images. Use --regenerate to force full regeneration.

Training-data check: before generation/KID, the script counts how many training samples
have each attribute. Attributes present in <= min_pct of the training set are skipped (default: 10%)
(so KID is only reported for well-represented attributes). Use --no-skip-low-count to disable.

Requires: torch-fidelity, matplotlib, conditional checkpoint (use_conditioning=True).

Usage:
  python scripts/kid_per_attribute.py --checkpoint path/to/conditional.pt --method ddpm --dataset-path data/celeba-subset/train/images
  python scripts/kid_per_attribute.py --checkpoint path/to/conditional.pt --method ddpm --attributes "Brown_Hair,Blond_Hair,Male" --num-samples 500 --no-plot
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CELEBA_40_ATTRIBUTES, save_image, create_dataloader_from_config
from src.methods import DDPM, FlowMatching
from src.models import create_model_from_config
from src.utils import EMA

try:
    from PIL import Image
except ImportError:
    Image = None


def remove_corrupt_images(dir_path: Path) -> int:
    """Remove image files that cannot be opened as RGB (corrupt). Returns number removed."""
    if Image is None:
        return 0
    removed = 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in dir_path.glob(ext):
            try:
                with Image.open(f) as img:
                    img.convert("RGB")
            except Exception:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def get_ref_paths_for_attribute(
    ref_images_dir: Path,
    attributes_csv_path: Path,
    attr_name: str,
    max_refs: int,
    rng: Optional[random.Random] = None,
) -> Optional[List[Path]]:
    """
    Return paths to real images that have the given attribute = 1.
    Randomly samples max_refs from the set of all training images with that attribute.
    CSV must have image_id index and a column matching attr_name (or with spaces as underscores).
    Returns None if CSV missing or attribute column not found.
    """
    if not attributes_csv_path.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    df = pd.read_csv(attributes_csv_path, index_col="image_id")
    # Column might be attr_name or with spaces
    col = attr_name
    if col not in df.columns:
        col = attr_name.replace("_", " ")
    if col not in df.columns:
        return None
    # All rows where attribute == 1, then sample n=max_refs
    all_ids = df.index[df[col].astype(int) == 1].tolist()
    if len(all_ids) < max_refs:
        ids = all_ids
    else:
        ids = (rng or random).sample(all_ids, max_refs)
    paths = []
    for i in ids:
        name = str(i)
        p = ref_images_dir / name
        if p.exists():
            paths.append(p)
            continue
        # CSV may have .jpg index but files saved as .png
        alt = (name.replace(".jpg", ".png") if name.endswith(".jpg") else name + ".png")
        p = ref_images_dir / alt
        if p.exists():
            paths.append(p)
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            if name.endswith(ext):
                continue
            p = ref_images_dir / (name + ext)
            if p.exists():
                paths.append(p)
                break
    return paths if paths else None


def get_training_attribute_counts(config, attr_order, num_attrs):
    """
    Load training dataset and count how many samples have each attribute = 1.
    Returns (dict attr_name -> count, total_size) or (None, 0) if dataset cannot be loaded.
    """
    data_cfg = config.get("data", {})
    if not data_cfg.get("use_attributes", False):
        return None, 0
    try:
        dl = create_dataloader_from_config(config, split="train")
        ds = dl.dataset
        if not getattr(ds, "num_attributes", 0):
            return None, 0
        total = len(ds)
        counts = [0] * num_attrs
        for batch in dl:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                attrs = batch[1]  # (B, num_attrs)
                for i in range(num_attrs):
                    if i < attrs.shape[1]:
                        counts[i] += (attrs[:, i] >= 0.5).sum().item()
        return {attr_order[i]: counts[i] for i in range(min(len(attr_order), num_attrs))}, total
    except Exception as e:
        print(f"Warning: could not load training data for attribute counts: {e}", file=sys.stderr)
        return None, 0


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = create_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(ckpt["ema"])
    return model, config, ema


def build_cond_one_attribute(attr_index: int, num_attributes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    cond = torch.zeros(batch_size, num_attributes, device=device, dtype=torch.float32)
    cond[:, attr_index] = 1.0
    return cond


def main():
    ap = argparse.ArgumentParser(description="Compute KID per attribute (conditional generation)")
    ap.add_argument("--checkpoint", required=True, help="Path to conditional checkpoint")
    ap.add_argument("--method", required=True, choices=["ddpm", "flow_matching"])
    ap.add_argument("--dataset-path", required=True, help="Reference image directory (e.g. train/images). KID uses real images with that attribute when attributes.csv is available.")
    ap.add_argument("--attributes-csv", default=None, help="Path to attributes.csv (default: <dataset-path>/../attributes.csv)")
    ap.add_argument("--output-dir", default=None, help="Base dir for per-attr samples and results (default: <checkpoint_dir>/kid_per_attr)")
    ap.add_argument("--num-samples", type=int, default=500, help="Samples per attribute (default: 100)")
    ap.add_argument("--batch-size", type=int, default=64, help="Generation batch size")
    ap.add_argument("--attributes", type=str, default=None, help="Comma-separated attributes (default: all 40)")
    ap.add_argument("--guidance-scale", type=float, default=2.0, help="Classifier-free guidance scale")
    ap.add_argument("--num-steps", type=int, default=None, help="Sampling steps (default: from config; use fewer with DDIM for speed)")
    ap.add_argument("--sampler", type=str, default=None, choices=["ddpm", "ddim"], help="Sampler (default: from config; ddim is faster)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--regenerate", action="store_true", help="Force regeneration for all attributes (default: skip generation if folder already has num_samples images)")
    ap.add_argument("--no-plot", action="store_true", help="Do not create bar chart (only CSV and table)")
    ap.add_argument("--no-skip-low-count", action="store_true", help="Do not skip attributes by prevalence (default: skip if <= min-pct)")
    ap.add_argument("--min-pct", type=float, default=5.0, help="Skip attributes present in <= this pct of training data (default: 10)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling ref images (default: none)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)
    ref_path = Path(args.dataset_path).resolve()
    if not ref_path.is_dir():
        print(f"Error: dataset-path must be a directory: {ref_path}", file=sys.stderr)
        sys.exit(1)
    attributes_csv = Path(args.attributes_csv).resolve() if args.attributes_csv else ref_path.parent / "attributes.csv"
    if attributes_csv.exists():
        print(f"Using attribute-specific reference: {attributes_csv} (KID vs. real images with that attribute)")
    else:
        print(f"Note: no attributes.csv at {attributes_csv}; KID will use full reference directory (all images)")

    out_base = args.output_dir or str(checkpoint.parent / "kid_per_attr")
    out_base = Path(out_base)
    os.makedirs(out_base, exist_ok=True)

    rng = random.Random(args.seed) if args.seed is not None else None

    print("Loading checkpoint...")
    model, config, ema = load_checkpoint(str(checkpoint), device)
    if not args.no_ema:
        ema.apply_shadow()

    if args.method == "ddpm":
        method = DDPM.from_config(model, config, device)
    else:
        method = FlowMatching.from_config(model, config, device)
    method.eval_mode()

    use_cond = config.get("model", {}).get("use_conditioning", False)
    if not use_cond:
        print("Warning: checkpoint has use_conditioning=False. KID will be unconditional per run.", file=sys.stderr)

    num_attrs = config.get("model", {}).get("num_attributes") or config.get("data", {}).get("num_attributes", 40)
    attr_order = config.get("data", {}).get("attribute_names") or CELEBA_40_ATTRIBUTES
    if args.attributes:
        attr_names = [a.strip() for a in args.attributes.split(",") if a.strip()]
        # Resolve to indices in attr_order
        attr_to_idx = {n: i for i, n in enumerate(attr_order)}
        indices = []
        for n in attr_names:
            if n in attr_to_idx:
                indices.append(attr_to_idx[n])
            else:
                print(f"Warning: unknown attribute '{n}' skipped.", file=sys.stderr)
        attr_list = [(i, attr_order[i]) for i in sorted(indices)]
    else:
        attr_list = [(i, attr_order[i]) for i in range(min(num_attrs, len(attr_order)))]

    data_config = config["data"]
    image_shape = (data_config["channels"], data_config["image_size"], data_config["image_size"])
    num_steps = args.num_steps or config.get("sampling", {}).get("num_steps", 1000 if args.method == "ddpm" else 100)
    sampler = args.sampler or config.get("sampling", {}).get("sampler", "ddpm")

    # Count how many training samples have each attribute (skip KID if attribute is rare)
    attr_counts = None
    train_total = 0
    if not args.no_skip_low_count:
        print("Counting attribute prevalence in training data...")
        attr_counts, train_total = get_training_attribute_counts(config, attr_order, num_attrs)
        if attr_counts and train_total > 0:
            for name, c in list(attr_counts.items())[:5]:
                pct = 100.0 * c / train_total
                print(f"  {name}: {c} ({pct:.1f}%)")
            if len(attr_counts) > 5:
                print(f"  ... and {len(attr_counts) - 5} more attributes")
            print(f"  Skip attributes with prevalence <= {args.min_pct}% (--min-pct)")
        else:
            print("  (skipped; could not load training data or no attributes)")

    results = []

    for attr_idx, attr_name in attr_list:
        # Skip if attribute is present in <= min_pct of training data (rare attributes)
        if attr_counts is not None and train_total > 0 and attr_name in attr_counts:
            train_count = attr_counts[attr_name]
            pct = 100.0 * train_count / train_total
            if pct <= args.min_pct:
                print(f"[{attr_name}] Skipping: present in {pct:.1f}% of training data (<= {args.min_pct}%)")
                continue

        attr_dir = out_base / f"samples_{attr_name.replace(' ', '_')}"
        os.makedirs(attr_dir, exist_ok=True)
        existing = list(attr_dir.glob("*.png")) + list(attr_dir.glob("*.jpg"))
        n_existing = len(existing)
        # Skip generation if we already have enough samples (unless --regenerate)
        if not args.regenerate and n_existing >= args.num_samples:
            print(f"[{attr_name}] Found {n_existing} existing images (>= {args.num_samples}), skipping generation")
            need_gen = False
        else:
            need_gen = True
            if existing:
                for f in existing:
                    try:
                        f.unlink()
                    except OSError:
                        pass

        if need_gen:
            print(f"[{attr_name}] Generating {args.num_samples} samples...")
            generated = 0
            with torch.no_grad():
                while generated < args.num_samples:
                    batch_size = min(args.batch_size, args.num_samples - generated)
                    cond = build_cond_one_attribute(attr_idx, num_attrs, batch_size, device)
                    samples = method.sample(
                        batch_size=batch_size,
                        image_shape=image_shape,
                        num_steps=num_steps,
                        sampler=sampler,
                        eta=config.get("sampling", {}).get("eta", 0.0),
                        cond=cond,
                        guidance_scale=args.guidance_scale,
                    )
                    samples = (samples + 1.0) / 2.0
                    samples = samples.clamp(0.0, 1.0)
                    for i in range(samples.shape[0]):
                        path = attr_dir / f"{generated + i:06d}.png"
                        save_image(samples[i : i + 1], str(path), nrow=1)
                    generated += batch_size
            print(f"  Saved to {attr_dir}")

        # KID: compare to real images that have this attribute (or full ref if no CSV / not enough)
        try:
            import torch_fidelity
        except ImportError:
            print("Error: torch-fidelity required. pip install torch-fidelity", file=sys.stderr)
            sys.exit(1)

        ref_dir = None
        if attributes_csv.exists():
            max_refs = args.num_samples  # 100 gens vs 100 real with that attribute
            ref_paths = get_ref_paths_for_attribute(ref_path, attributes_csv, attr_name, max_refs, rng=rng)
            if ref_paths and len(ref_paths) >= 10:
                ref_attr_dir = out_base / "ref" / attr_name.replace(" ", "_")
                ref_attr_dir.mkdir(parents=True, exist_ok=True)
                for fp in ref_paths:
                    lnk = ref_attr_dir / fp.name
                    if not lnk.exists():
                        try:
                            lnk.symlink_to(fp.resolve())
                        except OSError:
                            import shutil
                            shutil.copy2(fp, lnk)
                ref_dir = ref_attr_dir
                print(f"[{attr_name}] Reference: {len(ref_paths)} real images (sampled from training with this attribute)")
            else:
                print(f"[{attr_name}] Warning: not enough attribute-specific refs ({len(ref_paths) if ref_paths else 0}); using full reference dir")
        if ref_dir is None:
            ref_dir = ref_path

        # Remove corrupt images so torch_fidelity does not fail
        n_removed = remove_corrupt_images(attr_dir)
        if n_removed:
            print(f"[{attr_name}] Removed {n_removed} corrupt image(s) from samples dir")
        n_removed_ref = remove_corrupt_images(ref_dir)
        if n_removed_ref:
            print(f"[{attr_name}] Removed {n_removed_ref} corrupt image(s) from ref dir")

        n_gen = len(list(attr_dir.glob("*.png"))) + len(list(attr_dir.glob("*.jpg")))
        n_ref = len(ref_paths) if (ref_paths and ref_dir != ref_path) else None
        min_need = 10
        if n_gen < min_need:
            print(f"[{attr_name}] Skipping KID: only {n_gen} valid images (need >= {min_need})")
            continue
        if n_ref is not None and n_ref < min_need:
            print(f"[{attr_name}] Skipping KID: only {n_ref} valid ref images (need >= {min_need})")
            continue

        print(f"[{attr_name}] Computing KID...")
        cache_dir = str(out_base / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        # When using attribute-specific ref we have a small ref set; cap kid_subset_size so both sides have enough
        kid_subset_size = min(n_gen, n_ref) if n_ref is not None else min(n_gen, 1000)
        kid_subset_size = max(min_need, kid_subset_size)
        kid_kwargs = {"kid_subset_size": kid_subset_size} if kid_subset_size < 1000 else {}
        metrics = torch_fidelity.calculate_metrics(
            input1=str(attr_dir),
            input2=str(ref_dir),
            cuda=(device.type == "cuda"),
            kid=True,
            fid=False,
            isc=False,
            verbose=False,
            batch_size=args.batch_size,
            cache_root=cache_dir,
            **kid_kwargs,
        )
        kid_mean = metrics["kernel_inception_distance_mean"]
        kid_std = metrics["kernel_inception_distance_std"]
        train_count = attr_counts.get(attr_name, "") if attr_counts else ""
        results.append((attr_name, kid_mean, kid_std, train_count))
        print(f"  KID: {kid_mean:.6f} ± {kid_std:.6f}")

    if not args.no_ema:
        ema.restore()

    # Write CSV
    csv_path = out_base / "kid_per_attribute.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attribute", "kid_mean", "kid_std", "train_count"])
        for row in results:
            name, m, s, count = row[0], row[1], row[2], row[3]
            w.writerow([name, f"{m:.6f}", f"{s:.6f}", count if count != "" else ""])
    print(f"\nWrote {csv_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("KID per attribute (vs. reference)")
    print("=" * 70)
    print(f"{'Attribute':<25} {'KID mean':>12} {'KID std':>10} {'train_count':>12}")
    print("-" * 70)
    for row in results:
        name, m, s, count = row[0], row[1], row[2], row[3]
        print(f"{name:<25} {m:>12.6f} {s:>10.6f} {count:>12}")
    print("=" * 70)

    # Bar chart (and optional pie-style: share of total KID per attribute)
    if not args.no_plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Skipping plot (matplotlib not installed). Install with: pip install matplotlib", file=sys.stderr)
        else:
            names = [r[0] for r in results]
            means = np.array([r[1] for r in results])
            stds = np.array([r[2] for r in results])
            # r[3] is train_count (not used in plots)
            n = len(names)
            # Horizontal bar chart (readable for many attributes)
            fig, ax = plt.subplots(figsize=(10, max(6, n * 0.25)))
            y_pos = np.arange(n)
            bars = ax.barh(y_pos, means, xerr=stds, capsize=2, color="steelblue", alpha=0.85, error_kw={"elinewidth": 1})
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("KID (mean ± std)")
            ax.set_ylabel("Attribute")
            ax.set_title("KID per attribute (vs. reference)")
            ax.invert_yaxis()
            plt.tight_layout()
            plot_path = out_base / "kid_per_attribute.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved bar chart: {plot_path}")

            # Optional: pie chart of "share" (KID mean as share of sum - for relative comparison)
            # Pie is less standard for KID but user asked for it; we'll add a second plot.
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            total = means.sum()
            if total > 0:
                shares = means / total
                # Only show labels for attributes with share > 2% to avoid clutter
                labels = [n if s >= 0.02 else "" for n, s in zip(names, shares)]
                wedges, texts, autotexts = ax2.pie(
                    means, labels=labels, autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
                    startangle=90, pctdistance=0.75
                )
                for t in texts:
                    t.set_fontsize(7)
            ax2.set_title("KID share per attribute (fraction of total KID)")
            plt.tight_layout()
            pie_path = out_base / "kid_per_attribute_pie.png"
            plt.savefig(pie_path, dpi=150)
            plt.close()
            print(f"Saved pie chart: {pie_path}")


if __name__ == "__main__":
    main()
