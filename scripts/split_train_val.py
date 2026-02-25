#!/usr/bin/env python3
"""
Split CelebA-subset train into train (80%) and validation (20%) for classifier eval.

- **Classifier:** train on 80%, evaluate on 20% for realistic accuracy (no train-set overfitting).
- **DDPM (generation):** use all data by setting data.split to "all" in config so it loads train+val.

Creates data/celeba-subset/validation/ with images/ and attributes.csv.
Moves image files so train/images has 80%, validation/images has 20%.
Overwrites train/attributes.csv to only list the 80% train image_ids.

Usage:
  python scripts/split_train_val.py --root data/celeba-subset
  python scripts/split_train_val.py --root data/celeba-subset --val-frac 0.2 --seed 42
  python scripts/split_train_val.py --root data/celeba-subset --dry-run
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Split train into train/validation")
    ap.add_argument("--root", default="data/celeba-subset", help="Dataset root (contains train/)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Fraction for validation (default 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Print plan only, do not move files")
    args = ap.parse_args()

    root = Path(args.root)
    train_dir = root / "train"
    train_images_dir = train_dir / "images"
    train_csv = train_dir / "attributes.csv"
    val_dir = root / "validation"
    val_images_dir = val_dir / "images"

    if not train_dir.exists():
        print(f"Error: {train_dir} not found")
        sys.exit(1)
    if not train_csv.exists():
        print(f"Error: {train_csv} not found")
        sys.exit(1)
    if not train_images_dir.exists():
        print(f"Error: {train_images_dir} not found")
        sys.exit(1)

    # Image files on disk (names like 000002.png)
    image_files = list(train_images_dir.glob("*.png")) or list(train_images_dir.glob("*.jpg"))
    disk_ids = sorted(f.name for f in image_files)

    # CSV: index is image_id (e.g. 000002.png)
    df = pd.read_csv(train_csv, index_col="image_id")
    csv_ids = list(df.index.astype(str))
    # Use intersection so we only split images that have attributes
    common = sorted(set(disk_ids) & set(csv_ids))
    if len(common) == 0:
        # Try without extension match
        csv_stem = {Path(i).stem: i for i in csv_ids}
        common = [csv_stem[Path(d).stem] for d in disk_ids if Path(d).stem in csv_stem]
        disk_ids = common
    else:
        disk_ids = common
    n = len(disk_ids)
    if n == 0:
        print("Error: no images found in both train/images and attributes.csv")
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(disk_ids)
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val
    train_ids = disk_ids[:n_train]
    val_ids = disk_ids[n_train:]

    print(f"Total images: {n}")
    print(f"Train: {n_train} ({100 * n_train / n:.1f}%)")
    print(f"Validation: {n_val} ({100 * n_val / n:.1f}%)")

    if args.dry_run:
        print("Dry run: no files changed.")
        return

    val_images_dir.mkdir(parents=True, exist_ok=True)
    for i, img_id in enumerate(val_ids):
        src = train_images_dir / img_id
        if not src.exists():
            continue
        shutil.move(str(src), str(val_images_dir / img_id))
        if (i + 1) % 5000 == 0:
            print(f"  Moved {i + 1}/{n_val} to validation/")

    # validation/attributes.csv
    df_val = df.loc[val_ids]
    df_val.to_csv(val_dir / "attributes.csv")
    print(f"Wrote {val_dir / 'attributes.csv'} ({len(df_val)} rows)")

    # train/attributes.csv (only train_ids)
    df_train = df.loc[train_ids]
    df_train.to_csv(train_csv)
    print(f"Wrote {train_csv} ({len(df_train)} rows)")

    print("Done. Use --split validation to eval classifier; use split='all' in data config for DDPM to use full data.")


if __name__ == "__main__":
    main()
