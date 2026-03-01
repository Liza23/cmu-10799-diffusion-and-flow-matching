#!/usr/bin/env python3
"""Compute attribute correlation matrix for CelebA train attributes.

Reads `attributes.csv` (40 binary attributes) and computes the
Pearson correlation matrix (equivalent to phi coefficient
for binary variables).

Usage:
  python scripts/compute_attr_correlations.py \
    --csv data/celeba-subset/train/attributes.csv \
    --output logs/attr_correlations.csv

Options:
  --min-positive: only keep attributes with at least this many
                  positive examples (>0 by default).
  --top-k: print the top-k most correlated attribute pairs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute CelebA attribute correlation matrix")
    ap.add_argument(
        "--csv",
        default="data/celeba-subset/train/attributes.csv",
        help="Path to attributes.csv",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Optional path to save correlation matrix as CSV",
    )
    ap.add_argument(
        "--min-positive",
        type=int,
        default=1,
        help="Min positive count to keep an attribute (default 1)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Print top-k most correlated pairs (by |corr|)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image_id" in df.columns:
        df = df.drop(columns=["image_id"]).copy()

    # Only keep attributes that actually appear (sum >= min_positive)
    pos_counts = df.sum(axis=0)
    keep = pos_counts[pos_counts >= args.min_positive].index
    df = df[keep]

    print(
        f"Loaded {len(df)} samples, {df.shape[1]} attributes after filtering "
        f"(min_positive={args.min_positive})."
    )

    # Pearson correlation matrix
    corr = df.corr()

    # Optionally save full matrix
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(out_path)
        print(f"Saved correlation matrix to {out_path}")

    # Print top-k correlated attribute pairs (excluding self)
    names = corr.columns
    triu_idx = np.triu_indices(len(names), k=1)
    vals = corr.values[triu_idx]
    pairs = []
    for v, i, j in zip(vals, triu_idx[0], triu_idx[1]):
        pairs.append((abs(v), v, names[i], names[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])

    k = min(args.top_k, len(pairs))
    print(f"\nTop {k} attribute correlations (by |corr|):")
    for abs_v, v, a, b in pairs[:k]:
        print(f"  {a:20s} - {b:20s}: corr = {v:+.3f}")


if __name__ == "__main__":
    main()

