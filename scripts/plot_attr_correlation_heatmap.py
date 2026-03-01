#!/usr/bin/env python3
"""
Plot a beautified attribute correlation heatmap.

Reads the correlation matrix CSV produced by `compute_attr_correlations.py`
and plots a smaller, nicer heatmap for the top-K most entangled attributes.

Example:
  python scripts/plot_attr_correlation_heatmap.py \
    --corr logs/attr_correlations.csv \
    --top-k 25 \
    --output logs/attr_correlations_heatmap_top25.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot attribute correlation heatmap")
    ap.add_argument(
        "--corr",
        default="logs/attr_correlations.csv",
        help="Path to correlation CSV (from compute_attr_correlations.py)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of attributes to keep (by mean |corr|, default 25)",
    )
    ap.add_argument(
        "--output",
        default="logs/attr_correlations_heatmap_top25.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--cmap",
        default="RdBu_r",
        help="Matplotlib/Seaborn colormap (default: RdBu_r)",
    )
    args = ap.parse_args()

    corr_path = Path(args.corr)
    if not corr_path.exists():
        raise SystemExit(f"Correlation CSV not found: {corr_path}")

    corr = pd.read_csv(corr_path, index_col=0)

    # Rank attributes by average absolute correlation with others (ignore self-corr).
# Rank attributes by average absolute correlation with others (ignore self-corr).
    abs_corr = corr.abs()
    arr = abs_corr.to_numpy(copy=True)      # make a writable copy
    np.fill_diagonal(arr, 0.0)
    abs_corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    scores = abs_corr.mean(axis=1)
    top_k = min(args.top_k, len(scores))
    top_attrs = scores.sort_values(ascending=False).head(top_k).index
    sub = corr.loc[top_attrs, top_attrs]

    # Nicer style / fonts for poster.
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 8

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    hm = sns.heatmap(
        sub,
        ax=ax,
        cmap=args.cmap,
        center=0.0,
        square=True,
        linewidths=0.3,
        linecolor="lightgray",
        cbar_kws={"shrink": 0.7, "label": "Correlation"},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title("CelebA Attribute Correlations (top {} attrs)".format(top_k), fontsize=10)

    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()

