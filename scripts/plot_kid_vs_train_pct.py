#!/usr/bin/env python3
"""
Plot KID (y-axis) vs train_pct (x-axis) from a kid_per_attribute-style CSV.
Excludes attributes with 0 train_pct by default.

Usage:
  python scripts/plot_kid_vs_train_pct.py new.csv
  python scripts/plot_kid_vs_train_pct.py new.csv -o kid_vs_pct.png
"""

import argparse
import csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV with columns attribute, kid_mean, kid_std, train_pct")
    ap.add_argument("-o", "--output", default=None, help="Output image path (default: <csv_stem>_kid_vs_pct.png)")
    ap.add_argument("--no-labels", action="store_true", help="Do not plot attribute labels (less clutter)")
    ap.add_argument("--label-threshold", type=float, default=0, help="Only label points with train_pct >= this (default: 0)")
    ap.add_argument("--include-zero-pct", action="store_true", help="Include attributes with 0%% train_pct (default: exclude them)")
    ap.add_argument("--min-pct", type=float, default=1.0, help="Exclude attributes with train_pct < this (default: 1). Use 0 to include 0%%.")
    args = ap.parse_args()

    min_pct = 0.0 if args.include_zero_pct else args.min_pct

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")

    rows = []
    skipped_zero = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attr = row.get("attribute", "").strip()
            if not attr:
                continue
            try:
                kid_mean = float(row.get("kid_mean", 0))
                kid_std = float(row.get("kid_std", 0))
                pct_str = (row.get("train_pct") or "0").replace("%", "").strip()
                train_pct = float(pct_str) if pct_str else 0.0
            except (ValueError, TypeError):
                continue
            # Exclude attributes with train_pct below threshold (default 1%% so 0%% and <1%% are out)
            if train_pct < min_pct:
                skipped_zero += 1
                continue
            rows.append((attr, kid_mean, kid_std, train_pct))

    if skipped_zero:
        print(f"Excluded {skipped_zero} attributes (train_pct < {min_pct}%%).")
    if not rows:
        raise SystemExit("No valid rows in CSV (after filtering). Use --include-zero-pct to include 0%% attributes).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required: pip install matplotlib")

    attrs, kid_means, kid_stds, train_pcts = zip(*rows)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    # Apply Times New Roman after style so it sticks
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("#fafafa")

    ax.errorbar(
        train_pcts, kid_means, yerr=kid_stds,
        fmt="o", capsize=2.5, capthick=1,
        markersize=8, alpha=0.9,
        color="#2563eb", ecolor="#64748b", elinewidth=1,
        markeredgecolor="white", markeredgewidth=0.8,
    )
    ax.set_xlabel("Train % (prevalence in training set)", fontsize=12, fontweight="500")
    ax.set_ylabel("KID (mean)", fontsize=12, fontweight="500")
    ax.set_title("KID per attribute vs. training set prevalence", fontsize=14, fontweight="600", pad=12)
    ax.tick_params(axis="both", labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")
        label.set_fontname("Times New Roman")
    ax.xaxis.get_label().set_fontfamily("serif")
    ax.xaxis.get_label().set_fontname("Times New Roman")
    ax.yaxis.get_label().set_fontfamily("serif")
    ax.yaxis.get_label().set_fontname("Times New Roman")
    ax.title.set_fontfamily("serif")
    ax.title.set_fontname("Times New Roman")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35, linestyle="-")
    ax.set_axisbelow(True)

    if not args.no_labels:
        for attr, x, y in zip(attrs, train_pcts, kid_means):
            if x >= args.label_threshold:
                ax.annotate(
                    attr.replace("_", " "),
                    (x, y),
                    fontsize=7,
                    fontfamily="Times New Roman",
                    alpha=0.9,
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontweight="500",
                )

    fig.tight_layout()
    out = args.output or csv_path.with_name(csv_path.stem + "_kid_vs_pct.png")
    out = Path(out)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out} ({len(rows)} points)")


if __name__ == "__main__":
    main()
