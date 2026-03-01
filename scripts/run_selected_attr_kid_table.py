#!/usr/bin/env python3
"""
Run KID evaluation for a fixed set of 10 CelebA attributes (100 samples each),
then write a CSV and LaTeX table that pairs baseline KID values with newly
measured KID values.

This script wraps scripts/kid_per_attribute.py.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ATTR_BASELINE = [
    ("Blond_Hair", 0.011340),
    ("Double_Chin", 0.009479),
    ("Receding_Hairline", 0.007713),
    ("Male", 0.018930),
    ("No_Beard", 0.024145),
    ("Smiling", 0.008562),
    ("Young", 0.017018),
    ("Black_Hair", 0.015306),
    ("Wavy_Hair", 0.015442),
    ("Wearing_Necktie", 0.017552),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 100 samples for selected attributes and compute KID."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument(
        "--method",
        required=True,
        choices=["ddpm", "flow_matching"],
        help="Sampling method",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/celeba-subset/train/images",
        help="Reference image directory (default: data/celeba-subset/train/images)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: <checkpoint_dir>/kid_selected_attr_100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Samples per attribute (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for generation/KID (default: 64)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Guidance scale passed to kid_per_attribute.py (default: 2.0)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Sampling steps override",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["ddpm", "ddim"],
        help="Sampler override",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to run child script (default: current Python)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration even if existing images exist",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use non-EMA weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reference sampling",
    )
    return parser.parse_args()


def render_value(v: float | None) -> str:
    return f"{v:.6f}" if v is not None else "N/A"


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (checkpoint.parent / "kid_selected_attr_100")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    attributes = [name for name, _ in ATTR_BASELINE]
    attr_arg = ",".join(attributes)

    project_root = Path(__file__).resolve().parent.parent
    kid_script = project_root / "scripts" / "kid_per_attribute.py"
    if not kid_script.exists():
        raise FileNotFoundError(f"Missing script: {kid_script}")

    cmd = [
        args.python_bin,
        str(kid_script),
        "--checkpoint",
        str(checkpoint),
        "--method",
        args.method,
        "--dataset-path",
        args.dataset_path,
        "--output-dir",
        str(out_dir),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--attributes",
        attr_arg,
        "--guidance-scale",
        str(args.guidance_scale),
        "--device",
        args.device,
        "--no-plot",
        "--no-skip-low-count",
    ]
    if args.num_steps is not None:
        cmd += ["--num-steps", str(args.num_steps)]
    if args.sampler is not None:
        cmd += ["--sampler", args.sampler]
    if args.regenerate:
        cmd += ["--regenerate"]
    if args.no_ema:
        cmd += ["--no-ema"]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=project_root)

    kid_csv = out_dir / "kid_per_attribute.csv"
    if not kid_csv.exists():
        raise FileNotFoundError(f"Expected result file not found: {kid_csv}")

    measured: dict[str, float] = {}
    with kid_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("attribute", "").strip()
            m = row.get("kid_mean", "").strip()
            if name and m:
                try:
                    measured[name] = float(m)
                except ValueError:
                    pass

    summary_csv = out_dir / "selected_attr_kid_table.csv"
    rows_tex = out_dir / "selected_attr_kid_rows.tex"
    table_tex = out_dir / "selected_attr_kid_table.tex"

    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attribute", "baseline_kid", "new_kid"])
        for name, baseline in ATTR_BASELINE:
            w.writerow([name, f"{baseline:.6f}", render_value(measured.get(name))])

    with rows_tex.open("w") as f:
        for name, baseline in ATTR_BASELINE:
            disp = name.replace("_", "\\_")
            f.write(f"{disp} & {baseline:.6f} & {render_value(measured.get(name))} \\\\\n")

    with table_tex.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Attribute} & \\textbf{Baseline KID} & \\textbf{New KID (100 samples)} \\\\\n")
        f.write("\\midrule\n")
        for name, baseline in ATTR_BASELINE:
            disp = name.replace("_", "\\_")
            f.write(f"{disp} & {baseline:.6f} & {render_value(measured.get(name))} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Per-attribute KID for selected attributes. Lower is better.}\n")
        f.write("\\end{table}\n")

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {rows_tex}")
    print(f"Wrote: {table_tex}")

    missing = [name for name, _ in ATTR_BASELINE if name not in measured]
    if missing:
        print(
            "Warning: missing measured KID for: " + ", ".join(missing),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
