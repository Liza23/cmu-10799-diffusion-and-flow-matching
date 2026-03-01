#!/usr/bin/env python3
"""
Run KID evaluation for auxiliary-loss sweeps and write LaTeX tables.

Computes, for each configured checkpoint:
1) Mean KID (unconditional generation)
2) Blond_Hair KID (conditional generation with guidance)

Then writes:
- CSV summary
- lambda sweep LaTeX table
- method-ablation LaTeX table
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class KidStats:
    mean: float
    std: float


@dataclass
class VariantResult:
    label: str
    checkpoint: Path
    mean_kid: KidStats
    blond_kid: KidStats


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="Run KID sweeps for lambda and auxiliary-loss ablations.")
    p.add_argument("--method", default="ddpm", choices=["ddpm", "flow_matching"])
    p.add_argument("--dataset-path", default="data/celeba-subset/train/images")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--guidance",
        "--guidance-scale",
        dest="guidance",
        type=float,
        default=3.0,
        help="Guidance scale for Blond_Hair conditional KID (default: 3.0)",
    )
    p.add_argument(
        "--output-root",
        default=f"logs/auxloss_kid_tables_{ts}",
        help="Directory for generated samples, logs, CSV, and LaTeX tables",
    )
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--regenerate", action="store_true", help="Force sample regeneration")
    p.add_argument("--no-ema", action="store_true")

    # Lambda sweep defaults (based on runs currently in logs/)
    p.add_argument("--ckpt-baseline", default="logs/ddpm_20260222_052928")
    p.add_argument("--ckpt-lambda-002", default="logs/ddpm_20260227_191007")
    p.add_argument("--ckpt-lambda-005", default="logs/ddpm_20260225_222457")
    p.add_argument("--ckpt-lambda-010", default="logs/ddpm_20260225_184611")

    # Ablation table defaults
    p.add_argument(
        "--ckpt-t2",
        default=None,
        help="Checkpoint for '+ Auxiliary Loss (t < T/2)'. Default: --ckpt-lambda-010",
    )
    p.add_argument(
        "--ckpt-t10",
        default="logs/ddpm_20260226_014716",
        help="Checkpoint for '+ Auxiliary Loss (t < T/10)'",
    )
    return p.parse_args()


def resolve_checkpoint(path_or_run: str) -> Path:
    """Accepts checkpoint file, run dir, or checkpoints dir and resolves to one .pt file."""
    p = Path(path_or_run).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        return p

    # Directory resolution
    checkpoints_dir = p / "checkpoints" if (p / "checkpoints").is_dir() else p
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Could not find checkpoints directory in: {p}")

    final_candidates = sorted(checkpoints_dir.glob("*_final.pt"))
    if final_candidates:
        return final_candidates[0]

    pt_candidates = sorted(checkpoints_dir.glob("*.pt"))
    if not pt_candidates:
        raise FileNotFoundError(f"No checkpoint .pt files found in: {checkpoints_dir}")

    # Prefer highest numbered checkpoint like ddpm_0120000.pt
    def score(path: Path) -> tuple[int, int]:
        m = re.search(r"_(\d+)\.pt$", path.name)
        if m:
            return (1, int(m.group(1)))
        return (0, 0)

    return max(pt_candidates, key=score)


def run_eval(
    *,
    python_bin: str,
    repo_root: Path,
    checkpoint: Path,
    method: str,
    dataset_path: str,
    num_samples: int,
    batch_size: int,
    num_steps: int | None,
    device: str,
    output_dir: Path,
    log_path: Path,
    regenerate: bool,
    no_ema: bool,
    attributes: str | None,
    guidance: float | None,
) -> KidStats:
    cmd = [
        python_bin,
        "evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--method",
        method,
        "--dataset-path",
        dataset_path,
        "--num-samples",
        str(num_samples),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if num_steps is not None:
        cmd += ["--num-steps", str(num_steps)]
    if regenerate:
        cmd += ["--regenerate"]
    if no_ema:
        cmd += ["--no-ema"]
    if attributes:
        cmd += ["--attributes", attributes]
    if guidance is not None:
        cmd += ["--guidance", str(guidance)]

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    log_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))

    mean_match = re.search(r"KID mean:\s*([0-9]*\.?[0-9]+)", proc.stdout)
    std_match = re.search(r"KID std:\s*([0-9]*\.?[0-9]+)", proc.stdout)
    if not mean_match or not std_match:
        raise RuntimeError(
            f"Could not parse KID from output for checkpoint {checkpoint}. See {log_path}"
        )

    return KidStats(mean=float(mean_match.group(1)), std=float(std_match.group(1)))


def fmt_pm(stats: KidStats) -> str:
    return f"{stats.mean:.6f} ± {stats.std:.6f}"


def write_csv(path: Path, rows: list[VariantResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "checkpoint",
                "mean_kid_mean",
                "mean_kid_std",
                "blond_hair_kid_mean",
                "blond_hair_kid_std",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.label,
                    str(r.checkpoint),
                    f"{r.mean_kid.mean:.6f}",
                    f"{r.mean_kid.std:.6f}",
                    f"{r.blond_kid.mean:.6f}",
                    f"{r.blond_kid.std:.6f}",
                ]
            )


def write_lambda_table(path: Path, by_label: dict[str, VariantResult]) -> None:
    # Exactly the row labels requested by user.
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{$\\lambda$ (attr\\_loss\\_weight)} & \\textbf{Mean KID} & \\textbf{Blond\\_Hair KID} \\\\",
        "\\midrule",
        f"0 (baseline) & {by_label['lambda_0_baseline'].mean_kid.mean:.6f} & {by_label['lambda_0_baseline'].blond_kid.mean:.6f} \\\\",
        f"0.02 & {by_label['lambda_0.02'].mean_kid.mean:.6f} & {by_label['lambda_0.02'].blond_kid.mean:.6f} \\\\",
        f"0.05 & {by_label['lambda_0.05'].mean_kid.mean:.6f} & {by_label['lambda_0.05'].blond_kid.mean:.6f} \\\\",
        f"0.10 & {by_label['lambda_0.10'].mean_kid.mean:.6f} & {by_label['lambda_0.10'].blond_kid.mean:.6f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Hyperparameter sweep over auxiliary loss weight $\\lambda$. Lower KID is better.}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def write_ablation_table(path: Path, by_label: dict[str, VariantResult]) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Method Variant} & \\textbf{Mean KID} & \\textbf{Blond\\_Hair KID} & \\textbf{Notes} \\\\",
        "\\midrule",
        f"Baseline (HW3, no aux loss) & ${fmt_pm(by_label['lambda_0_baseline'].mean_kid)}$ & ${fmt_pm(by_label['lambda_0_baseline'].blond_kid)}$ & Standard CFG training \\\\",
        f"+ Auxiliary Loss ($t < T/2$) & ${fmt_pm(by_label['t_lt_T2'].mean_kid)}$ & ${fmt_pm(by_label['t_lt_T2'].blond_kid)}$ & Proposed method \\\\",
        f"+ Auxiliary Loss ($t < T/10$) & ${fmt_pm(by_label['t_lt_T10'].mean_kid)}$ & ${fmt_pm(by_label['t_lt_T10'].blond_kid)}$ & Aggressive gating \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Component ablation of the proposed auxiliary-loss method. Lower KID is better.}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_t2_arg = args.ckpt_t2 or args.ckpt_lambda_010

    # Default mapping for both requested tables.
    variants = [
        ("lambda_0_baseline", args.ckpt_baseline),
        ("lambda_0.02", args.ckpt_lambda_002),
        ("lambda_0.05", args.ckpt_lambda_005),
        ("lambda_0.10", args.ckpt_lambda_010),
        ("t_lt_T2", ckpt_t2_arg),   # per user note: use lambda=0.1 score if needed
        ("t_lt_T10", args.ckpt_t10),
    ]

    # Deduplicate evaluations if multiple labels map to same checkpoint.
    resolved: dict[str, Path] = {}
    for label, spec in variants:
        resolved[label] = resolve_checkpoint(spec)

    by_ckpt_key: dict[str, tuple[KidStats, KidStats]] = {}
    by_label: dict[str, VariantResult] = {}

    for label, _ in variants:
        ckpt = resolved[label]
        ckpt_key = str(ckpt)
        label_dir = out_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        if ckpt_key not in by_ckpt_key:
            mean_stats = run_eval(
                python_bin=args.python_bin,
                repo_root=repo_root,
                checkpoint=ckpt,
                method=args.method,
                dataset_path=args.dataset_path,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                device=args.device,
                output_dir=label_dir / "samples_mean",
                log_path=label_dir / "mean_eval.log",
                regenerate=args.regenerate,
                no_ema=args.no_ema,
                attributes=None,
                guidance=None,
            )
            blond_stats = run_eval(
                python_bin=args.python_bin,
                repo_root=repo_root,
                checkpoint=ckpt,
                method=args.method,
                dataset_path=args.dataset_path,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                device=args.device,
                output_dir=label_dir / "samples_blond_hair",
                log_path=label_dir / "blond_eval.log",
                regenerate=args.regenerate,
                no_ema=args.no_ema,
                attributes="Blond_Hair",
                guidance=args.guidance,
            )
            by_ckpt_key[ckpt_key] = (mean_stats, blond_stats)

        m, b = by_ckpt_key[ckpt_key]
        by_label[label] = VariantResult(label=label, checkpoint=ckpt, mean_kid=m, blond_kid=b)

    all_rows = [by_label[label] for label, _ in variants]
    csv_path = out_root / "auxloss_kid_summary.csv"
    lambda_tex = out_root / "table_lambda_sweep.tex"
    ablation_tex = out_root / "table_aux_ablation.tex"
    write_csv(csv_path, all_rows)
    write_lambda_table(lambda_tex, by_label)
    write_ablation_table(ablation_tex, by_label)

    # Also write a compact metadata file so run->checkpoint resolution is explicit.
    meta_path = out_root / "resolved_checkpoints.txt"
    with meta_path.open("w") as f:
        for label, _ in variants:
            f.write(f"{label}: {by_label[label].checkpoint}\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved LaTeX: {lambda_tex}")
    print(f"Saved LaTeX: {ablation_tex}")
    print(f"Saved mapping: {meta_path}")


if __name__ == "__main__":
    main()
