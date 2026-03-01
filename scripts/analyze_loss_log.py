#!/usr/bin/env python3
"""
Parse tqdm training logs and extract loss metrics into CSV + summary.

Supports logs that contain lines like:
  10%|...| 200/200000 [.., loss=0.0416, steps/s=4.43, loss_attr=0.4184]
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOSS_RE = re.compile(r"loss=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
LOSS_ATTR_RE = re.compile(r"loss_attr=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
STEPS_S_RE = re.compile(r"steps/s=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
STEP_RE = re.compile(r"(\d+)/(\d+)")


def _simple_moving_average(values: List[float], window: int) -> List[Optional[float]]:
    if window <= 1:
        return [v for v in values]
    out: List[Optional[float]] = [None] * len(values)
    csum = 0.0
    for i, v in enumerate(values):
        csum += v
        if i >= window:
            csum -= values[i - window]
        if i >= window - 1:
            out[i] = csum / window
    return out


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def parse_log(path: Path) -> Tuple[List[int], List[float], List[Optional[float]], List[Optional[float]], int]:
    text = path.read_text(errors="ignore").replace("\r", "\n")
    step_to_metrics: Dict[int, Tuple[float, Optional[float], Optional[float], int]] = {}

    for line in text.splitlines():
        if "loss=" not in line:
            continue

        loss_match = LOSS_RE.search(line)
        if not loss_match:
            continue
        loss = float(loss_match.group(1))

        step_match = STEP_RE.search(line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        total = int(step_match.group(2))

        loss_attr_match = LOSS_ATTR_RE.search(line)
        loss_attr = float(loss_attr_match.group(1)) if loss_attr_match else None

        steps_s_match = STEPS_S_RE.search(line)
        steps_s = float(steps_s_match.group(1)) if steps_s_match else None

        # Use the latest metrics for a step (tqdm updates often repeat steps)
        step_to_metrics[step] = (loss, loss_attr, steps_s, total)

    if not step_to_metrics:
        return [], [], [], [], 0

    steps_sorted = sorted(step_to_metrics.keys())
    loss_vals: List[float] = []
    loss_attr_vals: List[Optional[float]] = []
    steps_s_vals: List[Optional[float]] = []
    total = 0

    for step in steps_sorted:
        loss, loss_attr, steps_s, total = step_to_metrics[step]
        loss_vals.append(loss)
        loss_attr_vals.append(loss_attr)
        steps_s_vals.append(steps_s)

    return steps_sorted, loss_vals, loss_attr_vals, steps_s_vals, total


def write_csv(
    out_csv: Path,
    steps: List[int],
    total: int,
    loss_vals: List[float],
    loss_attr_vals: List[Optional[float]],
    steps_s_vals: List[Optional[float]],
    smooth_window: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    smooth = _simple_moving_average(loss_vals, smooth_window)

    headers = ["step", "total", "loss", "loss_attr", "steps_per_sec"]
    if smooth_window > 1:
        headers.append(f"loss_sma_{smooth_window}")

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, step in enumerate(steps):
            row = [
                step,
                total,
                f"{loss_vals[i]:.8f}",
                "" if loss_attr_vals[i] is None else f"{loss_attr_vals[i]:.8f}",
                "" if steps_s_vals[i] is None else f"{steps_s_vals[i]:.8f}",
            ]
            if smooth_window > 1:
                row.append("" if smooth[i] is None else f"{smooth[i]:.8f}")
            writer.writerow(row)


def write_summary(
    out_summary: Path,
    steps: List[int],
    loss_vals: List[float],
    loss_attr_vals: List[Optional[float]],
    steps_s_vals: List[Optional[float]],
) -> None:
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    if not loss_vals:
        out_summary.write_text("No loss entries found.\n")
        return

    loss_sorted = sorted(loss_vals)
    loss_attr_clean = [v for v in loss_attr_vals if v is not None]
    steps_s_clean = [v for v in steps_s_vals if v is not None]

    lines = []
    lines.append(f"Steps: {steps[0]} .. {steps[-1]} (n={len(steps)})")
    lines.append("")
    lines.append("Loss:")
    lines.append(f"  min: {min(loss_vals):.8f}")
    lines.append(f"  max: {max(loss_vals):.8f}")
    lines.append(f"  mean: {statistics.mean(loss_vals):.8f}")
    lines.append(f"  median: {statistics.median(loss_vals):.8f}")
    lines.append(f"  p25: {_percentile(loss_sorted, 25):.8f}")
    lines.append(f"  p75: {_percentile(loss_sorted, 75):.8f}")

    if loss_attr_clean:
        loss_attr_sorted = sorted(loss_attr_clean)
        lines.append("")
        lines.append("Loss_attr:")
        lines.append(f"  min: {min(loss_attr_clean):.8f}")
        lines.append(f"  max: {max(loss_attr_clean):.8f}")
        lines.append(f"  mean: {statistics.mean(loss_attr_clean):.8f}")
        lines.append(f"  median: {statistics.median(loss_attr_clean):.8f}")
        lines.append(f"  p25: {_percentile(loss_attr_sorted, 25):.8f}")
        lines.append(f"  p75: {_percentile(loss_attr_sorted, 75):.8f}")

    if steps_s_clean:
        steps_s_sorted = sorted(steps_s_clean)
        lines.append("")
        lines.append("Steps_per_sec:")
        lines.append(f"  min: {min(steps_s_clean):.8f}")
        lines.append(f"  max: {max(steps_s_clean):.8f}")
        lines.append(f"  mean: {statistics.mean(steps_s_clean):.8f}")
        lines.append(f"  median: {statistics.median(steps_s_clean):.8f}")
        lines.append(f"  p25: {_percentile(steps_s_sorted, 25):.8f}")
        lines.append(f"  p75: {_percentile(steps_s_sorted, 75):.8f}")

    out_summary.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract loss metrics from tqdm training logs."
    )
    parser.add_argument(
        "--log",
        required=True,
        action="append",
        help="Path to a training .log file (can be specified multiple times).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write outputs (default: same directory as each log).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help="Simple moving average window for loss (default: 200). Use 1 to disable.",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None

    for log_path_str in args.log:
        log_path = Path(log_path_str)
        if not log_path.exists():
            raise FileNotFoundError(log_path)

        steps, loss_vals, loss_attr_vals, steps_s_vals, total = parse_log(log_path)
        if not steps:
            print(f"[warn] No loss entries found in {log_path}")
            continue

        base = log_path.stem
        target_dir = out_dir if out_dir else log_path.parent
        out_csv = target_dir / f"{base}.loss.csv"
        out_summary = target_dir / f"{base}.loss.summary.txt"

        write_csv(
            out_csv=out_csv,
            steps=steps,
            total=total,
            loss_vals=loss_vals,
            loss_attr_vals=loss_attr_vals,
            steps_s_vals=steps_s_vals,
            smooth_window=args.smooth_window,
        )
        write_summary(
            out_summary=out_summary,
            steps=steps,
            loss_vals=loss_vals,
            loss_attr_vals=loss_attr_vals,
            steps_s_vals=steps_s_vals,
        )

        print(f"[ok] {log_path} -> {out_csv}")
        print(f"[ok] {log_path} -> {out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
