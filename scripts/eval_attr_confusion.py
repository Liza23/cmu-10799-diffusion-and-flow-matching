#!/usr/bin/env python3
"""
Evaluate CelebA attribute-classifier confusion metrics (TP/FP/FN/TN),
optionally by subgroup (e.g., Male=0 for women).

Examples:
  python scripts/eval_attr_confusion.py \
    --checkpoint checkpoints/attr_classifier.pt \
    --config configs/ddpm.yaml \
    --split validation \
    --target-attr Blond_Hair

  python scripts/eval_attr_confusion.py \
    --checkpoint checkpoints/attr_classifier.pt \
    --config configs/ddpm.yaml \
    --split validation \
    --target-attr Blond_Hair \
    --group-attr Male \
    --group-value 0
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CelebADataset
from src.models.attr_classifier import load_attr_classifier


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def resolve_attr_index(name_or_index: str, attr_names: List[str]) -> int:
    if name_or_index.isdigit():
        idx = int(name_or_index)
        if 0 <= idx < len(attr_names):
            return idx
        raise ValueError(f"Attribute index out of range: {idx}")

    if name_or_index in attr_names:
        return attr_names.index(name_or_index)

    lower_map = {n.lower(): i for i, n in enumerate(attr_names)}
    k = name_or_index.lower()
    if k in lower_map:
        return lower_map[k]

    raise ValueError(
        f"Unknown attribute '{name_or_index}'. "
        f"Expected one of: {', '.join(attr_names)}"
    )


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    pred_pos = tp + fp
    true_pos = tp + fn
    true_neg = tn + fp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": safe_div(tp + tn, total),
        "precision": safe_div(tp, pred_pos),
        "recall_tpr": safe_div(tp, true_pos),
        "specificity_tnr": safe_div(tn, true_neg),
        "fpr": safe_div(fp, true_neg),
        "fnr": safe_div(fn, true_pos),
        "pred_pos_rate": safe_div(pred_pos, total),
        "true_pos_rate": safe_div(true_pos, total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate attribute confusion metrics.")
    ap.add_argument("--checkpoint", required=True, help="Path to classifier checkpoint")
    ap.add_argument("--config", default="configs/ddpm.yaml", help="Path to YAML config")
    ap.add_argument("--split", default="validation", choices=["train", "validation"], help="Data split")
    ap.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: config value)")
    ap.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DataLoader pin_memory (default: config value)",
    )
    ap.add_argument("--max-batches", type=int, default=None, help="Optional cap on batches")
    ap.add_argument("--threshold", type=float, default=0.0, help="Logit threshold for positive prediction")
    ap.add_argument(
        "--target-attr",
        default="Blond_Hair",
        help="Target attribute name or index (ignored if --all-attributes)",
    )
    ap.add_argument("--all-attributes", action="store_true", help="Evaluate all attributes")
    ap.add_argument(
        "--group-attr",
        default=None,
        help="Optional subgroup attribute name or index (e.g., Male)",
    )
    ap.add_argument(
        "--group-value",
        type=int,
        choices=[0, 1],
        default=None,
        help="If set with --group-attr, evaluate only this subgroup value; else both 0 and 1",
    )
    ap.add_argument("-o", "--output", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CelebADataset(
        root=data_config.get("root", "./data/celeba-subset"),
        split=args.split,
        image_size=data_config.get("image_size", 64),
        augment=False,
        from_hub=data_config.get("from_hub", False),
        repo_name=data_config.get("repo_name", "electronickale/cmu-10799-celeba64-subset"),
        use_attributes=True,
        attribute_names=data_config.get("attribute_names"),
    )
    num_workers = args.num_workers if args.num_workers is not None else data_config.get("num_workers", 4)
    pin_memory = args.pin_memory if args.pin_memory is not None else data_config.get("pin_memory", True)

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    attr_names = getattr(dataset, "attribute_names", None) or [f"attr_{i}" for i in range(dataset.num_attributes)]
    num_attrs = dataset.num_attributes

    if args.all_attributes:
        target_indices = list(range(num_attrs))
    else:
        target_indices = [resolve_attr_index(args.target_attr, attr_names)]
    target_names = [attr_names[i] for i in target_indices]

    group_specs: List[Tuple[str, int, int]] = [("overall", -1, -1)]
    if args.group_attr is not None:
        group_idx = resolve_attr_index(args.group_attr, attr_names)
        group_name = attr_names[group_idx]
        if args.group_value is None:
            group_specs.extend(
                [
                    (f"{group_name}=0", group_idx, 0),
                    (f"{group_name}=1", group_idx, 1),
                ]
            )
        else:
            group_specs.append((f"{group_name}={args.group_value}", group_idx, int(args.group_value)))

    model = load_attr_classifier(
        args.checkpoint,
        device,
        num_attributes=num_attrs,
        image_size=data_config.get("image_size", 64),
    )
    model.eval()

    n_groups = len(group_specs)
    n_targets = len(target_indices)
    tp = torch.zeros((n_groups, n_targets), dtype=torch.long, device=device)
    fp = torch.zeros((n_groups, n_targets), dtype=torch.long, device=device)
    fn = torch.zeros((n_groups, n_targets), dtype=torch.long, device=device)
    tn = torch.zeros((n_groups, n_targets), dtype=torch.long, device=device)
    group_counts = torch.zeros(n_groups, dtype=torch.long, device=device)

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if args.max_batches is not None and bi >= args.max_batches:
                break

            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                continue

            images = batch[0].to(device)
            attrs = batch[1].to(device)
            logits = model(images)

            pred_all = logits > args.threshold
            true_all = attrs > 0.5

            pred = pred_all[:, target_indices]
            true = true_all[:, target_indices]

            for gi, (_, gidx, gval) in enumerate(group_specs):
                if gidx < 0:
                    mask = torch.ones(images.shape[0], dtype=torch.bool, device=device)
                else:
                    mask = true_all[:, gidx] == bool(gval)

                group_counts[gi] += mask.sum()
                if mask.any():
                    m = mask.unsqueeze(1)
                    tp[gi] += ((pred & true) & m).sum(dim=0)
                    fp[gi] += ((pred & (~true)) & m).sum(dim=0)
                    fn[gi] += (((~pred) & true) & m).sum(dim=0)
                    tn[gi] += (((~pred) & (~true)) & m).sum(dim=0)

    rows = []
    print("\nAttribute Classifier Confusion Evaluation")
    print("=" * 88)
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: logits > {args.threshold:.4f}")
    print(f"Samples processed (overall): {int(group_counts[0].item())}")
    print("-" * 88)

    for ti, tname in enumerate(target_names):
        print(f"\nTarget: {tname}")
        print("group            n      TP      FP      FN      TN    Acc    Prec    TPR    FPR  Pred+%  True+%")
        for gi, (glabel, _, _) in enumerate(group_specs):
            c = metrics_from_counts(
                int(tp[gi, ti].item()),
                int(fp[gi, ti].item()),
                int(fn[gi, ti].item()),
                int(tn[gi, ti].item()),
            )
            print(
                f"{glabel:<14} {c['total']:>6d} {c['tp']:>7d} {c['fp']:>7d} {c['fn']:>7d} {c['tn']:>7d} "
                f"{c['accuracy']:>6.3f} {c['precision']:>7.3f} {c['recall_tpr']:>6.3f} {c['fpr']:>6.3f} "
                f"{100.0*c['pred_pos_rate']:>6.2f} {100.0*c['true_pos_rate']:>7.2f}"
            )
            rows.append(
                {
                    "target_attr": tname,
                    "group": glabel,
                    **c,
                }
            )

    print("=" * 88)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "target_attr",
                    "group",
                    "total",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "accuracy",
                    "precision",
                    "recall_tpr",
                    "specificity_tnr",
                    "fpr",
                    "fnr",
                    "pred_pos_rate",
                    "true_pos_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()
