#!/usr/bin/env python3
"""
Evaluate the CelebA attribute classifier: accuracy per attribute and mean accuracy.

Usage:
  python scripts/eval_attr_classifier.py --checkpoint checkpoints/attr_classifier.pt --config configs/ddpm.yaml
  python scripts/eval_attr_classifier.py --checkpoint checkpoints/attr_classifier.pt --config configs/ddpm.yaml --split validation -o eval_results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import create_dataloader_from_config
from src.models.attr_classifier import load_attr_classifier


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Evaluate attribute classifier")
    ap.add_argument("--checkpoint", required=True, help="Path to classifier checkpoint")
    ap.add_argument("--config", default="configs/ddpm.yaml", help="Config for data")
    ap.add_argument("--split", default="train", choices=["train", "validation"], help="Data split to evaluate on")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=None, help="Limit batches (default: all)")
    ap.add_argument("-o", "--output", default=None, help="Save per-attribute accuracy to CSV")
    args = ap.parse_args()

    config = load_config(args.config)
    config.setdefault("training", {})["batch_size"] = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: no augment for eval, don't drop last batch
    data_config = config["data"]
    from src.data import CelebADataset
    from torch.utils.data import DataLoader
    dataset = CelebADataset(
        root=data_config.get("root", "./data/celeba-subset"),
        split=args.split,
        image_size=data_config["image_size"],
        augment=False,
        from_hub=data_config.get("from_hub", False),
        repo_name=data_config.get("repo_name", "electronickale/cmu-10799-celeba64-subset"),
        use_attributes=True,
        attribute_names=data_config.get("attribute_names"),
    )
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=False,
    )
    attr_names = getattr(dataset, "attribute_names", None) or [f"attr_{i}" for i in range(dataset.num_attributes)]
    num_attrs = dataset.num_attributes

    model = load_attr_classifier(
        args.checkpoint,
        device,
        num_attributes=num_attrs,
        image_size=data_config.get("image_size", 64),
    )
    model.eval()

    # Accumulate correct per attribute (binary: pred = logit > 0)
    correct = torch.zeros(num_attrs, device=device)
    total = torch.zeros(num_attrs, device=device)

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            if isinstance(batch, (tuple, list)):
                images, attrs = batch[0].to(device), batch[1].to(device)
            else:
                continue
            logits = model(images)
            pred = (logits > 0).float()
            correct += (pred == attrs).float().sum(dim=0)
            total += attrs.shape[0]

    total = total.clamp(min=1)
    acc_per = (correct / total).cpu().numpy()
    mean_acc = float(acc_per.mean())

    print(f"\nAttribute classifier evaluation ({args.split}, {int(total[0].item())} samples)")
    print("=" * 60)
    print(f"Mean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print("\nPer-attribute accuracy:")
    for i, name in enumerate(attr_names):
        print(f"  {name}: {acc_per[i]:.4f}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attribute", "accuracy", "correct", "total"])
            for i, name in enumerate(attr_names):
                w.writerow([name, f"{acc_per[i]:.4f}", int(correct[i].item()), int(total[i].item())])
            w.writerow(["mean", f"{mean_acc:.4f}", "", ""])
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
