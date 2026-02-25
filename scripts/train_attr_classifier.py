#!/usr/bin/env python3
"""
Train a CelebA 40-attribute classifier for attribute disentanglement regularization.

Run this before training DDPM with attr_loss. Saves checkpoint to --output.

Usage:
  python scripts/train_attr_classifier.py --config configs/ddpm.yaml --output checkpoints/attr_classifier.pt
  python scripts/train_attr_classifier.py --config configs/ddpm.yaml --output checkpoints/attr_classifier.pt --epochs 10
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import create_dataloader_from_config
from src.models.attr_classifier import CelebAAttrClassifier


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ddpm.yaml", help="Config with data paths")
    ap.add_argument("--output", default="checkpoints/attr_classifier.pt", help="Save path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    config = load_config(args.config)
    config.setdefault("training", {})["batch_size"] = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dl = create_dataloader_from_config(config, split="train")
    if not getattr(dl.dataset, "use_attributes", False):
        raise RuntimeError("Dataset must have use_attributes=True")
    num_attrs = dl.dataset.num_attributes
    image_size = config.get("data", {}).get("image_size", 64)

    # Model
    model = CelebAAttrClassifier(num_attributes=num_attrs, image_size=image_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if isinstance(batch, (tuple, list)):
                images, attrs = batch[0].to(device), batch[1].to(device)
            else:
                attrs = None
            if attrs is None:
                continue
            opt.zero_grad()
            logits = model(images)
            # BCE with logits; targets in {0, 1}
            loss = F.binary_cross_entropy_with_logits(logits, attrs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(n, 1)
        print(f"Epoch {epoch+1} loss: {avg:.4f}")

    torch.save({"model": model.state_dict(), "num_attributes": num_attrs}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
