#!/usr/bin/env python3
"""
Evaluate CelebA attribute-classifier accuracy on generated samples,
optionally restricted to top-K attributes by prevalence in the TRAIN set.

Usage:
  python eval_attr_acc.py \
    --ckpt_dir /scr/lizad/personal/10799/hw3/cmu-10799-diffusion/checkpoints \
    --samples_root /scr/lizad/personal/10799/hw3/logs/ddpm_20260215_231448/checkpoints/kid_per_attr \
    --train_stats_csv /path/to/kid_per_attribute.csv \
    --top_k 30 \
    --rank_by train_pct \
    --min_train_count 1 \
    --device cuda
"""

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


CELEBA_ATTRS = [
    "5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs",
    "Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows",
    "Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones",
    "Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin",
    "Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
    "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace",
    "Wearing_Necktie","Young"
]


# ----------------------------
# Model fallback (edit if needed)
# ----------------------------
class SmallAttrCNN(nn.Module):
    def __init__(self, num_attrs: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_attrs),
        )

    def forward(self, x):
        return self.net(x)


def build_model(repo_root: Optional[str], num_attrs: int = 40) -> nn.Module:
    if repo_root is not None:
        import sys
        sys.path.insert(0, repo_root)

        # EDIT if your classifier lives somewhere else
        possible_imports = [
            ("models.attribute_classifier", "AttributeClassifier"),
            ("models.classifier", "AttributeClassifier"),
            ("classifier", "AttributeClassifier"),
            ("models.attr_classifier", "AttributeClassifier"),
        ]
        for mod_name, cls_name in possible_imports:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls(num_attrs=num_attrs)
            except Exception:
                pass

    return SmallAttrCNN(num_attrs=num_attrs)


# ----------------------------
# Checkpoint loading helpers
# ----------------------------
def find_ckpt(ckpt_dir: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"--ckpt not found: {explicit}")
        return explicit

    patterns = [
        os.path.join(ckpt_dir, "*.pt"),
        os.path.join(ckpt_dir, "*.pth"),
        os.path.join(ckpt_dir, "*.ckpt"),
    ]
    cands = []
    for p in patterns:
        cands.extend(glob.glob(p))
    if not cands:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir} (pt/pth/ckpt).")
    cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0]


def extract_state_dict(ckpt_obj) -> Tuple[Dict[str, torch.Tensor], Optional[List[str]]]:
    attr_names = None

    if isinstance(ckpt_obj, dict):
        for k in ["attr_names", "attributes", "attribute_names", "attrs"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], (list, tuple)) and len(ckpt_obj[k]) == 40:
                attr_names = list(ckpt_obj[k])
                break

        for k in ["state_dict", "model_state_dict", "model", "net", "ema_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k], attr_names

        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, attr_names

    raise ValueError("Unrecognized checkpoint format; adjust extract_state_dict().")


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "model.", "net.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


# ----------------------------
# Train prevalence filtering
# ----------------------------
def parse_pct(s: str) -> float:
    """'7.8%' -> 7.8 ; '0.0%' -> 0.0"""
    s = s.strip()
    if s.endswith("%"):
        s = s[:-1]
    return float(s)


def load_topk_attrs_from_train_csv(
    train_csv_path: str,
    top_k: Optional[int],
    rank_by: str,
    min_train_count: int,
) -> List[str]:
    rows = []
    with open(train_csv_path, "r") as f:
        reader = csv.DictReader(f)
        required = {"attribute", "train_count", "train_pct"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{train_csv_path} missing required columns {required}. Found: {reader.fieldnames}")

        for r in reader:
            attr = r["attribute"].strip()
            train_count = int(str(r["train_count"]).strip())
            train_pct = parse_pct(str(r["train_pct"]))
            rows.append((attr, train_count, train_pct))

    # filter zeros / rare
    rows = [x for x in rows if x[1] >= min_train_count]

    if rank_by == "train_pct":
        rows.sort(key=lambda x: x[2], reverse=True)
    elif rank_by == "train_count":
        rows.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError("--rank_by must be train_pct or train_count")

    if top_k is not None:
        rows = rows[:top_k]

    return [x[0] for x in rows]


# ----------------------------
# Data helpers
# ----------------------------
def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files


def get_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1]
    ])


@dataclass
class AttrResult:
    attr: str
    n: int
    acc: float


@torch.no_grad()
def eval_one_attr(
    model: nn.Module,
    attr_idx: int,
    img_paths: List[str],
    tfm: transforms.Compose,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> AttrResult:
    correct = 0
    total = 0

    for i in range(0, len(img_paths), batch_size):
        batch_files = img_paths[i:i+batch_size]
        imgs = []
        for fp in batch_files:
            im = Image.open(fp).convert("RGB")
            imgs.append(tfm(im))
        x = torch.stack(imgs, dim=0).to(device)

        logits = model(x)  # [B,40]
        probs = torch.sigmoid(logits[:, attr_idx])
        preds = (probs >= threshold).long()

        correct += int((preds == 1).sum().item())  # conditioned label=1
        total += preds.numel()

    acc = (correct / total) if total else float("nan")
    return AttrResult(attr=CELEBA_ATTRS[attr_idx], n=total, acc=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--samples_root", type=str, required=True)
    ap.add_argument("--repo_root", type=str, default="/scr/lizad/personal/10799/hw3/cmu-10799-diffusion")

    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="attr_accuracy_on_samples.csv")

    # NEW: filtering
    ap.add_argument("--train_stats_csv", type=str, default=None,
                    help="CSV with columns: attribute,train_count,train_pct,... used to choose top-K attrs.")
    ap.add_argument("--top_k", type=int, default=None, help="Keep only top-K attrs by --rank_by (after min_train_count).")
    ap.add_argument("--rank_by", type=str, default="train_pct", choices=["train_pct", "train_count"])
    ap.add_argument("--min_train_count", type=int, default=1, help="Drop attrs with train_count < this.")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Determine which attrs to evaluate (optional)
    allowed_attrs = None
    if args.train_stats_csv is not None:
        allowed_attrs = set(load_topk_attrs_from_train_csv(
            train_csv_path=args.train_stats_csv,
            top_k=args.top_k,
            rank_by=args.rank_by,
            min_train_count=args.min_train_count,
        ))
        print(f"[INFO] Filtering enabled: evaluating {len(allowed_attrs)} attrs "
              f"(top_k={args.top_k}, rank_by={args.rank_by}, min_train_count={args.min_train_count})")

    ckpt_path = find_ckpt(args.ckpt_dir, args.ckpt)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict, ckpt_attr_names = extract_state_dict(ckpt_obj)
    state_dict = strip_prefix(state_dict)

    model = build_model(args.repo_root, num_attrs=40)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] load_state_dict strict=False -> missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device).eval()

    # attribute order mapping
    attr_list = ckpt_attr_names if (ckpt_attr_names and len(ckpt_attr_names) == 40) else CELEBA_ATTRS
    name_to_idx = {a: i for i, a in enumerate(attr_list)}

    tfm = get_transform(args.img_size)

    sample_dirs = sorted([d for d in glob.glob(os.path.join(args.samples_root, "samples_*")) if os.path.isdir(d)])
    if not sample_dirs:
        raise FileNotFoundError(f"No samples_* folders found under {args.samples_root}")

    results: List[AttrResult] = []
    for d in sample_dirs:
        folder = os.path.basename(d)
        attr = folder[len("samples_"):] if folder.startswith("samples_") else folder

        if allowed_attrs is not None and attr not in allowed_attrs:
            continue

        if attr not in name_to_idx:
            print(f"[WARN] Attribute '{attr}' not in classifier attr list; skipping folder {d}")
            continue

        img_paths = list_images(d)
        if not img_paths:
            print(f"[WARN] No images in {d}; skipping")
            continue

        idx = name_to_idx[attr]
        r = eval_one_attr(model, idx, img_paths, tfm, device, args.batch_size, args.threshold)
        r = AttrResult(attr=attr, n=r.n, acc=r.acc)
        results.append(r)
        print(f"[{attr:>22}] n={r.n:4d}  acc={r.acc:.4f}")

    valid = [r for r in results if r.n > 0 and r.acc == r.acc]
    mean_acc = sum(r.acc for r in valid) / len(valid) if valid else float("nan")
    print(f"\n[INFO] Mean accuracy over evaluated attrs ({len(valid)}): {mean_acc:.4f}")

    out_path = os.path.join(args.samples_root, args.out_csv)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attribute", "n_images", "accuracy"])
        for r in results:
            w.writerow([r.attr, r.n, r.acc])
        w.writerow(["MEAN", "", mean_acc])

    print(f"[INFO] Wrote: {out_path}")


if __name__ == "__main__":
    main()