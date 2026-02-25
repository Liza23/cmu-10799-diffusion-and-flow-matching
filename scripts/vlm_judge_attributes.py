#!/usr/bin/env python3
"""
Use a VLM (BLIP-2 or similar) to judge whether each generated image has the target
attribute. For each attribute we have N images (e.g. 100) generated WITH that attribute;
we ask the VLM "Does this person have [attribute]?" and report the fraction that answer "yes"
as the attribute accuracy for that attribute.

Usage:
  python scripts/vlm_judge_attributes.py --samples-dir logs/.../kid_per_attr --limit 10   # test on 10 imgs
  python scripts/vlm_judge_attributes.py --samples-dir logs/.../kid_per_attr -o vlm_accuracy.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import torch


def attr_to_question(attr: str) -> str:
    """Turn attribute name into a natural-language question."""
    # e.g. Blond_Hair -> "blond hair", 5_o_Clock_Shadow -> "5 o'clock shadow"
    s = attr.replace("_", " ").strip()
    if s.lower().startswith("5 o clock"):
        s = "5 o'clock shadow"
    return s


def parse_yes_no(answer: str) -> Optional[bool]:
    """Parse VLM output to yes (True) or no (False). None if unclear."""
    if not answer:
        return None
    a = answer.strip().lower()
    # Common patterns: "Yes.", "Yes, ...", "No.", "No, ..."
    if a.startswith("yes") or a == "y":
        return True
    if a.startswith("no") or a == "n":
        return False
    # Heuristic: first occurrence of yes vs no
    yes_i = a.find("yes")
    no_i = a.find("no")
    if yes_i >= 0 and (no_i < 0 or yes_i < no_i):
        return True
    if no_i >= 0:
        return False
    return None


def main():
    ap = argparse.ArgumentParser(description="VLM judge: attribute presence accuracy on generated samples")
    ap.add_argument("--samples-dir", required=True, help="Base dir containing samples_<Attribute> subdirs (e.g. kid_per_attr)")
    ap.add_argument("--attributes", type=str, default=None, help="Comma-separated attributes (default: all samples_* dirs)")
    ap.add_argument("--max-per-attr", type=int, default=100, help="Max images to evaluate per attribute (default: 100)")
    ap.add_argument("--limit", type=int, default=None, help="If set, only evaluate this many images per attr (for testing)")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (attribute, accuracy, n_eval, n_yes)")
    ap.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b", help="HuggingFace model for VQA")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1, help="Images per forward (1 for sequential; BLIP-2 is per-image)")
    args = ap.parse_args()

    samples_base = Path(args.samples_dir).resolve()
    if not samples_base.is_dir():
        print(f"Error: samples-dir not found: {samples_base}", file=sys.stderr)
        sys.exit(1)

    # Discover or use specified attributes
    if args.attributes:
        attr_names = [a.strip() for a in args.attributes.split(",") if a.strip()]
    else:
        attr_names = []
        for d in sorted(samples_base.iterdir()):
            if d.is_dir() and d.name.startswith("samples_"):
                name = d.name.replace("samples_", "", 1).replace(" ", "_")
                attr_names.append(name)
    if not attr_names:
        print("Error: no attribute dirs found (samples_<Name>). Use --attributes to list them.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading VLM: {args.model}...")
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError:
        print("Error: transformers required. pip install transformers", file=sys.stderr)
        sys.exit(1)
    from PIL import Image

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    n_per_attr = args.limit if args.limit is not None else args.max_per_attr
    prompt_template = "Does this person have {}? Answer only yes or no."

    results = []
    for attr in attr_names:
        attr_dir = samples_base / f"samples_{attr.replace(' ', '_')}"
        if not attr_dir.is_dir():
            print(f"[{attr}] Skipping: dir not found {attr_dir}")
            continue
        images = sorted(attr_dir.glob("*.png")) + sorted(attr_dir.glob("*.jpg"))[:n_per_attr]
        if not images:
            print(f"[{attr}] Skipping: no images")
            continue
        question = prompt_template.format(attr_to_question(attr))
        yes_count = 0
        n = 0
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            inputs = processor(images=img, text=question, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)
            answer = processor.decode(out[0], skip_special_tokens=True).strip()
            pred = parse_yes_no(answer)
            if pred is True:
                yes_count += 1
            n += 1
        acc = (yes_count / n) if n else 0.0
        results.append((attr, acc, n, yes_count))
        print(f"[{attr}] accuracy = {yes_count}/{n} = {acc:.2%}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attribute", "accuracy", "n_eval", "n_yes"])
            for attr, acc, n, yes_count in results:
                w.writerow([attr, f"{acc:.4f}", n, yes_count])
        print(f"Saved: {out_path}")

    # Summary
    if results:
        mean_acc = sum(r[1] for r in results) / len(results)
        print(f"\nMean accuracy over {len(results)} attributes: {mean_acc:.2%}")


if __name__ == "__main__":
    main()
