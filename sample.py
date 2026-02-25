"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

    # Conditional generation (requires a checkpoint trained with use_conditioning=True)
    python sample.py --checkpoint path/to/conditional_ddpm_final.pt --method ddpm --attributes "Eyeglasses,Brown_Hair,Male" --guidance_scale 2.0 --num_samples 16 --grid
    python sample.py --checkpoint path/to/conditional_ddpm_final.pt --method ddpm --list_attributes   # print valid attribute names

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging
"""

import os
import sys
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image, CELEBA_40_ATTRIBUTES
from src.methods import DDPM, FlowMatching
from src.utils import EMA


def build_condition_from_names(
    attribute_names_to_set: list,
    num_attributes: int,
    batch_size: int,
    device: torch.device,
    attr_order=None,
) -> torch.Tensor:
    """
    Build condition tensor (batch_size, num_attributes) from attribute names to set to 1.
    attr_order: list of attribute names in the ORDER the model was trained with (from checkpoint config).
                If None, uses CELEBA_40_ATTRIBUTES. Must match training order or conditioning is wrong.
    """
    order = attr_order if attr_order is not None else CELEBA_40_ATTRIBUTES
    attr_to_idx = {name: i for i, name in enumerate(order)}
    cond = torch.zeros(batch_size, num_attributes, device=device, dtype=torch.float32)
    for name in attribute_names_to_set:
        name = name.strip()
        if name in attr_to_idx and attr_to_idx[name] < num_attributes:
            cond[:, attr_to_idx[name]] = 1.0
        elif name:
            print(f"Warning: unknown attribute '{name}' (ignored). Use e.g. --list_attributes to see valid names.")
    return cond


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint['model'])
    
    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])
    ema.load_state_dict(checkpoint['ema'])
    
    return model, config, ema


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int,
    nrow: int | None = None,
) -> None:
    """
    TODO: save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples, used to calculate grid layout.
    """

    if samples is None:
        raise ValueError("samples is None")

    # Ensure [0, 1] range for saving
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    if nrow is None:
        nrow = max(1, int(num_samples ** 0.5))
    save_image(samples, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'flow_matching'], # You can add more later
                       help='Method used for training (ddpm or flow_matching)')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--sampler', type=str, default=None,
                       choices=['ddpm', 'ddim'],
                       help='Sampling method (default: from config)')
    parser.add_argument('--eta', type=float, default=None,
                       help='DDIM eta (default: from config or 0.0)')
    
    # Conditional generation (only for models trained with use_conditioning=True)
    parser.add_argument('--attributes', type=str, default=None,
                       help='Comma-separated attribute names to generate with (e.g. Eyeglasses,Brown_Hair,Male). Requires a conditional checkpoint.')
    parser.add_argument('--guidance_scale', type=float, default=None,
                       help='Classifier-free guidance scale (default: from config, often 1.0–5.0 for stronger attributes)')
    parser.add_argument('--list_attributes', action='store_true',
                       help='Print the 40 CelebA attribute names and exit (use these with --attributes)')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.list_attributes:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})
        order = cfg.get("data", {}).get("attribute_names") or CELEBA_40_ATTRIBUTES
        print("Attribute names and index (use these with --attributes; order must match checkpoint):")
        for i, name in enumerate(order):
            print(f"  {i:2d}: {name}")
        return
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    
    # Create method
    if args.method == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif args.method == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    else:
        raise ValueError(f"Unknown method: {args.method}.")
    
    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Condition for conditional generation (use checkpoint's attribute order so indices match training)
    use_conditioning = config.get('model', {}).get('use_conditioning', False)
    num_attributes = config.get('model', {}).get('num_attributes') or config.get('data', {}).get('num_attributes', 40)
    attr_order = config.get('data', {}).get('attribute_names')  # order the model was trained with
    if attr_order is None:
        attr_order = CELEBA_40_ATTRIBUTES
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else config.get('sampling', {}).get('guidance_scale', 1.0)
    attr_names_parsed = [a.strip() for a in args.attributes.split(',') if a.strip()] if args.attributes else []
    if args.attributes and not use_conditioning:
        print("WARNING: This checkpoint was trained WITHOUT attribute conditioning (use_conditioning=False).")
        print("         Your --attributes are being IGNORED; output is unconditional.")
        print("         To get attribute conditioning: train with data.use_attributes=true and model.use_conditioning=true, then sample from that checkpoint.")
        attr_names_parsed = []
    elif use_conditioning and attr_names_parsed:
        print(f"Conditional generation with attributes: {attr_names_parsed}, guidance_scale={guidance_scale}")
    elif use_conditioning and not attr_names_parsed:
        print("Conditional checkpoint but no --attributes given; sampling unconditionally (zeros).")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps or config['sampling']['num_steps']
            sampler = args.sampler or config['sampling'].get('sampler', 'ddpm')
            eta = args.eta if args.eta is not None else config['sampling'].get('eta', 0.0)

            # Build condition for this batch (same cond repeated for each sample when using --attributes)
            if use_conditioning and attr_names_parsed:
                cond = build_condition_from_names(
                    attr_names_parsed, num_attributes, batch_size, device, attr_order=attr_order
                )
            else:
                cond = None

            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                sampler=sampler,
                eta=eta,
                cond=cond,
                guidance_scale=guidance_scale,
            )

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    save_samples(samples[i:i+1], img_path, 1)
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, num_samples=args.num_samples, nrow=8)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
