#!/usr/bin/env python3
"""
Generate synthetic rock tiles using trained diffusion model.

This script generates high-quality synthetic images by iteratively denoising
random noise through the trained diffusion model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from PIL import Image
from tqdm import tqdm

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from diffusion_model import GaussianDiffusion
from diffusion_unet_simple import UNetSimple


def visualize_samples(images, output_file='diffusion_samples.png', n_cols=5):
    """Visualize generated samples."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Calculate figure size
    img_size = images[0].shape[1]  # H from (C, H, W)
    scale = max(3, img_size / 256 * 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        # Convert from (C, H, W) to (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'Sample {idx+1}', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Diffusion Model Synthetic Rock Tiles (Zoom 23)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


@torch.no_grad()
def generate_samples(model, diffusion, num_samples=20, img_size=256, batch_size=8, device='cuda'):
    """
    Generate synthetic images using diffusion model.
    
    Args:
        model: Trained UNet model
        diffusion: GaussianDiffusion instance
        num_samples: Number of samples to generate
        img_size: Image resolution
        batch_size: Generation batch size
        device: Device to use
    
    Returns:
        List of generated images as tensors
    """
    model.eval()
    all_samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        print(f"Generating batch {i+1}/{num_batches} ({current_batch_size} samples)...")
        
        # Generate samples
        shape = (current_batch_size, 3, img_size, img_size)
        samples = diffusion.p_sample_loop(model, shape, progress=True)
        
        # Convert from [-1, 1] to [0, 1]
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0, 1)
        
        all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def generate_with_ddim(model, diffusion, num_samples=20, img_size=256, 
                       ddim_steps=50, eta=0.0, batch_size=8, device='cuda'):
    """
    Generate samples using DDIM (faster sampling).
    
    DDIM allows for faster generation with fewer steps while maintaining quality.
    
    Args:
        model: Trained UNet model
        diffusion: GaussianDiffusion instance
        num_samples: Number of samples to generate
        img_size: Image resolution
        ddim_steps: Number of DDIM steps (much less than 1000)
        eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)
        batch_size: Generation batch size
        device: Device to use
    
    Returns:
        Generated images
    """
    model.eval()
    all_samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Create DDIM timestep schedule
    timesteps = torch.linspace(diffusion.num_timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        print(f"Generating batch {i+1}/{num_batches} ({current_batch_size} samples) with DDIM...")
        
        # Start from pure noise
        img = torch.randn(current_batch_size, 3, img_size, img_size, device=device)
        
        for step_idx, t in enumerate(tqdm(timesteps, desc='DDIM Sampling')):
            t_batch = t.repeat(current_batch_size)
            
            # Predict noise
            predicted_noise = model(img, t_batch)
            
            # Get alpha values
            alpha_t = diffusion.alphas_cumprod[t]
            
            if step_idx < len(timesteps) - 1:
                alpha_t_prev = diffusion.alphas_cumprod[timesteps[step_idx + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # Predict x_0
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * predicted_noise
            
            # Random noise
            if eta > 0 and step_idx < len(timesteps) - 1:
                noise = torch.randn_like(img)
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            else:
                noise = 0
                sigma_t = 0
            
            # Update image
            img = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0, 1)
        
        all_samples.append(img.cpu())
    
    return torch.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic rock tiles with diffusion model')
    parser.add_argument('--model_path', type=str, default='diffusion_model_zoom23_ema.pth',
                       help='Path to trained diffusion model')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to generate')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='diffusion_synthetic_zoom23',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM for faster sampling')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM steps (default: 50, faster than 1000)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                       help='DDIM eta (0=deterministic, 1=stochastic)')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Print GPU info
    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("DIFFUSION MODEL GENERATION - ZOOM LEVEL 23")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Resolution: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    if args.use_ddim:
        print(f"Sampling: DDIM ({args.ddim_steps} steps, eta={args.ddim_eta})")
    else:
        print(f"Sampling: DDPM (1000 steps)")
    print("=" * 80)
    print()
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = checkpoint.get('config', {})
    
    model = UNetSimple(
        img_channels=3,
        model_channels=config.get('base_channels', 128),
        channel_mults=(1, 2, 4, 8)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create diffusion
    print("Creating diffusion process...")
    num_timesteps = config.get('num_timesteps', 1000)
    diffusion = GaussianDiffusion(num_timesteps=num_timesteps, device=args.device)
    print(f"Diffusion timesteps: {num_timesteps}")
    print()
    
    # Generate samples
    print(f"Generating {args.num_samples} synthetic images...")
    print("This may take several minutes...")
    print()
    
    if args.use_ddim:
        synthetic_images = generate_with_ddim(
            model, diffusion,
            num_samples=args.num_samples,
            img_size=args.img_size,
            ddim_steps=args.ddim_steps,
            eta=args.ddim_eta,
            batch_size=args.batch_size,
            device=args.device
        )
        method_suffix = f"_ddim{args.ddim_steps}"
    else:
        synthetic_images = generate_samples(
            model, diffusion,
            num_samples=args.num_samples,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        )
        method_suffix = "_ddpm"
    
    print(f"Generated {len(synthetic_images)} images")
    print()
    
    # Visualize
    print("Creating visualization...")
    visualize_samples(
        synthetic_images,
        output_file=str(output_dir / f'diffusion_samples{method_suffix}.png'),
        n_cols=5
    )
    
    # Save individual images
    print(f"Saving individual images to {output_dir}/...")
    for idx, img in enumerate(synthetic_images):
        # Save as numpy array
        img_np = img.numpy()  # (3, H, W)
        np.save(output_dir / f'diffusion_{idx:03d}.npy', img_np)
        
        # Save as PNG
        img_pil = img.permute(1, 2, 0).numpy()
        img_pil = (np.clip(img_pil, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_pil).save(
            output_dir / f'diffusion_{idx:03d}.png',
            quality=95, optimize=True
        )
    
    print()
    print("=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Generated {args.num_samples} synthetic images")
    print(f"Resolution: {args.img_size}x{args.img_size} pixels")
    print(f"Files saved to: {output_dir}/")
    print()
    print("Output files:")
    print(f"  - diffusion_samples{method_suffix}.png (grid visualization)")
    print(f"  - diffusion_XXX.png (individual images)")
    print(f"  - diffusion_XXX.npy (raw numpy arrays)")
    
    if args.device == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print()
        print(f"Peak GPU memory usage: {max_memory:.2f} GB")


if __name__ == '__main__':
    main()

