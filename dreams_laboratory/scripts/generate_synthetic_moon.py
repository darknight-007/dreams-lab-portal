#!/usr/bin/env python3
"""
Generate synthetic moon tile images from trained autoencoder.

This script generates synthetic moon images by sampling from the latent space
learned during training.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from PIL import Image

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from multispectral_decoder import ReconstructionDecoder
from multispectral_vit import MultispectralViT


def sample_random_latents(num_samples: int, latents: np.ndarray,
                          method: str = 'gaussian') -> np.ndarray:
    """Sample random latents from distribution."""
    if method == 'gaussian':
        mean = latents.mean(axis=0)
        std = latents.std(axis=0)
        std = std * 0.5  # Reduce variance for more realistic samples
        samples = np.random.normal(mean, std, (num_samples, latents.shape[1]))
        
        # Clip to data range
        min_vals = latents.min(axis=0)
        max_vals = latents.max(axis=0)
        samples = np.clip(samples, min_vals, max_vals)
        
    elif method == 'uniform':
        min_vals = latents.min(axis=0)
        max_vals = latents.max(axis=0)
        samples = np.random.uniform(min_vals, max_vals, (num_samples, latents.shape[1]))
        
    elif method == 'real':
        indices = np.random.choice(len(latents), num_samples, replace=False)
        samples = latents[indices]
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return samples


def generate_synthetic_images(decoder, latents, device='cuda', num_samples=20, method='gaussian'):
    """Generate synthetic moon images from latents."""
    decoder.eval()
    
    # Sample latents
    sampled_latents = sample_random_latents(num_samples, latents, method=method)
    
    # Convert to tensor
    latent_tensor = torch.from_numpy(sampled_latents).float().to(device)
    
    # Generate images
    with torch.no_grad():
        synthetic_imgs = decoder(latent_tensor)  # (B, C, H, W)
    
    return synthetic_imgs.cpu()


def visualize_samples(images, output_file='synthetic_samples.png', n_cols=5, is_grayscale=False):
    """Visualize synthetic samples."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        if is_grayscale or img.shape[0] == 1:
            # Grayscale
            img_np = img[0].numpy() if img.shape[0] == 1 else img.mean(dim=0).numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[row, col].imshow(img_np, cmap='gray')
        else:
            # RGB - Convert from (C, H, W) to (H, W, C)
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[row, col].imshow(img_np)
        
        axes[row, col].set_title(f'Synthetic {idx+1}', fontsize=8)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Synthetic Moon Tiles (Zoom Level 9)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic moon tile images')
    parser.add_argument('--encoder_path', type=str, default='encoder_moon_zoom9.pth',
                       help='Path to trained encoder (for config)')
    parser.add_argument('--decoder_path', type=str, default='decoder_moon_zoom9.pth',
                       help='Path to trained decoder')
    parser.add_argument('--latent_file', type=str, default='latents_moon_zoom9.npy',
                       help='Pre-computed latent file')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of synthetic images to generate')
    parser.add_argument('--sample_method', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'real'],
                       help='Sampling method')
    parser.add_argument('--output_dir', type=str, default='synthetic_moon',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load encoder config
    encoder_checkpoint = torch.load(args.encoder_path, map_location=args.device)
    config = encoder_checkpoint['config']
    
    # Load decoder
    print("Loading decoder...")
    decoder_checkpoint = torch.load(args.decoder_path, map_location=args.device)
    decoder_config = decoder_checkpoint.get('config', config)
    
    decoder = ReconstructionDecoder(
        embed_dim=decoder_config['embed_dim'],
        img_size=decoder_config['img_size'],
        patch_size=decoder_config['patch_size'],
        in_channels=decoder_config.get('in_channels', 3)
    )
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder = decoder.to(args.device)
    decoder.eval()
    
    # Load latents
    print(f"Loading latents from {args.latent_file}...")
    latents = np.load(args.latent_file)
    print(f"Loaded {len(latents)} latent representations")
    
    # Generate synthetic images
    print(f"\nGenerating {args.num_samples} synthetic moon images using {args.sample_method} method...")
    synthetic_images = generate_synthetic_images(
        decoder, latents, args.device, args.num_samples, args.sample_method
    )
    
    # Determine if grayscale
    is_grayscale = (decoder_config.get('in_channels', 3) == 1)
    
    # Visualize
    visualize_samples(
        synthetic_images,
        output_file=str(output_dir / f'synthetic_samples_{args.sample_method}.png'),
        n_cols=5,
        is_grayscale=is_grayscale
    )
    
    # Save individual images
    print(f"\nSaving individual images to {output_dir}/...")
    for idx, img in enumerate(synthetic_images):
        # Save as numpy array
        img_np = img.numpy()  # (C, H, W)
        np.save(output_dir / f'synthetic_{idx:03d}.npy', img_np)
        
        # Save as PNG
        if img.shape[0] == 1:
            # Grayscale
            img_pil = img[0].numpy()
            img_pil = (np.clip(img_pil, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_pil, mode='L').save(output_dir / f'synthetic_{idx:03d}.png')
        else:
            # RGB
            img_pil = img.permute(1, 2, 0).numpy()
            img_pil = (np.clip(img_pil, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_pil).save(output_dir / f'synthetic_{idx:03d}.png')
    
    print(f"\nDone! Generated {args.num_samples} synthetic moon images")
    print(f"Files saved to: {output_dir}/")


if __name__ == '__main__':
    main()

