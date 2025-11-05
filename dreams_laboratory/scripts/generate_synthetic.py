#!/usr/bin/env python3
"""
Generate synthetic multispectral images from random latents using trained encoder-decoder.

IMPROVED VERSION:
- Better sampling strategies
- Test reconstruction first to verify decoder quality
- Sample from real latents as baseline
- Better visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, List
from sample_from_latent import (
    load_model, sample_random_latents, visualize_rgb_band
)
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder
from multispectral_vit import MultispectralTileDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


def test_reconstruction(encoder, decoder, dataloader, device='cuda', num_samples=5):
    """Test reconstruction quality first to verify decoder works."""
    encoder.eval()
    decoder.eval()
    
    print("\n" + "="*80)
    print("Testing Reconstruction Quality (to verify decoder works)")
    print("="*80)
    
    reconstructions = []
    originals = []
    
    with torch.no_grad():
        for idx, (images, paths) in enumerate(dataloader):
            if idx >= num_samples:
                break
            
            images = images.to(device)
            
            # Encode
            latents = encoder(images)
            
            # Decode
            reconstructed = decoder(latents)
            
            # Calculate MSE
            mse = F.mse_loss(reconstructed, images).item()
            print(f"Sample {idx+1}: MSE = {mse:.6f}")
            
            reconstructions.append(reconstructed[0].cpu())
            originals.append(images[0].cpu())
    
    # Visualize
    fig, axes = plt.subplots(2, len(reconstructions), figsize=(len(reconstructions)*3, 6))
    if len(reconstructions) == 1:
        axes = axes.reshape(2, 1)
    
    for idx in range(len(reconstructions)):
        # Original
        rgb_orig = visualize_rgb_band(originals[idx])
        axes[0, idx].imshow(rgb_orig)
        axes[0, idx].set_title('Original')
        axes[0, idx].axis('off')
        
        # Reconstructed
        rgb_recon = visualize_rgb_band(reconstructions[idx])
        axes[1, idx].imshow(rgb_recon)
        axes[1, idx].set_title('Reconstructed')
        axes[1, idx].axis('off')
    
    plt.suptitle('Reconstruction Test', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstruction_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved reconstruction test to: reconstruction_test.png")
    plt.close()
    
    return reconstructions, originals


def generate_synthetic_images(decoder: nn.Module, latents: np.ndarray,
                             device: str = 'cuda', num_samples: int = 20,
                             sample_method: str = 'gaussian',
                             use_real_latents: bool = False) -> List[torch.Tensor]:
    """Generate synthetic images from random latents."""
    decoder.eval()
    generated_images = []
    
    with torch.no_grad():
        if use_real_latents:
            # Use actual latents from real images (better quality)
            print(f"Using {num_samples} real latents from dataset...")
            indices = np.random.choice(len(latents), num_samples, replace=False)
            sampled_latents = latents[indices]
        else:
            # Sample random latents
            print(f"Sampling {num_samples} random latents using {sample_method} method...")
            sampled_latents = sample_random_latents(num_samples, latents, method=sample_method)
        
        # Convert to tensor
        latent_tensor = torch.from_numpy(sampled_latents).float().to(device)
        
        # Generate images
        synthetic_imgs = decoder(latent_tensor)  # (B, C, H, W) where C is number of channels
        
        for i in range(synthetic_imgs.shape[0]):
            generated_images.append(synthetic_imgs[i].cpu())
    
    return generated_images


def sample_from_latent_distribution(latents: np.ndarray, num_samples: int,
                                   method: str = 'gaussian', 
                                   clip_to_data_range: bool = True) -> np.ndarray:
    """Improved sampling with better control."""
    if method == 'gaussian':
        # Sample from Gaussian fitted to data
        mean = latents.mean(axis=0)
        std = latents.std(axis=0)
        
        # Use smaller std to stay closer to data distribution
        std = std * 0.5  # Reduce variance for more realistic samples
        
        samples = np.random.normal(mean, std, (num_samples, latents.shape[1]))
        
        if clip_to_data_range:
            # Clip to data range
            min_vals = latents.min(axis=0)
            max_vals = latents.max(axis=0)
            samples = np.clip(samples, min_vals, max_vals)
            
    elif method == 'uniform':
        # Sample from uniform distribution within data bounds
        min_vals = latents.min(axis=0)
        max_vals = latents.max(axis=0)
        samples = np.random.uniform(min_vals, max_vals, (num_samples, latents.shape[1]))
        
    elif method == 'sphere':
        # Sample from sphere with radius matching mean data norm
        mean_norm = np.linalg.norm(latents, axis=1).mean()
        samples = np.random.randn(num_samples, latents.shape[1])
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True) * mean_norm
        
    elif method == 'pca':
        # Sample using PCA projection (more structured)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(100, latents.shape[1]))
        latents_pca = pca.fit_transform(latents)
        
        # Sample in PCA space
        mean_pca = latents_pca.mean(axis=0)
        std_pca = latents_pca.std(axis=0) * 0.5
        samples_pca = np.random.normal(mean_pca, std_pca, (num_samples, latents_pca.shape[1]))
        
        # Transform back
        samples = pca.inverse_transform(samples_pca)
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return samples


def visualize_synthetic_samples(images: List[torch.Tensor],
                                title: str = "Synthetic Multispectral Images",
                                output_file: str = "synthetic_samples.png",
                                n_cols: int = 5):
    """Visualize a grid of synthetic images."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col]
        
        # Convert to RGB for visualization
        rgb = visualize_rgb_band(img)
        
        ax.imshow(rgb)
        ax.set_title(f"Synthetic {idx+1}", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic multispectral images')
    parser.add_argument('--encoder_path', type=str, default='multispectral_vit.pth',
                       help='Path to trained encoder')
    parser.add_argument('--decoder_path', type=str, default='decoder.pth',
                       help='Path to trained decoder')
    parser.add_argument('--latent_file', type=str, default='multispectral_latents.npy',
                       help='Pre-computed latent file for sampling distribution')
    
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of synthetic images to generate')
    parser.add_argument('--sample_method', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'sphere', 'pca', 'real'],
                       help='Random sampling method (real=use actual latents from dataset)')
    
    parser.add_argument('--test_reconstruction', action='store_true',
                       help='Test reconstruction quality first')
    parser.add_argument('--tile_dir', type=str,
                       default='/mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop',
                       help='Tile directory for reconstruction test')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load encoder
    print("Loading encoder...")
    encoder, config = load_model(args.encoder_path, args.device)
    
    # Load decoder
    print(f"Loading decoder from {args.decoder_path}")
    from multispectral_decoder import ReconstructionDecoder
    decoder = ReconstructionDecoder(
        embed_dim=config['embed_dim'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=config.get('in_channels', 3)  # Default to RGB (3 channels) if not in config
    )
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=args.device))
    decoder = decoder.to(args.device)
    decoder.eval()
    
    # Test reconstruction first if requested
    if args.test_reconstruction:
        print("\n" + "="*80)
        print("Testing Reconstruction Quality")
        print("="*80)
        dataset = MultispectralTileDataset(
            args.tile_dir, 
            img_size=config['img_size'],
            in_channels=config.get('in_channels', None)  # Auto-detect if not in config
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        test_reconstruction(encoder, decoder, dataloader, args.device, num_samples=5)
    
    # Load latents for sampling distribution
    if Path(args.latent_file).exists():
        print(f"\nLoading latent distribution from {args.latent_file}")
        latents = np.load(args.latent_file)
        print(f"Loaded {len(latents)} latent representations")
        print(f"Latent shape: {latents.shape}")
        print(f"Latent stats: mean={latents.mean():.4f}, std={latents.std():.4f}, "
              f"min={latents.min():.4f}, max={latents.max():.4f}")
    else:
        print("WARNING: No latent file found. Using default distribution.")
        latents = np.random.randn(100, config['embed_dim']) * 0.1  # Small values
    
    # Generate synthetic images
    print(f"\nGenerating {args.num_samples} synthetic images...")
    print(f"Sampling method: {args.sample_method}")
    
    use_real = (args.sample_method == 'real')
    if use_real:
        synthetic_images = generate_synthetic_images(
            decoder, latents, args.device, args.num_samples,
            sample_method='gaussian', use_real_latents=True
        )
    else:
        # Use improved sampling
        sampled_latents = sample_from_latent_distribution(
            latents, args.num_samples, method=args.sample_method,
            clip_to_data_range=True
        )
        
        # Convert to tensor and generate
        decoder.eval()
        with torch.no_grad():
            latent_tensor = torch.from_numpy(sampled_latents).float().to(args.device)
            synthetic_imgs = decoder(latent_tensor)
            synthetic_images = [synthetic_imgs[i].cpu() for i in range(synthetic_imgs.shape[0])]
    
    # Visualize
    visualize_synthetic_samples(
        synthetic_images,
        title=f"Synthetic Samples ({args.sample_method})",
        output_file=f"synthetic_samples_{args.sample_method}.png"
    )
    
    # Save individual images
    output_dir = Path("synthetic_images")
    output_dir.mkdir(exist_ok=True)
    
    for idx, img in enumerate(synthetic_images):
        # Save as numpy array (multispectral)
        img_np = img.numpy()  # (C, H, W) where C is number of channels
        np.save(output_dir / f"synthetic_{idx:03d}.npy", img_np)
        
        # Save RGB visualization
        rgb = visualize_rgb_band(img)
        plt.imsave(output_dir / f"synthetic_{idx:03d}_rgb.png", rgb)
    
    print(f"\nSaved {len(synthetic_images)} synthetic images to {output_dir}/")
    print("Files:")
    print("  - synthetic_XXX.npy: Multispectral data (variable channels)")
    print("  - synthetic_XXX_rgb.png: RGB visualization")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
