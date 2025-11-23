#!/usr/bin/env python3
"""
Extract latent representations from moon tiles.

This script extracts latent representations from all moon tile images in the dataset,
which are needed for generating synthetic samples.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import numpy as np

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from multispectral_vit import MultispectralViT
from train_moon_autoencoder import MoonTileDataset


def extract_latents(encoder, dataloader, device='cuda'):
    """Extract latent representations from all images."""
    encoder.eval()
    latents = []
    paths = []
    
    print("Extracting latent representations from moon tiles...")
    
    with torch.no_grad():
        for batch_idx, (images, image_paths) in enumerate(dataloader):
            images = images.to(device)
            
            # Encode to latent space
            latent = encoder(images)
            
            latents.append(latent.cpu().numpy())
            paths.extend(image_paths)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    # Concatenate all latents
    latents = np.concatenate(latents, axis=0)
    
    print(f"\nExtracted {len(latents)} latent representations")
    print(f"Latent shape: {latents.shape}")
    print(f"Statistics:")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std:  {latents.std():.4f}")
    print(f"  Min:  {latents.min():.4f}")
    print(f"  Max:  {latents.max():.4f}")
    
    return latents, paths


def main():
    parser = argparse.ArgumentParser(description='Extract latent representations from moon tiles')
    parser.add_argument('--encoder_path', type=str, default='encoder_moon_zoom9.pth',
                       help='Path to trained encoder')
    parser.add_argument('--tile_dir', type=str,
                       default='/mnt/22tb-hdd/2tbssdcx-bkup/deepgis/deepgis_moon/static-root/moon-tiles',
                       help='Root directory containing moon tiles')
    parser.add_argument('--zoom_level', type=int, default=9,
                       help='Zoom level (default: 9)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for extraction')
    parser.add_argument('--output', type=str, default='latents_moon_zoom9.npy',
                       help='Output file for latents')
    parser.add_argument('--output_paths', type=str, default='latent_paths_moon_zoom9.txt',
                       help='Output file for paths')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--convert_to_rgb', action='store_true',
                       help='Convert grayscale to RGB')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load encoder
    print("Loading encoder...")
    encoder_checkpoint = torch.load(args.encoder_path, map_location=args.device)
    config = encoder_checkpoint['config']
    
    # Filter out non-MultispectralViT parameters
    encoder_params = {
        'img_size': config['img_size'],
        'patch_size': config['patch_size'],
        'in_channels': config.get('in_channels', 3),
        'embed_dim': config['embed_dim'],
        'num_heads': config['num_heads'],
        'num_layers': config['num_layers'],
        'use_cross_band_attention': config.get('use_cross_band_attention', True)
    }
    
    encoder = MultispectralViT(**encoder_params)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder = encoder.to(args.device)
    encoder.eval()
    
    # Create dataset
    print("Loading moon tile dataset...")
    in_channels = config.get('in_channels', 3)
    convert_to_rgb = args.convert_to_rgb or (in_channels == 3)
    
    dataset = MoonTileDataset(
        tile_dir=args.tile_dir,
        zoom_level=args.zoom_level,
        img_size=args.img_size,
        normalize=True,
        convert_to_rgb=convert_to_rgb,
        in_channels=in_channels
    )
    
    print(f"Dataset size: {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Extract latents
    latents, paths = extract_latents(encoder, dataloader, device=args.device)
    
    # Save latents
    print(f"\nSaving latents to {args.output}...")
    np.save(args.output, latents)
    
    # Save paths
    print(f"Saving paths to {args.output_paths}...")
    with open(args.output_paths, 'w') as f:
        f.write('\n'.join(paths))
    
    print("\nDone!")
    print(f"Latents saved: {args.output}")
    print(f"Paths saved: {args.output_paths}")


if __name__ == '__main__':
    main()

