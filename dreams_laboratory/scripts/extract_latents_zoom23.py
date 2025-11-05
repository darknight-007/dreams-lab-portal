#!/usr/bin/env python3
"""
Extract latent representations from zoom level 23 tiles.

This script extracts latent representations from all images in the dataset,
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
from train_zoom23_autoencoder import ZoomLevelDataset


def extract_latents(encoder, dataloader, device='cuda'):
    """Extract latent representations from all images."""
    encoder.eval()
    latents = []
    paths = []
    
    print("Extracting latent representations...")
    
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
    parser = argparse.ArgumentParser(description='Extract latent representations')
    parser.add_argument('--encoder_path', type=str, default='encoder_zoom23.pth',
                       help='Path to trained encoder')
    parser.add_argument('--tile_dir', type=str,
                       default='/mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw',
                       help='Root directory containing tiles')
    parser.add_argument('--zoom_level', type=int, default=23,
                       help='Zoom level')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for extraction')
    parser.add_argument('--output', type=str, default='latents_zoom23.npy',
                       help='Output file for latents')
    parser.add_argument('--output_paths', type=str, default='latent_paths_zoom23.txt',
                       help='Output file for paths')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load encoder
    print("Loading encoder...")
    encoder_checkpoint = torch.load(args.encoder_path, map_location=args.device)
    config = encoder_checkpoint['config']
    
    encoder = MultispectralViT(**config)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder = encoder.to(args.device)
    encoder.eval()
    
    # Create dataset
    print("Loading dataset...")
    dataset = ZoomLevelDataset(
        tile_dir=args.tile_dir,
        zoom_level=args.zoom_level,
        img_size=args.img_size,
        normalize=True,
        in_channels=3
    )
    
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

