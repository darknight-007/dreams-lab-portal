#!/usr/bin/env python3
"""
Test reconstruction quality of trained moon autoencoder.

This script loads a trained encoder-decoder and tests reconstruction quality
on moon tiles (zoom level 9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from multispectral_vit import MultispectralViT, MultispectralTileDataset
from multispectral_decoder import ReconstructionDecoder
from train_moon_autoencoder import MoonTileDataset


def calculate_metrics(reconstructed, original):
    """Calculate reconstruction quality metrics."""
    mse = F.mse_loss(reconstructed, original).item()
    mae = F.l1_loss(reconstructed, original).item()
    
    # PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr
    }


def test_reconstruction(encoder, decoder, dataloader, device='cuda', num_samples=10, output_dir='reconstruction_test_moon'):
    """Test reconstruction quality."""
    encoder.eval()
    decoder.eval()
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("TESTING MOON TILE RECONSTRUCTION QUALITY")
    print("="*80)
    
    all_metrics = []
    reconstructions = []
    originals = []
    paths = []
    
    with torch.no_grad():
        for idx, (images, image_paths) in enumerate(dataloader):
            if idx >= num_samples:
                break
            
            images = images.to(device)
            
            # Encode
            latents = encoder(images)
            
            # Decode
            reconstructed = decoder(latents)
            
            # Calculate metrics
            metrics = calculate_metrics(reconstructed, images)
            all_metrics.append(metrics)
            
            print(f"Sample {idx+1}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, PSNR={metrics['psnr']:.2f} dB")
            
            reconstructions.append(reconstructed[0].cpu())
            originals.append(images[0].cpu())
            paths.append(image_paths[0])
    
    # Calculate average metrics
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    
    print("\n" + "-"*80)
    print(f"Average Metrics:")
    print(f"  MSE:  {avg_mse:.6f}")
    print(f"  MAE:  {avg_mae:.6f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print("-"*80)
    
    # Visualize
    n_samples = len(reconstructions)
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for idx in range(n_samples):
        # Original
        orig = originals[idx]
        if orig.shape[0] == 1:
            # Grayscale
            orig_img = orig[0].numpy()
            axes[0, idx].imshow(orig_img, cmap='gray')
        else:
            # RGB
            orig_img = orig.permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img, 0, 1)
            axes[0, idx].imshow(orig_img)
        axes[0, idx].set_title(f'Original {idx+1}')
        axes[0, idx].axis('off')
        
        # Reconstructed
        recon = reconstructions[idx]
        if recon.shape[0] == 1:
            # Grayscale
            recon_img = recon[0].numpy()
            axes[1, idx].imshow(recon_img, cmap='gray')
        else:
            # RGB
            recon_img = recon.permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img, 0, 1)
            axes[1, idx].imshow(recon_img)
        axes[1, idx].set_title(f'Reconstructed\nMSE: {all_metrics[idx]["mse"]:.4f}')
        axes[1, idx].axis('off')
    
    plt.suptitle('Moon Tile Reconstruction Test Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reconstruction_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_dir}/reconstruction_test.png")
    plt.close()
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Test moon tile reconstruction quality')
    parser.add_argument('--encoder_path', type=str, default='encoder_moon_zoom9.pth',
                       help='Path to trained encoder')
    parser.add_argument('--decoder_path', type=str, default='decoder_moon_zoom9.pth',
                       help='Path to trained decoder')
    parser.add_argument('--tile_dir', type=str,
                       default='/mnt/22tb-hdd/2tbssdcx-bkup/deepgis/deepgis_moon/static-root/moon-tiles',
                       help='Root directory containing moon tiles')
    parser.add_argument('--zoom_level', type=int, default=9,
                       help='Zoom level (default: 9)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (default: 256)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='reconstruction_test_moon',
                       help='Output directory')
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
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Test reconstruction
    metrics = test_reconstruction(
        encoder, decoder, dataloader,
        device=args.device,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

