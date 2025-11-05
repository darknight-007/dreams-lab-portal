#!/usr/bin/env python3
"""
Train transformer-based autoencoder on Zoom Level 23 GIS tiles for generative synthesis.

This script trains an encoder-decoder model specifically on zoom level 23 tiles,
which contain the highest detail and most tiles (8,083 images).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from typing import Optional

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from multispectral_vit import MultispectralViT, MultispectralTileDataset
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder


class ZoomLevelDataset(MultispectralTileDataset):
    """
    Dataset that filters tiles by zoom level for GIS tile structures.
    
    GIS tiles typically follow structure: zoom_level/x_coordinate/y_coordinate.png
    """
    
    def __init__(self, tile_dir: str, zoom_level: int, img_size: int = 256,
                 transform=None, normalize: bool = True, in_channels: Optional[int] = None):
        """
        Args:
            tile_dir: Root directory containing tiles
            zoom_level: Zoom level to filter (e.g., 23)
            img_size: Target image size
            transform: Optional transforms
            normalize: Whether to normalize
            in_channels: Number of channels (None=auto-detect)
        """
        # Temporarily set tile_dir to zoom level subdirectory
        zoom_dir = Path(tile_dir) / str(zoom_level)
        if not zoom_dir.exists():
            raise ValueError(f"Zoom level {zoom_level} directory not found: {zoom_dir}")
        
        # Initialize parent class with zoom-specific directory
        super().__init__(
            tile_dir=str(zoom_dir),
            img_size=img_size,
            transform=transform,
            normalize=normalize,
            in_channels=in_channels
        )
        self.zoom_level = zoom_level


def train_autoencoder(encoder, decoder, dataloader, num_epochs=30, lr=1e-4,
                     device='cuda', use_multi_gpu=False, save_checkpoints=True,
                     checkpoint_dir='checkpoints'):
    """Train encoder-decoder autoencoder with checkpointing."""
    # Create checkpoint directory
    if save_checkpoints:
        Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Create autoencoder
    autoencoder = MultispectralAutoencoder(encoder, decoder)
    
    # Multi-GPU setup
    if use_multi_gpu and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for training")
        device_ids = list(range(num_gpus))
        autoencoder = autoencoder.to(f'cuda:{device_ids[0]}')
        autoencoder = nn.DataParallel(autoencoder, device_ids=device_ids)
        primary_device = f'cuda:{device_ids[0]}'
        print(f"GPUs: {device_ids}")
    else:
        autoencoder = autoencoder.to(device)
        primary_device = device
        if torch.cuda.device_count() > 1:
            print(f"Note: {torch.cuda.device_count()} GPUs available but multi-GPU not enabled.")
            print(f"      Add --multi_gpu flag to use all GPUs (faster training).")
    
    optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    autoencoder.train()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(primary_device)
            optimizer.zero_grad()
            
            # Forward pass: encode then decode
            reconstructed = autoencoder(images)
            
            # Reconstruction loss
            loss = criterion(reconstructed, images)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if save_checkpoints:
            # Unwrap DataParallel if used
            if isinstance(autoencoder, nn.DataParallel):
                model_to_save = autoencoder.module
            else:
                model_to_save = autoencoder
            
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': model_to_save.encoder.state_dict(),
                'decoder_state_dict': model_to_save.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            
            # Save latest checkpoint
            torch.save(checkpoint, Path(checkpoint_dir) / 'latest_checkpoint.pth')
            
            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, Path(checkpoint_dir) / 'best_checkpoint.pth')
                print(f"  → Saved best checkpoint (loss: {avg_loss:.6f})")
        
        print()
    
    # Unwrap DataParallel if used
    if isinstance(autoencoder, nn.DataParallel):
        autoencoder = autoencoder.module
    
    return autoencoder.encoder, autoencoder.decoder


def main():
    parser = argparse.ArgumentParser(
        description='Train transformer autoencoder on Zoom Level 23 GIS tiles'
    )
    parser.add_argument(
        '--tile_dir',
        type=str,
        default='/mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw',
        help='Root directory containing tile structure'
    )
    parser.add_argument(
        '--zoom_level',
        type=int,
        default=23,
        help='Zoom level to use (default: 23)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Image size (default: 256, matches tile size)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=16,
        help='Patch size for ViT (default: 16)'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=512,
        help='Embedding dimension (default: 512)'
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        help='Number of attention heads (default: 8)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=6,
        help='Number of transformer layers (default: 6)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (default: 16 for 256x256 images)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Use all available GPUs'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints_zoom23',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--no_checkpoints',
        action='store_true',
        help='Disable checkpoint saving'
    )
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.multi_gpu = False
    
    print("=" * 80)
    print("TRANSFORMER AUTOENCODER TRAINING - ZOOM LEVEL 23")
    print("=" * 80)
    print(f"Tile Directory: {args.tile_dir}")
    print(f"Zoom Level: {args.zoom_level}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Create dataset
    print("Loading Zoom Level 23 dataset...")
    try:
        dataset = ZoomLevelDataset(
            tile_dir=args.tile_dir,
            zoom_level=args.zoom_level,
            img_size=args.img_size,
            normalize=True,
            in_channels=3  # RGB
        )
        print(f"Found {len(dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if args.multi_gpu else 2,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    print(f"Batches per epoch: {len(dataloader)}")
    print()
    
    # Create encoder
    print("Creating encoder...")
    encoder = MultispectralViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,  # RGB
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_cross_band_attention=True
    )
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    
    # Create decoder
    print("Creating decoder...")
    decoder = ReconstructionDecoder(
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,  # RGB
        num_layers=3,
        hidden_dim=512
    )
    
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    print()
    
    # Train model
    print("Starting training...")
    encoder, decoder = train_autoencoder(
        encoder,
        decoder,
        dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_multi_gpu=args.multi_gpu,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save final models
    print()
    print("Saving final models...")
    
    config = {
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'in_channels': 3,
        'zoom_level': args.zoom_level
    }
    
    # Save encoder
    encoder_path = 'encoder_zoom23.pth'
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'config': config
    }, encoder_path)
    print(f"Encoder saved to: {encoder_path}")
    
    # Save decoder
    decoder_path = 'decoder_zoom23.pth'
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'config': config
    }, decoder_path)
    print(f"Decoder saved to: {decoder_path}")
    
    # Save autoencoder (combined)
    autoencoder = MultispectralAutoencoder(encoder, decoder)
    autoencoder_path = 'autoencoder_zoom23.pth'
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'encoder_config': config
    }, autoencoder_path)
    print(f"Autoencoder saved to: {autoencoder_path}")
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Generate synthetic samples:")
    print(f"   python3 generate_synthetic_zoom23.py --encoder_path {encoder_path} --decoder_path {decoder_path}")
    print()
    print("2. Test reconstruction quality:")
    print(f"   python3 test_reconstruction.py --encoder_path {encoder_path} --decoder_path {decoder_path}")


if __name__ == '__main__':
    main()

