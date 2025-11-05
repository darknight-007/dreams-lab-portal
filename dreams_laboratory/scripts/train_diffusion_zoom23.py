#!/usr/bin/env python3
"""
Train diffusion model on Zoom Level 23 rock tiles.

This script trains a DDPM (Denoising Diffusion Probabilistic Model) for
generating high-quality synthetic rock tiles.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from diffusion_model import GaussianDiffusion
from diffusion_unet_simple import UNetSimple
from multispectral_vit import MultispectralTileDataset


def train_diffusion(model, diffusion, dataloader, num_epochs=100, lr=2e-4,
                   device='cuda', use_multi_gpu=False, save_checkpoints=True,
                   checkpoint_dir='checkpoints_diffusion', ema_decay=0.9999):
    """Train diffusion model with EMA."""
    
    # Create checkpoint directory
    if save_checkpoints:
        Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Multi-GPU setup
    if use_multi_gpu and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for training")
        device_ids = list(range(num_gpus))
        model = model.to(f'cuda:{device_ids[0]}')
        model = nn.DataParallel(model, device_ids=device_ids)
        primary_device = f'cuda:{device_ids[0]}'
        print(f"GPUs: {device_ids}")
    else:
        model = model.to(device)
        primary_device = device
        if torch.cuda.device_count() > 1:
            print(f"Note: {torch.cuda.device_count()} GPUs available but multi-GPU not enabled.")
            print(f"      Add --multi_gpu flag to use all GPUs.")
    
    # Exponential Moving Average (EMA) for better generation
    if ema_decay > 0:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    else:
        ema_model = None
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(primary_device)
            
            # Normalize to [-1, 1] for diffusion
            images = images * 2.0 - 1.0
            
            optimizer.zero_grad()
            
            # Calculate loss
            loss = diffusion.training_losses(model, images)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA
            if ema_model is not None:
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.6e}')
        
        # Save checkpoint
        if save_checkpoints and (epoch + 1) % 5 == 0:
            # Unwrap DataParallel if used
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
                ema_to_save = ema_model.module if ema_model is not None else None
            else:
                model_to_save = model
                ema_to_save = ema_model
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if ema_to_save is not None else None,
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
    if isinstance(model, nn.DataParallel):
        model = model.module
        ema_model = ema_model.module if ema_model is not None else None
    
    return model, ema_model


def main():
    parser = argparse.ArgumentParser(
        description='Train diffusion model on Zoom Level 23 rock tiles'
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
        help='Image size (default: 256)'
    )
    parser.add_argument(
        '--base_channels',
        type=int,
        default=128,
        help='Base number of channels in UNet (default: 128)'
    )
    parser.add_argument(
        '--num_timesteps',
        type=int,
        default=1000,
        help='Number of diffusion timesteps (default: 1000)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
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
        default='checkpoints_diffusion_zoom23',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--no_ema',
        action='store_true',
        help='Disable EMA (Exponential Moving Average)'
    )
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.multi_gpu = False
    
    print("=" * 80)
    print("DIFFUSION MODEL TRAINING - ZOOM LEVEL 23")
    print("=" * 80)
    print(f"Tile Directory: {args.tile_dir}")
    print(f"Zoom Level: {args.zoom_level}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Diffusion Timesteps: {args.num_timesteps}")
    print(f"Base Channels: {args.base_channels}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"EMA: {'Disabled' if args.no_ema else 'Enabled'}")
    print("=" * 80)
    print()
    
    # Create dataset
    print("Loading Zoom Level 23 dataset...")
    try:
        from train_zoom23_autoencoder import ZoomLevelDataset
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
    
    # Create UNet model
    print("Creating UNet model...")
    model = UNetSimple(
        img_channels=3,
        model_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8)
    )
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    print()
    
    # Create diffusion
    print("Creating diffusion process...")
    diffusion = GaussianDiffusion(
        num_timesteps=args.num_timesteps,
        device=args.device
    )
    print(f"Diffusion timesteps: {args.num_timesteps}")
    print()
    
    # Train model
    print("Starting training...")
    ema_decay = 0.0 if args.no_ema else 0.9999
    model, ema_model = train_diffusion(
        model,
        diffusion,
        dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_multi_gpu=args.multi_gpu,
        save_checkpoints=True,
        checkpoint_dir=args.checkpoint_dir,
        ema_decay=ema_decay
    )
    
    # Save final models
    print()
    print("Saving final models...")
    
    config = {
        'img_size': args.img_size,
        'base_channels': args.base_channels,
        'num_timesteps': args.num_timesteps,
        'zoom_level': args.zoom_level
    }
    
    # Save main model
    model_path = 'diffusion_model_zoom23.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save EMA model if available
    if ema_model is not None:
        ema_path = 'diffusion_model_zoom23_ema.pth'
        torch.save({
            'model_state_dict': ema_model.state_dict(),
            'config': config
        }, ema_path)
        print(f"EMA model saved to: {ema_path}")
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Generate synthetic samples:")
    print(f"   python3 generate_diffusion_zoom23.py --model_path {model_path}")
    print()
    print("2. Use EMA model for better quality:")
    if ema_model is not None:
        print(f"   python3 generate_diffusion_zoom23.py --model_path {ema_path}")
    print()


if __name__ == '__main__':
    main()

