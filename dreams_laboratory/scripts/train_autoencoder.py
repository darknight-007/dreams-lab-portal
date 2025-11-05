#!/usr/bin/env python3
"""
Train Multispectral Autoencoder (Encoder + Decoder) for reconstruction.

This script trains both the encoder and decoder together to learn to reconstruct
multispectral images, enabling synthetic image generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multispectral_vit import MultispectralViT, MultispectralTileDataset
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder
import argparse
from pathlib import Path


def train_autoencoder(encoder, decoder, dataloader, num_epochs=10, lr=1e-4,
                     device='cuda', use_multi_gpu=False):
    """Train encoder-decoder autoencoder."""
    # Create autoencoder
    autoencoder = MultispectralAutoencoder(encoder, decoder)
    
    # Multi-GPU setup
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        autoencoder = nn.DataParallel(autoencoder)
        device_ids = list(range(torch.cuda.device_count()))
        autoencoder = autoencoder.to(f'cuda:{device_ids[0]}')
        primary_device = f'cuda:{device_ids[0]}'
    else:
        autoencoder = autoencoder.to(device)
        primary_device = device
    
    optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    autoencoder.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
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
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
    
    # Unwrap DataParallel if used
    if isinstance(autoencoder, nn.DataParallel):
        autoencoder = autoencoder.module
    
    return autoencoder.encoder, autoencoder.decoder


def main():
    parser = argparse.ArgumentParser(description='Train Multispectral Autoencoder')
    parser.add_argument('--tile_dir', type=str, 
                       default='/mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop',
                       help='Directory containing TIFF tiles')
    parser.add_argument('--encoder_path', type=str, default='multispectral_vit.pth',
                       help='Path to pre-trained encoder (optional)')
    parser.add_argument('--img_size', type=int, default=960,
                       help='Image size')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--in_channels', type=int, default=None,
                       help='Number of input channels (None=auto-detect, 3=RGB, 5=multispectral)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use all available GPUs')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load or create encoder
    if Path(args.encoder_path).exists():
        print(f"Loading pre-trained encoder from {args.encoder_path}")
        checkpoint = torch.load(args.encoder_path, map_location=args.device)
        encoder = MultispectralViT(**checkpoint['config'])
        encoder.load_state_dict(checkpoint['model_state_dict'])
        config = checkpoint['config']
        print(f"Loaded encoder with config: {config}")
    else:
        print("Creating new encoder...")
        encoder = MultispectralViT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels or 3,  # Default to RGB (3 channels)
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers
        )
        config = {
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'in_channels': args.in_channels or 3  # Default to RGB (3 channels)
        }
    
    # Create decoder (memory-efficient version)
    print("Creating decoder...")
    decoder = ReconstructionDecoder(
        embed_dim=config['embed_dim'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=config.get('in_channels', 3),  # Use config value, default to RGB (3 channels)
        num_layers=3,
        hidden_dim=512  # Small hidden dimension to save memory
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = MultispectralTileDataset(
        tile_dir=args.tile_dir,
        img_size=config['img_size'],
        normalize=True,
        in_channels=config.get('in_channels', None)  # Auto-detect if not in config
    )
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if args.multi_gpu else 2
    )
    
    # Train
    print("Training autoencoder...")
    encoder, decoder = train_autoencoder(
        encoder, decoder, dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_multi_gpu=args.multi_gpu
    )
    
    # Save models
    print("Saving models...")
    
    # Save encoder
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'config': config
    }, 'multispectral_vit.pth')
    
    # Save decoder
    torch.save(decoder.state_dict(), 'decoder.pth')
    
    # Save autoencoder (combined)
    autoencoder = MultispectralAutoencoder(encoder, decoder)
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'encoder_config': config
    }, 'multispectral_autoencoder.pth')
    
    print("Models saved:")
    print("  - multispectral_vit.pth: Encoder")
    print("  - decoder.pth: Decoder")
    print("  - multispectral_autoencoder.pth: Combined model")
    print("\nYou can now generate synthetic images using:")
    print("  python3 generate_synthetic.py --encoder_path multispectral_vit.pth --decoder_path decoder.pth")


if __name__ == '__main__':
    main()

