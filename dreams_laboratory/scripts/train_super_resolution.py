#!/usr/bin/env python3
"""
Train a super-resolution network to upscale autoencoder outputs.

This network learns to upscale 256x256 generated images to higher resolutions
by training on real tile data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import sys

scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class ResidualBlock(nn.Module):
    """Residual block for super-resolution."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class SuperResolutionNet(nn.Module):
    """
    Enhanced Deep Residual Network for Super-Resolution (EDSR-style).
    Upscales images by 2x, 4x, or 8x using sub-pixel convolution.
    """
    
    def __init__(self, scale_factor=2, num_channels=3, num_features=64, num_blocks=16):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.input_conv = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        
        # Mid-level feature processing
        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.mid_bn = nn.BatchNorm2d(num_features)
        
        # Upsampling using sub-pixel convolution (PixelShuffle)
        # For 2x: one upsampling layer
        # For 4x: two upsampling layers
        # For 8x: three upsampling layers
        num_upsample = int(np.log2(scale_factor))
        
        self.upsample_layers = nn.ModuleList()
        for _ in range(num_upsample):
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # 2x upsampling
                nn.ReLU(inplace=True)
            ))
        
        # Final output layer
        self.output_conv = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Extract features
        feat = self.input_conv(x)
        
        # Residual learning
        res = self.residual_blocks(feat)
        res = self.mid_bn(self.mid_conv(res))
        feat = feat + res
        
        # Upsample
        for upsample_layer in self.upsample_layers:
            feat = upsample_layer(feat)
        
        # Output
        out = self.output_conv(feat)
        
        # Add skip connection from input (bicubic upsampled)
        x_upsampled = F.interpolate(x, scale_factor=self.scale_factor, 
                                     mode='bicubic', align_corners=False)
        out = out + x_upsampled
        
        return torch.clamp(out, 0, 1)


class SuperResolutionDataset(Dataset):
    """Dataset for training super-resolution on real tiles."""
    
    def __init__(self, tile_paths, low_res_size=256, high_res_size=512):
        self.tile_paths = tile_paths
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.scale_factor = high_res_size // low_res_size
        
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        # Load high-res image
        img_path = self.tile_paths[idx]
        img_hr = Image.open(img_path).convert('RGB')
        
        # Resize to target high-res size
        img_hr = img_hr.resize((self.high_res_size, self.high_res_size), Image.BICUBIC)
        img_hr = np.array(img_hr).astype(np.float32) / 255.0
        img_hr = torch.from_numpy(img_hr).permute(2, 0, 1)  # (C, H, W)
        
        # Create low-res version
        img_lr = F.interpolate(img_hr.unsqueeze(0), size=(self.low_res_size, self.low_res_size),
                               mode='bicubic', align_corners=False).squeeze(0)
        
        return img_lr, img_hr


def train_super_resolution(model, dataloader, num_epochs=100, lr=1e-4, device='cuda'):
    """Train super-resolution model."""
    model = model.to(device)
    
    # Loss functions
    criterion_pixel = nn.L1Loss()  # L1 loss for pixel-level
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Generate super-resolved images
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion_pixel(sr_imgs, hr_imgs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
        
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'super_resolution_best.pth')
            print(f"  → Saved best model (loss: {avg_loss:.6f})")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train super-resolution network')
    parser.add_argument('--tile_dir', type=str, required=True,
                       help='Directory containing training tiles')
    parser.add_argument('--low_res', type=int, default=256,
                       help='Low resolution size')
    parser.add_argument('--high_res', type=int, default=512,
                       help='High resolution size')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_blocks', type=int, default=16,
                       help='Number of residual blocks')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Find all tile images
    tile_paths = []
    tile_dir = Path(args.tile_dir)
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        tile_paths.extend(list(tile_dir.rglob(ext)))
    
    print(f"Found {len(tile_paths)} tiles for training")
    
    scale_factor = args.high_res // args.low_res
    print(f"Training {scale_factor}x super-resolution: {args.low_res} → {args.high_res}")
    
    # Create dataset and dataloader
    dataset = SuperResolutionDataset(tile_paths, args.low_res, args.high_res)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=4, pin_memory=True)
    
    # Create model
    model = SuperResolutionNet(
        scale_factor=scale_factor,
        num_channels=3,
        num_features=64,
        num_blocks=args.num_blocks
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model = train_super_resolution(
        model, dataloader, num_epochs=args.epochs, 
        lr=args.lr, device=args.device
    )
    
    # Save final model
    torch.save(model.state_dict(), 'super_resolution_final.pth')
    print("Training complete! Model saved.")


if __name__ == '__main__':
    main()

