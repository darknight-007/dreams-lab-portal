"""
Multispectral Vision Transformer for Bishop Rocky Scarp Dataset

Adapts Vision Transformer to handle 5-band MicaSense RedEdge-MX imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import argparse
import rasterio
from rasterio.windows import Window


class MultispectralPatchEmbedding(nn.Module):
    """Patch embedding for multispectral images (5 bands)."""
    
    def __init__(self, img_size: int = 960, patch_size: int = 16, 
                 in_channels: int = 5, embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Convolution to create patches and project to embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, H, W) multispectral image tensor
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        # (B, 5, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # Flatten spatial dimensions
        B, C, H, W = x.shape
        # Reshape to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class CrossBandAttention(nn.Module):
    """Attention mechanism to learn relationships between spectral bands."""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, embed_dim) patch embeddings
        Returns:
            (B, num_patches, embed_dim) attended embeddings
        """
        B, N, E = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out


class MultispectralViT(nn.Module):
    """Multispectral Vision Transformer for 5-band imagery."""
    
    def __init__(self, img_size: int = 960, patch_size: int = 16, 
                 in_channels: int = 5, embed_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 use_cross_band_attention: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.use_cross_band_attention = use_cross_band_attention
        
        # Patch embedding
        self.patch_embed = MultispectralPatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Optional: Cross-band attention before transformer
        if use_cross_band_attention:
            self.cross_band_attn = CrossBandAttention(embed_dim, num_heads)
            self.cross_band_norm = nn.LayerNorm(embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            if len(m.shape) == 2 and m.shape[-1] == self.embed_dim:
                torch.nn.init.trunc_normal_(m, std=0.02)
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, 5, H, W) multispectral image tensor
            return_all: If True, return all patch embeddings; if False, return CLS token
        Returns:
            If return_all=False: (B, embed_dim) latent representation
            If return_all=True: (B, num_patches+1, embed_dim) all embeddings
        """
        B = x.shape[0]
        
        # Patch embedding: (B, 5, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Optional cross-band attention
        if self.use_cross_band_attention:
            x = x + self.cross_band_attn(self.cross_band_norm(x))
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add positional encoding
        pos_embed_with_cls = torch.cat([
            torch.zeros(1, 1, self.embed_dim, device=x.device),
            self.pos_embed
        ], dim=1)
        x = x + pos_embed_with_cls
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Layer norm
        x = self.norm(x)
        
        if return_all:
            return x  # (B, num_patches+1, embed_dim)
        else:
            # Return CLS token as latent representation
            return x[:, 0]  # (B, embed_dim)


class MultispectralTileDataset(Dataset):
    """Dataset for loading multispectral TIFF tiles."""
    
    def __init__(self, tile_dir: str, img_size: int = 960, 
                 transform: Optional[transforms.Compose] = None,
                 normalize: bool = True):
        """
        Args:
            tile_dir: Directory containing TIFF files
            img_size: Target image size (will be resized/cropped)
            transform: Optional torchvision transforms
            normalize: Whether to normalize to [0, 1] or use z-score
        """
        self.tile_dir = Path(tile_dir)
        self.img_size = img_size
        self.normalize = normalize
        
        # Find all TIFF files
        self.tiff_files = sorted(list(self.tile_dir.glob('**/*.tif')))
        self.tiff_files = [f for f in self.tiff_files if '__MACOSX' not in str(f)]
        
        if len(self.tiff_files) == 0:
            raise ValueError(f"No TIFF files found in {tile_dir}")
        
        print(f"Found {len(self.tiff_files)} TIFF files")
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.tiff_files)
    
    def __getitem__(self, idx):
        tiff_path = self.tiff_files[idx]
        
        try:
            # Read multispectral TIFF (expecting 5 bands)
            with rasterio.open(tiff_path) as src:
                # Read all bands
                bands = []
                for i in range(1, min(6, src.count + 1)):  # Read up to 5 bands
                    band = src.read(i)
                    bands.append(band)
                
                # If fewer than 5 bands, pad with zeros or repeat last band
                while len(bands) < 5:
                    bands.append(bands[-1] if bands else np.zeros_like(bands[0]))
                
                # Stack bands: (5, H, W)
                multispectral = np.stack(bands[:5], axis=0)
                
                # Convert to float32
                multispectral = multispectral.astype(np.float32)
                
                # Normalize
                if self.normalize:
                    # Clip to reasonable range and normalize
                    # For 16-bit data, typical range is 0-65535
                    multispectral = np.clip(multispectral, 0, 65535)
                    multispectral = multispectral / 65535.0  # Normalize to [0, 1]
                else:
                    # Z-score normalization per band
                    for i in range(5):
                        band = multispectral[i]
                        mean = band.mean()
                        std = band.std()
                        if std > 0:
                            multispectral[i] = (band - mean) / std
            
            # Convert to tensor
            multispectral = torch.from_numpy(multispectral)
            
            # Apply transforms (resize, etc.)
            # Note: transforms work on PIL Images, so we'll handle resizing manually
            if multispectral.shape[1] != self.img_size or multispectral.shape[2] != self.img_size:
                multispectral = F.interpolate(
                    multispectral.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            return multispectral, str(tiff_path)
            
        except Exception as e:
            print(f"Error loading {tiff_path}: {e}")
            # Return zeros on error
            return torch.zeros(5, self.img_size, self.img_size), str(tiff_path)


def train_multispectral_vit(model: nn.Module, dataloader: DataLoader, 
                           num_epochs: int = 10, lr: float = 1e-4,
                           device: str = 'cuda', use_multi_gpu: bool = False):
    """Train the multispectral Vision Transformer."""
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
        device_ids = list(range(torch.cuda.device_count()))
        model = model.to(f'cuda:{device_ids[0]}')
        primary_device = f'cuda:{device_ids[0]}'
    else:
        model = model.to(device)
        primary_device = device
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Self-supervised learning: masked patch prediction
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(primary_device)
            optimizer.zero_grad()
            
            # Extract latent representation
            latent = model(images)
            
            # Simple self-supervised loss: L2 regularization on latent space
            # In practice, you might want:
            # - Masked patch prediction
            # - Contrastive learning
            # - Reconstruction loss
            loss = torch.mean(latent ** 2) + 0.1 * torch.mean(torch.abs(latent))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return model


def extract_latent_representations(model: nn.Module, dataloader: DataLoader,
                                   device: str = 'cuda', num_samples: Optional[int] = None,
                                   use_multi_gpu: bool = False):
    """Extract latent representations for all tiles."""
    model.eval()
    latents = []
    tile_paths = []
    
    # Determine primary device
    if use_multi_gpu and isinstance(model, nn.DataParallel):
        primary_device = next(model.parameters()).device
    else:
        primary_device = device
    
    with torch.no_grad():
        for idx, (images, paths) in enumerate(dataloader):
            if num_samples and idx * dataloader.batch_size >= num_samples:
                break
            
            images = images.to(primary_device)
            latent = model(images)
            latents.append(latent.cpu())
            tile_paths.extend(paths)
    
    latents = torch.cat(latents, dim=0).numpy()
    return latents, tile_paths


def main():
    parser = argparse.ArgumentParser(description='Multispectral Vision Transformer')
    parser.add_argument('--tile_dir', type=str, 
                       default='/mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop',
                       help='Directory containing TIFF tiles')
    parser.add_argument('--img_size', type=int, default=960,
                       help='Image size (default: 960, matches tile size)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Embedding dimension (default: 512)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4, smaller due to 5-channel input)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use all available GPUs (DataParallel)')
    parser.add_argument('--extract_latents', action='store_true',
                       help='Extract latent representations after training')
    
    args = parser.parse_args()
    
    # Check device and GPUs
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.multi_gpu = False
    elif args.multi_gpu:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Found {num_gpus} GPUs")
            print(f"  {torch.cuda.get_device_name(0)}")
            for i in range(1, num_gpus):
                print(f"  {torch.cuda.get_device_name(i)}")
        else:
            print(f"Only 1 GPU found, multi-GPU disabled")
            args.multi_gpu = False
    
    # Create dataset and dataloader
    print("Loading multispectral tiles...")
    dataset = MultispectralTileDataset(
        tile_dir=args.tile_dir,
        img_size=args.img_size,
        normalize=True
    )
    
    # Adjust batch size for multi-GPU
    effective_batch_size = args.batch_size
    if args.multi_gpu and torch.cuda.device_count() > 1:
        effective_batch_size = args.batch_size * torch.cuda.device_count()
        print(f"Multi-GPU: Effective batch size = {effective_batch_size} "
              f"({args.batch_size} per GPU × {torch.cuda.device_count()} GPUs)")
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4 if args.multi_gpu else 2  # More workers for multi-GPU
    )
    
    # Create model
    print("Creating Multispectral Vision Transformer...")
    model = MultispectralViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=5,  # 5 bands from RedEdge-MX
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_cross_band_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Training model...")
    model = train_multispectral_vit(
        model, dataloader, num_epochs=args.epochs, 
        lr=args.lr, device=args.device, use_multi_gpu=args.multi_gpu
    )
    
    # Extract latent representations
    if args.extract_latents:
        print("Extracting latent representations...")
        latents, paths = extract_latent_representations(
            model, dataloader, device=args.device, use_multi_gpu=args.multi_gpu
        )
        print(f"Extracted {latents.shape[0]} latent representations")
        print(f"Latent dimension: {latents.shape[1]}")
        
        # Save latents
        np.save('multispectral_latents.npy', latents)
        with open('multispectral_tile_paths.txt', 'w') as f:
            f.write('\n'.join(paths))
        print("Saved latents to 'multispectral_latents.npy'")
        print("Saved paths to 'multispectral_tile_paths.txt'")
    
    # Save model (unwrap DataParallel if used)
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    
    save_path = 'multispectral_vit.pth'
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'config': {
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'in_channels': 5
        }
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    import torch.optim as optim
    main()

