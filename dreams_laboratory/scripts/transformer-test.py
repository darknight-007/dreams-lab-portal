"""
Transformer Encoder Testing Script for Image Latent Space Learning

This script implements a Vision Transformer (ViT)-style encoder that:
1. Splits images into patches
2. Processes patches through a transformer encoder
3. Learns a latent space representation
4. Can be used for downstream tasks or visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Tuple, Optional, List
import argparse


class ImagePatchDataset(Dataset):
    """Dataset for loading images and converting them to patches."""
    
    def __init__(self, image_dir: str, img_size: int = 224, patch_size: int = 16, 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir: Directory containing images
            img_size: Target image size (will be resized to this)
            patch_size: Size of each patch (patches will be patch_size x patch_size)
            transform: Optional torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Find all image files
        self.image_files = []
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in extensions:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        return img_tensor, str(img_path)


class PatchEmbedding(nn.Module):
    """Converts image patches into embeddings."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 512):
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
            x: (B, C, H, W) image tensor
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, num_patches_h, num_patches_w)
        B, C, H, W = x.shape
        # Reshape to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer Encoder for learning image latent spaces."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                         in_channels, embed_dim)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        # Optional: class token (for classification tasks)
        # For latent space learning, we can use CLS token or mean pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
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
            if m.shape[-1] == self.embed_dim:
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
            return_all: If True, return all patch embeddings; if False, return CLS token
        Returns:
            If return_all=False: (B, embed_dim) latent representation
            If return_all=True: (B, num_patches+1, embed_dim) all embeddings
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
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


class AutoencoderHead(nn.Module):
    """Simple decoder head for reconstruction task (optional)."""
    
    def __init__(self, embed_dim: int = 512, img_size: int = 224, 
                 patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Decoder: map latent back to patches
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * in_channels)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, embed_dim) latent representation
        Returns:
            (B, C, H, W) reconstructed image
        """
        B = latent.shape[0]
        # Decode to patches
        patches = self.decoder(latent)  # (B, patch_size^2 * C)
        
        # Reshape to image
        C = 3  # RGB
        patches = patches.view(B, self.num_patches, C, 
                              self.patch_size, self.patch_size)
        
        # Rearrange patches into image
        H_patches = W_patches = int(np.sqrt(self.num_patches))
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, C, num_patches, p, p)
        patches = patches.contiguous().view(B, C, H_patches, 
                                          self.patch_size, W_patches, 
                                          self.patch_size)
        patches = patches.permute(0, 1, 2, 4, 3, 5)  # (B, C, H_patches, W_patches, p, p)
        img = patches.contiguous().view(B, C, self.img_size, self.img_size)
        
        return torch.sigmoid(img)  # Normalize to [0, 1]


def train_model(model: nn.Module, dataloader: DataLoader, 
                num_epochs: int = 10, lr: float = 1e-4,
                device: str = 'cuda', use_decoder: bool = False):
    """Train the transformer encoder."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    decoder = None
    if use_decoder:
        decoder = AutoencoderHead(
            embed_dim=model.embed_dim,
            img_size=224,
            patch_size=16
        ).to(device)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(decoder.parameters()),
            lr=lr, weight_decay=0.01
        )
    
    model.train()
    if decoder:
        decoder.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            
            if use_decoder and decoder:
                # Reconstruction task
                latent = model(images)
                reconstructed = decoder(latent)
                # Resize reconstructed to match input
                reconstructed = nn.functional.interpolate(
                    reconstructed, size=images.shape[2:], mode='bilinear'
                )
                # Denormalize images for comparison
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                images_denorm = images * std + mean
                images_denorm = torch.clamp(images_denorm, 0, 1)
                loss = criterion(reconstructed, images_denorm)
            else:
                # Self-supervised: try to predict next patch or use contrastive loss
                # Simple version: mean squared error on embeddings
                latent = model(images)
                # Use a simple regularization loss
                loss = torch.mean(latent ** 2)  # L2 regularization on latent space
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return model, decoder


def visualize_latent_space(model: nn.Module, dataloader: DataLoader, 
                          device: str = 'cuda', num_samples: int = 100):
    """Extract and visualize latent space representations."""
    model.eval()
    latents = []
    image_paths = []
    
    with torch.no_grad():
        for idx, (images, paths) in enumerate(dataloader):
            if idx * dataloader.batch_size >= num_samples:
                break
            
            images = images.to(device)
            latent = model(images)
            latents.append(latent.cpu())
            image_paths.extend(paths)
    
    latents = torch.cat(latents, dim=0).numpy()
    
    # Use PCA or t-SNE for visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print(f"Extracted {latents.shape[0]} latent representations of dimension {latents.shape[1]}")
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d_pca = pca.fit_transform(latents)
    
    # t-SNE to 2D (may take longer)
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d_tsne = tsne.fit_transform(latents)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].scatter(latents_2d_pca[:, 0], latents_2d_pca[:, 1], alpha=0.6)
    axes[0].set_title('Latent Space (PCA)')
    axes[0].set_xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(latents_2d_tsne[:, 0], latents_2d_tsne[:, 1], alpha=0.6)
    axes[1].set_title('Latent Space (t-SNE)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_space_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'latent_space_visualization.png'")
    plt.close()
    
    return latents


def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer Encoder')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Embedding dimension (default: 512)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers (default: 6)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--use_decoder', action='store_true',
                       help='Use autoencoder reconstruction loss')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize latent space after training')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = ImagePatchDataset(
        image_dir=args.image_dir,
        img_size=args.img_size,
        patch_size=args.patch_size
    )
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    
    # Create model
    print("Creating model...")
    model = VisionTransformerEncoder(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Training model...")
    model, decoder = train_model(
        model, dataloader, num_epochs=args.epochs, 
        lr=args.lr, device=args.device, use_decoder=args.use_decoder
    )
    
    # Visualize latent space
    if args.visualize:
        print("Visualizing latent space...")
        visualize_latent_space(model, dataloader, device=args.device)
    
    # Save model
    save_path = 'transformer_encoder.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'img_size': args.img_size,
            'patch_size': args.patch_size,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers
        }
    }, save_path)
    print(f"Model saved to {save_path}")
    
    if decoder:
        decoder_path = 'transformer_decoder.pth'
        torch.save(decoder.state_dict(), decoder_path)
        print(f"Decoder saved to {decoder_path}")


if __name__ == '__main__':
    main()
