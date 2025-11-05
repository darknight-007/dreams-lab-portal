"""
Decoder implementations for Multispectral Vision Transformer Encoder.

This module provides different decoder architectures that can be paired with
the MultispectralViT encoder for various tasks:
1. Reconstruction Decoder - Reconstruct images from latent representations
2. Segmentation Decoder - Pixel-level classification
3. Transformer Decoder - Sequence generation/translation

Supports variable-channel imagery (RGB: 3 channels, Multispectral: 5+ channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ReconstructionDecoder(nn.Module):
    """
    Decoder for reconstructing multispectral images from latent representations.
    Supports variable-channel imagery (RGB: 3 channels, Multispectral: 5+ channels).
    
    Architecture: CLS token → MLP → Patches → Image
    Memory-efficient version using progressive upsampling.
    """
    
    def __init__(self, embed_dim: int = 512, img_size: int = 960, 
                 patch_size: int = 16, in_channels: int = 3,
                 num_layers: int = 3, hidden_dim: int = 2048):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Memory-efficient decoder: use much smaller hidden dimensions
        # Cap hidden_dim to save memory
        hidden_dim = min(hidden_dim, 512)  # Max 512 to keep memory manageable
        
        # Step 1: Latent → Patch embeddings (much smaller)
        # Use smaller intermediate dimension
        self.latent_to_patches = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_patches * 32)  # Even smaller patch embeddings (32-D)
        )
        
        # Step 2: Patch embeddings → Pixel patches (shared across all patches)
        self.patch_to_pixels = nn.Sequential(
            nn.Linear(32, patch_size * patch_size * in_channels),
            nn.GELU()
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, embed_dim) latent representation from encoder
        Returns:
            (B, C, H, W) reconstructed multispectral image where C is in_channels
        """
        B = latent.shape[0]
        
        # Step 1: Decode to patch embeddings (much smaller)
        # (B, embed_dim) -> (B, num_patches * 32)
        patch_embeddings_flat = self.latent_to_patches(latent)
        
        # Reshape to patches: (B, num_patches, 32)
        patch_embeddings = patch_embeddings_flat.view(
            B, self.num_patches, 32
        )
        
        # Step 2: Decode each patch to pixels (memory efficient)
        # (B, num_patches, 32) -> (B, num_patches, patch_size^2 * C)
        patches_flat = self.patch_to_pixels(patch_embeddings)
        
        # Reshape to patches: (B, num_patches, C, patch_size, patch_size)
        patches = patches_flat.view(
            B, self.num_patches, self.in_channels,
            self.patch_size, self.patch_size
        )
        
        # Rearrange patches into image
        H_patches = W_patches = int(np.sqrt(self.num_patches))
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, C, num_patches, p, p)
        patches = patches.contiguous().view(
            B, self.in_channels, H_patches, self.patch_size,
            W_patches, self.patch_size
        )
        patches = patches.permute(0, 1, 2, 4, 3, 5)  # (B, C, H_patches, W_patches, p, p)
        img = patches.contiguous().view(
            B, self.in_channels, self.img_size, self.img_size
        )
        
        # Normalize to [0, 1] using sigmoid
        return torch.sigmoid(img)


class PatchBasedDecoder(nn.Module):
    """
    Decoder that uses transformer decoder architecture.
    Reconstructs images by decoding patch embeddings.
    Supports variable-channel imagery (RGB: 3 channels, Multispectral: 5+ channels).
    
    Architecture: All patch embeddings → Transformer Decoder → Patches → Image
    """
    
    def __init__(self, embed_dim: int = 512, img_size: int = 960,
                 patch_size: int = 16, in_channels: int = 3,
                 num_layers: int = 6, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # Project patches back to image pixels
        self.patch_to_pixels = nn.Linear(
            embed_dim, patch_size * patch_size * in_channels
        )
        
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, num_patches+1, embed_dim) from encoder with return_all=True
        Returns:
            (B, C, H, W) reconstructed image where C is in_channels
        """
        B = encoder_output.shape[0]
        
        # Remove CLS token, keep only patch embeddings
        patch_embeddings = encoder_output[:, 1:]  # (B, num_patches, embed_dim)
        
        # Use CLS token as memory for decoder
        memory = encoder_output[:, 0:1]  # (B, 1, embed_dim)
        memory = memory.expand(-1, self.num_patches, -1)  # (B, num_patches, embed_dim)
        
        # Transformer decoder: self-attention + cross-attention
        decoded_patches = self.transformer_decoder(
            patch_embeddings,  # tgt (query)
            memory  # memory (key, value)
        )  # (B, num_patches, embed_dim)
        
        # Decode patches to pixels
        patches_flat = self.patch_to_pixels(decoded_patches)  # (B, num_patches, patch_size^2 * C)
        
        # Reshape to image
        patches = patches_flat.view(
            B, self.num_patches, self.in_channels,
            self.patch_size, self.patch_size
        )
        
        # Rearrange patches
        H_patches = W_patches = int(np.sqrt(self.num_patches))
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(
            B, self.in_channels, H_patches, self.patch_size,
            W_patches, self.patch_size
        )
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        img = patches.view(B, self.in_channels, self.img_size, self.img_size)
        
        return torch.sigmoid(img)


class SegmentationDecoder(nn.Module):
    """
    Decoder for pixel-level semantic segmentation.
    
    Architecture: Patch embeddings → Upsampling → Dense prediction
    """
    
    def __init__(self, embed_dim: int = 512, img_size: int = 960,
                 patch_size: int = 16, num_classes: int = 10,
                 num_layers: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        
        # Start from patch size
        current_size = img_size // patch_size  # e.g., 60 for 960/16
        current_dim = embed_dim
        
        # Progressive upsampling
        for i in range(num_layers):
            next_dim = max(current_dim // 2, 64)
            
            # Upsample and project
            self.upsample_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_dim, next_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(next_dim),
                nn.ReLU(inplace=True)
            ))
            
            current_dim = next_dim
            current_size *= 2
        
        # Final classification head
        self.classifier = nn.Conv2d(current_dim, num_classes, kernel_size=1)
        
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, num_patches+1, embed_dim) from encoder
        Returns:
            (B, num_classes, H, W) segmentation mask
        """
        B = encoder_output.shape[0]
        
        # Remove CLS token
        patch_embeddings = encoder_output[:, 1:]  # (B, num_patches, embed_dim)
        
        # Reshape to spatial grid: (B, num_patches, embed_dim) -> (B, embed_dim, H_patches, W_patches)
        H_patches = W_patches = int(np.sqrt(self.num_patches))
        x = patch_embeddings.view(B, H_patches, W_patches, self.embed_dim)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H_patches, W_patches)
        
        # Progressive upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        
        # Final classification
        segmentation = self.classifier(x)  # (B, num_classes, H, W)
        
        # Upsample to exact image size if needed
        if x.shape[2] != self.img_size:
            segmentation = F.interpolate(
                segmentation, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )
        
        return segmentation


class MultispectralAutoencoder(nn.Module):
    """
    Complete encoder-decoder architecture for reconstruction.
    """
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) multispectral image where C is number of channels
        Returns:
            (B, C, H, W) reconstructed image
        """
        # Encode
        latent = self.encoder(x)  # (B, embed_dim)
        
        # Decode
        reconstructed = self.decoder(latent)  # (B, C, H, W)
        
        return reconstructed


class MultispectralSegmentationModel(nn.Module):
    """
    Encoder-decoder for semantic segmentation.
    """
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) multispectral image where C is number of channels
        Returns:
            (B, num_classes, H, W) segmentation logits
        """
        # Encode with all patch embeddings
        encoder_output = self.encoder(x, return_all=True)  # (B, num_patches+1, embed_dim)
        
        # Decode to segmentation
        segmentation = self.decoder(encoder_output)  # (B, num_classes, H, W)
        
        return segmentation


def create_reconstruction_decoder(encoder_config: dict) -> ReconstructionDecoder:
    """Create a decoder matching the encoder configuration."""
    return ReconstructionDecoder(
        embed_dim=encoder_config.get('embed_dim', 512),
        img_size=encoder_config.get('img_size', 960),
        patch_size=encoder_config.get('patch_size', 16),
        in_channels=encoder_config.get('in_channels', 3),  # Default to RGB (3 channels)
        num_layers=3
    )


def create_segmentation_decoder(encoder_config: dict, num_classes: int = 10) -> SegmentationDecoder:
    """Create a segmentation decoder matching the encoder configuration."""
    return SegmentationDecoder(
        embed_dim=encoder_config.get('embed_dim', 512),
        img_size=encoder_config.get('img_size', 960),
        patch_size=encoder_config.get('patch_size', 16),
        num_classes=num_classes,
        num_layers=4
    )


if __name__ == '__main__':
    # Example usage
    from multispectral_vit import MultispectralViT
    
    # Create encoder
    encoder = MultispectralViT(
        img_size=960,
        patch_size=16,
        embed_dim=512,
        num_heads=8,
        num_layers=6
    )
    
    # Create decoder
    decoder = ReconstructionDecoder(
        embed_dim=512,
        img_size=960,
        patch_size=16,
        in_channels=3  # RGB example (can be 5 for multispectral)
    )
    
    # Create autoencoder
    autoencoder = MultispectralAutoencoder(encoder, decoder)
    
    # Test forward pass
    x = torch.randn(2, 3, 960, 960)  # RGB example (can be 5 for multispectral)
    reconstructed = autoencoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")


