#!/usr/bin/env python3
"""
Denoising Diffusion Probabilistic Model (DDPM) for Rock Tile Generation.

Implements a diffusion model for generating high-quality synthetic rock tiles.
Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: (B,) tensor of timestep indices
        embedding_dim: Dimension of embedding
    
    Returns:
        (B, embedding_dim) tensor of embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # Zero pad if odd dimension
        emb = F.pad(emb, (0, 1, 0, 0))
    
    return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_emb_dim)
        """
        residual = x
        
        # First conv
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]
        
        # Second conv
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        # Residual connection
        return x + self.residual_conv(residual)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        residual = x
        x = self.norm(x)
        
        # QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, C/heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Output
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj_out(out)
        
        return out + residual


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2, 
                 downsample=True, use_attention=False):
        super().__init__()
        
        self.resblocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])
        
        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = None
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x, time_emb):
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(self, in_channels, out_channels, skip_channels, time_emb_dim, num_layers=2,
                 upsample=True, use_attention=False):
        super().__init__()
        
        # First resblock handles concatenated skip connection
        self.resblocks = nn.ModuleList([
            ResidualBlock(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])
        
        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = None
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = None
    
    def forward(self, x, time_emb, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        for resblock in self.resblocks:
            x = resblock(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net architecture for diffusion model.
    Predicts noise added to images at each diffusion timestep.
    """
    
    def __init__(self, img_channels=3, base_channels=128, channel_mult=(1, 2, 2, 4),
                 num_res_blocks=2, attention_resolutions=(16,), dropout=0.1,
                 time_emb_dim=512):
        super().__init__()
        
        self.img_channels = img_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    DownBlock(
                        now_channels,
                        out_channels,
                        time_emb_dim,
                        num_layers=1,
                        downsample=False,
                        use_attention=False  # Can enable for certain resolutions
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            # Downsample (except for last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    DownBlock(
                        now_channels,
                        now_channels,
                        time_emb_dim,
                        num_layers=1,
                        downsample=True,
                        use_attention=False
                    )
                )
                channels.append(now_channels)
        
        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim)
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_channels = channels[-1] if channels else 0
                if channels:
                    channels.pop()
                
                self.up_blocks.append(
                    UpBlock(
                        now_channels,
                        out_channels,
                        skip_channels,
                        time_emb_dim,
                        num_layers=1,
                        upsample=False,
                        use_attention=False
                    )
                )
                now_channels = out_channels
            
            # Upsample (except for first level)
            if level != 0:
                self.up_blocks.append(
                    UpBlock(
                        now_channels,
                        now_channels,
                        0,  # No skip connection for upsampling blocks
                        time_emb_dim,
                        num_layers=1,
                        upsample=True,
                        use_attention=False
                    )
                )
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, img_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps):
        """
        Args:
            x: (B, C, H, W) noisy images
            timesteps: (B,) timestep indices
        
        Returns:
            (B, C, H, W) predicted noise
        """
        # Time embedding
        time_emb = get_timestep_embedding(timesteps, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Downsampling with skip connections
        skips = [x]
        for block in self.down_blocks:
            x = block(x, time_emb)
            skips.append(x)
        
        # Middle
        for block in self.middle:
            if isinstance(block, ResidualBlock):
                x = block(x, time_emb)
            else:
                x = block(x)
        
        # Upsampling with skip connections
        for block in self.up_blocks:
            skip = skips.pop() if skips else None
            x = block(x, time_emb, skip)
        
        # Output
        x = self.conv_out(x)
        
        return x


class GaussianDiffusion:
    """
    Gaussian Diffusion Process for DDPM.
    Handles noise scheduling and sampling.
    """
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to images.
        
        Args:
            x_start: (B, C, H, W) clean images
            t: (B,) timestep indices
            noise: Optional pre-generated noise
        
        Returns:
            (B, C, H, W) noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, clip_denoised=True):
        """
        Reverse diffusion process: denoise images by one step.
        
        Args:
            model: UNet model
            x_t: (B, C, H, W) noisy images at timestep t
            t: (B,) timestep indices
            clip_denoised: Whether to clip output to [-1, 1]
        
        Returns:
            (B, C, H, W) denoised images at timestep t-1
        """
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Calculate x_0
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        x_0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        
        if clip_denoised:
            x_0 = torch.clamp(x_0, -1, 1)
        
        # Calculate mean of posterior
        posterior_mean_coef1_t = self.posterior_mean_coef1[t][:, None, None, None]
        posterior_mean_coef2_t = self.posterior_mean_coef2[t][:, None, None, None]
        
        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        
        # Calculate variance
        posterior_variance_t = self.posterior_variance[t][:, None, None, None]
        
        # Sample
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float()[:, None, None, None]
        
        return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, progress=True):
        """
        Generate samples by iteratively denoising from pure noise.
        
        Args:
            model: UNet model
            shape: (B, C, H, W) shape of images to generate
            progress: Whether to show progress
        
        Returns:
            (B, C, H, W) generated images
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        timesteps = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc='Sampling')
        
        for i in timesteps:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
        
        return img
    
    def training_losses(self, model, x_start):
        """
        Calculate training loss for batch.
        
        Args:
            model: UNet model
            x_start: (B, C, H, W) clean images
        
        Returns:
            loss: scalar loss value
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to images
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # Calculate loss (simple MSE)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


if __name__ == '__main__':
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = UNet(img_channels=3, base_channels=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create diffusion
    diffusion = GaussianDiffusion(num_timesteps=1000, device=device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test training loss
    loss = diffusion.training_losses(model, x)
    print(f"Loss: {loss.item():.6f}")

