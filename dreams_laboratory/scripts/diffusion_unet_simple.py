#!/usr/bin/env python3
"""
Simplified U-Net for diffusion model with proper skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(min(32, out_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNetSimple(nn.Module):
    """Simplified U-Net for diffusion."""
    
    def __init__(self, img_channels=3, model_channels=128, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        
        time_emb_dim = model_channels * 4
        
        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input
        self.conv_in = nn.Conv2d(img_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down = nn.ModuleList()
        ch = model_channels
        chs = [ch]
        for mult in channel_mults:
            out_ch = model_channels * mult
            self.down.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_emb_dim),
                ResBlock(out_ch, out_ch, time_emb_dim),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if mult != channel_mults[-1] else nn.Identity()
            ]))
            ch = out_ch
            chs.append(ch)
        
        # Middle
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim),
            ResBlock(ch, ch, time_emb_dim)
        ])
        
        # Upsampling
        self.up = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = model_channels * mult
            self.up.append(nn.ModuleList([
                ResBlock(ch + chs.pop(), out_ch, time_emb_dim),
                ResBlock(out_ch, out_ch, time_emb_dim),
                nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1) if mult != channel_mults[0] else nn.Identity()
            ]))
            ch = out_ch
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, img_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, self.conv_in.out_channels)
        t_emb = self.time_embed(t_emb)
        
        # Input
        h = self.conv_in(x)
        
        # Downsampling
        hs = [h]
        for res1, res2, downsample in self.down:
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            hs.append(h)
            h = downsample(h)
        
        # Middle
        for block in self.mid:
            h = block(h, t_emb)
        
        # Upsampling
        for res1, res2, upsample in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = upsample(h)
        
        # Output
        return self.conv_out(h)


if __name__ == '__main__':
    model = UNetSimple()
    x = torch.randn(2, 3, 256, 256)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

