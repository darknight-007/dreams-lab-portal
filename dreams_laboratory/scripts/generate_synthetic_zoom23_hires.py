#!/usr/bin/env python3
"""
Generate high-resolution synthetic RGB images from trained zoom level 23 autoencoder.

This script generates synthetic images at higher resolution than training by:
1. Generating at native decoder resolution (256x256)
2. Upscaling using high-quality interpolation methods
3. Optional: Using Real-ESRGAN or similar super-resolution if available

Optimized for Titan RTX (24GB VRAM) - can handle larger batches and higher resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from PIL import Image

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from multispectral_decoder import ReconstructionDecoder
from multispectral_vit import MultispectralViT

# Try to import super-resolution model
try:
    from train_super_resolution import SuperResolutionNet
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


class HighResUpsampler(nn.Module):
    """
    High-quality upsampler using learned convolutions.
    Uses sub-pixel convolution (pixel shuffle) for better quality.
    """
    
    def __init__(self, in_channels=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Upsampling using sub-pixel convolution
        self.upsample = nn.Conv2d(64, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Extract features
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        feat = self.relu(self.conv3(feat))
        
        # Upsample using pixel shuffle
        upsampled = self.upsample(feat)
        upsampled = self.pixel_shuffle(upsampled)
        
        # Residual connection: upsample input and add
        x_upsampled = F.interpolate(x, scale_factor=self.scale_factor, 
                                     mode='bicubic', align_corners=False)
        return torch.clamp(upsampled + x_upsampled, 0, 1)


def sample_random_latents(num_samples: int, latents: np.ndarray,
                          method: str = 'gaussian') -> np.ndarray:
    """Sample random latents from distribution."""
    if method == 'gaussian':
        mean = latents.mean(axis=0)
        std = latents.std(axis=0)
        std = std * 0.5  # Reduce variance for more realistic samples
        samples = np.random.normal(mean, std, (num_samples, latents.shape[1]))
        
        # Clip to data range
        min_vals = latents.min(axis=0)
        max_vals = latents.max(axis=0)
        samples = np.clip(samples, min_vals, max_vals)
        
    elif method == 'uniform':
        min_vals = latents.min(axis=0)
        max_vals = latents.max(axis=0)
        samples = np.random.uniform(min_vals, max_vals, (num_samples, latents.shape[1]))
        
    elif method == 'real':
        indices = np.random.choice(len(latents), num_samples, replace=False)
        samples = latents[indices]
        
    elif method == 'interpolate':
        # Interpolate between random pairs
        samples = []
        for _ in range(num_samples):
            idx1, idx2 = np.random.choice(len(latents), 2, replace=False)
            alpha = np.random.uniform(0, 1)
            sample = alpha * latents[idx1] + (1 - alpha) * latents[idx2]
            samples.append(sample)
        samples = np.array(samples)
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return samples


def generate_synthetic_images(decoder, latents, device='cuda', num_samples=20, 
                              method='gaussian', batch_size=32, target_size=None,
                              upscale_method='bicubic', use_learned_upsampler=False):
    """
    Generate synthetic images from latents with optional upscaling.
    
    Args:
        decoder: Trained decoder model
        latents: Latent representations from training data
        device: Device to use
        num_samples: Number of images to generate
        method: Sampling method ('gaussian', 'uniform', 'real', 'interpolate')
        batch_size: Batch size for generation (Titan RTX can handle larger batches)
        target_size: Target size for upscaling (None = no upscaling)
        upscale_method: Upscaling method ('bicubic', 'lanczos', 'learned')
        use_learned_upsampler: Use learned upsampler (experimental)
    
    Returns:
        Synthetic images tensor
    """
    decoder.eval()
    
    # Sample latents
    sampled_latents = sample_random_latents(num_samples, latents, method=method)
    
    # Initialize learned upsampler if requested
    learned_upsampler = None
    if use_learned_upsampler and target_size is not None:
        native_size = decoder.img_size
        scale_factor = target_size // native_size
        if scale_factor > 1:
            learned_upsampler = HighResUpsampler(in_channels=3, scale_factor=scale_factor).to(device)
            learned_upsampler.eval()
    
    # Generate in batches
    synthetic_imgs_list = []
    
    for i in range(0, num_samples, batch_size):
        batch_latents = sampled_latents[i:i+batch_size]
        latent_tensor = torch.from_numpy(batch_latents).float().to(device)
        
        # Generate images at native resolution
        with torch.no_grad():
            batch_imgs = decoder(latent_tensor)  # (B, 3, H, W)
            
            # Upscale if requested
            if target_size is not None and target_size > decoder.img_size:
                if learned_upsampler is not None:
                    # Use learned upsampler
                    batch_imgs = learned_upsampler(batch_imgs)
                else:
                    # Use traditional upsampling
                    if upscale_method == 'bicubic':
                        batch_imgs = F.interpolate(
                            batch_imgs, size=(target_size, target_size),
                            mode='bicubic', align_corners=False
                        )
                    elif upscale_method == 'lanczos':
                        # Use PIL for Lanczos (higher quality)
                        batch_imgs_pil = []
                        for img in batch_imgs:
                            img_np = img.cpu().permute(1, 2, 0).numpy()
                            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_np)
                            img_pil = img_pil.resize((target_size, target_size), Image.LANCZOS)
                            img_np = np.array(img_pil).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                            batch_imgs_pil.append(img_tensor)
                        batch_imgs = torch.stack(batch_imgs_pil).to(device)
                    else:
                        raise ValueError(f"Unknown upscale method: {upscale_method}")
                
                # Ensure values are in [0, 1]
                batch_imgs = torch.clamp(batch_imgs, 0, 1)
        
        synthetic_imgs_list.append(batch_imgs.cpu())
    
    return torch.cat(synthetic_imgs_list, dim=0)


def visualize_samples(images, output_file='synthetic_samples.png', n_cols=5):
    """Visualize synthetic samples."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Calculate figure size based on image resolution
    img_size = images[0].shape[1]  # H from (C, H, W)
    scale = max(3, img_size / 256 * 3)  # Scale figure size with image size
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        # Convert from (C, H, W) to (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'Synthetic {idx+1}\n{img_size}x{img_size}px', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'High-Resolution Synthetic RGB Images (Zoom Level 23)', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate high-resolution synthetic RGB images')
    parser.add_argument('--encoder_path', type=str, default='encoder_zoom23.pth',
                       help='Path to trained encoder (for config)')
    parser.add_argument('--decoder_path', type=str, default='decoder_zoom23.pth',
                       help='Path to trained decoder')
    parser.add_argument('--latent_file', type=str, default='latents_zoom23.npy',
                       help='Pre-computed latent file')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of synthetic images to generate')
    parser.add_argument('--sample_method', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'real', 'interpolate'],
                       help='Sampling method')
    parser.add_argument('--target_size', type=int, default=512,
                       help='Target resolution (default: 512, native is 256)')
    parser.add_argument('--upscale_method', type=str, default='bicubic',
                       choices=['bicubic', 'lanczos', 'learned'],
                       help='Upscaling method')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for generation (Titan RTX can handle 32+)')
    parser.add_argument('--output_dir', type=str, default='synthetic_zoom23_hires',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--no_upscale', action='store_true',
                       help='Generate at native resolution (256x256)')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Print GPU info
    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("HIGH-RESOLUTION SYNTHETIC IMAGE GENERATION - ZOOM LEVEL 23")
    print("=" * 80)
    print(f"Number of samples: {args.num_samples}")
    print(f"Sampling method: {args.sample_method}")
    if not args.no_upscale:
        print(f"Native resolution: 256x256")
        print(f"Target resolution: {args.target_size}x{args.target_size}")
        print(f"Upscale method: {args.upscale_method}")
        print(f"Upscale factor: {args.target_size / 256:.1f}x")
    else:
        print(f"Resolution: 256x256 (native)")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    print()
    
    # Load encoder config
    print("Loading model configuration...")
    encoder_checkpoint = torch.load(args.encoder_path, map_location=args.device)
    config = encoder_checkpoint['config']
    
    # Filter out non-MultispectralViT parameters
    config = {k: v for k, v in config.items() 
              if k not in ['zoom_level']}
    
    print(f"Native image size: {config['img_size']}x{config['img_size']}")
    
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
    print(f"Decoder loaded successfully")
    
    # Load latents
    print(f"\nLoading latents from {args.latent_file}...")
    latents = np.load(args.latent_file)
    print(f"Loaded {len(latents)} latent representations")
    print(f"Latent dimension: {latents.shape[1]}")
    
    # Determine target size
    target_size = None if args.no_upscale else args.target_size
    use_learned = args.upscale_method == 'learned'
    
    # Generate synthetic images
    print(f"\nGenerating {args.num_samples} synthetic images...")
    synthetic_images = generate_synthetic_images(
        decoder, latents, args.device, 
        num_samples=args.num_samples,
        method=args.sample_method,
        batch_size=args.batch_size,
        target_size=target_size,
        upscale_method=args.upscale_method,
        use_learned_upsampler=use_learned
    )
    
    print(f"Generated images shape: {synthetic_images.shape}")
    
    # Visualize
    print("\nCreating visualization...")
    suffix = f"_{args.sample_method}"
    if not args.no_upscale:
        suffix += f"_{args.target_size}px"
    visualize_samples(
        synthetic_images,
        output_file=str(output_dir / f'synthetic_samples{suffix}.png'),
        n_cols=5
    )
    
    # Save individual images
    print(f"\nSaving individual images to {output_dir}/...")
    for idx, img in enumerate(synthetic_images):
        # Save as numpy array
        img_np = img.numpy()  # (3, H, W)
        np.save(output_dir / f'synthetic_{idx:03d}.npy', img_np)
        
        # Save as PNG (high quality)
        img_pil = img.permute(1, 2, 0).numpy()
        img_pil = (np.clip(img_pil, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_pil).save(
            output_dir / f'synthetic_{idx:03d}.png',
            quality=95, optimize=True
        )
    
    # Print statistics
    print()
    print("=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Generated {args.num_samples} synthetic images")
    final_size = synthetic_images.shape[2]
    print(f"Resolution: {final_size}x{final_size} pixels")
    print(f"Files saved to: {output_dir}/")
    print()
    print("Output files:")
    print(f"  - synthetic_samples{suffix}.png (grid visualization)")
    print(f"  - synthetic_XXX.png (individual high-res images)")
    print(f"  - synthetic_XXX.npy (raw numpy arrays)")
    
    if args.device == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print()
        print(f"Peak GPU memory usage: {max_memory:.2f} GB")


if __name__ == '__main__':
    main()

