"""
Sample and visualize images from trained Multispectral ViT latent space.

This script allows you to:
1. Sample random points in latent space and find nearest real images
2. Interpolate between images in latent space
3. Visualize what different regions of latent space represent
4. Generate image samples by decoding latent representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import rasterio
import argparse
import json
from typing import List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


class MultispectralPatchEmbedding(nn.Module):
    """Patch embedding for multispectral images (5 bands)."""
    
    def __init__(self, img_size: int = 960, patch_size: int = 16, 
                 in_channels: int = 5, embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        B, C, H, W = x.shape
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
        B, N, E = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
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
        
        self.patch_embed = MultispectralPatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if use_cross_band_attention:
            self.cross_band_attn = CrossBandAttention(embed_dim, num_heads)
            self.cross_band_norm = nn.LayerNorm(embed_dim)
        
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
        
        self.norm = nn.LayerNorm(embed_dim)
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
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        if self.use_cross_band_attention:
            x = x + self.cross_band_attn(self.cross_band_norm(x))
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        pos_embed_with_cls = torch.cat([
            torch.zeros(1, 1, self.embed_dim, device=x.device),
            self.pos_embed
        ], dim=1)
        x = x + pos_embed_with_cls
        
        x = self.transformer_encoder(x)
        x = self.norm(x)
        
        if return_all:
            return x
        else:
            return x[:, 0]


class MultispectralTileDataset(Dataset):
    """Dataset for loading multispectral TIFF tiles."""
    
    def __init__(self, tile_dir: str, img_size: int = 960, 
                 normalize: bool = True):
        self.tile_dir = Path(tile_dir)
        self.img_size = img_size
        self.normalize = normalize
        
        self.tiff_files = sorted(list(self.tile_dir.glob('**/*.tif')))
        self.tiff_files = [f for f in self.tiff_files if '__MACOSX' not in str(f)]
        
        if len(self.tiff_files) == 0:
            raise ValueError(f"No TIFF files found in {tile_dir}")
        
        print(f"Found {len(self.tiff_files)} TIFF files")
    
    def __len__(self):
        return len(self.tiff_files)
    
    def __getitem__(self, idx):
        tiff_path = self.tiff_files[idx]
        
        try:
            with rasterio.open(tiff_path) as src:
                bands = []
                for i in range(1, min(6, src.count + 1)):
                    band = src.read(i)
                    bands.append(band)
                
                while len(bands) < 5:
                    bands.append(bands[-1] if bands else np.zeros_like(bands[0]))
                
                multispectral = np.stack(bands[:5], axis=0).astype(np.float32)
                
                if self.normalize:
                    multispectral = np.clip(multispectral, 0, 65535) / 65535.0
                else:
                    for i in range(5):
                        band = multispectral[i]
                        mean = band.mean()
                        std = band.std()
                        if std > 0:
                            multispectral[i] = (band - mean) / std
            
            multispectral = torch.from_numpy(multispectral)
            
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
            return torch.zeros(5, self.img_size, self.img_size), str(tiff_path)


def load_model(model_path: str, device: str = 'cuda') -> Tuple[nn.Module, dict]:
    """Load trained model and config."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = MultispectralViT(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Config: {config}")
    
    return model, config


def extract_all_latents(model: nn.Module, dataloader: DataLoader, 
                       device: str = 'cuda') -> Tuple[np.ndarray, List[str]]:
    """Extract latent representations for all images."""
    model.eval()
    latents = []
    paths = []
    
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.to(device)
            latent = model(images)
            latents.append(latent.cpu().numpy())
            paths.extend(image_paths)
    
    latents = np.vstack(latents)
    return latents, paths


def sample_random_latents(num_samples: int, latents: np.ndarray, 
                          method: str = 'gaussian') -> np.ndarray:
    """Sample random points in latent space."""
    if method == 'gaussian':
        # Sample from Gaussian fitted to data
        mean = latents.mean(axis=0)
        std = latents.std(axis=0)
        samples = np.random.normal(mean, std, (num_samples, latents.shape[1]))
    elif method == 'uniform':
        # Sample from uniform distribution within data bounds
        min_vals = latents.min(axis=0)
        max_vals = latents.max(axis=0)
        samples = np.random.uniform(min_vals, max_vals, (num_samples, latents.shape[1]))
    elif method == 'sphere':
        # Sample from sphere with radius matching mean data norm
        mean_norm = np.linalg.norm(latents, axis=1).mean()
        samples = np.random.randn(num_samples, latents.shape[1])
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True) * mean_norm
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return samples


def interpolate_latents(latent1: np.ndarray, latent2: np.ndarray, 
                       num_steps: int = 10) -> np.ndarray:
    """Interpolate between two latent representations."""
    alphas = np.linspace(0, 1, num_steps)
    interpolated = []
    
    for alpha in alphas:
        interp = (1 - alpha) * latent1 + alpha * latent2
        interpolated.append(interp)
    
    return np.array(interpolated)


def find_nearest_images(query_latents: np.ndarray, database_latents: np.ndarray,
                       database_paths: List[str], k: int = 5) -> List[List[Tuple[str, float]]]:
    """Find k nearest images in latent space."""
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
    nbrs.fit(database_latents)
    
    distances, indices = nbrs.kneighbors(query_latents)
    
    results = []
    for i in range(len(query_latents)):
        neighbors = []
        for j, idx in enumerate(indices[i]):
            neighbors.append((database_paths[idx], 1 - distances[i][j]))  # cosine similarity
        results.append(neighbors)
    
    return results


def visualize_rgb_band(image: torch.Tensor, band_indices: Tuple[int, int, int] = (2, 1, 0)) -> np.ndarray:
    """Convert multispectral image to RGB for visualization."""
    # Ensure we have a tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    
    # Handle different input shapes
    if len(image.shape) == 2:
        # Single band (H, W) - shouldn't happen but handle it
        raise ValueError(f"Expected 3D tensor (C, H, W), got 2D: {image.shape}")
    elif len(image.shape) == 3:
        # Expected shape: (C, H, W)
        if image.shape[0] < max(band_indices) + 1:
            # Not enough bands, use available bands
            band_indices = (min(2, image.shape[0]-1), min(1, image.shape[0]-1), 0)
        # Use list indexing for proper selection
        rgb = image[list(band_indices)].cpu().numpy()
    elif len(image.shape) == 4:
        # Batch dimension: (B, C, H, W) - take first item
        if image.shape[1] < max(band_indices) + 1:
            band_indices = (min(2, image.shape[1]-1), min(1, image.shape[1]-1), 0)
        rgb = image[0, list(band_indices)].cpu().numpy()
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    # Ensure we have the right shape (C, H, W)
    if len(rgb.shape) != 3:
        raise ValueError(f"RGB extraction failed. Expected 3D array, got shape: {rgb.shape}")
    
    # Check if we need to add a channel dimension
    if rgb.shape[0] != 3:
        # Try to understand the shape
        if rgb.shape[0] == 1 and len(rgb.shape) == 3:
            # Single channel, need to replicate
            rgb = np.repeat(rgb, 3, axis=0)
        else:
            raise ValueError(f"RGB extraction failed. Expected 3 channels, got shape: {rgb.shape}")
    
    # Normalize to [0, 1] if needed
    if rgb.max() > 1.0:
        rgb = np.clip(rgb, 0, 1)
    
    # Convert to uint8
    rgb = (rgb * 255).astype(np.uint8)
    
    # Transpose to (H, W, C) - ensure we have 3 dimensions
    if len(rgb.shape) == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    else:
        raise ValueError(f"Cannot transpose. Array shape: {rgb.shape}")
    
    return rgb


def visualize_samples(images: List[torch.Tensor], paths: List[str], 
                     title: str = "Sampled Images", 
                     output_file: str = "sampled_images.png",
                     n_cols: int = 5):
    """Visualize a grid of images."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, path) in enumerate(zip(images, paths)):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = axes[row, col]
        
        # Convert to RGB (using R=band2, G=band1, B=band0)
        rgb = visualize_rgb_band(img)
        
        ax.imshow(rgb)
        ax.set_title(Path(path).name[:20], fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def visualize_interpolation(images: List[torch.Tensor], paths: List[str],
                            output_file: str = "interpolation.png"):
    """Visualize interpolation between images."""
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 3, 3))
    
    if len(images) == 1:
        axes = [axes]
    
    for idx, (img, path) in enumerate(zip(images, paths)):
        rgb = visualize_rgb_band(img)
        axes[idx].imshow(rgb)
        axes[idx].set_title(f"Step {idx}\n{Path(path).name[:15]}", fontsize=8)
        axes[idx].axis('off')
    
    plt.suptitle("Latent Space Interpolation", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved interpolation to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Sample images from trained Multispectral ViT')
    parser.add_argument('--model_path', type=str, default='multispectral_vit.pth',
                       help='Path to trained model')
    parser.add_argument('--tile_dir', type=str, 
                       default='/mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop',
                       help='Directory containing TIFF tiles')
    parser.add_argument('--latent_file', type=str, default=None,
                       help='Pre-computed latent file (optional, speeds up if available)')
    parser.add_argument('--paths_file', type=str, default=None,
                       help='Pre-computed paths file (optional)')
    
    parser.add_argument('--sample_random', type=int, default=0,
                       help='Number of random samples to generate')
    parser.add_argument('--sample_method', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'sphere'],
                       help='Random sampling method')
    
    parser.add_argument('--interpolate', type=int, nargs=2, default=None,
                       help='Interpolate between two image indices (e.g., --interpolate 0 100)')
    parser.add_argument('--interp_steps', type=int, default=10,
                       help='Number of interpolation steps')
    
    parser.add_argument('--query_indices', type=int, nargs='+', default=None,
                       help='Find nearest neighbors for specific image indices')
    parser.add_argument('--k_neighbors', type=int, default=5,
                       help='Number of nearest neighbors to find')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.model_path, args.device)
    
    # Load or compute latents
    if args.latent_file and Path(args.latent_file).exists():
        print(f"Loading pre-computed latents from {args.latent_file}")
        latents = np.load(args.latent_file)
        with open(args.paths_file, 'r') as f:
            paths = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(latents)} latent representations")
    else:
        print("Extracting latents from images...")
        dataset = MultispectralTileDataset(args.tile_dir, 
                                          img_size=config['img_size'],
                                          normalize=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        latents, paths = extract_all_latents(model, dataloader, args.device)
        print(f"Extracted {len(latents)} latent representations")
    
    # Load dataset for image retrieval
    dataset = MultispectralTileDataset(args.tile_dir, 
                                      img_size=config['img_size'],
                                      normalize=True)
    
    # Random sampling
    if args.sample_random > 0:
        print(f"\nGenerating {args.sample_random} random samples...")
        random_latents = sample_random_latents(args.sample_random, latents, 
                                             method=args.sample_method)
        
        # Find nearest images
        nearest = find_nearest_images(random_latents, latents, paths, 
                                    k=args.k_neighbors)
        
        # Load and visualize images
        sampled_images = []
        sampled_paths = []
        for neighbors in nearest:
            # Use the closest neighbor
            closest_path = neighbors[0][0]
            closest_idx = paths.index(closest_path)
            img, _ = dataset[closest_idx]
            sampled_images.append(img)
            sampled_paths.append(closest_path)
        
        visualize_samples(sampled_images, sampled_paths,
                        title=f"Random Samples ({args.sample_method})",
                        output_file=f"random_samples_{args.sample_method}.png")
    
    # Interpolation
    if args.interpolate:
        idx1, idx2 = args.interpolate
        print(f"\nInterpolating between images {idx1} and {idx2}...")
        
        latent1 = latents[idx1]
        latent2 = latents[idx2]
        
        interpolated_latents = interpolate_latents(latent1, latent2, 
                                                  num_steps=args.interp_steps)
        
        # Find nearest images for each interpolation step
        nearest = find_nearest_images(interpolated_latents, latents, paths, k=1)
        
        # Load images
        interp_images = []
        interp_paths = []
        for neighbors in nearest:
            closest_path = neighbors[0][0]
            closest_idx = paths.index(closest_path)
            img, _ = dataset[closest_idx]
            interp_images.append(img)
            interp_paths.append(closest_path)
        
        visualize_interpolation(interp_images, interp_paths,
                               output_file="interpolation.png")
    
    # Query specific images
    if args.query_indices:
        print(f"\nFinding nearest neighbors for {len(args.query_indices)} query images...")
        
        query_latents = latents[args.query_indices]
        nearest = find_nearest_images(query_latents, latents, paths, 
                                    k=args.k_neighbors)
        
        # Visualize query + neighbors
        all_images = []
        all_paths = []
        all_titles = []
        
        for i, query_idx in enumerate(args.query_indices):
            # Query image
            query_img, _ = dataset[query_idx]
            all_images.append(query_img)
            all_paths.append(paths[query_idx])
            all_titles.append(f"Query {i+1}")
            
            # Neighbors
            for j, (neighbor_path, similarity) in enumerate(nearest[i]):
                neighbor_idx = paths.index(neighbor_path)
                neighbor_img, _ = dataset[neighbor_idx]
                all_images.append(neighbor_img)
                all_paths.append(neighbor_path)
                all_titles.append(f"Neighbor {j+1}\n(sim: {similarity:.3f})")
        
        # Visualize in grid
        visualize_samples(all_images, all_paths,
                         title="Query Images and Nearest Neighbors",
                         output_file="query_neighbors.png",
                         n_cols=args.k_neighbors + 1)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

