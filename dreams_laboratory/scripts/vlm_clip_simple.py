#!/usr/bin/env python3
"""
Simplified CLIP embeddings using sentence-transformers.
More stable and easier to use than raw transformers.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Tuple
import json
from tqdm import tqdm

# Use sentence-transformers CLIP (more stable)
from sentence_transformers import SentenceTransformer, util


class SimpleCLIPEmbedder:
    """Simple CLIP embeddings using sentence-transformers."""
    
    def __init__(self, model_name='clip-ViT-B-32', device='cuda', multi_gpu=False):
        """
        Initialize CLIP model.
        
        Args:
            model_name: Model name (clip-ViT-B-32 or clip-ViT-L-14)
            device: Device to use
            multi_gpu: Use DataParallel for multi-GPU
        """
        print(f"Loading CLIP model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to(device)
            
            # Multi-GPU support
            if multi_gpu and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
                # sentence-transformers handles multi-GPU internally
        
        self.device = device
        print(f"Model loaded on {device}")
        
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")
    
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for encoding
        
        Returns:
            (N, embed_dim) array of image embeddings
        """
        print(f"Encoding {len(image_paths)} images...")
        
        # Load images
        images = []
        valid_paths = []
        for p in tqdm(image_paths, desc="Loading images"):
            try:
                img = Image.open(p).convert('RGB')
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Warning: Could not load {p}: {e}")
        
        print(f"Loaded {len(images)} valid images")
        
        # Encode with progress bar
        embeddings = self.model.encode(
            images,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        return embeddings, valid_paths
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text descriptions to embeddings.
        
        Args:
            texts: List of text descriptions
        
        Returns:
            (N, embed_dim) array of text embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device
        )
        return embeddings
    
    def text_to_image_search(self, query: str, image_embeddings: np.ndarray,
                            image_paths: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for images matching text query."""
        text_emb = self.encode_text([query])[0]
        
        # Cosine similarity
        similarities = util.cos_sim(text_emb, image_embeddings)[0].cpu().numpy()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(image_paths[idx], float(similarities[idx])) 
                  for idx in top_indices]
        
        return results
    
    def image_to_image_search(self, query_image: str, image_embeddings: np.ndarray,
                             image_paths: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar images."""
        query_emb, _ = self.encode_images([query_image])
        query_emb = query_emb[0]
        
        # Cosine similarity
        similarities = util.cos_sim(query_emb, image_embeddings)[0].cpu().numpy()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(image_paths[idx], float(similarities[idx])) 
                  for idx in top_indices]
        
        return results
    
    def zero_shot_classification(self, image_paths: List[str],
                                 class_descriptions: List[str],
                                 batch_size: int = 32) -> np.ndarray:
        """Classify images using text descriptions."""
        # Encode
        image_embeddings, _ = self.encode_images(image_paths, batch_size)
        text_embeddings = self.encode_text(class_descriptions)
        
        # Calculate similarities
        similarities = util.cos_sim(image_embeddings, text_embeddings).cpu().numpy()
        
        # Softmax to get probabilities
        exp_sim = np.exp(similarities)
        probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)
        
        return probs


def main():
    parser = argparse.ArgumentParser(description='Simple CLIP embeddings for rock tiles')
    parser.add_argument('--tile_dir', type=str, required=True,
                       help='Directory containing tile images')
    parser.add_argument('--model', type=str, default='clip-ViT-B-32',
                       choices=['clip-ViT-B-32', 'clip-ViT-L-14'],
                       help='CLIP model to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--output_dir', type=str, default='clip_embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use all available GPUs')
    
    # Operation mode
    parser.add_argument('--mode', type=str, default='extract',
                       choices=['extract', 'search', 'classify'],
                       help='Operation mode')
    parser.add_argument('--query', type=str, default=None,
                       help='Search query (for search mode)')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                       help='Class descriptions (for classify mode)')
    
    args = parser.parse_args()
    
    # Find all images
    tile_dir = Path(args.tile_dir)
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(list(tile_dir.rglob(ext)))
    image_paths = [str(p) for p in image_paths]
    
    print(f"Found {len(image_paths)} images in {tile_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize CLIP
    embedder = SimpleCLIPEmbedder(
        model_name=args.model,
        device=args.device,
        multi_gpu=args.multi_gpu
    )
    
    if args.mode == 'extract':
        # Extract embeddings
        print(f"\nExtracting embeddings with batch_size={args.batch_size}...")
        embeddings, valid_paths = embedder.encode_images(
            image_paths,
            batch_size=args.batch_size
        )
        
        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Save
        np.save(output_dir / 'image_embeddings.npy', embeddings)
        with open(output_dir / 'image_paths.json', 'w') as f:
            json.dump(valid_paths, f, indent=2)
        
        # Save statistics
        stats = {
            'num_images': len(valid_paths),
            'embedding_dim': int(embeddings.shape[1]),
            'model': args.model,
            'mean': float(embeddings.mean()),
            'std': float(embeddings.std()),
            'min': float(embeddings.min()),
            'max': float(embeddings.max())
        }
        with open(output_dir / 'embedding_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Saved embeddings to {output_dir}/")
        print(f"  - image_embeddings.npy: {embeddings.shape}")
        print(f"  - image_paths.json: {len(valid_paths)} paths")
        print(f"  - embedding_stats.json")
    
    elif args.mode == 'search':
        if not args.query:
            print("Error: --query required for search mode")
            return
        
        # Load embeddings
        embeddings = np.load(output_dir / 'image_embeddings.npy')
        with open(output_dir / 'image_paths.json') as f:
            paths = json.load(f)
        
        print(f"\nSearching for: '{args.query}'")
        results = embedder.text_to_image_search(args.query, embeddings, paths, top_k=10)
        
        print("\nTop 10 results:")
        for i, (path, score) in enumerate(results, 1):
            print(f"{i}. {Path(path).name} (similarity: {score:.4f})")
    
    elif args.mode == 'classify':
        if not args.classes:
            print("Error: --classes required for classify mode")
            return
        
        print(f"\nClassifying with classes: {args.classes}")
        
        # Use first 100 images for demo
        sample_size = min(100, len(image_paths))
        sample_paths = image_paths[:sample_size]
        
        probs = embedder.zero_shot_classification(
            sample_paths,
            args.classes,
            batch_size=args.batch_size
        )
        
        print(f"\nClassification Results (first 10 images):")
        for i in range(min(10, len(probs))):
            pred_class = args.classes[np.argmax(probs[i])]
            confidence = np.max(probs[i])
            print(f"{Path(sample_paths[i]).name}: {pred_class} ({confidence:.2%})")
        
        # Save full results
        results = []
        for i, path in enumerate(sample_paths):
            pred_idx = np.argmax(probs[i])
            results.append({
                'image': str(path),
                'predicted_class': args.classes[pred_idx],
                'confidence': float(probs[i, pred_idx]),
                'all_probabilities': {cls: float(probs[i, j]) 
                                     for j, cls in enumerate(args.classes)}
            })
        
        with open(output_dir / 'classification_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved classification results to {output_dir}/classification_results.json")


if __name__ == '__main__':
    main()

