#!/usr/bin/env python3
"""
CLIP Vision-Language Embeddings for Environmental Research.

Extract and analyze embeddings from rock tile images for:
- Semantic search (text → find similar images)
- Image similarity
- Zero-shot classification
- Environmental pattern analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
import json

# Import CLIP differently to handle version compatibility
try:
    from transformers import CLIPModel, CLIPProcessor
    USE_TRANSFORMERS = True
except ImportError:
    # Fallback to sentence-transformers CLIP
    from sentence_transformers import SentenceTransformer, util
    USE_TRANSFORMERS = False


class CLIPEmbedder:
    """CLIP model for vision-language embeddings."""
    
    def __init__(self, model_name='openai/clip-vit-large-patch14', device='cuda', multi_gpu=False):
        """
        Initialize CLIP model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            multi_gpu: Use all available GPUs
        """
        self.device = device
        self.multi_gpu = multi_gpu
        
        print(f"Loading CLIP model: {model_name}...")
        
        if USE_TRANSFORMERS:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
            except:
                # Fallback for older versions
                from transformers import CLIPFeatureExtractor, CLIPTokenizer
                self.processor = {
                    'feature_extractor': CLIPFeatureExtractor.from_pretrained(model_name),
                    'tokenizer': CLIPTokenizer.from_pretrained(model_name)
                }
            
            self.model = AutoModel.from_pretrained(model_name)
            
            # Multi-GPU support
            if multi_gpu and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
                self.model = nn.DataParallel(self.model)
                self.model = self.model.to(device)
            else:
                self.model = self.model.to(device)
            
            self.model.eval()
        else:
            # Fallback to sentence-transformers
            print("Using sentence-transformers CLIP implementation")
            self.model = SentenceTransformer('clip-ViT-B-32')
            if device == 'cuda':
                self.model = self.model.to(device)
        
        print(f"Model loaded on {device}")
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
    
    @torch.no_grad()
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for encoding
        
        Returns:
            (N, embed_dim) array of image embeddings
        """
        if not USE_TRANSFORMERS:
            # Use sentence-transformers encode
            from PIL import Image
            images = [Image.open(p) for p in image_paths]
            embeddings = self.model.encode(images, batch_size=batch_size, show_progress_bar=True)
            return embeddings
        
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = [Image.open(p).convert('RGB') for p in batch_paths]
            
            # Handle different processor types
            if isinstance(self.processor, dict):
                inputs = self.processor['feature_extractor'](images=images, return_tensors="pt")
            else:
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model (handle DataParallel wrapper)
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            # Get image features
            outputs = model(**inputs)
            if hasattr(outputs, 'image_embeds'):
                image_features = outputs.image_embeds
            elif hasattr(model, 'get_image_features'):
                image_features = model.get_image_features(**inputs)
            else:
                image_features = outputs.last_hidden_state[:, 0, :]
            
            image_features = F.normalize(image_features, p=2, dim=1)
            
            embeddings.append(image_features.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i+len(batch_paths)}/{len(image_paths)} images")
        
        return np.vstack(embeddings)
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text descriptions to embeddings.
        
        Args:
            texts: List of text descriptions
        
        Returns:
            (N, embed_dim) array of text embeddings
        """
        if not USE_TRANSFORMERS:
            # Use sentence-transformers encode
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        
        # Handle different processor types
        if isinstance(self.processor, dict):
            inputs = self.processor['tokenizer'](texts, return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model (handle DataParallel wrapper)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # Get text features
        outputs = model(**inputs)
        if hasattr(outputs, 'text_embeds'):
            text_features = outputs.text_embeds
        elif hasattr(model, 'get_text_features'):
            text_features = model.get_text_features(**inputs)
        else:
            text_features = outputs.last_hidden_state[:, 0, :]
        
        text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features.cpu().numpy()
    
    def text_to_image_search(self, query: str, image_embeddings: np.ndarray, 
                            image_paths: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for images matching a text query.
        
        Args:
            query: Text search query
            image_embeddings: Precomputed image embeddings
            image_paths: Corresponding image paths
            top_k: Number of results to return
        
        Returns:
            List of (image_path, similarity_score) tuples
        """
        text_emb = self.encode_text([query])[0]
        
        # Cosine similarity
        similarities = image_embeddings @ text_emb
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(image_paths[idx], float(similarities[idx])) 
                  for idx in top_indices]
        
        return results
    
    def image_to_image_search(self, query_image: str, image_embeddings: np.ndarray,
                             image_paths: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar images.
        
        Args:
            query_image: Path to query image
            image_embeddings: Precomputed image embeddings
            image_paths: Corresponding image paths
            top_k: Number of results to return
        
        Returns:
            List of (image_path, similarity_score) tuples
        """
        query_emb = self.encode_images([query_image])[0]
        
        similarities = image_embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(image_paths[idx], float(similarities[idx])) 
                  for idx in top_indices]
        
        return results
    
    def zero_shot_classification(self, image_paths: List[str], 
                                 class_descriptions: List[str]) -> np.ndarray:
        """
        Classify images using text descriptions (zero-shot).
        
        Args:
            image_paths: List of image paths
            class_descriptions: List of class descriptions (e.g., ["granite", "sandstone"])
        
        Returns:
            (N, num_classes) array of probabilities
        """
        # Encode
        image_embeddings = self.encode_images(image_paths)
        text_embeddings = self.encode_text(class_descriptions)
        
        # Calculate similarities
        logits = image_embeddings @ text_embeddings.T
        
        # Softmax to get probabilities
        probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        
        return probs


def main():
    parser = argparse.ArgumentParser(description='CLIP embeddings for rock tiles')
    parser.add_argument('--tile_dir', type=str, required=True,
                       help='Directory containing tile images')
    parser.add_argument('--model', type=str, default='openai/clip-vit-large-patch14',
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
    
    print(f"Found {len(image_paths)} images")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize CLIP
    embedder = CLIPEmbedder(model_name=args.model, device=args.device, multi_gpu=args.multi_gpu)
    
    if args.mode == 'extract':
        # Extract embeddings
        print(f"\nExtracting embeddings...")
        embeddings = embedder.encode_images(image_paths, batch_size=args.batch_size)
        
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save
        np.save(output_dir / 'image_embeddings.npy', embeddings)
        with open(output_dir / 'image_paths.json', 'w') as f:
            json.dump(image_paths, f, indent=2)
        
        print(f"\nSaved embeddings to {output_dir}/")
        print(f"  - image_embeddings.npy: {embeddings.shape}")
        print(f"  - image_paths.json: {len(image_paths)} paths")
    
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
        probs = embedder.zero_shot_classification(
            image_paths[:100],  # Sample for demo
            args.classes
        )
        
        print("\nPredictions (first 10 images):")
        for i in range(min(10, len(probs))):
            pred_class = args.classes[np.argmax(probs[i])]
            confidence = np.max(probs[i])
            print(f"{Path(image_paths[i]).name}: {pred_class} ({confidence:.2%})")


if __name__ == '__main__':
    main()

