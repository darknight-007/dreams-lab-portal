#!/usr/bin/env python3
"""
Segmentation-Assisted Labeling for DeepGIS Label App

This script uses a pre-trained segmentation model to generate initial labels,
which can then be refined in the DeepGIS label app. This significantly speeds up
the labeling process.

Workflow:
1. Load images from DeepGIS database
2. Run segmentation model to generate initial masks
3. Convert masks to GeoJSON format compatible with DeepGIS label app
4. Save as "suggested labels" that can be loaded and refined
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import json
from pathlib import Path
import argparse
import sys
import os
from typing import List, Dict, Tuple, Optional

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Django setup
try:
    import django
    django_path = Path(__file__).parent.parent.parent / 'deepgis-xr'
    if django_path.exists():
        sys.path.insert(0, str(django_path))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepgis_xr.settings')
        django.setup()
        from deepgis_xr.apps.core.models import Image, ImageLabel, CategoryType
        DJANGO_AVAILABLE = True
    else:
        DJANGO_AVAILABLE = False
except Exception as e:
    print(f"Warning: Django not available: {e}")
    DJANGO_AVAILABLE = False

# Import segmentation models
from multispectral_vit import MultispectralViT
from multispectral_decoder import SegmentationDecoder, MultispectralSegmentationModel


def load_segmentation_model(model_path: str, config: Dict, device: str = 'cuda'):
    """Load trained segmentation model."""
    encoder = MultispectralViT(**{k: v for k, v in config.items() 
                                   if k not in ['zoom_level', 'dataset']})
    decoder = SegmentationDecoder(
        embed_dim=config['embed_dim'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config.get('num_classes', 10)
    )
    
    model = MultispectralSegmentationModel(encoder, decoder)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict_segmentation(model: nn.Module, image_path: Path, device: str = 'cuda',
                         img_size: int = 512) -> np.ndarray:
    """Run segmentation model on image."""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img.resize((img_size, img_size)))
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)  # (1, num_classes, H, W)
        predictions = torch.argmax(logits, dim=1)  # (1, H, W)
        mask = predictions[0].cpu().numpy()
    
    return mask


def mask_to_geojson(mask: np.ndarray, categories: List[str], 
                   original_size: Tuple[int, int],
                   confidence_threshold: float = 0.5) -> Dict:
    """
    Convert segmentation mask to GeoJSON format for DeepGIS label app.
    
    Args:
        mask: Segmentation mask (H, W) with class IDs
        categories: List of category names
        original_size: Original image size (height, width)
        confidence_threshold: Minimum confidence (not used for segmentation, kept for API)
    """
    from skimage import measure
    
    features = []
    mask_h, mask_w = mask.shape
    orig_h, orig_w = original_size
    
    # Process each class
    for class_id in range(len(categories)):
        class_mask = (mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
        
        # Find contours
        try:
            contours = measure.find_contours(class_mask, 0.5)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Convert to normalized coordinates [0, 1]
                normalized_coords = []
                for point in contour:
                    y, x = point
                    # Normalize
                    x_norm = x / mask_w
                    y_norm = y / mask_h
                    normalized_coords.append([x_norm, y_norm])
                
                # Close polygon
                if len(normalized_coords) > 0:
                    normalized_coords.append(normalized_coords[0])
                
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [normalized_coords]
                    },
                    'properties': {
                        'category': categories[class_id],
                        'confidence': 1.0,  # Segmentation doesn't have per-pixel confidence
                        'auto_generated': True
                    }
                })
        except ImportError:
            print("Warning: scikit-image not installed. Using simple contour detection.")
            # Fallback: use bounding box
            y_coords, x_coords = np.where(class_mask > 0)
            if len(x_coords) > 0:
                x_min, x_max = x_coords.min() / mask_w, x_coords.max() / mask_w
                y_min, y_max = y_coords.min() / mask_h, y_coords.max() / mask_h
                
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max],
                            [x_min, y_min]
                        ]]
                    },
                    'properties': {
                        'category': categories[class_id],
                        'confidence': 1.0,
                        'auto_generated': True
                    }
                })
    
    return {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'auto_generated': True,
            'model': 'segmentation_assisted',
            'num_objects': len(features)
        }
    }


def save_suggested_labels(image: Image, geojson: Dict, output_path: Path):
    """Save suggested labels in format compatible with DeepGIS label app."""
    # Add image metadata
    geojson['metadata'] = geojson.get('metadata', {})
    geojson['metadata']['image'] = image.name
    geojson['metadata']['image_width'] = image.width
    geojson['metadata']['image_height'] = image.height
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Saved suggested labels to: {output_path}")


def process_images_for_labeling(model_path: str, config_path: str, 
                               image_ids: Optional[List[int]] = None,
                               output_dir: Path = Path('suggested_labels'),
                               device: str = 'cuda'):
    """Process images and generate suggested labels."""
    if not DJANGO_AVAILABLE:
        raise RuntimeError("Django not available")
    
    # Load model
    import torch
    checkpoint = torch.load(config_path, map_location=device)
    config = checkpoint.get('config', checkpoint)
    
    model = load_segmentation_model(model_path, config, device)
    
    # Get categories
    categories = ['background'] + [cat.name for cat in CategoryType.objects.all().order_by('id')]
    config['num_classes'] = len(categories)
    
    # Get images
    if image_ids is None:
        images = list(Image.objects.all())
    else:
        images = list(Image.objects.filter(id__in=image_ids))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(images)} images...")
    
    for idx, image in enumerate(images):
        try:
            # Get image path
            image_path = Path(image.path)
            if not image_path.exists():
                # Try alternative paths
                image_path = Path(f"static/images/{image.path}")
                if not image_path.exists():
                    print(f"Skipping {image.name}: file not found")
                    continue
            
            # Predict segmentation
            mask = predict_segmentation(model, image_path, device, config['img_size'])
            
            # Convert to GeoJSON
            geojson = mask_to_geojson(mask, categories, (image.height, image.width))
            
            # Save
            output_file = output_dir / f"{image.id}_suggested_labels.json"
            save_suggested_labels(image, geojson, output_file)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(images)} images...")
        
        except Exception as e:
            print(f"Error processing image {image.id}: {e}")
    
    print(f"Completed! Generated {len(list(output_dir.glob('*_suggested_labels.json')))} suggested label files")


def main():
    parser = argparse.ArgumentParser(
        description='Generate suggested labels using segmentation model for DeepGIS label app'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained segmentation model')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model config (encoder checkpoint)')
    parser.add_argument('--image_ids', type=int, nargs='+', default=None,
                       help='Specific image IDs to process (None = all)')
    parser.add_argument('--output_dir', type=str, default='suggested_labels',
                       help='Output directory for suggested labels')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    process_images_for_labeling(
        args.model_path,
        args.config_path,
        args.image_ids,
        Path(args.output_dir),
        args.device
    )


if __name__ == '__main__':
    main()

