#!/usr/bin/env python3
"""
Segment Anything Model (SAM) for Rock Analysis

SAM is Meta's foundation model that can segment ANY object without class labels.
Perfect for preliminary analysis of geological imagery.

Features:
- Segments all regions in an image automatically
- No class labels needed (unsupervised)
- Can find fractures, regions, boundaries in rocks
- Outputs masks that can be analyzed or labeled later

Use cases:
1. Exploratory analysis - "What regions exist in this rock?"
2. Preprocessing for labeling - Auto-segment, then classify
3. Boundary detection - Find cracks, fractures, mineral boundaries
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import json
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Tuple, Optional
import random

# Check for SAM availability
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  Segment Anything (SAM) not installed.")
    print()
    print("Install with:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
    print()


class SegmentAnythingRocks:
    """Segment Anything Model wrapper for rock analysis."""
    
    MODELS = {
        'vit_h': {
            'checkpoint': 'sam_vit_h_4b8939.pth',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'description': 'Huge (2.4GB) - Best quality, slower'
        },
        'vit_l': {
            'checkpoint': 'sam_vit_l_0b3195.pth',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'description': 'Large (1.2GB) - Good balance'
        },
        'vit_b': {
            'checkpoint': 'sam_vit_b_01ec64.pth',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'description': 'Base (375MB) - Fastest'
        }
    }
    
    def __init__(self, model_type='vit_b', device='cuda'):
        """Initialize SAM model."""
        if not SAM_AVAILABLE:
            raise RuntimeError("Segment Anything not installed")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        
        # Check if checkpoint exists
        checkpoint_path = Path(self.MODELS[model_type]['checkpoint'])
        
        if not checkpoint_path.exists():
            print(f"⬇️  Downloading SAM model: {model_type}")
            print(f"   URL: {self.MODELS[model_type]['url']}")
            print(f"   Size: {self.MODELS[model_type]['description']}")
            print()
            self._download_checkpoint(model_type)
        
        print(f"🔧 Loading SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.sam.to(self.device)
        
        # Automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires at least 100 pixels
        )
        
        print(f"✓ Model loaded on {self.device}")
        print()
    
    def _download_checkpoint(self, model_type):
        """Download SAM checkpoint."""
        import urllib.request
        
        url = self.MODELS[model_type]['url']
        checkpoint_path = self.MODELS[model_type]['checkpoint']
        
        print("Downloading... (this may take a few minutes)")
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"✓ Downloaded to {checkpoint_path}")
        print()
    
    def segment_image(self, image_path: Path) -> List[Dict]:
        """
        Segment all objects in an image.
        
        Returns:
            List of masks, each with:
            - segmentation: binary mask (H, W)
            - area: number of pixels
            - bbox: [x, y, w, h]
            - predicted_iou: quality score
            - stability_score: stability score
        """
        print(f"📸 Processing: {image_path.name}")
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        print(f"   Image size: {img.size}")
        print(f"   Generating masks...")
        
        # Generate masks
        masks = self.mask_generator.generate(img_array)
        
        print(f"   ✓ Found {len(masks)} segments")
        print()
        
        return masks, img
    
    def visualize_masks(self, image: Image.Image, masks: List[Dict], 
                       output_path: Optional[Path] = None) -> Image.Image:
        """Visualize all masks with different colors."""
        img_array = np.array(image)
        
        # Sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Create overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Generate random colors
        colors = []
        for i in range(len(masks)):
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
                100  # Alpha
            )
            colors.append(color)
        
        # Draw masks
        for mask_data, color in zip(masks, colors):
            mask = mask_data['segmentation']
            
            # Create colored mask
            colored_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))
            mask_array = np.zeros((image.height, image.width, 4), dtype=np.uint8)
            mask_array[mask] = color
            colored_mask = Image.fromarray(mask_array)
            
            overlay = Image.alpha_composite(overlay, colored_mask)
        
        # Composite with original image
        result = image.convert('RGBA')
        result = Image.alpha_composite(result, overlay)
        result = result.convert('RGB')
        
        if output_path:
            result.save(output_path)
            print(f"✓ Saved visualization to {output_path}")
        
        return result
    
    def analyze_segments(self, masks: List[Dict]) -> Dict:
        """Analyze segmentation results."""
        total_area = sum(m['area'] for m in masks)
        avg_area = total_area / len(masks) if masks else 0
        
        # Sort by area
        masks_by_area = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        analysis = {
            'num_segments': len(masks),
            'total_area': total_area,
            'avg_area': avg_area,
            'avg_iou': np.mean([m['predicted_iou'] for m in masks]),
            'avg_stability': np.mean([m['stability_score'] for m in masks]),
            'largest_segments': [
                {
                    'area': m['area'],
                    'bbox': m['bbox'],
                    'iou': float(m['predicted_iou']),
                    'stability': float(m['stability_score'])
                }
                for m in masks_by_area[:10]
            ]
        }
        
        return analysis
    
    def export_to_geojson(self, masks: List[Dict], img_size: Tuple[int, int]) -> Dict:
        """Convert masks to GeoJSON format."""
        from skimage import measure
        
        features = []
        width, height = img_size
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            
            try:
                # Find contours
                contours = measure.find_contours(mask.astype(np.uint8), 0.5)
                
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    
                    # Normalize coordinates
                    coords = []
                    for point in contour:
                        y, x = point
                        coords.append([float(x / width), float(y / height)])
                    
                    # Close polygon
                    if len(coords) > 0:
                        coords.append(coords[0])
                    
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [coords]
                        },
                        'properties': {
                            'segment_id': i,
                            'area': int(mask_data['area']),
                            'bbox': [float(x) for x in mask_data['bbox']],
                            'iou': float(mask_data['predicted_iou']),
                            'stability': float(mask_data['stability_score']),
                            'auto_generated': True,
                            'model': 'segment_anything'
                        }
                    })
            except Exception as e:
                print(f"Warning: Could not process mask {i}: {e}")
        
        return {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'model': 'segment_anything',
                'num_segments': len(masks)
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description='Segment Anything Model (SAM) for rock analysis'
    )
    parser.add_argument('image_path', type=str, help='Path to rock image')
    parser.add_argument('--model', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
                       default='vit_b', help='SAM model size')
    parser.add_argument('--output_dir', type=str, default='sam_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--geojson', action='store_true',
                       help='Export to GeoJSON format')
    
    args = parser.parse_args()
    
    if not SAM_AVAILABLE:
        print("❌ Segment Anything not installed")
        print()
        print("Install with:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        return
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" SEGMENT ANYTHING MODEL (SAM) - Rock Analysis")
    print("="*70)
    print()
    print("SAM segments EVERYTHING in an image without class labels.")
    print("Perfect for exploratory analysis of geological imagery!")
    print()
    print("="*70)
    print()
    
    # Initialize SAM
    try:
        sam = SegmentAnythingRocks(model_type=args.model, device=args.device)
    except Exception as e:
        print(f"❌ Error loading SAM: {e}")
        return
    
    # Segment image
    masks, img = sam.segment_image(image_path)
    
    # Analyze
    print("📊 Analyzing segments...")
    analysis = sam.analyze_segments(masks)
    
    print()
    print("="*70)
    print(" SEGMENTATION RESULTS")
    print("="*70)
    print()
    print(f"Total segments found: {analysis['num_segments']}")
    print(f"Average segment area: {analysis['avg_area']:.1f} pixels")
    print(f"Average IoU: {analysis['avg_iou']:.3f}")
    print(f"Average stability: {analysis['avg_stability']:.3f}")
    print()
    
    print("Top 10 largest segments:")
    for i, seg in enumerate(analysis['largest_segments'], 1):
        print(f"  {i}. Area: {seg['area']:>8} px | "
              f"IoU: {seg['iou']:.3f} | "
              f"Stability: {seg['stability']:.3f}")
    print()
    
    # Visualize
    print("🎨 Creating visualization...")
    vis_path = output_dir / f"{image_path.stem}_sam_segments.jpg"
    sam.visualize_masks(img, masks, vis_path)
    print()
    
    # Save analysis
    analysis_path = output_dir / f"{image_path.stem}_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved analysis to {analysis_path}")
    
    # Export to GeoJSON
    if args.geojson:
        print("🗺️  Exporting to GeoJSON...")
        geojson = sam.export_to_geojson(masks, img.size)
        geojson_path = output_dir / f"{image_path.stem}_segments.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"✓ Saved GeoJSON to {geojson_path}")
    
    print()
    print("="*70)
    print(" NEXT STEPS")
    print("="*70)
    print()
    print("1. Review segments:")
    print(f"   xdg-open {vis_path}")
    print()
    print("2. Use segments for labeling:")
    print("   - Import GeoJSON into DeepGIS label app")
    print("   - Assign class labels to each segment")
    print("   - Train custom classifier")
    print()
    print("3. Analyze specific regions:")
    print("   - Extract largest segments")
    print("   - Measure textures, colors, patterns")
    print("   - Correlate with geological properties")
    print()
    print("="*70)
    print()


if __name__ == '__main__':
    main()

