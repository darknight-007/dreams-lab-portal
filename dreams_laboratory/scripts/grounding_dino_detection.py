#!/usr/bin/env python3
"""
Grounding DINO - Text-Based Object Detection

Grounding DINO enables zero-shot object detection using natural language prompts.
Unlike traditional detectors limited to fixed classes, you can detect ANY object
by simply describing it in text.

Features:
- Text-based detection: "solar panel", "damaged roof", "vehicle", etc.
- Zero-shot: No training needed for new objects
- High accuracy with visual grounding
- Multiple objects in one query: "car . truck . bus"

Use cases:
1. Infrastructure inspection - "crack . corrosion . damage"
2. Urban analysis - "building . road . parking lot"  
3. Environmental monitoring - "tree . water body . vegetation"
4. Flexible exploration - describe what you're looking for
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Check for Grounding DINO availability
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util import box_ops
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("⚠️  Grounding DINO not installed.")
    print()
    print("Install with:")
    print("  pip install groundingdino-py")
    print()
    print("Or build from source:")
    print("  git clone https://github.com/IDEA-Research/GroundingDINO.git")
    print("  cd GroundingDINO")
    print("  pip install -e .")
    print()


class GroundingDINODetector:
    """Grounding DINO wrapper for text-based object detection."""
    
    # Model configurations
    MODELS = {
        'swin_t': {
            'config': '/app/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
            'checkpoint': 'groundingdino_swint_ogc.pth',
            'url': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
            'description': 'Swin Transformer Tiny - Faster, good accuracy'
        },
        'swin_b': {
            'config': '/app/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py',
            'checkpoint': 'groundingdino_swinb_cogcoor.pth',
            'url': 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth',
            'description': 'Swin Transformer Base - Best accuracy'
        }
    }
    
    def __init__(self, model_type='swin_t', device='cuda', model_dir='/app/models'):
        """
        Initialize Grounding DINO detector.
        
        Args:
            model_type: 'swin_t' (faster) or 'swin_b' (more accurate)
            device: 'cuda' or 'cpu'
            model_dir: Directory to store model weights (default: /app/models in Docker)
        """
        if not GROUNDING_DINO_AVAILABLE:
            raise ImportError("Grounding DINO is not installed. See instructions above.")
        
        self.device = device
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model config
        model_config = self.MODELS[model_type]
        config_path = model_config['config']
        checkpoint_path = self.model_dir / model_config['checkpoint']
        
        # Download checkpoint if needed
        if not checkpoint_path.exists():
            print(f"📥 Downloading {model_type} checkpoint...")
            self._download_checkpoint(model_config['url'], checkpoint_path)
        
        # Load model
        print(f"🔧 Loading Grounding DINO ({model_type})...")
        self.model = load_model(config_path, str(checkpoint_path), device=device)
        print(f"✓ Model loaded on {device}")
    
    def _download_checkpoint(self, url: str, save_path: Path):
        """Download model checkpoint."""
        import urllib.request
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ Downloaded to {save_path}")
    
    def detect(self, 
               image: Image.Image,
               text_prompt: str,
               box_threshold: float = 0.35,
               text_threshold: float = 0.25) -> Dict:
        """
        Detect objects in image based on text prompt.
        
        Args:
            image: PIL Image
            text_prompt: Text description of objects to find (e.g., "car . truck . bus")
            box_threshold: Confidence threshold for bounding boxes (0-1)
            text_threshold: Confidence threshold for text matching (0-1)
            
        Returns:
            Dictionary with boxes, logits, phrases
        """
        # Convert PIL to format expected by Grounding DINO
        image_array = np.array(image)
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_array,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert boxes from normalized [0,1] to pixel coordinates
        h, w = image_array.shape[:2]
        boxes_pixel = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        
        # Convert to xyxy format
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_pixel)
        
        return {
            'boxes': boxes_xyxy.cpu().numpy(),  # [x1, y1, x2, y2]
            'logits': logits.cpu().numpy(),      # Confidence scores
            'phrases': phrases,                   # Detected object labels
            'num_detections': len(boxes)
        }
    
    def visualize(self,
                  image: Image.Image,
                  detections: Dict,
                  show_labels: bool = True,
                  show_confidence: bool = True) -> Image.Image:
        """
        Visualize detections on image.
        
        Args:
            image: Original PIL Image
            detections: Detection results from detect()
            show_labels: Whether to show text labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            PIL Image with visualizations
        """
        # Convert to array for drawing
        img_array = np.array(image)
        img_draw = Image.fromarray(img_array).convert('RGB')
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a nice font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw each detection
        boxes = detections['boxes']
        logits = detections['logits']
        phrases = detections['phrases']
        
        # Color palette
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
        ]
        
        for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label with background
            if show_labels:
                label = phrase
                if show_confidence:
                    label += f" {score:.2f}"
                
                # Get text size
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw background rectangle
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width + 6, y1],
                    fill=color
                )
                
                # Draw text
                draw.text((x1 + 3, y1 - text_height - 2), label, fill='white', font=font)
        
        return img_draw
    
    def to_geojson(self,
                   detections: Dict,
                   image_width: int,
                   image_height: int,
                   viewport_bounds: Optional[Dict] = None) -> Dict:
        """
        Convert detections to GeoJSON format.
        
        Args:
            detections: Detection results from detect()
            image_width: Image width in pixels
            image_height: Image height in pixels
            viewport_bounds: Dict with {min_lon, max_lon, min_lat, max_lat}
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        boxes = detections['boxes']
        logits = detections['logits']
        phrases = detections['phrases']
        
        for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
            x1, y1, x2, y2 = box
            
            # Normalize coordinates to [0, 1]
            x1_norm = float(x1 / image_width)
            y1_norm = float(y1 / image_height)
            x2_norm = float(x2 / image_width)
            y2_norm = float(y2 / image_height)
            
            # If viewport bounds provided, convert to geographic coordinates
            if viewport_bounds:
                min_lon = viewport_bounds['min_lon']
                max_lon = viewport_bounds['max_lon']
                min_lat = viewport_bounds['min_lat']
                max_lat = viewport_bounds['max_lat']
                
                # Map pixel coordinates to geographic coordinates
                lon1 = min_lon + x1_norm * (max_lon - min_lon)
                lat1 = max_lat - y1_norm * (max_lat - min_lat)  # Flip Y
                lon2 = min_lon + x2_norm * (max_lon - min_lon)
                lat2 = max_lat - y2_norm * (max_lat - min_lat)  # Flip Y
                
                # Create polygon for bounding box
                coordinates = [[
                    [lon1, lat1],
                    [lon2, lat1],
                    [lon2, lat2],
                    [lon1, lat2],
                    [lon1, lat1]
                ]]
                
                geometry = {
                    'type': 'Polygon',
                    'coordinates': coordinates
                }
            else:
                # Use pixel coordinates
                geometry = {
                    'type': 'Polygon',
                    'coordinates': [[
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                        [x1, y1]
                    ]]
                }
            
            feature = {
                'type': 'Feature',
                'geometry': geometry,
                'properties': {
                    'detection_id': i + 1,
                    'class': phrase,
                    'confidence': float(score),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'bbox_normalized': [x1_norm, y1_norm, x2_norm, y2_norm]
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features,
            'properties': {
                'num_detections': len(features),
                'model': 'grounding_dino',
                'image_size': [image_width, image_height]
            }
        }


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grounding DINO Object Detection')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--prompt', required=True, help='Text prompt (e.g., "car . truck . bus")')
    parser.add_argument('--model', choices=['swin_t', 'swin_b'], default='swin_t',
                       help='Model type: swin_t (faster) or swin_b (more accurate)')
    parser.add_argument('--box-threshold', type=float, default=0.35,
                       help='Box confidence threshold (0-1)')
    parser.add_argument('--text-threshold', type=float, default=0.25,
                       help='Text matching threshold (0-1)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to run on')
    parser.add_argument('--output', default='output_grounding_dino.jpg',
                       help='Output image path')
    parser.add_argument('--save-json', action='store_true',
                       help='Save detections as JSON')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grounding DINO - Text-Based Object Detection")
    print("=" * 60)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Load image
    print(f"📷 Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    print(f"   Size: {image.width} x {image.height}")
    
    # Initialize detector
    detector = GroundingDINODetector(
        model_type=args.model,
        device=args.device
    )
    
    # Run detection
    print(f"\n🔍 Detecting: '{args.prompt}'")
    print(f"   Box threshold: {args.box_threshold}")
    print(f"   Text threshold: {args.text_threshold}")
    
    detections = detector.detect(
        image,
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    
    print(f"\n✓ Found {detections['num_detections']} objects:")
    for phrase, score in zip(detections['phrases'], detections['logits']):
        print(f"   - {phrase}: {score:.3f}")
    
    # Visualize
    print(f"\n🎨 Creating visualization...")
    vis_image = detector.visualize(image, detections)
    vis_image.save(args.output, quality=95)
    print(f"✓ Saved to: {args.output}")
    
    # Save JSON if requested
    if args.save_json:
        geojson = detector.to_geojson(detections, image.width, image.height)
        json_path = Path(args.output).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"✓ Saved GeoJSON to: {json_path}")
    
    print("\n" + "=" * 60)
    print("Detection complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

