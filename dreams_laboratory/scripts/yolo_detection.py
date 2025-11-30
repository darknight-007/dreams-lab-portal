#!/usr/bin/env python3
"""
YOLOv8 - Object Detection

YOLOv8 (Ultralytics) provides fast, accurate object detection with built-in
support for 80 COCO classes and easy fine-tuning for custom datasets.

Features:
- Real-time detection speeds
- 80 pre-trained COCO classes (person, car, truck, etc.)
- Multiple model sizes (n, s, m, l, x) for speed/accuracy tradeoff
- Segmentation and pose estimation variants available
- Easy to use - just pip install ultralytics

Use cases:
1. Urban analysis - vehicles, people, traffic signs
2. Infrastructure inspection - detect objects in imagery
3. Environmental monitoring - animals, vegetation areas
4. General object detection with known classes
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Check for YOLOv8 availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 (Ultralytics) not installed.")
    print()
    print("Install with:")
    print("  pip install ultralytics")
    print()


# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class YOLODetector:
    """YOLOv8 wrapper for object detection."""
    
    # Model configurations
    MODELS = {
        'yolov8n': {
            'name': 'yolov8n.pt',
            'description': 'Nano - Fastest, smallest (3.2M params)'
        },
        'yolov8s': {
            'name': 'yolov8s.pt',
            'description': 'Small - Fast, good accuracy (11.2M params)'
        },
        'yolov8m': {
            'name': 'yolov8m.pt',
            'description': 'Medium - Balanced speed/accuracy (25.9M params)'
        },
        'yolov8l': {
            'name': 'yolov8l.pt',
            'description': 'Large - High accuracy (43.7M params)'
        },
        'yolov8x': {
            'name': 'yolov8x.pt',
            'description': 'Extra Large - Best accuracy (68.2M params)'
        },
        # Segmentation models
        'yolov8n-seg': {
            'name': 'yolov8n-seg.pt',
            'description': 'Nano Segmentation - Instance segmentation'
        },
        'yolov8s-seg': {
            'name': 'yolov8s-seg.pt',
            'description': 'Small Segmentation - Instance segmentation'
        },
    }
    
    def __init__(self, model_type='yolov8n', device='cuda', model_dir='/app/models'):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_type: Model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: 'cuda' or 'cpu'
            model_dir: Directory to store model weights
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 (Ultralytics) is not installed. Run: pip install ultralytics")
        
        self.device = device
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model config
        if model_type not in self.MODELS:
            available = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
        
        model_config = self.MODELS[model_type]
        model_path = self.model_dir / model_config['name']
        
        # Load model (auto-downloads if not present)
        print(f"🔧 Loading YOLOv8 ({model_type})...")
        
        # YOLO auto-downloads to ~/.cache/ultralytics or uses provided path
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            # Let YOLO download automatically
            self.model = YOLO(model_config['name'])
        
        # Move to device
        self.model.to(device)
        
        print(f"✓ Model loaded on {device}")
        print(f"   {model_config['description']}")
    
    def detect(self, 
               image: Image.Image,
               confidence: float = 0.25,
               iou_threshold: float = 0.45,
               classes: Optional[List[int]] = None,
               class_names: Optional[List[str]] = None) -> Dict:
        """
        Detect objects in image.
        
        Args:
            image: PIL Image
            confidence: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            classes: List of class indices to detect (None = all)
            class_names: List of class names to detect (alternative to indices)
            
        Returns:
            Dictionary with boxes, scores, class_ids, class_names
        """
        # Convert class names to indices if provided
        if class_names and not classes:
            classes = []
            for name in class_names:
                name_lower = name.lower()
                for i, coco_name in enumerate(COCO_CLASSES):
                    if name_lower in coco_name.lower() or coco_name.lower() in name_lower:
                        classes.append(i)
                        break
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )
        
        # Extract results
        result = results[0]  # First (and only) image
        
        boxes = []
        scores = []
        class_ids = []
        detected_names = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            detected_names = [COCO_CLASSES[i] for i in class_ids]
        
        return {
            'boxes': np.array(boxes) if len(boxes) > 0 else np.array([]).reshape(0, 4),
            'scores': np.array(scores),
            'class_ids': np.array(class_ids),
            'class_names': detected_names,
            'num_detections': len(boxes)
        }
    
    def detect_with_segmentation(self,
                                  image: Image.Image,
                                  confidence: float = 0.25,
                                  iou_threshold: float = 0.45,
                                  classes: Optional[List[int]] = None) -> Dict:
        """
        Detect objects with instance segmentation masks.
        Requires a segmentation model (yolov8n-seg, yolov8s-seg, etc.)
        
        Args:
            image: PIL Image
            confidence: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            classes: List of class indices to detect (None = all)
            
        Returns:
            Dictionary with boxes, scores, class_ids, class_names, masks
        """
        # Run inference
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        boxes = []
        scores = []
        class_ids = []
        detected_names = []
        masks = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            detected_names = [COCO_CLASSES[i] for i in class_ids]
            
            # Get masks if available (segmentation model)
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
        
        return {
            'boxes': np.array(boxes) if len(boxes) > 0 else np.array([]).reshape(0, 4),
            'scores': np.array(scores),
            'class_ids': np.array(class_ids),
            'class_names': detected_names,
            'masks': masks,
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
        img_draw = image.copy().convert('RGB')
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a nice font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Color palette
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
        ]
        
        boxes = detections['boxes']
        scores = detections['scores']
        class_names = detections['class_names']
        
        for i, (box, score, class_name) in enumerate(zip(boxes, scores, class_names)):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label with background
            if show_labels:
                label = class_name
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
        scores = detections['scores']
        class_names = detections['class_names']
        class_ids = detections['class_ids']
        
        for i, (box, score, class_name, class_id) in enumerate(zip(boxes, scores, class_names, class_ids)):
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
                        [float(x1), float(y1)],
                        [float(x2), float(y1)],
                        [float(x2), float(y2)],
                        [float(x1), float(y2)],
                        [float(x1), float(y1)]
                    ]]
                }
            
            feature = {
                'type': 'Feature',
                'geometry': geometry,
                'properties': {
                    'detection_id': i + 1,
                    'class': class_name,
                    'class_id': int(class_id),
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
                'model': f'yolov8_{self.model_type}',
                'image_size': [image_width, image_height]
            }
        }
    
    @staticmethod
    def get_available_classes() -> List[str]:
        """Return list of all available COCO classes."""
        return COCO_CLASSES.copy()
    
    @staticmethod
    def search_classes(query: str) -> List[Tuple[int, str]]:
        """
        Search for classes matching a query.
        
        Args:
            query: Search string
            
        Returns:
            List of (class_id, class_name) tuples
        """
        query_lower = query.lower()
        matches = []
        for i, name in enumerate(COCO_CLASSES):
            if query_lower in name.lower():
                matches.append((i, name))
        return matches


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', choices=list(YOLODetector.MODELS.keys()), default='yolov8n',
                       help='Model variant (default: yolov8n)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--classes', type=str, default=None,
                       help='Comma-separated class names to detect (e.g., "car,truck,person")')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to run on')
    parser.add_argument('--output', default='output_yolo.jpg',
                       help='Output image path')
    parser.add_argument('--save-json', action='store_true',
                       help='Save detections as JSON')
    parser.add_argument('--list-classes', action='store_true',
                       help='List all available COCO classes')
    
    args = parser.parse_args()
    
    if args.list_classes:
        print("Available COCO classes:")
        print("=" * 40)
        for i, name in enumerate(COCO_CLASSES):
            print(f"  {i:2d}: {name}")
        return
    
    print("=" * 60)
    print("YOLOv8 - Object Detection")
    print("=" * 60)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Load image
    print(f"📷 Loading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    print(f"   Size: {image.width} x {image.height}")
    
    # Parse class names if provided
    class_names = None
    if args.classes:
        class_names = [c.strip() for c in args.classes.split(',')]
        print(f"   Filtering for classes: {class_names}")
    
    # Initialize detector
    detector = YOLODetector(
        model_type=args.model,
        device=args.device
    )
    
    # Run detection
    print(f"\n🔍 Running detection...")
    print(f"   Confidence threshold: {args.confidence}")
    print(f"   IoU threshold: {args.iou}")
    
    detections = detector.detect(
        image,
        confidence=args.confidence,
        iou_threshold=args.iou,
        class_names=class_names
    )
    
    print(f"\n✓ Found {detections['num_detections']} objects:")
    for name, score in zip(detections['class_names'], detections['scores']):
        print(f"   - {name}: {score:.3f}")
    
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

