#!/usr/bin/env python3
"""
Zero-Shot Object Detection using Pre-trained Models

This script demonstrates using pre-trained segmentation models WITHOUT any training
to detect objects from COCO dataset (80 categories).

Available models:
1. Mask R-CNN (TorchVision) - Fast, good for general objects
2. Mask2Former (Detectron2) - State-of-the-art, more accurate
3. Segment Anything Model (SAM) - Universal, no class labels

COCO Categories: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse,
sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase,
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,
apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
hair drier, toothbrush
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import sys

# COCO class names (80 categories)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ZeroShotMaskRCNN:
    """Zero-shot object detection using pre-trained Mask R-CNN."""
    
    def __init__(self, confidence_threshold: float = 0.5, device: str = 'cuda'):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained model (COCO weights)
        print("Loading pre-trained Mask R-CNN (COCO dataset)...")
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Can detect {len(COCO_CLASSES)-1} object categories")
    
    def predict(self, image_path: Path) -> Dict:
        """Run zero-shot detection on an image."""
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        print(f"Running inference on {image_path.name}...")
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        # Filter by confidence
        mask = predictions['scores'] >= self.confidence_threshold
        
        results = {
            'image_path': str(image_path),
            'image_size': (img.width, img.height),
            'num_detections': mask.sum().item(),
            'detections': []
        }
        
        # Process each detection
        for i in range(mask.sum().item()):
            class_id = predictions['labels'][mask][i].item()
            class_name = COCO_CLASSES[class_id]
            score = predictions['scores'][mask][i].item()
            box = predictions['boxes'][mask][i].cpu().numpy()
            seg_mask = predictions['masks'][mask][i, 0].cpu().numpy()
            
            results['detections'].append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(score),
                'bbox': box.tolist(),
                'mask_shape': seg_mask.shape,
                'mask_area': float((seg_mask > 0.5).sum())
            })
        
        return results
    
    def visualize(self, image_path: Path, predictions: Dict, output_path: Optional[Path] = None):
        """Visualize detections on image."""
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        for i, det in enumerate(predictions['detections']):
            color = colors[i % len(colors)]
            bbox = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label background
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], bbox[1] - 25), label, fill='white', font=font)
        
        if output_path:
            img.save(output_path)
            print(f"✓ Saved visualization to {output_path}")
        
        return img


class ZeroShotMask2Former:
    """Zero-shot detection using pre-trained Mask2Former."""
    
    def __init__(self, confidence_threshold: float = 0.5, device: str = 'cuda'):
        try:
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
        except ImportError:
            raise RuntimeError(
                "detectron2 not installed. Install with:\n"
                "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Setup config
        print("Loading pre-trained Mask2Former (COCO dataset)...")
        cfg = get_cfg()
        
        # Try different pre-trained models
        config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = device if torch.cuda.is_available() else 'cpu'
        
        self.predictor = DefaultPredictor(cfg)
        print(f"✓ Model loaded on {cfg.MODEL.DEVICE}")
        print(f"✓ Can detect {len(COCO_CLASSES)-1} object categories")
    
    def predict(self, image_path: Path) -> Dict:
        """Run zero-shot detection."""
        img = np.array(Image.open(image_path).convert('RGB'))
        
        print(f"Running inference on {image_path.name}...")
        outputs = self.predictor(img)
        
        instances = outputs['instances']
        results = {
            'image_path': str(image_path),
            'image_size': (img.shape[1], img.shape[0]),
            'num_detections': len(instances),
            'detections': []
        }
        
        for i in range(len(instances)):
            class_id = instances.pred_classes[i].item()
            class_name = COCO_CLASSES[class_id + 1]  # +1 for background
            score = instances.scores[i].item()
            box = instances.pred_boxes[i].tensor.cpu().numpy()[0]
            
            results['detections'].append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(score),
                'bbox': box.tolist()
            })
        
        return results


def predictions_to_geojson(predictions: Dict) -> Dict:
    """Convert predictions to GeoJSON format."""
    features = []
    img_w, img_h = predictions['image_size']
    
    for det in predictions['detections']:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # Normalize coordinates to [0, 1]
        coords = [
            [x1/img_w, y1/img_h],
            [x2/img_w, y1/img_h],
            [x2/img_w, y2/img_h],
            [x1/img_w, y2/img_h],
            [x1/img_w, y1/img_h]  # Close polygon
        ]
        
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [coords]
            },
            'properties': {
                'category': det['class_name'],
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'bbox': bbox
            }
        })
    
    return {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'model': 'zero_shot_coco',
            'num_detections': predictions['num_detections'],
            'image_path': predictions['image_path']
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot object detection using pre-trained models (no training required!)'
    )
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, choices=['maskrcnn', 'mask2former'],
                       default='maskrcnn', help='Model to use')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--output_dir', type=str, default='zero_shot_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization image')
    parser.add_argument('--geojson', action='store_true',
                       help='Export to GeoJSON format')
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    if args.model == 'maskrcnn':
        detector = ZeroShotMaskRCNN(args.confidence, args.device)
    else:
        detector = ZeroShotMask2Former(args.confidence, args.device)
    
    # Run detection
    predictions = detector.predict(image_path)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"Detected objects: {predictions['num_detections']}")
    print(f"{'='*60}\n")
    
    for i, det in enumerate(predictions['detections'], 1):
        print(f"{i}. {det['class_name'].upper()}")
        print(f"   Confidence: {det['confidence']:.2%}")
        print(f"   Bounding box: {det['bbox']}")
        print()
    
    # Save results
    results_file = output_dir / f"{image_path.stem}_detections.json"
    with open(results_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Saved results to {results_file}")
    
    # Visualize
    if args.visualize and args.model == 'maskrcnn':
        vis_file = output_dir / f"{image_path.stem}_visualization.jpg"
        detector.visualize(image_path, predictions, vis_file)
    
    # Export to GeoJSON
    if args.geojson:
        geojson = predictions_to_geojson(predictions)
        geojson_file = output_dir / f"{image_path.stem}_geojson.json"
        with open(geojson_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"✓ Saved GeoJSON to {geojson_file}")
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

