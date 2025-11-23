#!/usr/bin/env python3
"""
Mask2Former Training Script with DeepGIS Label App Integration

This script trains Mask2Former for instance segmentation using labels from DeepGIS label app.
It also includes functionality to use a pre-trained segmentation model to assist labeling.

Mask2Former is a state-of-the-art segmentation model that can perform:
- Semantic segmentation
- Instance segmentation
- Panoptic segmentation

Features:
- Load labels from DeepGIS database (ImageLabel, CategoryLabel)
- Convert GeoJSON labels to segmentation masks
- Train Mask2Former model
- Use pre-trained model to generate initial labels for faster annotation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import json
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Tuple, Optional
import os

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Try to import detectron2 (Mask2Former implementation)
try:
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.utils.visualizer import Visualizer, ColorMode
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("Warning: detectron2 not installed. Install with: pip install detectron2")
    print("For CUDA support: pip install 'git+https://github.com/facebookresearch/detectron2.git'")

# Django setup for accessing DeepGIS models
try:
    import django
    django_path = Path(__file__).parent.parent.parent / 'deepgis-xr'
    if django_path.exists():
        sys.path.insert(0, str(django_path))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepgis_xr.settings')
        django.setup()
        from deepgis_xr.apps.core.models import Image, ImageLabel, CategoryLabel, CategoryType
        DJANGO_AVAILABLE = True
    else:
        DJANGO_AVAILABLE = False
except Exception as e:
    print(f"Warning: Django not available: {e}")
    DJANGO_AVAILABLE = False


class DeepGISLabelDataset(Dataset):
    """
    Dataset that loads images and labels from DeepGIS database.
    Converts GeoJSON labels to segmentation masks.
    """
    
    def __init__(self, image_dir: str, label_ids: Optional[List[int]] = None,
                 img_size: int = 512, num_classes: int = None):
        """
        Args:
            image_dir: Directory containing images
            label_ids: List of ImageLabel IDs to use (None = use all)
            img_size: Target image size
            num_classes: Number of classes (None = auto-detect from categories)
        """
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        
        if not DJANGO_AVAILABLE:
            raise RuntimeError("Django not available. Cannot load DeepGIS labels.")
        
        # Get labels from database
        if label_ids is None:
            self.labels = list(ImageLabel.objects.all().order_by('id'))
        else:
            self.labels = list(ImageLabel.objects.filter(id__in=label_ids))
        
        if len(self.labels) == 0:
            raise ValueError("No labels found in database")
        
        # Get all categories
        self.categories = list(CategoryType.objects.all().order_by('id'))
        if num_classes is None:
            self.num_classes = len(self.categories) + 1  # +1 for background
        else:
            self.num_classes = num_classes
        
        # Create category mapping
        self.category_to_id = {cat.name: idx + 1 for idx, cat in enumerate(self.categories)}
        self.id_to_category = {idx + 1: cat.name for idx, cat in enumerate(self.categories)}
        self.id_to_category[0] = 'background'
        
        print(f"Loaded {len(self.labels)} labeled images")
        print(f"Categories: {[cat.name for cat in self.categories]}")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_label = self.labels[idx]
        image = image_label.image
        
        # Load image
        image_path = self.image_dir / image.path if not image.path.startswith('http') else None
        if image_path is None or not image_path.exists():
            # Try alternative paths
            image_path = Path(image.path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image.path}")
        
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        original_size = img_array.shape[:2]
        
        # Resize image
        img_resized = Image.fromarray(img_array).resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        
        # Create segmentation mask from GeoJSON labels
        mask = self._create_mask_from_labels(image_label, original_size)
        
        # Resize mask
        mask_resized = Image.fromarray(mask).resize((self.img_size, self.img_size), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized)).long()
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'image_id': image.id,
            'label_id': image_label.id,
            'original_size': original_size
        }
    
    def _create_mask_from_labels(self, image_label, img_size: Tuple[int, int]) -> np.ndarray:
        """Convert GeoJSON labels to segmentation mask."""
        mask = np.zeros((img_size[0], img_size[1]), dtype=np.int32)
        
        try:
            label_data = json.loads(image_label.combined_label_shapes)
            features = label_data.get('features', [])
            
            for feature in features:
                geometry = feature.get('geometry', {})
                category = feature.get('properties', {}).get('category', 'unknown')
                
                if category not in self.category_to_id:
                    continue
                
                class_id = self.category_to_id[category]
                
                # Handle different geometry types
                geom_type = geometry.get('type', '')
                coordinates = geometry.get('coordinates', [])
                
                if geom_type == 'Polygon':
                    # Convert polygon coordinates to mask
                    polygon_coords = coordinates[0]  # Exterior ring
                    self._draw_polygon(mask, polygon_coords, class_id, img_size)
                elif geom_type == 'Circle':
                    # Convert circle to polygon
                    center = coordinates[0]
                    radius = coordinates[1] if len(coordinates) > 1 else 10
                    self._draw_circle(mask, center, radius, class_id, img_size)
                elif geom_type == 'Point':
                    # Point becomes small circle
                    center = coordinates
                    self._draw_circle(mask, center, 5, class_id, img_size)
        
        except Exception as e:
            print(f"Error creating mask for label {image_label.id}: {e}")
        
        return mask
    
    def _draw_polygon(self, mask: np.ndarray, coords: List, class_id: int, img_size: Tuple[int, int]):
        """Draw polygon on mask."""
        try:
            # Convert coordinates to pixel coordinates
            # Assuming coordinates are normalized [0, 1] or pixel coordinates
            img = Image.new('L', (img_size[1], img_size[0]), 0)
            draw = ImageDraw.Draw(img)
            
            # Convert coordinates
            polygon = []
            for coord in coords:
                if len(coord) == 2:
                    x, y = coord
                    # Handle normalized coordinates
                    if x <= 1.0 and y <= 1.0:
                        x = int(x * img_size[1])
                        y = int(y * img_size[0])
                    else:
                        x = int(x)
                        y = int(y)
                    polygon.append((x, y))
            
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=class_id)
                mask_array = np.array(img)
                mask[mask_array == class_id] = class_id
        except Exception as e:
            print(f"Error drawing polygon: {e}")
    
    def _draw_circle(self, mask: np.ndarray, center: List, radius: float, class_id: int, img_size: Tuple[int, int]):
        """Draw circle on mask."""
        try:
            img = Image.new('L', (img_size[1], img_size[0]), 0)
            draw = ImageDraw.Draw(img)
            
            # Convert center coordinates
            if len(center) == 2:
                x, y = center
                if x <= 1.0 and y <= 1.0:
                    x = int(x * img_size[1])
                    y = int(y * img_size[0])
                    radius = int(radius * min(img_size))
                else:
                    x = int(x)
                    y = int(y)
                    radius = int(radius)
                
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=class_id)
                mask_array = np.array(img)
                mask[mask_array == class_id] = class_id
        except Exception as e:
            print(f"Error drawing circle: {e}")


def convert_labels_to_coco_format(dataset: DeepGISLabelDataset, output_dir: Path):
    """Convert DeepGIS labels to COCO format for Mask2Former training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Add categories
    for idx, cat in enumerate(dataset.categories):
        coco_data['categories'].append({
            'id': idx + 1,
            'name': cat.name,
            'supercategory': 'object'
        })
    
    annotation_id = 1
    
    for idx, image_label in enumerate(dataset.labels):
        image = image_label.image
        
        # Add image
        image_id = idx + 1
        coco_data['images'].append({
            'id': image_id,
            'file_name': image.name,
            'width': image.width,
            'height': image.height
        })
        
        # Add annotations
        try:
            label_data = json.loads(image_label.combined_label_shapes)
            features = label_data.get('features', [])
            
            for feature in features:
                category = feature.get('properties', {}).get('category', 'unknown')
                if category not in dataset.category_to_id:
                    continue
                
                category_id = dataset.category_to_id[category]
                geometry = feature.get('geometry', {})
                
                # Convert geometry to segmentation format
                segmentation = convert_geometry_to_segmentation(geometry, image.width, image.height)
                
                if segmentation:
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'segmentation': segmentation,
                        'area': calculate_area(segmentation, image.width, image.height),
                        'bbox': calculate_bbox(segmentation, image.width, image.height),
                        'iscrowd': 0
                    })
                    annotation_id += 1
        except Exception as e:
            print(f"Error processing label {image_label.id}: {e}")
    
    # Save COCO format JSON
    output_file = output_dir / 'annotations.json'
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Converted {len(coco_data['images'])} images to COCO format")
    print(f"Saved to: {output_file}")
    
    return output_file


def convert_geometry_to_segmentation(geometry: Dict, width: int, height: int) -> Optional[List]:
    """Convert GeoJSON geometry to COCO segmentation format."""
    geom_type = geometry.get('type', '')
    coordinates = geometry.get('coordinates', [])
    
    if geom_type == 'Polygon':
        # COCO format: list of polygons, each as [x1, y1, x2, y2, ...]
        polygon_coords = coordinates[0]  # Exterior ring
        segmentation = []
        for coord in polygon_coords:
            if len(coord) == 2:
                x, y = coord
                # Normalize if needed
                if x <= 1.0 and y <= 1.0:
                    x = x * width
                    y = y * height
                segmentation.extend([float(x), float(y)])
        return [segmentation] if len(segmentation) >= 6 else None
    
    elif geom_type == 'Circle':
        # Convert circle to polygon
        center = coordinates[0] if coordinates else [0.5, 0.5]
        radius = coordinates[1] if len(coordinates) > 1 else 0.1
        
        # Create polygon approximation of circle
        num_points = 32
        segmentation = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            if x <= 1.0 and y <= 1.0:
                x = x * width
                y = y * height
            segmentation.extend([float(x), float(y)])
        return [segmentation]
    
    return None


def calculate_area(segmentation: List, width: int, height: int) -> float:
    """Calculate area of segmentation polygon."""
    if not segmentation or len(segmentation[0]) < 6:
        return 0.0
    
    coords = segmentation[0]
    x_coords = coords[::2]
    y_coords = coords[1::2]
    
    # Shoelace formula
    area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i]
                         for i in range(len(x_coords)-1)))
    return float(area)


def calculate_bbox(segmentation: List, width: int, height: int) -> List[float]:
    """Calculate bounding box from segmentation."""
    if not segmentation or len(segmentation[0]) < 6:
        return [0.0, 0.0, 0.0, 0.0]
    
    coords = segmentation[0]
    x_coords = coords[::2]
    y_coords = coords[1::2]
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def train_mask2former(coco_annotations: Path, image_dir: Path, output_dir: Path,
                      num_classes: int, num_epochs: int = 50, batch_size: int = 4):
    """Train Mask2Former model using detectron2."""
    if not DETECTRON2_AVAILABLE:
        raise RuntimeError("detectron2 not installed. Install with: pip install detectron2")
    
    # Register dataset
    dataset_name = "deepgis_dataset"
    register_coco_instances(
        dataset_name,
        {},
        str(coco_annotations),
        str(image_dir)
    )
    
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    
    # Use Mask2Former config
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    
    # Training settings
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = num_epochs * 1000  # Approximate
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    
    # Output directory
    cfg.OUTPUT_DIR = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return cfg


def generate_initial_labels(model_path: Path, image_path: Path, output_geojson: Path,
                           confidence_threshold: float = 0.5):
    """Use trained Mask2Former to generate initial labels for DeepGIS label app."""
    if not DETECTRON2_AVAILABLE:
        raise RuntimeError("detectron2 not available")
    
    # Load model
    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    predictor = DefaultPredictor(cfg)
    
    # Predict
    img = np.array(Image.open(image_path).convert('RGB'))
    outputs = predictor(img)
    
    # Convert predictions to GeoJSON format
    geojson = predictions_to_geojson(outputs, img.shape[:2])
    
    # Save
    with open(output_geojson, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Generated initial labels: {output_geojson}")
    return geojson


def predictions_to_geojson(outputs, img_size: Tuple[int, int]) -> Dict:
    """Convert Mask2Former predictions to GeoJSON format for DeepGIS label app."""
    instances = outputs['instances']
    
    features = []
    for i in range(len(instances)):
        mask = instances.pred_masks[i].cpu().numpy()
        score = instances.scores[i].item()
        class_id = instances.pred_classes[i].item()
        
        # Convert mask to polygon
        polygons = mask_to_polygons(mask)
        
        for polygon in polygons:
            # Normalize coordinates to [0, 1]
            normalized_coords = [[x / img_size[1], y / img_size[0]] for x, y in polygon]
            normalized_coords.append(normalized_coords[0])  # Close polygon
            
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [normalized_coords]
                },
                'properties': {
                    'category': f'class_{class_id}',
                    'confidence': score
                }
            })
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def mask_to_polygons(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Convert binary mask to polygon coordinates."""
    from skimage import measure
    
    try:
        contours = measure.find_contours(mask, 0.5)
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                polygon = [(int(x), int(y)) for y, x in contour]
                polygons.append(polygon)
        return polygons
    except ImportError:
        print("Warning: scikit-image not installed. Using simple contour detection.")
        # Simple fallback
        return []


def main():
    parser = argparse.ArgumentParser(description='Train Mask2Former with DeepGIS labels')
    parser.add_argument('--mode', type=str, choices=['train', 'convert', 'predict'],
                       default='train', help='Mode: train, convert, or predict')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--label_ids', type=int, nargs='+', default=None,
                       help='Specific label IDs to use (None = all)')
    parser.add_argument('--output_dir', type=str, default='mask2former_output',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (for predict mode)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to image (for predict mode)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'convert':
        # Convert labels to COCO format
        print("Converting DeepGIS labels to COCO format...")
        dataset = DeepGISLabelDataset(args.image_dir, args.label_ids, args.img_size)
        coco_file = convert_labels_to_coco_format(dataset, output_dir / 'coco_format')
        print(f"Conversion complete: {coco_file}")
    
    elif args.mode == 'train':
        # Train Mask2Former
        if not DETECTRON2_AVAILABLE:
            print("Error: detectron2 not installed")
            print("Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
            return
        
        print("Loading dataset...")
        dataset = DeepGISLabelDataset(args.image_dir, args.label_ids, args.img_size)
        
        print("Converting to COCO format...")
        coco_file = convert_labels_to_coco_format(dataset, output_dir / 'coco_format')
        
        print("Training Mask2Former...")
        cfg = train_mask2former(
            coco_file,
            Path(args.image_dir),
            output_dir / 'checkpoints',
            dataset.num_classes,
            args.num_epochs,
            args.batch_size
        )
        
        print(f"Training complete! Model saved to: {output_dir / 'checkpoints'}")
    
    elif args.mode == 'predict':
        # Generate initial labels
        if args.model_path is None or args.image_path is None:
            print("Error: --model_path and --image_path required for predict mode")
            return
        
        print("Generating initial labels...")
        geojson = generate_initial_labels(
            Path(args.model_path),
            Path(args.image_path),
            output_dir / 'initial_labels.json',
            args.confidence
        )
        print(f"Initial labels saved to: {output_dir / 'initial_labels.json'}")
        print(f"Found {len(geojson['features'])} objects")


if __name__ == '__main__':
    main()

