#!/usr/bin/env python3
"""
Test Zero-Shot Detection on Rock Tiles

This script demonstrates the limitations of zero-shot COCO models
when applied to specialized domains like geological imagery.

Expected Result: 
- COCO models won't detect rocks well (not in 80 COCO categories)
- May detect generic "stone" or nothing at all
- Demonstrates why custom training is needed for geological features
"""

import requests
import numpy as np
from PIL import Image
from pathlib import Path
import io
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

def download_rock_tile(z, x, y, save_path=None):
    """Download a rock tile from rocks.deepgis.org"""
    url = f'https://rocks.deepgis.org/{z}/{x}/{y}.png'
    print(f"🌐 Downloading rock tile from:")
    print(f"   {url}")
    print()
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open as image
        img = Image.open(io.BytesIO(response.content))
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)
            print(f"✓ Saved tile to: {save_path}")
        
        return img, save_path if save_path else None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading tile: {e}")
        print()
        print("Note: Rock tiles may not be accessible from this location.")
        print("You can also provide a local rock image instead.")
        return None, None


def test_zero_shot_on_rocks():
    """Main test function"""
    print("="*70)
    print(" TESTING ZERO-SHOT DETECTION ON GEOLOGICAL IMAGERY")
    print("="*70)
    print()
    print("This test demonstrates:")
    print("  ✅ Zero-shot works on common objects (people, cars, animals)")
    print("  ❌ Zero-shot fails on specialized objects (rocks, minerals)")
    print()
    print("="*70)
    print()
    
    # Sample rock tile coordinates (example - adjust for your dataset)
    sample_tiles = [
        (23, 5892, 12745, "High-res rock surface detail"),
        (22, 2946, 6372, "Medium-res rock texture"),
        (21, 1473, 3186, "Lower-res rock overview")
    ]
    
    results_dir = scripts_dir / "zero_shot_rocks_test"
    results_dir.mkdir(exist_ok=True)
    
    all_images = []
    
    # Download sample tiles
    print("📥 STEP 1: Downloading Sample Rock Tiles")
    print("-"*70)
    for i, (z, x, y, description) in enumerate(sample_tiles, 1):
        print(f"{i}. {description} (zoom {z})")
        save_path = results_dir / f"rock_tile_z{z}_x{x}_y{y}.png"
        
        img, path = download_rock_tile(z, x, y, save_path)
        if img and path:
            all_images.append((path, description))
            print(f"   Size: {img.size}")
            print()
        else:
            print(f"   ⚠️  Failed to download, skipping...")
            print()
    
    if not all_images:
        print()
        print("❌ No tiles downloaded successfully.")
        print()
        print("Alternative: Run on a local rock image:")
        print("  python zero_shot_detection.py /path/to/your/rock/image.jpg --visualize")
        print()
        return
    
    # Run zero-shot detection
    print()
    print("🔍 STEP 2: Running Zero-Shot Detection (COCO Model)")
    print("-"*70)
    print()
    print("Testing if pre-trained COCO model can detect rocks...")
    print("(Spoiler: It probably won't! Rocks aren't in COCO's 80 classes)")
    print()
    
    try:
        from zero_shot_detection import ZeroShotMaskRCNN
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector = ZeroShotMaskRCNN(confidence_threshold=0.3, device=device)
        
        for img_path, description in all_images:
            print(f"📸 Testing: {description}")
            print(f"   File: {img_path.name}")
            
            predictions = detector.predict(img_path)
            
            print(f"   Detections: {predictions['num_detections']}")
            
            if predictions['num_detections'] > 0:
                print("   Found:")
                for det in predictions['detections']:
                    print(f"     • {det['class_name']}: {det['confidence']:.2%}")
                
                # Visualize
                vis_path = results_dir / f"{img_path.stem}_detected.jpg"
                detector.visualize(img_path, predictions, vis_path)
                print(f"   Visualization: {vis_path}")
            else:
                print("   ❌ No objects detected!")
                print("   (Expected: Rocks aren't in COCO dataset)")
            
            print()
    
    except ImportError as e:
        print(f"❌ Error importing detection module: {e}")
        print()
        print("Make sure you have the required dependencies:")
        print("  pip install torch torchvision")
        return
    
    except Exception as e:
        print(f"❌ Error running detection: {e}")
        return
    
    # Summary
    print()
    print("="*70)
    print(" RESULTS SUMMARY")
    print("="*70)
    print()
    print("🎯 Key Findings:")
    print()
    print("1. Zero-Shot COCO Model:")
    print("   • Trained on: 80 common object categories")
    print("   • Categories: person, car, dog, chair, pizza, etc.")
    print("   • Rock types: ❌ NOT INCLUDED")
    print()
    print("2. For Geological Detection, You Need:")
    print("   ✅ Custom-trained model on labeled rock data")
    print("   ✅ Categories like: granite, basalt, sandstone, etc.")
    print("   ✅ Training script: train_mask2former_deepgis.py")
    print()
    print("3. Alternative Approach:")
    print("   • Use zero-shot to bootstrap labels for OTHER objects")
    print("     (e.g., detect 'person' or 'car' in field photos)")
    print("   • Then manually label rocks and train custom model")
    print()
    print("="*70)
    print()
    print(f"📁 All results saved to: {results_dir}")
    print()
    print("Next Steps:")
    print("  1. Label rock images in DeepGIS label app")
    print("  2. Train custom Mask2Former:")
    print("     python train_mask2former_deepgis.py --mode train")
    print("  3. Use custom model for rock detection")
    print()


if __name__ == '__main__':
    test_zero_shot_on_rocks()

