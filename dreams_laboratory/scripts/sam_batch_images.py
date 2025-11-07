#!/usr/bin/env python3
"""
SAM Batch Processing - Generalized

Runs Segment Anything Model on a random sample of images from any directory.
Works with rock tiles, moon tiles, or any other image dataset.

Usage:
    # Rock tiles
    python3 sam_batch_images.py /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw --num-samples 20

    # Moon tiles
    python3 sam_batch_images.py /mnt/22tb-hdd/2tbssdcx-bkup/deepgis/deepgis_moon/static-root/moon-tiles/9 --num-samples 20

    # Specific zoom levels only
    python3 sam_batch_images.py /path/to/tiles --num-samples 20 --zoom-levels 21 22 23

    # All images in directory (no sampling)
    python3 sam_batch_images.py /path/to/images --all
"""

import random
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Dict, Optional
import time
import argparse


def find_all_images(base_dir: Path, zoom_levels: Optional[List[int]] = None, 
                   extensions: List[str] = None) -> List[Path]:
    """
    Find all image files in the directory.
    
    Args:
        base_dir: Base directory to search
        zoom_levels: If provided, only search in these zoom subdirectories
        extensions: Image extensions to search for (default: png, jpg, jpeg, tif, tiff)
    
    Returns:
        List of image paths
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    all_images = []
    
    if zoom_levels is not None:
        # Search within zoom level subdirectories (e.g., /tiles/23/x/y.png)
        print(f"🔍 Searching in zoom levels: {zoom_levels}")
        for zoom in zoom_levels:
            zoom_dir = base_dir / str(zoom)
            if zoom_dir.exists():
                for ext in extensions:
                    images = list(zoom_dir.glob(f"**/*{ext}"))
                    all_images.extend(images)
                print(f"  Zoom {zoom}: {len([i for i in all_images if f'/{zoom}/' in str(i)])} images")
    else:
        # Search entire directory recursively
        print(f"🔍 Searching entire directory recursively")
        for ext in extensions:
            images = list(base_dir.glob(f"**/*{ext}"))
            all_images.extend(images)
        print(f"  Found {len(all_images)} images")
    
    return all_images


def get_zoom_level(image_path: Path, base_dir: Path) -> Optional[int]:
    """Extract zoom level from image path if available."""
    try:
        relative = image_path.relative_to(base_dir)
        parts = relative.parts
        # Try to find a numeric directory in the path
        for part in parts:
            if part.isdigit():
                return int(part)
    except:
        pass
    return None


def run_sam_on_image(image_path: Path, output_dir: Path, base_dir: Path) -> Dict:
    """Run SAM on a single image and return results."""
    script_dir = Path(__file__).parent
    sam_script = script_dir / "segment_anything_rocks.py"
    
    # Run SAM
    cmd = [
        "python3",
        str(sam_script),
        str(image_path),
        "--model", "vit_b",
        "--geojson",
        "--output_dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse analysis results
        analysis_file = output_dir / f"{image_path.stem}_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            zoom = get_zoom_level(image_path, base_dir)
            
            return {
                'success': True,
                'image': image_path.name,
                'path': str(image_path.relative_to(base_dir)) if image_path.is_relative_to(base_dir) else str(image_path),
                'zoom': zoom,
                'analysis': analysis,
                'error': None
            }
        else:
            return {
                'success': False,
                'image': image_path.name,
                'path': str(image_path.relative_to(base_dir)) if image_path.is_relative_to(base_dir) else str(image_path),
                'error': 'Analysis file not created'
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'image': image_path.name,
            'error': 'Timeout (>60s)'
        }
    except Exception as e:
        return {
            'success': False,
            'image': image_path.name,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run SAM on a batch of images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rock tiles (20 random samples)
  python3 sam_batch_images.py /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw --num-samples 20

  # Moon tiles (20 random samples)
  python3 sam_batch_images.py /mnt/22tb-hdd/2tbssdcx-bkup/deepgis/deepgis_moon/static-root/moon-tiles/9 --num-samples 20

  # Specific zoom levels
  python3 sam_batch_images.py /path/to/tiles --num-samples 20 --zoom-levels 21 22 23

  # Process all images (no sampling)
  python3 sam_batch_images.py /path/to/images --all

  # Custom output directory
  python3 sam_batch_images.py /path/to/images --num-samples 10 --output-dir my_results
        """
    )
    
    parser.add_argument('dataset_path', type=str,
                       help='Path to dataset directory')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of random samples to process (default: 20)')
    parser.add_argument('--all', action='store_true',
                       help='Process all images (ignore --num-samples)')
    parser.add_argument('--zoom-levels', type=int, nargs='+',
                       help='Only process these zoom levels (e.g., --zoom-levels 21 22 23)')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory (default: sam_batch_results_<dataset_name>)')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
                       help='Image file extensions to search for (default: .png .jpg .jpeg .tif .tiff)')
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dataset_name = dataset_path.name
        output_dir = Path(__file__).parent / f"sam_batch_results_{dataset_name}"
    
    output_dir.mkdir(exist_ok=True)
    
    # Print header
    print("="*70)
    print(" SAM BATCH PROCESSING - Image Segmentation")
    print("="*70)
    print()
    print(f"📂 Dataset: {dataset_path}")
    print(f"📊 Sample size: {'ALL images' if args.all else f'{args.num_samples} random samples'}")
    print(f"📁 Output: {output_dir}")
    if args.zoom_levels:
        print(f"🔢 Zoom levels: {args.zoom_levels}")
    print()
    
    # Find all images
    print("🔍 Finding images...")
    all_images = find_all_images(dataset_path, args.zoom_levels, args.extensions)
    
    if len(all_images) == 0:
        print("❌ No images found!")
        print()
        print("Tips:")
        print("  - Check the path is correct")
        print("  - Try different zoom levels: --zoom-levels 9 10 11")
        print("  - Try different extensions: --extensions .png .tif")
        sys.exit(1)
    
    print(f"✓ Found {len(all_images)} total images")
    print()
    
    # Select sample
    if args.all or len(all_images) <= args.num_samples:
        if args.all:
            print(f"📸 Processing ALL {len(all_images)} images")
        else:
            print(f"⚠️  Only {len(all_images)} images available, processing all")
        sample_images = all_images
    else:
        sample_images = random.sample(all_images, args.num_samples)
        print(f"📸 Selected {len(sample_images)} random images")
    
    print()
    print("="*70)
    print()
    
    # Process each image
    results = []
    start_time = time.time()
    
    for i, img_path in enumerate(sample_images, 1):
        rel_path = img_path.relative_to(dataset_path) if img_path.is_relative_to(dataset_path) else img_path
        
        print(f"[{i}/{len(sample_images)}] Processing: {img_path.name}")
        print(f"    Path: {rel_path}")
        
        result = run_sam_on_image(img_path, output_dir, dataset_path)
        results.append(result)
        
        if result['success']:
            analysis = result['analysis']
            print(f"    ✓ Segments: {analysis['num_segments']}")
            print(f"      Avg area: {analysis['avg_area']:.1f} px")
            print(f"      Avg IoU: {analysis['avg_iou']:.3f}")
        else:
            print(f"    ❌ Error: {result['error']}")
        
        print()
    
    elapsed = time.time() - start_time
    
    # Summary statistics
    print()
    print("="*70)
    print(" BATCH PROCESSING COMPLETE")
    print("="*70)
    print()
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✓ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/len(results):.1f}s per image)")
    print()
    
    if successful:
        print("📊 AGGREGATE STATISTICS:")
        print()
        
        total_segments = sum(r['analysis']['num_segments'] for r in successful)
        avg_segments = total_segments / len(successful)
        min_segments = min(r['analysis']['num_segments'] for r in successful)
        max_segments = max(r['analysis']['num_segments'] for r in successful)
        
        avg_iou = sum(r['analysis']['avg_iou'] for r in successful) / len(successful)
        avg_stability = sum(r['analysis']['avg_stability'] for r in successful) / len(successful)
        avg_area = sum(r['analysis']['avg_area'] for r in successful) / len(successful)
        
        print(f"  Segments per image:")
        print(f"    • Average: {avg_segments:.1f}")
        print(f"    • Min: {min_segments}")
        print(f"    • Max: {max_segments}")
        print(f"    • Total: {total_segments}")
        print()
        print(f"  Quality metrics:")
        print(f"    • Average IoU: {avg_iou:.3f}")
        print(f"    • Average Stability: {avg_stability:.3f}")
        print(f"    • Average Area: {avg_area:.1f} pixels")
        print()
        
        # By zoom level (if available)
        by_zoom = {}
        for r in successful:
            if r['zoom'] is not None:
                zoom = r['zoom']
                if zoom not in by_zoom:
                    by_zoom[zoom] = []
                by_zoom[zoom].append(r)
        
        if by_zoom:
            print(f"  By zoom level:")
            for zoom in sorted(by_zoom.keys()):
                tiles = by_zoom[zoom]
                avg_segs = sum(t['analysis']['num_segments'] for t in tiles) / len(tiles)
                print(f"    • Zoom {zoom}: {len(tiles)} images, avg {avg_segs:.1f} segments")
            print()
    
    if failed:
        print("❌ FAILED IMAGES:")
        for r in failed:
            print(f"  • {r['image']}: {r['error']}")
        print()
    
    # Save summary
    summary = {
        'dataset': str(dataset_path),
        'num_samples': len(sample_images),
        'successful': len(successful),
        'failed': len(failed),
        'elapsed_seconds': elapsed,
        'avg_time_per_image': elapsed / len(results) if results else 0,
        'results': results
    }
    
    if successful:
        summary['statistics'] = {
            'avg_segments_per_image': avg_segments,
            'min_segments': min_segments,
            'max_segments': max_segments,
            'total_segments': total_segments,
            'avg_iou': avg_iou,
            'avg_stability': avg_stability,
            'avg_area': avg_area
        }
        
        if by_zoom:
            summary['statistics']['by_zoom_level'] = {
                zoom: {
                    'count': len(tiles),
                    'avg_segments': sum(t['analysis']['num_segments'] for t in tiles) / len(tiles)
                }
                for zoom, tiles in by_zoom.items()
            }
    
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_file}")
    print()
    print("="*70)
    print()
    print("📁 View all results:")
    print(f"   cd {output_dir}")
    print(f"   ls -lh *_sam_segments.jpg")
    print()
    print("   # View visualizations")
    print(f"   xdg-open {output_dir}/*_sam_segments.jpg")
    print()
    print("   # View one analysis file")
    print(f"   cat {output_dir}/*_analysis.json | head -50")
    print()
    print("💡 Next steps:")
    print("   1. Review segmentation visualizations")
    print("   2. Import GeoJSON files into DeepGIS for labeling")
    print("   3. Train custom model on labeled data")
    print()


if __name__ == '__main__':
    main()

