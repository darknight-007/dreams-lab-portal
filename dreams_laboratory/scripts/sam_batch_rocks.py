#!/usr/bin/env python3
"""
SAM Batch Processing for Rock Tiles

Runs Segment Anything Model on a random sample of rock images
and generates comprehensive analysis and statistics.
"""

import random
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Dict
import time
import os

def find_all_rock_tiles(base_dir: Path, zoom_levels: List[int] = None) -> List[Path]:
    """Find all PNG files in the rock tiles directory."""
    if zoom_levels is None:
        zoom_levels = [21, 22, 23]  # High resolution tiles
    
    all_tiles = []
    for zoom in zoom_levels:
        zoom_dir = base_dir / str(zoom)
        if zoom_dir.exists():
            tiles = list(zoom_dir.glob("**/*.png"))
            all_tiles.extend(tiles)
            print(f"  Zoom {zoom}: {len(tiles)} tiles")
    
    return all_tiles


def run_sam_on_image(image_path: Path, output_dir: Path) -> Dict:
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
            return {
                'success': True,
                'image': image_path.name,
                'zoom': int(str(image_path.parent.parent.name)),
                'analysis': analysis,
                'error': None
            }
        else:
            return {
                'success': False,
                'image': image_path.name,
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
    # Get dataset path from environment or use default
    rocks_dir = Path(os.environ.get('ROCKS_DIR', '/path/to/rock-tiles/raw'))
    output_dir = Path(__file__).parent / "sam_batch_results"
    output_dir.mkdir(exist_ok=True)
    
    num_samples = 20
    
    print("="*70)
    print(" SAM BATCH PROCESSING - Random Sample of Rock Tiles")
    print("="*70)
    print()
    print(f"📂 Dataset: {rocks_dir}")
    print(f"📊 Sample size: {num_samples} images")
    print(f"📁 Output: {output_dir}")
    print()
    
    # Find all tiles
    print("🔍 Finding rock tiles...")
    all_tiles = find_all_rock_tiles(rocks_dir, zoom_levels=[21, 22, 23])
    print(f"✓ Found {len(all_tiles)} total tiles")
    print()
    
    # Random sample
    if len(all_tiles) < num_samples:
        print(f"⚠️  Only {len(all_tiles)} tiles available, using all")
        sample_tiles = all_tiles
    else:
        sample_tiles = random.sample(all_tiles, num_samples)
    
    print(f"📸 Selected {len(sample_tiles)} random tiles")
    print()
    print("="*70)
    print()
    
    # Process each tile
    results = []
    start_time = time.time()
    
    for i, tile_path in enumerate(sample_tiles, 1):
        print(f"[{i}/{len(sample_tiles)}] Processing: {tile_path.name}")
        print(f"    Path: .../{tile_path.parent.parent.name}/{tile_path.parent.name}/{tile_path.name}")
        
        result = run_sam_on_image(tile_path, output_dir)
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
        
        # By zoom level
        by_zoom = {}
        for r in successful:
            zoom = r['zoom']
            if zoom not in by_zoom:
                by_zoom[zoom] = []
            by_zoom[zoom].append(r)
        
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
        'dataset': str(rocks_dir),
        'num_samples': len(sample_tiles),
        'successful': len(successful),
        'failed': len(failed),
        'elapsed_seconds': elapsed,
        'avg_time_per_image': elapsed / len(results),
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
            'avg_area': avg_area,
            'by_zoom_level': {
                zoom: {
                    'count': len(tiles),
                    'avg_segments': sum(t['analysis']['num_segments'] for t in tiles) / len(tiles)
                }
                for zoom, tiles in by_zoom.items()
            }
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
    print(f"   xdg-open *_sam_segments.jpg")
    print()
    print("💡 Next steps:")
    print("   1. Review segmentation visualizations")
    print("   2. Import GeoJSON files into DeepGIS for labeling")
    print("   3. Train custom model on labeled data")
    print()


if __name__ == '__main__':
    main()

