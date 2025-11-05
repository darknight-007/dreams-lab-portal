"""
Script to analyze the contents of the bishop folder, including image tiles.
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Tuple
import sys


def get_file_info(filepath: Path) -> Dict:
    """Get information about a file."""
    stat = filepath.stat()
    return {
        'name': filepath.name,
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'is_dir': filepath.is_dir(),
        'extension': filepath.suffix.lower()
    }


def analyze_image(image_path: Path) -> Dict:
    """Analyze an image file and return its properties."""
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_bytes': image_path.stat().st_size,
                'error': None
            }
    except Exception as e:
        return {
            'width': None,
            'height': None,
            'format': None,
            'mode': None,
            'size_bytes': image_path.stat().st_size,
            'error': str(e)
        }


def analyze_folder(folder_path: str) -> Dict:
    """Analyze the entire folder structure."""
    folder = Path(folder_path)
    
    if not folder.exists():
        return {'error': f'Folder does not exist: {folder_path}'}
    
    if not folder.is_dir():
        return {'error': f'Path is not a directory: {folder_path}'}
    
    results = {
        'folder_path': str(folder.absolute()),
        'total_files': 0,
        'total_dirs': 0,
        'total_size': 0,
        'file_types': Counter(),
        'image_files': [],
        'non_image_files': [],
        'subdirectories': [],
        'image_statistics': {
            'total_images': 0,
            'formats': Counter(),
            'dimensions': Counter(),
            'modes': Counter(),
            'total_image_size': 0,
            'errors': []
        },
        'directory_structure': []
    }
    
    # Image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
                       '.gif', '.webp', '.svg', '.tif', '.tiff'}
    
    print(f"Analyzing folder: {folder_path}")
    print("=" * 80)
    
    # Walk through the directory
    for root, dirs, files in os.walk(folder):
        root_path = Path(root)
        relative_path = root_path.relative_to(folder)
        
        # Count subdirectories
        for d in dirs:
            dir_path = root_path / d
            results['subdirectories'].append({
                'path': str(relative_path / d) if relative_path != Path('.') else d,
                'info': get_file_info(dir_path)
            })
            results['total_dirs'] += 1
        
        # Analyze files
        for file in files:
            file_path = root_path / file
            file_info = get_file_info(file_path)
            file_info['relative_path'] = str(relative_path / file) if relative_path != Path('.') else file
            
            results['total_files'] += 1
            results['total_size'] += file_info['size']
            results['file_types'][file_info['extension']] += 1
            
            # Check if it's an image
            if file_info['extension'] in image_extensions:
                print(f"Analyzing image: {file_info['relative_path']}")
                image_info = analyze_image(file_path)
                image_info.update(file_info)
                
                results['image_files'].append(image_info)
                results['image_statistics']['total_images'] += 1
                results['image_statistics']['total_image_size'] += image_info['size_bytes']
                
                if image_info['error']:
                    results['image_statistics']['errors'].append({
                        'file': file_info['relative_path'],
                        'error': image_info['error']
                    })
                else:
                    results['image_statistics']['formats'][image_info['format']] += 1
                    results['image_statistics']['modes'][image_info['mode']] += 1
                    dim_key = f"{image_info['width']}x{image_info['height']}"
                    results['image_statistics']['dimensions'][dim_key] += 1
            else:
                results['non_image_files'].append(file_info)
        
        # Store directory structure
        if relative_path != Path('.'):
            results['directory_structure'].append({
                'path': str(relative_path),
                'file_count': len(files),
                'subdir_count': len(dirs)
            })
    
    return results


def print_summary(results: Dict):
    """Print a human-readable summary of the analysis."""
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    print("\n" + "=" * 80)
    print("FOLDER ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nFolder Path: {results['folder_path']}")
    print(f"Total Files: {results['total_files']:,}")
    print(f"Total Directories: {results['total_dirs']:,}")
    print(f"Total Size: {results['total_size'] / (1024**3):.2f} GB")
    
    print("\n" + "-" * 80)
    print("FILE TYPES")
    print("-" * 80)
    for ext, count in results['file_types'].most_common(10):
        ext_name = ext if ext else '(no extension)'
        print(f"  {ext_name:15s}: {count:,} files")
    
    print("\n" + "-" * 80)
    print("IMAGE STATISTICS")
    print("-" * 80)
    img_stats = results['image_statistics']
    print(f"Total Images: {img_stats['total_images']:,}")
    print(f"Total Image Size: {img_stats['total_image_size'] / (1024**3):.2f} GB")
    
    if img_stats['formats']:
        print("\nImage Formats:")
        for fmt, count in img_stats['formats'].most_common():
            print(f"  {fmt:10s}: {count:,} files")
    
    if img_stats['modes']:
        print("\nColor Modes:")
        for mode, count in img_stats['modes'].most_common():
            print(f"  {mode:10s}: {count:,} files")
    
    if img_stats['dimensions']:
        print("\nCommon Dimensions:")
        for dim, count in img_stats['dimensions'].most_common(10):
            print(f"  {dim:15s}: {count:,} files")
    
    if img_stats['errors']:
        print(f"\nErrors reading {len(img_stats['errors'])} images:")
        for err_info in img_stats['errors'][:5]:
            print(f"  {err_info['file']}: {err_info['error']}")
        if len(img_stats['errors']) > 5:
            print(f"  ... and {len(img_stats['errors']) - 5} more errors")
    
    print("\n" + "-" * 80)
    print("DIRECTORY STRUCTURE")
    print("-" * 80)
    if results['subdirectories']:
        print(f"Total Subdirectories: {len(results['subdirectories'])}")
        print("\nTop-level structure:")
        top_level = [d for d in results['subdirectories'] if '/' not in d['path']]
        for d in sorted(top_level, key=lambda x: x['info']['modified'], reverse=True)[:10]:
            print(f"  {d['path']}")
    else:
        print("No subdirectories found (all files are in root)")
    
    print("\n" + "-" * 80)
    print("NON-IMAGE FILES")
    print("-" * 80)
    if results['non_image_files']:
        print(f"Total: {len(results['non_image_files'])} files")
        print("\nSample files:")
        for f in results['non_image_files'][:10]:
            size_mb = f['size'] / (1024**2)
            print(f"  {f['relative_path']:60s} ({size_mb:.2f} MB)")
        if len(results['non_image_files']) > 10:
            print(f"  ... and {len(results['non_image_files']) - 10} more files")
    else:
        print("All files are images!")


def save_json_report(results: Dict, output_path: str):
    """Save detailed results to a JSON file."""
    # Convert Counter objects to dicts for JSON serialization
    json_results = results.copy()
    json_results['file_types'] = dict(results['file_types'])
    json_results['image_statistics']['formats'] = dict(results['image_statistics']['formats'])
    json_results['image_statistics']['dimensions'] = dict(results['image_statistics']['dimensions'])
    json_results['image_statistics']['modes'] = dict(results['image_statistics']['modes'])
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed JSON report saved to: {output_path}")


def main():
    folder_path = "/mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop/"
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    
    print(f"Starting analysis of: {folder_path}")
    print("This may take a while for large folders...\n")
    
    # Analyze folder
    results = analyze_folder(folder_path)
    
    # Print summary
    print_summary(results)
    
    # Save JSON report
    output_file = f"bishop_folder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json_report(results, output_file)
    
    # Additional analysis: check for tile patterns
    if results['image_statistics']['total_images'] > 0:
        print("\n" + "=" * 80)
        print("TILE PATTERN ANALYSIS")
        print("=" * 80)
        
        # Group images by dimensions
        dim_groups = defaultdict(list)
        for img in results['image_files']:
            if img['width'] and img['height']:
                dim_key = (img['width'], img['height'])
                dim_groups[dim_key].append(img['relative_path'])
        
        print("\nImages grouped by dimensions:")
        for (w, h), files in sorted(dim_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"  {w}x{h}: {len(files)} images")
            # Check if filenames suggest tile coordinates
            sample_names = files[:3]
            print(f"    Sample files: {', '.join([Path(f).name for f in sample_names])}")


if __name__ == '__main__':
    main()

