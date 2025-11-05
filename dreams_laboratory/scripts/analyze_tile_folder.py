#!/usr/bin/env python3
"""
Analyze GIS tile folder structure and count leaf-node image files.

This script provides detailed statistics about image tiles in a GIS tile directory structure.
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple


def count_image_files(tile_dir: str, extensions: List[str] = None) -> Dict:
    """
    Count image files in tile directory structure.
    
    Args:
        tile_dir: Root directory containing tiles
        extensions: List of file extensions to count (default: common image formats)
    
    Returns:
        Dictionary with statistics
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp']
    
    tile_path = Path(tile_dir)
    if not tile_path.exists():
        raise ValueError(f"Directory does not exist: {tile_dir}")
    
    stats = {
        'total_files': 0,
        'by_extension': defaultdict(int),
        'by_zoom_level': defaultdict(int),
        'by_directory': defaultdict(int),
        'file_sizes': [],
        'zoom_levels': set(),
        'file_paths': []
    }
    
    # Walk through directory structure
    for root, dirs, files in os.walk(tile_dir):
        root_path = Path(root)
        
        # Skip common non-image files
        skip_files = {'tilemapresource.xml', 'googlemaps.html', 'leaflet.html', 'openlayers.html'}
        
        for file in files:
            if file in skip_files:
                continue
            
            file_path = root_path / file
            file_ext = file_path.suffix.lower()
            
            if file_ext in extensions or any(file.lower().endswith(ext) for ext in extensions):
                stats['total_files'] += 1
                stats['by_extension'][file_ext] += 1
                stats['file_paths'].append(str(file_path))
                
                # Try to determine zoom level from path structure
                # GIS tiles often follow: zoom_level/x/y.ext or zoom_level/x/y.ext
                parts = root_path.parts
                if len(parts) >= 2:
                    # Check if first part after root is numeric (zoom level)
                    try:
                        zoom_level = int(parts[-2])  # Second to last part (x coordinate)
                        # Actually, check if any part looks like zoom level
                        for part in reversed(parts):
                            if part.isdigit() and len(part) <= 2:  # Zoom levels are typically 0-20
                                zoom_level = int(part)
                                stats['zoom_levels'].add(zoom_level)
                                stats['by_zoom_level'][zoom_level] += 1
                                break
                    except (ValueError, IndexError):
                        pass
                
                # Track directory depth
                depth = len(root_path.relative_to(tile_path).parts)
                stats['by_directory'][depth] += 1
                
                # Get file size
                try:
                    file_size = file_path.stat().st_size
                    stats['file_sizes'].append(file_size)
                except OSError:
                    pass
    
    return stats


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_statistics(stats: Dict, verbose: bool = False):
    """Print formatted statistics."""
    print("=" * 80)
    print("GIS TILE FOLDER ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"Total Image Files: {stats['total_files']:,}")
    print()
    
    # File count by extension
    if stats['by_extension']:
        print("Files by Extension:")
        for ext, count in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_files']) * 100
            print(f"  {ext:10s}: {count:8,} ({percentage:5.2f}%)")
        print()
    
    # Files by zoom level
    if stats['by_zoom_level']:
        print("Files by Zoom Level:")
        for zoom in sorted(stats['by_zoom_level'].keys()):
            count = stats['by_zoom_level'][zoom]
            percentage = (count / stats['total_files']) * 100
            print(f"  Zoom {zoom:2d}: {count:8,} files ({percentage:5.2f}%)")
        print()
    
    # Files by directory depth
    if stats['by_directory']:
        print("Files by Directory Depth:")
        for depth in sorted(stats['by_directory'].keys()):
            count = stats['by_directory'][depth]
            percentage = (count / stats['total_files']) * 100
            print(f"  Depth {depth}: {count:8,} files ({percentage:5.2f}%)")
        print()
    
    # File size statistics
    if stats['file_sizes']:
        total_size = sum(stats['file_sizes'])
        avg_size = total_size / len(stats['file_sizes'])
        min_size = min(stats['file_sizes'])
        max_size = max(stats['file_sizes'])
        
        print("File Size Statistics:")
        print(f"  Total Size:    {format_size(total_size)}")
        print(f"  Average Size:   {format_size(avg_size)}")
        print(f"  Minimum Size:   {format_size(min_size)}")
        print(f"  Maximum Size:   {format_size(max_size)}")
        print()
    
    # Zoom level summary
    if stats['zoom_levels']:
        print(f"Zoom Levels Found: {sorted(stats['zoom_levels'])}")
        print(f"Number of Zoom Levels: {len(stats['zoom_levels'])}")
        print()
    
    # Sample file paths (if verbose)
    if verbose and stats['file_paths']:
        print("Sample File Paths:")
        for path in stats['file_paths'][:10]:
            print(f"  {path}")
        if len(stats['file_paths']) > 10:
            print(f"  ... and {len(stats['file_paths']) - 10} more")
        print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GIS tile folder structure and count image files'
    )
    parser.add_argument(
        'tile_dir',
        type=str,
        help='Root directory containing tile structure'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'],
        help='File extensions to count (default: .png .jpg .jpeg .tif .tiff .webp)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output including sample file paths'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to file (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Count files
    print(f"Analyzing tile directory: {args.tile_dir}")
    print("This may take a while for large directories...")
    print()
    
    stats = count_image_files(args.tile_dir, args.extensions)
    
    # Print statistics
    print_statistics(stats, verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        import json
        # Convert sets to lists for JSON serialization
        stats_json = stats.copy()
        stats_json['zoom_levels'] = sorted(list(stats['zoom_levels']))
        stats_json['by_extension'] = dict(stats['by_extension'])
        stats_json['by_zoom_level'] = dict(stats['by_zoom_level'])
        stats_json['by_directory'] = dict(stats['by_directory'])
        
        # Don't save all file paths to JSON (too large)
        stats_json.pop('file_paths', None)
        
        with open(args.output, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"Statistics saved to: {args.output}")


if __name__ == '__main__':
    main()

