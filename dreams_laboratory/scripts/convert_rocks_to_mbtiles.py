#!/usr/bin/env python3
"""
Convert rock tiles to MBTiles format for TileServer GL.
"""

import os
import sqlite3
from pathlib import Path
import argparse

def create_mbtiles(tiles_dir, output_file, name="Rock Tiles", description="DeepGIS Rock Imagery"):
    """
    Convert XYZ tile directory to MBTiles.
    
    Args:
        tiles_dir: Path to tiles (contains zoom/x/y.png structure)
        output_file: Output .mbtiles file
        name: Layer name
        description: Layer description
    """
    
    print(f"Creating MBTiles: {output_file}")
    print(f"Source: {tiles_dir}\n")
    
    # Create MBTiles database
    conn = sqlite3.connect(output_file)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            name TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB,
            PRIMARY KEY (zoom_level, tile_column, tile_row)
        )
    """)
    
    # Get bounds from tiles
    tiles_path = Path(tiles_dir)
    zoom_dirs = [z for z in tiles_path.iterdir() if z.is_dir() and z.name.isdigit()]
    
    if not zoom_dirs:
        print(f"Error: No zoom directories found in {tiles_dir}")
        return
    
    zoom_levels = sorted([int(z.name) for z in zoom_dirs])
    
    min_zoom = min(zoom_levels)
    max_zoom = max(zoom_levels)
    
    print(f"Zoom levels found: {min_zoom} - {max_zoom}")
    
    # Metadata (adjust bounds to your specific region if known)
    metadata = {
        'name': name,
        'description': description,
        'version': '1.0',
        'type': 'baselayer',
        'format': 'png',
        'minzoom': str(min_zoom),
        'maxzoom': str(max_zoom),
        'bounds': '-180,-85.0511,180,85.0511',  # World bounds (adjust if you know exact region)
        'center': '-111.26513,33.78215,18',  # Default center (adjust to your area)
        'attribution': 'DeepGIS Rock Tiles'
    }
    
    for key, value in metadata.items():
        cursor.execute("INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)",
                      (key, value))
    
    conn.commit()
    
    # Import tiles
    print(f"\nConverting tiles...")
    total_tiles = 0
    
    for zoom in zoom_levels:
        zoom_path = tiles_path / str(zoom)
        if not zoom_path.exists():
            continue
            
        print(f"Processing zoom level {zoom}...", end=' ', flush=True)
        zoom_tile_count = 0
        
        for x_dir in zoom_path.iterdir():
            if not x_dir.is_dir():
                continue
            
            x = int(x_dir.name)
            
            for tile_file in x_dir.glob('*.png'):
                y = int(tile_file.stem)
                
                # Convert from XYZ (Google/OSM) to TMS (MBTiles) y-coordinate
                # TMS uses inverted Y: y_tms = (2^zoom - 1) - y
                y_tms = (1 << zoom) - 1 - y
                
                # Read tile data
                try:
                    with open(tile_file, 'rb') as f:
                        tile_data = f.read()
                    
                    # Insert into database
                    cursor.execute("""
                        INSERT OR REPLACE INTO tiles 
                        (zoom_level, tile_column, tile_row, tile_data)
                        VALUES (?, ?, ?, ?)
                    """, (zoom, x, y_tms, tile_data))
                    
                    total_tiles += 1
                    zoom_tile_count += 1
                    
                    # Commit every 1000 tiles for progress
                    if total_tiles % 1000 == 0:
                        conn.commit()
                        print(f"\r  Processed {total_tiles} tiles...", end='', flush=True)
                        
                except Exception as e:
                    print(f"\nWarning: Could not process {tile_file}: {e}")
                    continue
        
        print(f" {zoom_tile_count} tiles")
    
    # Final commit and optimize
    print("\nOptimizing database...")
    conn.commit()
    cursor.execute("CREATE INDEX IF NOT EXISTS tiles_idx ON tiles (zoom_level, tile_column, tile_row)")
    cursor.execute("ANALYZE")
    conn.commit()
    conn.close()
    
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total tiles: {total_tiles:,}")
    print(f"   Zoom range: {min_zoom} - {max_zoom}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"\nNext steps:")
    print(f"   1. Move to tileserver: cp {output_file} /home/jdas/dreams-lab-website-server/deepgis-xr/data/")
    print(f"   2. Restart tileserver: docker restart deepgis-xr_tileserver_1")
    print(f"   3. View at: https://mbtiles.deepgis.org/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XYZ tiles to MBTiles')
    parser.add_argument('--tiles_dir', type=str, required=True,
                       help='Directory containing XYZ tiles (with zoom/x/y.png structure)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output .mbtiles file path')
    parser.add_argument('--name', type=str, default='Rock Tiles',
                       help='Layer name (shown in TileServer)')
    parser.add_argument('--description', type=str, default='DeepGIS Rock Imagery',
                       help='Layer description')
    
    args = parser.parse_args()
    
    # Validate inputs
    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.exists():
        print(f"Error: Tiles directory not found: {tiles_dir}")
        exit(1)
    
    output_file = Path(args.output)
    if output_file.exists():
        response = input(f"Output file {output_file} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)
    
    create_mbtiles(
        str(tiles_dir),
        str(output_file),
        args.name,
        args.description
    )
