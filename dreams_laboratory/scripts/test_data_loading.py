"""
Quick test script to verify multispectral TIFF data loading works correctly.
This should be run before training the full model.
"""

import sys
from pathlib import Path

# Check dependencies
missing_deps = []
try:
    import torch
    print("✓ PyTorch available")
except ImportError:
    missing_deps.append("torch")
    print("✗ PyTorch not found")

try:
    import rasterio
    print("✓ Rasterio available")
except ImportError:
    missing_deps.append("rasterio")
    print("✗ Rasterio not found")

try:
    import numpy as np
    print("✓ NumPy available")
except ImportError:
    missing_deps.append("numpy")
    print("✗ NumPy not found")

try:
    from PIL import Image
    print("✓ PIL available")
except ImportError:
    missing_deps.append("Pillow")
    print("✗ PIL not found")

if missing_deps:
    print(f"\nMissing dependencies: {', '.join(missing_deps)}")
    print("\nInstall with:")
    if "rasterio" in missing_deps:
        print("  pip install rasterio")
    if "torch" in missing_deps:
        print("  pip install torch")
    if "numpy" in missing_deps:
        print("  pip install numpy")
    if "Pillow" in missing_deps:
        print("  pip install Pillow")
    sys.exit(1)

# Test data loading
print("\n" + "="*80)
print("Testing data loading...")
print("="*80)

tile_dir = Path("/mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop")

if not tile_dir.exists():
    print(f"ERROR: Directory does not exist: {tile_dir}")
    sys.exit(1)

# Find a sample TIFF file
tiff_files = sorted(list(tile_dir.glob("**/*.tif")))
tiff_files = [f for f in tiff_files if "__MACOSX" not in str(f)]

if len(tiff_files) == 0:
    print(f"ERROR: No TIFF files found in {tile_dir}")
    sys.exit(1)

print(f"\nFound {len(tiff_files)} TIFF files")
print(f"Testing with: {tiff_files[0].name}")

# Try to read the first TIFF
try:
    import rasterio
    import numpy as np
    
    with rasterio.open(tiff_files[0]) as src:
        print(f"\nFile: {tiff_files[0].name}")
        print(f"  Bands: {src.count}")
        print(f"  Width: {src.width}, Height: {src.height}")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  CRS: {src.crs}")
        
        # Read first band
        band1 = src.read(1)
        print(f"\n  Band 1 stats:")
        print(f"    Min: {band1.min()}, Max: {band1.max()}")
        print(f"    Mean: {band1.mean():.2f}, Std: {band1.std():.2f}")
        
        # Read all bands
        print(f"\n  Reading all {src.count} bands...")
        all_bands = []
        for i in range(1, min(6, src.count + 1)):
            band = src.read(i)
            all_bands.append(band)
            print(f"    Band {i}: shape {band.shape}, range [{band.min()}, {band.max()}]")
        
        if len(all_bands) >= 5:
            print("\n✓ Successfully read 5+ bands (suitable for MicaSense RedEdge-MX)")
        elif len(all_bands) == 1:
            print("\n⚠ Only 1 band found - may be grayscale or single-band TIFF")
        else:
            print(f"\n⚠ Found {len(all_bands)} bands (expected 5 for RedEdge-MX)")
        
        # Stack bands
        multispectral = np.stack(all_bands[:5], axis=0)
        if len(all_bands) < 5:
            # Pad with last band if needed
            while multispectral.shape[0] < 5:
                multispectral = np.concatenate([multispectral, multispectral[-1:]], axis=0)
        
        print(f"\n  Multispectral stack shape: {multispectral.shape}")
        print(f"  Data type: {multispectral.dtype}")
        
        # Test normalization
        normalized = multispectral.astype(np.float32) / 65535.0
        print(f"  After normalization: range [{normalized.min():.4f}, {normalized.max():.4f}]")
        
        print("\n✓ Data loading test PASSED!")
        print("\nYou can now run the full training script:")
        print("  python3 multispectral_vit.py --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop")
        
except Exception as e:
    print(f"\n✗ ERROR loading TIFF file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

