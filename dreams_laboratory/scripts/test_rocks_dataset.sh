#!/bin/bash
# Test Zero-Shot Detection on Rock Tiles Dataset
#
# Usage:
#   export ROCKS_DIR=/path/to/your/rock-tiles/raw
#   bash test_rocks_dataset.sh

ROCKS_DIR="${ROCKS_DIR:-/path/to/rock-tiles/raw}"  # Set via environment variable or use default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/rocks_dataset_test_results"

echo "======================================================================"
echo "  TESTING ZERO-SHOT DETECTION ON GEOLOGICAL IMAGERY"
echo "======================================================================"
echo ""
echo "This test will demonstrate:"
echo "  ✅ How pre-trained COCO models work"
echo "  ❌ Why they don't work well on rocks (not in COCO categories)"
echo "  💡 When custom training is necessary"
echo ""
echo "======================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📂 Rock Tiles Dataset: $ROCKS_DIR"
echo "📁 Output Directory: $OUTPUT_DIR"
echo ""

# Find sample tiles from different zoom levels
echo "🔍 Finding sample rock tiles..."
echo ""

# High-res tile (zoom 23)
TILE_23=$(find "$ROCKS_DIR/23" -name "*.png" | head -1)
# Medium-res tile (zoom 21)
TILE_21=$(find "$ROCKS_DIR/21" -name "*.png" | head -1)
# Lower-res tile (zoom 19)
TILE_19=$(find "$ROCKS_DIR/19" -name "*.png" | head -1)

echo "Sample tiles selected:"
echo "  1. Zoom 23 (highest detail): $(basename $TILE_23)"
echo "  2. Zoom 21 (medium detail):  $(basename $TILE_21)"
echo "  3. Zoom 19 (lower detail):   $(basename $TILE_19)"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Test each tile
for TILE in "$TILE_23" "$TILE_21" "$TILE_19"; do
    BASENAME=$(basename "$TILE" .png)
    ZOOM=$(echo "$TILE" | grep -o '/[0-9]\+/' | tr -d '/')
    
    echo "🪨 Testing: Zoom $ZOOM - $BASENAME"
    echo ""
    
    # Run zero-shot detection
    python "$SCRIPT_DIR/zero_shot_detection.py" "$TILE" \
        --model maskrcnn \
        --confidence 0.3 \
        --visualize \
        --geojson \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | grep -E "(Loading|Running|Detected|class_name|Confidence|Saved)"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo ""
done

echo ""
echo "======================================================================"
echo "  TEST COMPLETE"
echo "======================================================================"
echo ""
echo "📊 EXPECTED RESULTS:"
echo ""
echo "  ❌ Few or NO detections"
echo "     Reason: Rocks aren't in the 80 COCO categories"
echo ""
echo "  🎯 COCO categories include:"
echo "     person, car, dog, cat, chair, laptop, pizza, etc."
echo ""
echo "  🪨 What's MISSING:"
echo "     rock, stone, granite, basalt, sandstone, etc."
echo ""
echo "======================================================================"
echo ""
echo "💡 SOLUTION: Train Custom Model"
echo ""
echo "To detect geological features, you need to:"
echo ""
echo "1. Label your rock images with categories:"
echo "   • granite, basalt, sandstone, limestone, etc."
echo "   • Or: fracture, mineral, texture, etc."
echo ""
echo "2. Train custom Mask2Former model:"
echo "   cd $SCRIPT_DIR"
echo "   python train_mask2former_deepgis.py --mode train \\"
echo "     --image_dir /path/to/labeled/images \\"
echo "     --num_epochs 50"
echo ""
echo "3. Or use Multispectral ViT for multispectral data:"
echo "   python train_autoencoder.py --in_channels 5"
echo ""
echo "======================================================================"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "View visualizations:"
echo "  xdg-open $OUTPUT_DIR/*_visualization.jpg"
echo ""

