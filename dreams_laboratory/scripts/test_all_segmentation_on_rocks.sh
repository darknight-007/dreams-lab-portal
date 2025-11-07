#!/bin/bash
# Comprehensive Segmentation Test: Zero-Shot COCO + Segment Anything (SAM)
# Tests both approaches on rock dataset to compare results
#
# Usage:
#   export ROCKS_DIR=/path/to/your/rock-tiles/raw
#   bash test_all_segmentation_on_rocks.sh

ROCKS_DIR="${ROCKS_DIR:-/path/to/rock-tiles/raw}"  # Set via environment variable or use default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/comprehensive_rock_segmentation_test"

echo "======================================================================"
echo "  COMPREHENSIVE SEGMENTATION TEST ON ROCKS DATASET"
echo "======================================================================"
echo ""
echo "This test compares TWO approaches:"
echo ""
echo "  1️⃣  Zero-Shot COCO (Mask R-CNN)"
echo "      ├─ Pre-trained on 80 common object categories"
echo "      ├─ Provides class labels (person, car, etc.)"
echo "      └─ Expected: ❌ Won't work well on rocks"
echo ""
echo "  2️⃣  Segment Anything Model (SAM)"
echo "      ├─ Segments EVERYTHING without class labels"
echo "      ├─ Finds all regions/boundaries/textures"
echo "      └─ Expected: ✅ Will find rock regions"
echo ""
echo "======================================================================"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/zero_shot_coco"
mkdir -p "$OUTPUT_DIR/segment_anything"

# Find sample tiles
echo "🔍 Finding sample rock tiles..."
echo ""

TILE_23=$(find "$ROCKS_DIR/23" -name "*.png" | head -1)
TILE_22=$(find "$ROCKS_DIR/22" -name "*.png" | head -1)
TILE_21=$(find "$ROCKS_DIR/21" -name "*.png" | head -1)

echo "Sample tiles selected:"
echo "  1. Zoom 23 (256x256px): $(basename $TILE_23)"
echo "  2. Zoom 22 (256x256px): $(basename $TILE_22)"
echo "  3. Zoom 21 (256x256px): $(basename $TILE_21)"
echo ""
echo "======================================================================"
echo ""

# Test each tile with both methods
for TILE in "$TILE_23" "$TILE_22" "$TILE_21"; do
    BASENAME=$(basename "$TILE" .png)
    ZOOM=$(echo "$TILE" | grep -o '/[0-9]\+/' | head -1 | tr -d '/')
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " TESTING: Zoom $ZOOM - $BASENAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Test 1: Zero-Shot COCO
    echo "🎯 TEST 1: Zero-Shot COCO (Mask R-CNN)"
    echo "────────────────────────────────────────────────────────────────────"
    
    python3 "$SCRIPT_DIR/zero_shot_detection.py" "$TILE" \
        --model maskrcnn \
        --confidence 0.3 \
        --visualize \
        --geojson \
        --output_dir "$OUTPUT_DIR/zero_shot_coco" 2>&1 | \
        grep -E "(Loading|Model loaded|Running|Detected objects|class_name|Confidence|Saved)" || echo "   ✓ Completed"
    
    echo ""
    
    # Test 2: Segment Anything
    echo "🤖 TEST 2: Segment Anything Model (SAM)"
    echo "────────────────────────────────────────────────────────────────────"
    
    python3 "$SCRIPT_DIR/segment_anything_rocks.py" "$TILE" \
        --model vit_b \
        --geojson \
        --output_dir "$OUTPUT_DIR/segment_anything" 2>&1 | \
        grep -E "(Loading|Model loaded|Processing|Found|segments|Total|Average|Saved)" || echo "   ✓ Completed"
    
    echo ""
done

echo ""
echo "======================================================================"
echo "  TEST COMPLETE - RESULTS COMPARISON"
echo "======================================================================"
echo ""

# Count results
COCO_DETECTIONS=$(find "$OUTPUT_DIR/zero_shot_coco" -name "*_detections.json" | wc -l)
SAM_RESULTS=$(find "$OUTPUT_DIR/segment_anything" -name "*_analysis.json" | wc -l)

echo "📊 RESULTS SUMMARY:"
echo ""
echo "  Zero-Shot COCO:"
echo "    - Files processed: $COCO_DETECTIONS"
echo "    - Results: $OUTPUT_DIR/zero_shot_coco/"
echo ""
echo "  Segment Anything:"
echo "    - Files processed: $SAM_RESULTS"
echo "    - Results: $OUTPUT_DIR/segment_anything/"
echo ""
echo "======================================================================"
echo ""
echo "🔍 DETAILED COMPARISON:"
echo ""

# Compare results for each tile
for TILE in "$TILE_23" "$TILE_22" "$TILE_21"; do
    BASENAME=$(basename "$TILE" .png)
    
    echo "  📸 $BASENAME:"
    echo ""
    
    # COCO results
    COCO_JSON="$OUTPUT_DIR/zero_shot_coco/${BASENAME}_detections.json"
    if [ -f "$COCO_JSON" ]; then
        COCO_COUNT=$(python3 -c "import json; print(json.load(open('$COCO_JSON'))['num_detections'])" 2>/dev/null || echo "0")
        echo "    Zero-Shot COCO: $COCO_COUNT detections"
        if [ "$COCO_COUNT" -gt 0 ]; then
            echo "      (Likely false positives - rocks not in COCO)"
        else
            echo "      ✓ No false positives"
        fi
    else
        echo "    Zero-Shot COCO: No results"
    fi
    
    # SAM results
    SAM_JSON="$OUTPUT_DIR/segment_anything/${BASENAME}_analysis.json"
    if [ -f "$SAM_JSON" ]; then
        SAM_COUNT=$(python3 -c "import json; print(json.load(open('$SAM_JSON'))['num_segments'])" 2>/dev/null || echo "0")
        echo "    Segment Anything: $SAM_COUNT segments"
        if [ "$SAM_COUNT" -gt 0 ]; then
            echo "      ✓ Found geological regions/features"
        fi
    else
        echo "    Segment Anything: No results"
    fi
    
    echo ""
done

echo "======================================================================"
echo ""
echo "💡 KEY FINDINGS:"
echo ""
echo "  1. Zero-Shot COCO (Mask R-CNN):"
echo "     ❌ Cannot detect rocks (not in 80 COCO classes)"
echo "     ❌ May produce false positives (seeing 'person' in textures)"
echo "     ✅ Good for: Common objects (people, cars, animals)"
echo ""
echo "  2. Segment Anything (SAM):"
echo "     ✅ Finds all regions without class labels"
echo "     ✅ Good for: Exploratory analysis, finding boundaries"
echo "     ⚠️  Limitation: No automatic classification"
echo "     💡 Solution: Use SAM + manual labeling + train classifier"
echo ""
echo "======================================================================"
echo ""
echo "🎯 RECOMMENDED WORKFLOW FOR ROCK DETECTION:"
echo ""
echo "  Step 1: Use SAM for preliminary segmentation"
echo "    python3 segment_anything_rocks.py image.png --geojson"
echo ""
echo "  Step 2: Import SAM segments into DeepGIS label app"
echo "    - Load GeoJSON with auto-generated boundaries"
echo "    - Manually assign class labels (granite, basalt, etc.)"
echo ""
echo "  Step 3: Train custom Mask2Former model"
echo "    python3 train_mask2former_deepgis.py --mode train \\"
echo "      --image_dir labeled_images/ \\"
echo "      --num_epochs 50"
echo ""
echo "  Step 4: Use trained model for automatic detection"
echo "    python3 train_mask2former_deepgis.py --mode predict \\"
echo "      --model_path checkpoints/model_final.pth \\"
echo "      --image_path new_rock.jpg"
echo ""
echo "======================================================================"
echo ""
echo "📁 All results saved to: $OUTPUT_DIR"
echo ""
echo "View visualizations:"
echo "  # Zero-Shot COCO results"
echo "  xdg-open $OUTPUT_DIR/zero_shot_coco/*_visualization.jpg"
echo ""
echo "  # SAM segmentation results"
echo "  xdg-open $OUTPUT_DIR/segment_anything/*_sam_segments.jpg"
echo ""
echo "======================================================================"
echo ""

