#!/bin/bash
# Demo: Zero-Shot Object Detection (No Training Required!)

echo "======================================================================"
echo "  ZERO-SHOT OBJECT DETECTION DEMO"
echo "======================================================================"
echo ""
echo "This demo shows that Mask R-CNN and Mask2Former CAN detect objects"
echo "WITHOUT any training, using pre-trained COCO weights."
echo ""
echo "🎯 Can detect 80 categories:"
echo "   - People & Animals: person, dog, cat, bird, horse..."
echo "   - Vehicles: car, truck, bus, bicycle, motorcycle..."
echo "   - Indoor: chair, couch, tv, laptop, keyboard..."
echo "   - Food: pizza, banana, apple, sandwich..."
echo "   - And 60+ more!"
echo ""
echo "======================================================================"
echo ""

# Check if user provided an image
if [ -z "$1" ]; then
    echo "❌ No image provided!"
    echo ""
    echo "Usage:"
    echo "  $0 /path/to/your/image.jpg"
    echo ""
    echo "Example:"
    echo "  $0 ~/Pictures/street_scene.jpg"
    echo "  $0 ~/Pictures/living_room.jpg"
    echo "  $0 ~/Pictures/parking_lot.jpg"
    echo ""
    echo "Tip: Works best with images containing common objects like:"
    echo "  - People"
    echo "  - Cars/vehicles"
    echo "  - Furniture"
    echo "  - Animals"
    echo "  - Food items"
    echo ""
    exit 1
fi

IMAGE_PATH="$1"

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Image not found: $IMAGE_PATH"
    exit 1
fi

echo "📸 Image: $(basename "$IMAGE_PATH")"
echo ""
echo "🔍 Running zero-shot detection..."
echo "   Model: Mask R-CNN (pre-trained on COCO)"
echo "   Confidence threshold: 0.5"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run zero-shot detection
python "$SCRIPT_DIR/zero_shot_detection.py" "$IMAGE_PATH" \
    --model maskrcnn \
    --confidence 0.5 \
    --visualize \
    --geojson \
    --output_dir "$SCRIPT_DIR/zero_shot_demo_results"

EXIT_CODE=$?

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! Zero-shot detection completed."
    echo ""
    echo "📁 Results saved to: $SCRIPT_DIR/zero_shot_demo_results/"
    echo ""
    echo "Files created:"
    echo "  - *_detections.json     → Detection results (classes, scores, boxes)"
    echo "  - *_visualization.jpg   → Image with bounding boxes"
    echo "  - *_geojson.json        → GeoJSON format (for GIS integration)"
    echo ""
    echo "🎨 View the visualization:"
    echo "  xdg-open \$SCRIPT_DIR/zero_shot_demo_results/*_visualization.jpg"
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo ""
    echo "💡 KEY TAKEAWAY:"
    echo ""
    echo "   ✅ YES - Models CAN detect objects WITHOUT training!"
    echo "   ✅ Works on 80 COCO categories (common objects)"
    echo "   ❌ For custom objects (rocks, minerals, etc.) → Need training"
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
else
    echo "❌ Detection failed. Check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  - PyTorch not installed: pip install torch torchvision"
    echo "  - Image format not supported: Try converting to .jpg"
    echo "  - Out of memory: Try --device cpu"
fi

echo ""

