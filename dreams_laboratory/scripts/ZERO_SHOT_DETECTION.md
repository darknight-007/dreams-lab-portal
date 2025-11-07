# Zero-Shot Object Detection Guide

## 🎯 Quick Answer: YES! You can detect objects WITHOUT training!

Both **Mask R-CNN** and **Mask2Former** come pre-trained on the **COCO dataset** and can detect **80 common object categories** out of the box.

---

## 📦 What Can Be Detected (COCO Categories)

### People & Animals
`person`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`

### Vehicles
`bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`

### Outdoor Objects
`traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`

### Indoor Objects
`chair`, `couch`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `keyboard`, `mouse`, `cell phone`, `book`, `clock`

### Kitchen Items
`bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `pizza`, `donut`, `cake`, `microwave`, `oven`, `refrigerator`

### Accessories
`backpack`, `umbrella`, `handbag`, `tie`, `suitcase`

### Sports
`frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`

### Other
`potted plant`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

**Total: 80 categories** (no training required!)

---

## 🚀 Quick Start

### 1. Detect Objects in Any Image

```bash
cd /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts

# Basic detection
python zero_shot_detection.py /path/to/image.jpg

# With visualization
python zero_shot_detection.py /path/to/image.jpg --visualize

# Export to GeoJSON
python zero_shot_detection.py /path/to/image.jpg --geojson

# High confidence only
python zero_shot_detection.py /path/to/image.jpg --confidence 0.8 --visualize
```

### 2. Use Mask2Former (More Accurate)

```bash
python zero_shot_detection.py /path/to/image.jpg --model mask2former --visualize
```

### 3. CPU Mode (No GPU Required)

```bash
python zero_shot_detection.py /path/to/image.jpg --device cpu --visualize
```

---

## 📊 Model Comparison

| Feature | Mask R-CNN | Mask2Former |
|---------|------------|-------------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Slower |
| **Accuracy** | Good | ✨ Better |
| **Memory** | 💾 2-4 GB | 💾 4-8 GB |
| **Best For** | Real-time, demos | High accuracy |
| **Dependencies** | TorchVision only | Detectron2 required |

---

## 🎨 Example Output

```json
{
  "image_path": "street_scene.jpg",
  "num_detections": 5,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.98,
      "bbox": [120, 50, 280, 400]
    },
    {
      "class_name": "car",
      "confidence": 0.95,
      "bbox": [300, 200, 650, 450]
    },
    {
      "class_name": "bicycle",
      "confidence": 0.87,
      "bbox": [50, 150, 180, 350]
    }
  ]
}
```

---

## 🔧 Integration with DeepGIS

### Use Zero-Shot Detection to Bootstrap Labeling

Instead of starting from scratch, use pre-trained models to generate initial labels:

```bash
# 1. Detect objects in all images
for img in /path/to/images/*.jpg; do
    python zero_shot_detection.py "$img" --geojson --output_dir zero_shot_labels
done

# 2. Import GeoJSON into DeepGIS label app
# 3. Refine and correct labels manually
# 4. Train custom model on refined labels
```

### Example: Label Cars in Satellite Images

```bash
# Detect all vehicles
python zero_shot_detection.py satellite_image.jpg \
    --model mask2former \
    --confidence 0.7 \
    --geojson \
    --visualize
```

The script will find: `car`, `truck`, `bus`, `motorcycle`, etc.

---

## 💡 When to Use Zero-Shot vs Custom Training

### ✅ Use Zero-Shot When:
- Detecting **common objects** (people, cars, animals)
- Need **quick results** without training time
- **Prototyping** or exploring data
- Limited training data available

### ⚠️ Train Custom Model When:
- Need **domain-specific classes** (geological features, specialized equipment)
- Working with **non-standard imagery** (multispectral, medical, microscopy)
- Need **higher accuracy** on specific objects
- Have sufficient labeled training data

---

## 🔬 Technical Details

### How It Works

```python
┌─────────────────────────────────────────────────┐
│  Pre-trained Model (Trained on COCO Dataset)    │
│  - 118,000 training images                      │
│  - 80 object categories                         │
│  - Millions of annotations                      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Your Image → Model → Predictions                │
│  (No training required!)                         │
└─────────────────────────────────────────────────┘
```

### Model Architecture

**Mask R-CNN:**
- Backbone: ResNet-50 + FPN
- Two-stage detector (RPN + RoI heads)
- Outputs: Bounding boxes + segmentation masks
- Speed: ~5-10 FPS on GPU

**Mask2Former:**
- Transformer-based architecture
- Unified panoptic segmentation
- More accurate but slower
- Better for complex scenes

---

## 🛠️ Advanced Usage

### 1. Batch Processing

```bash
#!/bin/bash
# Process all images in directory
for img in /path/to/images/*.jpg; do
    echo "Processing $img..."
    python zero_shot_detection.py "$img" \
        --visualize \
        --geojson \
        --output_dir batch_results
done
```

### 2. Filter Specific Classes

Modify the script to only keep certain detections:

```python
# Only keep vehicles
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
filtered_detections = [d for d in detections if d['class_name'] in vehicle_classes]
```

### 3. Post-Processing

```python
# Merge overlapping detections
# Apply NMS (Non-Maximum Suppression)
# Filter by size, aspect ratio, etc.
```

---

## 🆚 Comparison: Zero-Shot vs Fine-Tuned

### Example: Detecting Rocks in Geological Images

| Approach | Result |
|----------|--------|
| **Zero-Shot COCO** | May detect as "stone", "rock" (generic) or miss completely |
| **Fine-Tuned Custom** | Can distinguish: "granite", "basalt", "sandstone", "limestone" |

### When Zero-Shot Works Well:
- **Urban scenes**: Cars, people, buildings
- **Indoor scenes**: Furniture, electronics
- **Common animals**: Dogs, cats, birds
- **Everyday objects**: Food, utensils, tools

### When Custom Training Needed:
- **Geological features**: Rock types, minerals
- **Medical imaging**: Tumors, organs, cells
- **Industrial**: Defects, parts, components
- **Agricultural**: Crop diseases, pests, plants
- **Multispectral**: Vegetation indices, land cover

---

## 🚨 Limitations

### 1. Limited to COCO Classes
- Can't detect custom objects (rocks, minerals, specialized equipment)
- Only 80 categories

### 2. RGB Images Only
- Pre-trained on 3-channel RGB images
- Won't work on multispectral (5+ bands) without fine-tuning

### 3. Domain Gap
- Trained on everyday photos
- May struggle with:
  - Aerial/satellite imagery
  - Microscopy images
  - Night vision / IR images
  - Scientific visualizations

### 4. No Fine-Grained Categories
- Can detect "bird" but not "robin" vs "sparrow"
- Can detect "car" but not "Honda Civic" vs "Tesla Model 3"

---

## 🎓 Summary

| Question | Answer |
|----------|--------|
| **Can detect without training?** | ✅ YES - 80 COCO categories |
| **What models?** | Mask R-CNN, Mask2Former |
| **Custom objects?** | ❌ NO - Need training |
| **Multispectral?** | ❌ NO - RGB only |
| **Best use case** | Common objects, prototyping, bootstrapping labels |

---

## 📚 Next Steps

1. **Try zero-shot detection** on your images:
   ```bash
   python zero_shot_detection.py your_image.jpg --visualize
   ```

2. **If COCO classes work well** → Use as-is or fine-tune on your data

3. **If need custom classes** → Train custom model:
   - Collect training data
   - Label manually or use zero-shot as starting point
   - Fine-tune on custom categories

4. **If using multispectral** → Use custom ViT segmentation model:
   - Train from scratch (no pre-trained weights)
   - Use `train_autoencoder.py` pipeline

---

## 🔗 Related Scripts

- `train_mask2former_deepgis.py` - Train custom Mask2Former
- `segmentation_assisted_labeling.py` - Custom ViT segmentation
- `zero_shot_detection.py` - This script (no training!)

---

**🚀 Ready to try? Run:**

```bash
cd $PROJECT_ROOT/dreams_laboratory/scripts
python zero_shot_detection.py --help
```

