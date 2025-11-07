# Segmentation Stack Comparison

## 🎯 Quick Reference

| Approach | Training Required? | Input Type | Best For |
|----------|-------------------|------------|----------|
| **Zero-Shot COCO** | ❌ NO | RGB (3ch) | Common objects (80 COCO classes) |
| **Fine-Tuned Mask2Former** | ✅ YES | RGB (3ch) | Custom object categories |
| **Custom ViT Segmentation** | ✅ YES | RGB/Multispectral (3-5+ ch) | Geological/remote sensing |

---

## 📊 Detailed Comparison

### 1. Zero-Shot Detection (NEW!)

**Script:** `zero_shot_detection.py`

**No Training Required!** ✨

```bash
python zero_shot_detection.py image.jpg --visualize
```

| Pros | Cons |
|------|------|
| ✅ No training needed | ❌ Limited to 80 COCO classes |
| ✅ Instant results | ❌ RGB images only |
| ✅ High accuracy on common objects | ❌ Not customizable |
| ✅ Pre-trained on 118K images | ❌ May not work on specialized domains |

**Use Cases:**
- Detecting people, cars, animals in photos
- Quick prototyping
- Bootstrapping label datasets
- General object detection

**Categories:**
```
person, car, dog, cat, chair, laptop, phone, bicycle, truck, bird,
horse, sheep, cow, bottle, cup, knife, spoon, bowl, banana, apple,
sandwich, orange, pizza, cake, couch, tv, book, clock, vase, etc.
```

---

### 2. Fine-Tuned Mask2Former

**Script:** `train_mask2former_deepgis.py`

**Training Required:** Yes (custom categories)

```bash
# Convert labels to COCO format
python train_mask2former_deepgis.py --mode convert --image_dir /path/to/images

# Train on custom categories
python train_mask2former_deepgis.py --mode train --image_dir /path/to/images --num_epochs 50
```

| Pros | Cons |
|------|------|
| ✅ Custom object categories | ❌ Requires labeled training data |
| ✅ State-of-the-art accuracy | ❌ Training time (hours) |
| ✅ Transfer learning from COCO | ❌ RGB only (3 channels) |
| ✅ Panoptic segmentation | ❌ Requires detectron2 |

**Use Cases:**
- Custom object detection (rocks, minerals, equipment)
- High-accuracy instance segmentation
- When COCO classes aren't enough

**Example Custom Categories:**
```python
categories = ['granite', 'basalt', 'sandstone', 'limestone', 'shale']
```

**Architecture:**
```
Pre-trained COCO → Fine-tune on custom data → Custom predictions
```

---

### 3. Multispectral ViT Segmentation

**Scripts:** `multispectral_vit.py` + `multispectral_decoder.py` + `segmentation_assisted_labeling.py`

**Training Required:** Yes (from scratch, no pre-training)

```bash
# Train autoencoder first
python train_autoencoder.py --img_size 960 --in_channels 5

# Then train segmentation decoder
# (requires custom training script)
```

| Pros | Cons |
|------|------|
| ✅ Multispectral support (5+ bands) | ❌ No pre-trained weights |
| ✅ Cross-band attention (unique!) | ❌ Train from scratch (slow) |
| ✅ Designed for remote sensing | ❌ Requires large dataset |
| ✅ Handles NIR, RedEdge bands | ❌ Complex architecture |

**Use Cases:**
- Multispectral/hyperspectral imagery
- Geological feature mapping
- Vegetation analysis (NDVI, NDRE)
- Drone/satellite imagery
- When spectral information is critical

**Input Bands:**
```
Band 1: Blue (475nm)
Band 2: Green (560nm)
Band 3: Red (668nm)
Band 4: Red Edge (717nm)
Band 5: Near-Infrared (840nm)
```

**Unique Feature: Cross-Band Attention**
```python
# Learns relationships between spectral bands
# E.g., NIR/Red ratio for vegetation
x = x + self.cross_band_attn(x)
```

---

### 4. Baseline Mask R-CNN

**Script:** `deepgis-xr/deepgis_xr/apps/ml/services/predictor.py`

**Training Required:** Optional (can use pre-trained)

| Pros | Cons |
|------|------|
| ✅ Simple, well-documented | ❌ Less accurate than Mask2Former |
| ✅ Fast inference | ❌ Not state-of-the-art |
| ✅ TorchVision (easy install) | ❌ Instance segmentation only |
| ✅ Pre-trained COCO weights | ❌ No panoptic segmentation |

**Use Cases:**
- Quick baseline model
- Real-time applications
- When speed > accuracy

---

## 🔀 Decision Tree

```
Do you need to detect objects in an image?
│
├─ Are they common objects? (people, cars, animals, furniture)
│  │
│  ├─ YES → ✅ Use Zero-Shot Detection (no training!)
│  │         Script: zero_shot_detection.py
│  │
│  └─ NO → Continue below...
│
├─ Is the image RGB (3 channels)?
│  │
│  ├─ YES → Do you need custom categories?
│  │        │
│  │        ├─ YES → ✅ Fine-Tune Mask2Former
│  │        │         Script: train_mask2former_deepgis.py
│  │        │
│  │        └─ NO → ✅ Use Zero-Shot Detection
│  │
│  └─ NO (Multispectral/5+ bands) → ✅ Custom ViT Segmentation
│                                     Scripts: multispectral_vit.py
│
└─ Do you have labeled training data?
   │
   ├─ NO → 
   │      ├─ Start with Zero-Shot to bootstrap labels
   │      └─ Then refine and train custom model
   │
   └─ YES →
          ├─ < 100 images → Use Zero-Shot or augment data
          ├─ 100-1000 images → Fine-Tune Mask2Former
          └─ 1000+ images → Custom ViT (if multispectral)
```

---

## 💻 Example Workflows

### Workflow 1: Quick Object Detection (No Training)

```bash
# Detect objects in any image
python zero_shot_detection.py street_scene.jpg --visualize

# Result: Detects people, cars, bicycles, etc.
# Time: ~1-5 seconds per image
# Training time: 0 hours ✨
```

**Output:**
```json
{
  "detections": [
    {"class_name": "person", "confidence": 0.98},
    {"class_name": "car", "confidence": 0.95},
    {"class_name": "bicycle", "confidence": 0.87}
  ]
}
```

---

### Workflow 2: Custom Object Categories (RGB)

```bash
# 1. Label your data in DeepGIS
# Categories: ['rock_type_A', 'rock_type_B', 'rock_type_C']

# 2. Convert to COCO format
python train_mask2former_deepgis.py --mode convert --image_dir images/

# 3. Train Mask2Former
python train_mask2former_deepgis.py --mode train \
    --image_dir images/ \
    --num_epochs 50 \
    --batch_size 4

# 4. Predict on new images
python train_mask2former_deepgis.py --mode predict \
    --model_path checkpoints/model_final.pth \
    --image_path test_image.jpg

# Training time: 2-6 hours (depends on dataset size)
```

---

### Workflow 3: Multispectral Segmentation

```bash
# 1. Train encoder (unsupervised)
python train_autoencoder.py \
    --img_size 960 \
    --in_channels 5 \
    --num_epochs 100

# 2. Train segmentation decoder (supervised)
# (Requires custom script with labeled multispectral data)

# 3. Run inference
python segmentation_assisted_labeling.py \
    --model_path multispectral_segmentation_model.pth \
    --config_path multispectral_vit.pth

# Training time: 10-20 hours (from scratch)
```

---

## 📈 Performance Comparison

### Accuracy (on respective domains)

| Model | Common Objects | Custom Objects | Multispectral |
|-------|----------------|----------------|---------------|
| Zero-Shot COCO | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| Fine-Tuned Mask2Former | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Custom ViT | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Speed (inference)

| Model | GPU (FPS) | CPU (FPS) | Memory |
|-------|-----------|-----------|--------|
| Zero-Shot Mask R-CNN | 10-15 | 2-3 | 2-4 GB |
| Mask2Former | 5-8 | 1-2 | 4-8 GB |
| Custom ViT | 8-12 | 1-2 | 3-6 GB |

### Training Time

| Model | Dataset Size | Training Time | Labels Required |
|-------|--------------|---------------|-----------------|
| Zero-Shot | N/A | **0 hours** ✨ | 0 |
| Mask2Former | 500 images | 2-4 hours | 500+ |
| Custom ViT | 5000 tiles | 10-20 hours | 5000+ |

---

## 🎓 Summary Table

| Model | Input | Training | Classes | Accuracy | Speed | Best For |
|-------|-------|----------|---------|----------|-------|----------|
| **Zero-Shot** | RGB | ❌ None | 80 COCO | ⭐⭐⭐⭐ | ⚡⚡⚡ | Common objects, prototyping |
| **Mask2Former** | RGB | ✅ Fine-tune | Custom | ⭐⭐⭐⭐⭐ | ⚡⚡ | Custom categories, high accuracy |
| **ViT Segmentation** | Multi | ✅ From scratch | Custom | ⭐⭐⭐⭐ | ⚡⚡ | Multispectral, remote sensing |
| **Mask R-CNN** | RGB | ⚙️ Optional | 80/Custom | ⭐⭐⭐ | ⚡⚡⚡ | Baseline, real-time |

---

## 🚀 Getting Started

### New to Segmentation?
```bash
# Start here - no training required!
python zero_shot_detection.py your_image.jpg --visualize
```

### Have Labeled RGB Data?
```bash
# Fine-tune on your categories
python train_mask2former_deepgis.py --mode train --image_dir images/
```

### Working with Multispectral?
```bash
# Train custom ViT
python train_autoencoder.py --in_channels 5
```

---

## 📚 Documentation

- **Zero-Shot:** [ZERO_SHOT_DETECTION.md](ZERO_SHOT_DETECTION.md)
- **Mask2Former:** `train_mask2former_deepgis.py` (docstring)
- **ViT Segmentation:** [ENCODER_DECODER_GUIDE.md](ENCODER_DECODER_GUIDE.md)
- **Architecture:** [UNIQUE_FEATURES.md](UNIQUE_FEATURES.md)

---

## ❓ FAQ

### Q: Can I detect custom objects without training?
**A:** No. Zero-shot only works for the 80 COCO categories. For custom objects, you need to fine-tune or train from scratch.

### Q: Which model should I use?
**A:** 
- Common objects → Zero-Shot
- Custom RGB categories → Mask2Former
- Multispectral → Custom ViT

### Q: How much training data do I need?
**A:**
- Zero-Shot: 0 images ✨
- Fine-tuning: 100-1000 images (more is better)
- From scratch: 1000+ images

### Q: Can I use multispectral with Mask2Former?
**A:** Not directly. Mask2Former expects RGB (3 channels). For multispectral, use the custom ViT segmentation model.

---

**🎯 Bottom Line:**

- **Detection without training?** → ✅ YES (80 COCO classes only)
- **Custom objects?** → ⚠️ NO (training required)
- **Multispectral?** → ⚠️ NO (custom model required)

---

**Created:** 2025-11-07  
**Last Updated:** 2025-11-07

