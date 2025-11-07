# Segment Anything Model (SAM) Guide

## 🎯 What is SAM?

**Segment Anything Model (SAM)** is Meta's foundation model that can segment **ANY** object in an image without requiring class labels or training.

### Key Features
- ✅ **Zero-shot segmentation** - No training required
- ✅ **No class labels** - Finds all regions automatically
- ✅ **Universal** - Works on any image domain
- ✅ **High quality** - State-of-the-art boundaries
- ⚠️ **No classification** - Only segments, doesn't classify

---

## 🆚 SAM vs. Zero-Shot COCO vs. Custom Training

| Feature | Zero-Shot COCO | SAM | Custom Training |
|---------|----------------|-----|-----------------|
| **Training Required** | ❌ No | ❌ No | ✅ Yes |
| **Class Labels** | ✅ Yes (80 COCO) | ❌ No | ✅ Yes (custom) |
| **Works on Rocks** | ❌ No | ✅ Yes | ✅ Yes |
| **Output** | Labeled masks | Unlabeled masks | Labeled masks |
| **Best For** | Common objects | Exploration, boundaries | Domain-specific |

### Simple Comparison

```
Input: Rock image with fractures and mineral boundaries

┌─────────────────────────────────────────────────────────────┐
│                    Zero-Shot COCO                            │
├─────────────────────────────────────────────────────────────┤
│ Result: 0-2 detections (likely false positives)             │
│ "person" detected in rock texture ❌                         │
│ Cannot detect geological features ❌                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Segment Anything (SAM)                     │
├─────────────────────────────────────────────────────────────┤
│ Result: 50-200 segments                                      │
│ ✅ Finds all regions, fractures, boundaries                  │
│ ✅ High-quality masks                                        │
│ ⚠️  No labels (just "segment_1", "segment_2", etc.)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Custom Trained Mask2Former                     │
├─────────────────────────────────────────────────────────────┤
│ Result: 5-20 detections                                      │
│ ✅ Labeled: "granite", "basalt", "fracture", etc.           │
│ ✅ Domain-specific knowledge                                │
│ ⚠️  Requires labeled training data                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Dependencies
pip install opencv-python matplotlib scikit-image
```

### Basic Usage

```bash
cd $PROJECT_ROOT/dreams_laboratory/scripts

# Segment a rock image
python3 segment_anything_rocks.py /path/to/rock_image.png --geojson

# Use larger model (better quality, slower)
python3 segment_anything_rocks.py rock.png --model vit_l

# Use huge model (best quality)
python3 segment_anything_rocks.py rock.png --model vit_h
```

### Test on Rock Dataset

```bash
# Run comprehensive test (COCO + SAM)
bash test_all_segmentation_on_rocks.sh

# Results saved to: comprehensive_rock_segmentation_test/
```

---

## 📊 Model Sizes

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `vit_b` | 375 MB | ⚡⚡⚡ Fast | Good | Quick tests |
| `vit_l` | 1.2 GB | ⚡⚡ Medium | Better | Production |
| `vit_h` | 2.4 GB | ⚡ Slow | Best | High accuracy |

**Recommendation**: Start with `vit_b` for testing, use `vit_l` or `vit_h` for final analysis.

---

## 🎨 Example Outputs

### Rock Tile Segmentation

**Input**: 256x256 rock surface tile (zoom 23)

**SAM Output**: 
- 150-200 segments
- Finds:
  - Mineral boundaries
  - Fractures and cracks
  - Texture regions
  - Color variations
  - Surface features

**File Outputs**:
```
rock_tile_sam_segments.jpg    # Visualization (colored masks)
rock_tile_analysis.json        # Segment statistics
rock_tile_segments.geojson     # GeoJSON for GIS integration
```

---

## 🔧 Advanced Usage

### 1. Adjust Segmentation Parameters

Edit `segment_anything_rocks.py`:

```python
self.mask_generator = SamAutomaticMaskGenerator(
    model=self.sam,
    points_per_side=32,          # More points = more segments (default: 32)
    pred_iou_thresh=0.86,        # IoU threshold (default: 0.86)
    stability_score_thresh=0.92, # Stability threshold (default: 0.92)
    min_mask_region_area=100,    # Minimum area in pixels (default: 100)
)
```

**Adjust for rocks**:
- Small fractures: `min_mask_region_area=50`, `points_per_side=48`
- Large regions: `min_mask_region_area=500`, `points_per_side=24`

### 2. Filter Segments by Size

```python
# Only keep large segments (> 1000 pixels)
large_segments = [m for m in masks if m['area'] > 1000]

# Only keep small segments (< 500 pixels) - for fractures
small_segments = [m for m in masks if m['area'] < 500]
```

### 3. Combine SAM with CLIP for Classification

```python
# 1. Segment with SAM (finds regions)
masks = sam.segment_image(image_path)

# 2. Extract each segment
for mask in masks:
    segment_img = extract_segment(image, mask['segmentation'])
    
    # 3. Classify with CLIP
    label = clip_classify(segment_img, 
                         classes=['granite', 'basalt', 'sandstone'])
    
    # 4. Assign label to segment
    mask['class'] = label
```

---

## 💡 Use Cases for Geological Analysis

### 1. Exploratory Analysis
**Question**: "What features exist in this rock?"

```bash
python3 segment_anything_rocks.py mystery_rock.jpg --geojson
```

**Output**: All regions segmented, ready for visual inspection

### 2. Fracture Detection
**Question**: "Where are all the cracks/fractures?"

```python
# Filter small, elongated segments
fractures = [m for m in masks 
             if m['area'] < 500 and is_elongated(m['bbox'])]
```

### 3. Mineral Boundary Mapping
**Question**: "Where do different minerals meet?"

```bash
python3 segment_anything_rocks.py rock_sample.jpg --model vit_h --geojson
# Import GeoJSON into QGIS for analysis
```

### 4. Pre-Labeling for Training Data
**Question**: "How to label 1000 rock images faster?"

**Workflow**:
1. Run SAM on all images → Auto-segment
2. Import segments into DeepGIS label app
3. Manually assign class labels (much faster than drawing!)
4. Train custom model on labeled data

### 5. Texture Analysis
**Question**: "How many distinct texture regions?"

```python
# Count segments in different size ranges
fine_texture = len([m for m in masks if m['area'] < 100])
medium_texture = len([m for m in masks if 100 <= m['area'] < 500])
coarse_texture = len([m for m in masks if m['area'] >= 500])
```

---

## 🔄 Recommended Workflow for Rock Detection

### Full Pipeline: SAM → Manual Labeling → Custom Training

```
┌──────────────────────────────────────────────────────────┐
│ Step 1: Auto-Segment with SAM                            │
├──────────────────────────────────────────────────────────┤
│ python3 segment_anything_rocks.py rock*.png --geojson    │
│                                                           │
│ Output: ~100-200 segments per image (no labels)          │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Manual Classification in DeepGIS                 │
├──────────────────────────────────────────────────────────┤
│ • Import GeoJSON files                                   │
│ • Assign class labels:                                   │
│   - "granite"                                            │
│   - "basalt"                                             │
│   - "fracture"                                           │
│   - "mineral_vein"                                       │
│   - etc.                                                 │
│                                                           │
│ Time: ~5-10 min per image (much faster than manual!)    │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: Train Custom Mask2Former                         │
├──────────────────────────────────────────────────────────┤
│ python3 train_mask2former_deepgis.py --mode train \      │
│   --image_dir labeled_images/ \                          │
│   --num_epochs 50                                        │
│                                                           │
│ Output: Custom model for automatic rock detection       │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: Automatic Detection on New Images                │
├──────────────────────────────────────────────────────────┤
│ python3 train_mask2former_deepgis.py --mode predict \    │
│   --model_path model_final.pth \                         │
│   --image_path new_rock.jpg                              │
│                                                           │
│ Output: Labeled detections (granite, basalt, etc.)      │
└──────────────────────────────────────────────────────────┘
```

**Time Investment**:
- SAM segmentation: ~5 sec per image
- Manual labeling: ~10 min per image × 100 images = ~17 hours
- Training: 2-6 hours
- **Total**: ~1 day of work for 100 images

**Benefit**: After training, automatic detection on unlimited images!

---

## ⚖️ When to Use Each Approach

### Use SAM When:
✅ Exploring unknown geological samples  
✅ Need to find all boundaries/regions  
✅ Pre-processing for manual labeling  
✅ Analyzing textures and patterns  
✅ No training data available yet  

### Use Zero-Shot COCO When:
✅ Detecting common objects (people, vehicles) in field photos  
✅ Quick sanity check  
✅ Want to see what pre-trained models can do  
❌ NOT for geological features  

### Use Custom Training When:
✅ Have labeled training data  
✅ Need specific class labels (granite, basalt, etc.)  
✅ Domain-specific detection required  
✅ Production deployment  

---

## 🐛 Troubleshooting

### SAM not installed
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Model download fails
```bash
# Manually download
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# Place in: /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts/
```

### Out of memory
```bash
# Use smaller model
python3 segment_anything_rocks.py rock.png --model vit_b --device cpu

# Or reduce image size
python3 segment_anything_rocks.py rock.png --model vit_b
```

### Too many/few segments
Adjust parameters in `segment_anything_rocks.py`:
```python
points_per_side=32,          # More = more segments
min_mask_region_area=100,    # Higher = fewer segments
```

---

## 📚 Additional Resources

- **SAM Paper**: https://arxiv.org/abs/2304.02643
- **SAM GitHub**: https://github.com/facebookresearch/segment-anything
- **Demo**: https://segment-anything.com/demo

---

## 🎯 Summary

| Question | Answer |
|----------|--------|
| **Can SAM detect rocks?** | ✅ Yes - segments regions (no labels) |
| **Does SAM classify?** | ❌ No - only segments |
| **Training required?** | ❌ No |
| **Works on multispectral?** | ⚠️ RGB only (3 channels) |
| **Best use case** | Exploration + pre-labeling |

**Bottom Line**: SAM is perfect for **preliminary analysis** and **speeding up manual labeling**. Combine with custom training for full automation.

---

**Created**: 2025-11-07  
**Last Updated**: 2025-11-07

