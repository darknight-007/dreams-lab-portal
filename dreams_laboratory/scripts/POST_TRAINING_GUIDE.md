# Post-Training Workflow Guide

After running `multispectral_vit.py`, you'll have:
- `multispectral_vit.pth` - Trained model
- `multispectral_latents.npy` - Latent representations (if --extract_latents used)
- `multispectral_tile_paths.txt` - Paths to corresponding tiles

## Step-by-Step Next Steps

### 1. **Visualize the Latent Space**
See how tiles are distributed in the learned feature space:

```bash
python3 analyze_latents.py --visualize
```

This creates:
- `latent_space_visualization.png` - PCA and t-SNE plots showing tile relationships

**What to look for:**
- Clusters indicate similar geological features
- Outliers may be anomalies or unique formations
- Spatial structure suggests the model learned meaningful patterns

---

### 2. **Cluster Similar Tiles**
Group tiles by similarity to discover geological zones:

```bash
python3 analyze_latents.py --cluster --n_clusters 10
```

This creates:
- `tile_clusters.json` - Cluster assignments for each tile
- `clustered_latent_space.png` - Visualization colored by cluster

**What to look for:**
- Clusters might correspond to:
  - Different rock types
  - Weathering patterns
  - Vegetation zones
  - Lighting conditions
  - Geological units

**Try different cluster counts:**
```bash
python3 analyze_latents.py --cluster --n_clusters 5   # Broad categories
python3 analyze_latents.py --cluster --n_clusters 20  # Fine-grained groups
```

---

### 3. **Find Similar Tiles**
Use a specific tile as a query to find similar ones:

```bash
# Find tiles similar to tile at index 0
python3 analyze_latents.py --similarity 0 --n_similar 20

# Or find by filename pattern
python3 analyze_latents.py --similarity $(grep -n "IMG_0188" multispectral_tile_paths.txt | cut -d: -f1) --n_similar 10
```

**Use cases:**
- Find all tiles with similar rock formations
- Locate similar spectral signatures
- Quality control (find anomalies)

---

### 4. **Run All Analyses**
Run everything at once:

```bash
python3 analyze_latents.py --all
```

---

## Advanced Analyses

### Create a Geological Feature Map

Use the clusters to create a map showing different geological zones:

```python
# Load clusters
import json
with open('tile_clusters.json') as f:
    clusters = json.load(f)

# Extract GPS coordinates from tile paths/metadata
# Map each cluster to a color
# Create georeferenced visualization
```

### Downstream Tasks

#### A. **Geological Classification**
Fine-tune the model for specific rock types:

```python
# Use latent representations as features
# Train a classifier on labeled samples
# Predict rock types for all tiles
```

#### B. **Segmentation**
Add a segmentation head to identify features at pixel level:

```python
# Modify multispectral_vit.py
# Add decoder head for dense prediction
# Train on labeled segmentation masks
```

#### C. **Anomaly Detection**
Find unusual tiles:

```python
# Use reconstruction error
# Or distance from cluster centers
# Flag outliers for manual inspection
```

### Integration with GIS

1. **Extract GPS coordinates** from `paramlog.dat` files
2. **Map latent representations** to geographic locations
3. **Create georeferenced maps** showing:
   - Clusters as different zones
   - Similarity heatmaps
   - Feature distributions

## Quick Reference Commands

```bash
# 1. Train model
python3 multispectral_vit.py \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --extract_latents

# 2. Visualize results
python3 analyze_latents.py --visualize

# 3. Cluster tiles
python3 analyze_latents.py --cluster --n_clusters 10

# 4. Find similar tiles
python3 analyze_latents.py --similarity 0 --n_similar 20

# 5. Run everything
python3 analyze_latents.py --all
```

## What Each Output Means

### `multispectral_vit.pth`
- Trained model weights
- Can be loaded for inference on new tiles
- Can be fine-tuned for specific tasks

### `multispectral_latents.npy`
- Each row = one tile's latent representation
- 512-dimensional vector (or your embed_dim)
- Encodes learned features about the tile

### `tile_clusters.json`
- Maps cluster IDs to tile paths
- Use for identifying geological zones
- Can create maps colored by cluster

### Visualization files
- Help understand data structure
- Identify patterns and anomalies
- Guide further analysis

## Next Steps After Analysis

1. **Validate clusters**: Check if clusters correspond to known geological features
2. **Label samples**: Manually label tiles from each cluster
3. **Fine-tune model**: Train classifier on labeled data
4. **Create maps**: Generate georeferenced visualizations
5. **Extract features**: Use latents for downstream ML tasks
6. **Publish findings**: Document geological patterns discovered

## Questions to Answer

After running analyses, you should be able to answer:
- Are there distinct geological zones?
- Which tiles are most similar/dissimilar?
- Are there anomalies or unusual formations?
- How do features vary across the scarp?
- What spectral patterns are most informative?

