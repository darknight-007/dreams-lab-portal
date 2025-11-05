# Visual-Language Model Research Guide

## Overview

This guide covers using Vision-Language Models (VLMs) for environmental research on rock tile imagery.

**⚠️ Important**: Run VLM research when NOT training autoencoders/diffusion models (alternating use of GPUs).

---

## Installed Tools

✅ **Transformers** - HuggingFace models (CLIP, etc.)  
✅ **Sentence-Transformers** - Semantic embeddings  
✅ **CLIP** - Vision-language embeddings  

---

## Research Applications

### 1. **Semantic Search** (Text → Images)
Find rock tiles matching text descriptions:
- "rough textured granite with quartz crystals"
- "weathered sandstone with erosion patterns"
- "volcanic rock with vesicular structure"

### 2. **Image Similarity** (Image → Images)
Find visually similar tiles for:
- Pattern recognition
- Geological feature clustering
- Anomaly detection

### 3. **Zero-Shot Classification**
Classify rocks without training:
- Rock types (granite, sandstone, basalt, etc.)
- Geological features (fractures, weathering, minerals)
- Environmental conditions (wet, dry, weathered)

### 4. **Multimodal Analysis**
Combine visual and textual information for:
- Automated documentation
- Dataset annotation
- Environmental monitoring

---

## Usage Examples

### Extract CLIP Embeddings (One-Time Setup)

```bash
cd /home/jdas/dreams-lab-portal/dreams_laboratory/scripts

# Extract embeddings from all zoom 23 tiles
python3 vlm_clip_embeddings.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --mode extract \
    --batch_size 32 \
    --output_dir clip_embeddings_zoom23
```

**Output**:
- `clip_embeddings_zoom23/image_embeddings.npy` - Embeddings (8083 x 768)
- `clip_embeddings_zoom23/image_paths.json` - Image paths

**Time**: ~10-15 minutes for 8083 images  
**GPU Memory**: ~4-6 GB

---

### Semantic Search

Find images matching text descriptions:

```bash
# Search for specific rock types
python3 vlm_clip_embeddings.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --mode search \
    --query "granite with visible quartz crystals" \
    --output_dir clip_embeddings_zoom23

# More searches
python3 vlm_clip_embeddings.py --mode search --query "smooth weathered surface"
python3 vlm_clip_embeddings.py --mode search --query "fractured rock with cracks"
python3 vlm_clip_embeddings.py --mode search --query "porous volcanic texture"
```

**Example Output**:
```
Top 10 results:
1. tile_12345.png (similarity: 0.8234)
2. tile_67890.png (similarity: 0.7891)
...
```

---

### Zero-Shot Classification

Classify rocks without training a classifier:

```bash
python3 vlm_clip_embeddings.py \
    --mode classify \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --classes "granite" "sandstone" "basalt" "limestone" "marble" \
    --output_dir clip_embeddings_zoom23
```

**Use Cases**:
- Geological mapping
- Automated rock type identification
- Dataset organization

---

## Advanced Research Applications

### 1. Environmental Pattern Discovery

```python
# Custom script for pattern analysis
import numpy as np
from sklearn.cluster import KMeans
import umap

# Load embeddings
embeddings = np.load('clip_embeddings_zoom23/image_embeddings.npy')

# Reduce dimensions for visualization
reducer = umap.UMAP(n_components=2)
embedding_2d = reducer.fit_transform(embeddings)

# Cluster similar patterns
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings)

# Analyze each cluster
for i in range(10):
    cluster_images = image_paths[clusters == i]
    print(f"Cluster {i}: {len(cluster_images)} images")
```

### 2. Temporal/Spatial Analysis

Compare embeddings across:
- Different zoom levels
- Geographic locations
- Time periods (if available)

### 3. Synthetic vs Real Discrimination

Compare embeddings from:
- Real rock tiles
- Autoencoder-generated tiles
- Diffusion-generated tiles

```python
# Analyze embedding distributions
real_embeddings = embeddings_real
synthetic_embeddings = embeddings_synthetic

# Calculate distribution similarity
from scipy.spatial.distance import cosine
similarity = 1 - cosine(real_embeddings.mean(0), synthetic_embeddings.mean(0))
```

---

## Model Options

### CLIP Models (Vision-Language)

| Model | Size | VRAM | Performance |
|-------|------|------|-------------|
| `openai/clip-vit-base-patch32` | 151M | ~2GB | Fast, good |
| `openai/clip-vit-large-patch14` | 428M | ~4GB | Better quality ⭐ |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 986M | ~8GB | Best quality |

### Multimodal LLMs (Future)

When training completes, you can deploy:
- **LLaVA** (7B-13B) - Image captioning, Q&A
- **Qwen-VL** (7B) - Multilingual VLM
- **CogVLM** (17B) - Advanced reasoning

---

## When to Run VLM Research

### ✅ Good Times:
- After autoencoder training completes
- After diffusion training completes
- During breaks in training
- When evaluating generated samples

### ❌ Avoid:
- During autoencoder training (GPUs busy)
- During diffusion training (GPUs busy)
- When GPUs are at capacity

---

## GPU Memory Management

### Check Available Memory:
```bash
watch -n 1 nvidia-smi
```

### Current Usage (as of now):
- GPU 0: 12.5GB / 24GB (Autoencoder)
- GPU 1: 11.9GB / 24GB (Diffusion)

### CLIP Memory Needs:
- Base model: ~2GB
- Large model: ~4GB
- Embeddings: ~300MB (for 8K images)

### Strategy:
1. Wait for one training to complete
2. Use freed GPU for VLM research
3. Or run CLIP on CPU (slower but possible)

---

## Research Workflow

### Phase 1: Extract Embeddings (One-Time)

```bash
# After training completes, extract embeddings
python3 vlm_clip_embeddings.py \
    --mode extract \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23
```

### Phase 2: Exploratory Analysis

```bash
# Try various search queries
for query in "granite" "sandstone" "fractured" "weathered" "smooth"; do
    python3 vlm_clip_embeddings.py --mode search --query "$query"
done
```

### Phase 3: Classification

```bash
# Zero-shot classification
python3 vlm_clip_embeddings.py \
    --mode classify \
    --classes "igneous rock" "sedimentary rock" "metamorphic rock"
```

### Phase 4: Integration

Combine with your generative models:
- Compare embeddings: real vs synthetic
- Guide generation with text prompts
- Quality assessment of generated tiles

---

## Research Questions

### 1. **Synthetic Quality Assessment**
- How similar are synthetic embeddings to real ones?
- Do diffusion models capture more semantic features than autoencoders?

### 2. **Pattern Discovery**
- What natural clusters exist in the rock tile dataset?
- Can CLIP identify geological features unsupervised?

### 3. **Text-Guided Generation**
- Can we condition diffusion on CLIP embeddings?
- Use text prompts to generate specific rock types

### 4. **Environmental Monitoring**
- Track changes in rock textures over time/space
- Identify environmental degradation patterns

---

## Example Research Pipeline

```bash
# 1. Extract embeddings from real data
python3 vlm_clip_embeddings.py --mode extract \
    --tile_dir /mnt/dreamslab-store/.../raw/23 \
    --output_dir clip_real

# 2. Generate synthetic samples (after training)
python3 generate_diffusion_zoom23.py --num_samples 100

# 3. Extract embeddings from synthetic data
python3 vlm_clip_embeddings.py --mode extract \
    --tile_dir diffusion_synthetic_zoom23 \
    --output_dir clip_synthetic

# 4. Compare distributions
python3 analyze_embedding_similarity.py \
    --real_emb clip_real/image_embeddings.npy \
    --synthetic_emb clip_synthetic/image_embeddings.npy
```

---

## Integration with Your Work

### With Autoencoder:
- Encode images → CLIP embeddings
- Compare CLIP space vs latent space
- Text-guided latent space navigation

### With Diffusion Model:
- Condition diffusion on CLIP embeddings
- Text-to-image generation for rocks
- Guided sampling with semantic constraints

### Combined:
- Use autoencoder for fast generation
- Use CLIP for semantic filtering
- Use diffusion for highest quality on filtered set

---

## Next Steps

### Immediate (After Training):
1. Extract CLIP embeddings from zoom 23 tiles
2. Experiment with semantic search
3. Try zero-shot classification

### Short-term:
1. Cluster embeddings to discover patterns
2. Compare real vs synthetic embeddings
3. Document findings

### Long-term:
1. Deploy LLaVA for image captioning
2. Build text-guided generation pipeline
3. Create environmental monitoring tools

---

## Additional Resources

### Documentation:
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- CLIP Paper: https://arxiv.org/abs/2103.00020
- Sentence-Transformers: https://www.sbert.net/

### Models to Explore:
- CLIP variants (different sizes)
- SigLIP (improved CLIP)
- ImageBind (multi-modal embeddings)
- LLaVA (visual question answering)

---

## GPU Schedule Recommendation

| Time Period | GPU 0 | GPU 1 |
|-------------|-------|-------|
| **Now** | Autoencoder training | Diffusion training |
| **After autoencoder done** | VLM research | Diffusion training |
| **After all training** | VLM + generation | VLM + generation |

Your Titan RTX GPUs are perfect for this research! 🚀

