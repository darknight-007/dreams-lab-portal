# VLM Research Quick Start

## ⚠️ Run AFTER Training Completes

Your VLM environment is installed and ready. Use this guide when your GPU training finishes.

---

## Check GPU Availability

```bash
nvidia-smi
```

**Wait until**:
- GPU memory usage is low (~1-2GB)
- No training processes running
- OR one GPU is free while other trains

---

## Quick Commands

### 1. Extract Embeddings (10-15 minutes)

```bash
cd /home/jdas/dreams-lab-portal/dreams_laboratory/scripts

python3 vlm_clip_embeddings.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --mode extract \
    --batch_size 32 \
    --output_dir clip_embeddings_zoom23
```

### 2. Search for Rock Types

```bash
# Find granite
python3 vlm_clip_embeddings.py \
    --mode search \
    --query "granite with visible crystals" \
    --output_dir clip_embeddings_zoom23

# Find weathered rocks
python3 vlm_clip_embeddings.py \
    --mode search \
    --query "weathered surface with erosion patterns" \
    --output_dir clip_embeddings_zoom23

# Find fractured rocks
python3 vlm_clip_embeddings.py \
    --mode search \
    --query "fractured rock with visible cracks" \
    --output_dir clip_embeddings_zoom23
```

### 3. Classify Rock Types

```bash
python3 vlm_clip_embeddings.py \
    --mode classify \
    --classes "granite" "sandstone" "basalt" "limestone" \
    --output_dir clip_embeddings_zoom23
```

---

## What You Can Research

✅ **Semantic Search**: Find images by description  
✅ **Pattern Discovery**: Cluster similar geological features  
✅ **Zero-Shot Classification**: Identify rock types without training  
✅ **Quality Assessment**: Compare real vs synthetic tiles  
✅ **Environmental Analysis**: Track patterns and changes  

---

## Example Research Session

```bash
# 1. Extract embeddings
python3 vlm_clip_embeddings.py --mode extract

# 2. Search for different features
for feature in "smooth" "rough" "fractured" "weathered" "crystalline"; do
    echo "Searching for: $feature"
    python3 vlm_clip_embeddings.py --mode search --query "$feature rock texture"
    echo "---"
done

# 3. Classify by rock type
python3 vlm_clip_embeddings.py --mode classify \
    --classes "igneous" "sedimentary" "metamorphic"
```

---

## Output Files

After extraction:
```
clip_embeddings_zoom23/
├── image_embeddings.npy    # 8083 x 768 embedding matrix
└── image_paths.json        # Corresponding image paths
```

These embeddings can be reused for multiple analyses!

---

## Tips

1. **Extract once**: Embeddings can be reused for many queries
2. **Batch queries**: Process multiple searches at once
3. **Save results**: Document interesting findings
4. **Visualize**: Use UMAP/t-SNE to visualize embedding space

---

## Next: Advanced Research

See `VLM_RESEARCH_GUIDE.md` for:
- Custom analysis scripts
- Integration with generative models
- Deployment of larger VLMs (LLaVA, etc.)
- Environmental monitoring pipelines

---

## Currently Available

✅ CLIP (installed)  
✅ Transformers (installed)  
✅ Sentence-Transformers (installed)  
⏳ LLaVA (deploy after training)  
⏳ Ollama (deploy after training)  

Ready to go! 🚀

