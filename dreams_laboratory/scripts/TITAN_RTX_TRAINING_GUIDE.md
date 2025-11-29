# Training with Two Titan RTX GPUs (48GB Total VRAM)

## GPU Specifications

**Titan RTX:**
- VRAM per GPU: 24 GB
- Total VRAM: 48 GB (2×24GB)
- CUDA Cores: 4,608 per GPU
- Memory Bandwidth: 672 GB/s per GPU

**Comparison:**
- Your Setup: 2× Titan RTX (48GB total)
- RTX 3090: 1× 24GB
- RTX 4090: 1× 24GB
- A6000: 1× 48GB

**You have excellent hardware for high-resolution training!**

## Feasible Resolutions

### 512×512 Resolution ⭐⭐⭐⭐⭐ RECOMMENDED

**Memory Requirements:**
- Single GPU: ~16 GB per batch
- With 2 GPUs (DataParallel): Can split batch across GPUs
- Effective batch size: 16 (8 per GPU)

**Training Configuration:**
```bash
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --patch_size 16 \
    --batch_size 16 \
    --epochs 50 \
    --multi_gpu \
    --device cuda
```

**Expected Performance:**
- Memory per GPU: ~12-14 GB (comfortable)
- Training time: ~6-8 hours (vs 8-12 hours on single GPU)
- Batch size: 16 (8 per GPU)
- Quality: Excellent (native 512×512)

**Why This Works:**
- DataParallel splits batch across 2 GPUs
- Each GPU processes 8 images at a time
- Memory is shared efficiently
- Training is ~1.5-2x faster than single GPU

### 768×768 Resolution ⭐⭐⭐⭐ FEASIBLE

**Memory Requirements:**
- Single GPU: ~36 GB per batch
- With 2 GPUs: Can work with smaller batches

**Training Configuration:**
```bash
python3 train_zoom23_autoencoder.py \
    --img_size 768 \
    --patch_size 16 \
    --batch_size 4 \
    --epochs 50 \
    --multi_gpu \
    --device cuda
```

**Expected Performance:**
- Memory per GPU: ~20-22 GB (close to limit)
- Training time: ~12-16 hours
- Batch size: 4 (2 per GPU) - may need gradient accumulation
- Quality: Excellent (native 768×768)

**Considerations:**
- Batch size of 2 per GPU is small (may need gradient accumulation)
- Close to memory limit (may need to reduce other parameters)
- Still feasible with careful tuning

### 1024×1024 Resolution ⭐⭐⭐ CHALLENGING

**Memory Requirements:**
- Single GPU: ~64 GB per batch
- With 2 GPUs: Very tight, would need batch_size=1

**Training Configuration:**
```bash
python3 train_zoom23_autoencoder.py \
    --img_size 1024 \
    --patch_size 16 \
    --batch_size 2 \
    --epochs 50 \
    --multi_gpu \
    --device cuda
```

**Expected Performance:**
- Memory per GPU: ~24 GB (at limit)
- Training time: ~24-36 hours
- Batch size: 2 (1 per GPU) - requires gradient accumulation
- Quality: Best (native 1024×1024)

**Challenges:**
- Batch size of 1 per GPU is very small
- Need gradient accumulation to simulate larger batch
- May need to reduce embed_dim or num_layers
- Close to memory limits

## Optimal Configuration for 2× Titan RTX

### Recommended: 512×512 with Batch Size 16

```bash
python3 train_zoom23_autoencoder.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23 \
    --img_size 512 \
    --patch_size 16 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --batch_size 16 \
    --epochs 50 \
    --device cuda \
    --multi_gpu
```

**Why This Works Best:**
- ✅ Comfortable memory usage (~12-14 GB per GPU)
- ✅ Good batch size (8 per GPU)
- ✅ Fast training (~6-8 hours)
- ✅ Excellent quality (2x resolution)
- ✅ No memory pressure

### Advanced: 768×768 with Gradient Accumulation

If you want even higher resolution:

```bash
# Would need to modify training script to support gradient accumulation
python3 train_zoom23_autoencoder.py \
    --img_size 768 \
    --patch_size 16 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --multi_gpu
```

This would:
- Use batch_size=4 (2 per GPU)
- Accumulate gradients over 4 steps
- Effective batch size: 16
- Better convergence than batch_size=2

## Memory Breakdown (512×512 Example)

### Per GPU (with batch_size=16, 8 per GPU):

```
Encoder:
- Input images: 8 × 3 × 512 × 512 × 4 bytes = 25 MB
- Patch embeddings: 8 × 1,024 × 512 × 4 bytes = 16 MB
- Positional embeddings: 1 × 1,024 × 512 × 4 bytes = 2 MB
- Transformer layers: ~200 MB
- Gradients: ~200 MB
Subtotal: ~450 MB

Decoder:
- Latent: 8 × 512 × 4 bytes = 16 KB
- Patch embeddings: 8 × 1,024 × 32 × 4 bytes = 1 MB
- Pixel patches: 8 × 1,024 × 16 × 16 × 3 × 4 bytes = 25 MB
- Gradients: ~100 MB
Subtotal: ~125 MB

PyTorch overhead: ~500 MB
Intermediate computations: ~200 MB
Total per GPU: ~12-14 GB

With 2 GPUs: 24-28 GB total (plenty of headroom!)
```

## Multi-GPU Performance

### DataParallel vs DistributedDataParallel

**Current Script Uses DataParallel:**
```python
if use_multi_gpu and torch.cuda.device_count() > 1:
    autoencoder = nn.DataParallel(autoencoder, device_ids=[0, 1])
```

**Performance:**
- Speedup: ~1.7-1.9x (not perfect 2x due to overhead)
- Memory: Split across both GPUs
- Batch size: Automatically divided

**Alternative: DistributedDataParallel (Better but more complex)**
- Speedup: ~1.9-1.95x (more efficient)
- Requires: More code changes
- Benefit: Better for large models

## Training Time Estimates

| Resolution | Batch Size | Time (2× Titan RTX) | Time (1× Titan RTX) | Speedup |
|------------|------------|---------------------|---------------------|---------|
| 256×256    | 16         | 1-2 hours           | 2-4 hours           | 2x      |
| 512×512    | 16         | 6-8 hours           | 8-12 hours          | 1.5x    |
| 768×768    | 4          | 12-16 hours         | 18-24 hours         | 1.5x    |
| 1024×1024  | 2          | 24-36 hours         | 36-48 hours         | 1.5x    |

## Recommended Approach

### Option 1: 512×512 (Best Balance) ⭐⭐⭐⭐⭐

```bash
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --batch_size 16 \
    --multi_gpu \
    --epochs 50
```

**Why:**
- Excellent quality (2x resolution)
- Comfortable memory usage
- Reasonable training time
- Good batch size for convergence

### Option 2: 768×768 (Maximum Quality) ⭐⭐⭐⭐

```bash
python3 train_zoom23_autoencoder.py \
    --img_size 768 \
    --batch_size 4 \
    --multi_gpu \
    --epochs 50
```

**Why:**
- Highest quality (3x resolution)
- Still feasible with 2 GPUs
- Longer training time
- May need gradient accumulation

### Option 3: Progressive Training ⭐⭐⭐⭐⭐

Train at 512×512 first, then fine-tune at 768×768:

```bash
# Stage 1: Train at 512×512
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --batch_size 16 \
    --multi_gpu \
    --epochs 30

# Stage 2: Fine-tune at 768×768
python3 train_zoom23_autoencoder.py \
    --img_size 768 \
    --batch_size 4 \
    --encoder_path encoder_512.pth \
    --decoder_path decoder_512.pth \
    --multi_gpu \
    --epochs 20
```

## Memory Optimization Tips

### If Running Out of Memory:

1. **Reduce Batch Size:**
   ```bash
   --batch_size 8  # Instead of 16
   ```

2. **Use Gradient Checkpointing:**
   ```python
   # In model, use torch.utils.checkpoint
   # Trade compute for memory
   ```

3. **Reduce Embedding Dimension:**
   ```bash
   --embed_dim 384  # Instead of 512
   ```

4. **Use Mixed Precision:**
   ```python
   # Use torch.cuda.amp for automatic mixed precision
   # Can reduce memory by ~30-40%
   ```

5. **Larger Patch Size:**
   ```bash
   --patch_size 32  # Instead of 16
   # Reduces num_patches by 4x
   ```

## Summary

**With 2× Titan RTX (48GB total):**

✅ **512×512**: Highly recommended - Comfortable, fast, excellent quality
✅ **768×768**: Feasible - Close to limits but workable
⚠️ **1024×1024**: Challenging - Requires optimization, small batches

**Recommended Command:**
```bash
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --batch_size 16 \
    --multi_gpu \
    --epochs 50
```

**Expected Results:**
- Memory: ~12-14 GB per GPU (comfortable)
- Training time: ~6-8 hours
- Quality: Excellent (2x current resolution)
- Batch size: 16 (good for convergence)

You have excellent hardware for high-resolution training! 🚀



