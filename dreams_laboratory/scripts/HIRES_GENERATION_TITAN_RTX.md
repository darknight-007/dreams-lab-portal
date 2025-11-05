# High-Resolution Synthetic Generation on Titan RTX

## GPU Comparison

| GPU | VRAM | Native Batch Size | HR Batch Size |
|-----|------|-------------------|---------------|
| RTX 2080 Ti | 11GB | 16 | 8 (512px) |
| Titan RTX | 24GB | 32+ | 32 (512px), 16 (1024px) |

## Quick Start Commands

### 1. Generate 512x512 images (2x native resolution)

**High Quality - Bicubic (Default, Fast)**
```bash
python3 generate_synthetic_zoom23_hires.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 50 \
    --target_size 512 \
    --batch_size 32 \
    --sample_method gaussian
```

**Highest Quality - Lanczos (Slower, Best)**
```bash
python3 generate_synthetic_zoom23_hires.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 50 \
    --target_size 512 \
    --batch_size 32 \
    --upscale_method lanczos \
    --sample_method gaussian
```

**Experimental - Learned Upsampling (Neural)**
```bash
python3 generate_synthetic_zoom23_hires.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 50 \
    --target_size 512 \
    --batch_size 32 \
    --upscale_method learned \
    --sample_method gaussian
```

### 2. Generate 1024x1024 images (4x native resolution)

**For Titan RTX - Maximum Quality**
```bash
python3 generate_synthetic_zoom23_hires.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 20 \
    --target_size 1024 \
    --batch_size 16 \
    --upscale_method lanczos \
    --sample_method gaussian
```

### 3. Generate at Native Resolution (256x256)

```bash
python3 generate_synthetic_zoom23_hires.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 100 \
    --batch_size 64 \
    --no_upscale
```

## Sampling Methods

### 1. Gaussian (Default - Most Realistic)
Samples from learned distribution, stays close to training data.
```bash
--sample_method gaussian
```

### 2. Uniform (Exploration)
Explores full latent space range, more variety but may be unrealistic.
```bash
--sample_method uniform
```

### 3. Real (Reconstruction)
Uses actual latent codes from training data.
```bash
--sample_method real
```

### 4. Interpolate (Smooth Transitions)
Interpolates between training samples for smooth variations.
```bash
--sample_method interpolate
```

## Upscaling Methods

### 1. Bicubic (Fast, Good Quality)
- **Speed**: ⚡⚡⚡ Very Fast
- **Quality**: ⭐⭐⭐ Good
- **Best for**: Quick iterations, testing

### 2. Lanczos (Slower, Best Quality)
- **Speed**: ⚡⚡ Medium
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Best for**: Final production, highest quality

### 3. Learned (Experimental)
- **Speed**: ⚡ Slower
- **Quality**: ⭐⭐⭐⭐ Very Good (depends on training)
- **Best for**: Neural enhancement experiments

## Recommended Workflows

### Workflow 1: Quick Testing
```bash
# Generate 10 samples at 512px for quick review
python3 generate_synthetic_zoom23_hires.py \
    --num_samples 10 \
    --target_size 512 \
    --batch_size 32 \
    --sample_method gaussian
```

### Workflow 2: Production Quality
```bash
# Generate 100 high-quality samples at 1024px
python3 generate_synthetic_zoom23_hires.py \
    --num_samples 100 \
    --target_size 1024 \
    --batch_size 16 \
    --upscale_method lanczos \
    --sample_method gaussian \
    --output_dir synthetic_production_1024
```

### Workflow 3: Variety Pack
```bash
# Generate with different methods
for method in gaussian uniform interpolate; do
    python3 generate_synthetic_zoom23_hires.py \
        --num_samples 25 \
        --target_size 512 \
        --sample_method $method \
        --output_dir synthetic_${method}_512
done
```

### Workflow 4: Multi-Resolution Export
```bash
# 256px (native)
python3 generate_synthetic_zoom23_hires.py \
    --num_samples 50 --no_upscale \
    --output_dir synthetic_256px

# 512px
python3 generate_synthetic_zoom23_hires.py \
    --num_samples 50 --target_size 512 \
    --output_dir synthetic_512px

# 1024px
python3 generate_synthetic_zoom23_hires.py \
    --num_samples 50 --target_size 1024 \
    --batch_size 16 --output_dir synthetic_1024px
```

## Performance Tips for Titan RTX

### Maximize Throughput
```bash
# Larger batch sizes (up to 64 for native res)
--batch_size 64  # for 256px
--batch_size 32  # for 512px
--batch_size 16  # for 1024px
```

### Monitor GPU Usage
```bash
# In another terminal
watch -n 0.5 nvidia-smi
```

### Generate Large Batches
```bash
# Generate 1000 samples efficiently
python3 generate_synthetic_zoom23_hires.py \
    --num_samples 1000 \
    --target_size 512 \
    --batch_size 32 \
    --output_dir synthetic_large_batch
```

## Expected Performance (Titan RTX)

| Resolution | Batch Size | Speed (imgs/sec) | Memory Usage |
|------------|------------|------------------|--------------|
| 256x256 | 64 | ~100 | ~8 GB |
| 512x512 | 32 | ~40 | ~12 GB |
| 1024x1024 | 16 | ~15 | ~18 GB |

## Output Files

Each run produces:
```
synthetic_zoom23_hires/
├── synthetic_samples_gaussian_512px.png  # Grid visualization
├── synthetic_000.png                     # High-res PNG
├── synthetic_000.npy                     # NumPy array
├── synthetic_001.png
├── synthetic_001.npy
└── ...
```

## Troubleshooting

### Out of Memory
Reduce batch size or target resolution:
```bash
--batch_size 8 --target_size 512
```

### Slow Generation
Use faster upscaling:
```bash
--upscale_method bicubic
```

### Poor Quality
Try Lanczos or increase resolution:
```bash
--upscale_method lanczos --target_size 1024
```

## Next Steps

After generation, you can:

1. **Analyze quality**: Use `test_reconstruction_zoom23.py`
2. **Compare methods**: Generate with different sampling methods
3. **Scale up**: Generate thousands of samples for training datasets
4. **Export**: Convert to other formats or create tilesets

## Example: Complete Workflow

```bash
# 1. Extract latents (if not done)
python3 extract_latents_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --output_file latents_zoom23.npy

# 2. Generate high-res samples
python3 generate_synthetic_zoom23_hires.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 100 \
    --target_size 1024 \
    --batch_size 16 \
    --upscale_method lanczos \
    --output_dir synthetic_final_1024

# 3. View results
eog synthetic_final_1024/synthetic_samples_gaussian_1024px.png
```

