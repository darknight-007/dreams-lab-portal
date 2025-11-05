# Guide: Improving Synthetic Sample Resolution

## Current Status
- **Native resolution**: 256×256
- **Compression**: 384:1 (196,608 values → 512 values)
- **Quality**: ~21 dB PSNR

## Methods to Improve Resolution

### Method 1: **Retrain at Higher Native Resolution** ⭐ BEST for Quality

Train the autoencoder at 512×512 or 1024×1024 from scratch.

```bash
# 512x512 training (recommended for Titan RTX)
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --patch_size 16 \
    --batch_size 16 \
    --embed_dim 768 \
    --num_layers 8 \
    --epochs 50 \
    --multi_gpu \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw
```

**Pros:**
✅ Native high-res, no upscaling needed  
✅ Best quality - no interpolation artifacts  
✅ Titan RTX can handle 512×512 @ batch 16  
✅ Learns fine details directly

**Cons:**
❌ 4x longer training (for 512×512)  
❌ Needs more VRAM  
❌ Larger model files

**Expected Results:**
- PSNR: 23-25 dB (better than 256×256)
- True high-resolution details
- Training time: ~2-3x longer

---

### Method 2: **Train Separate Super-Resolution Network** ⭐ BEST for Flexibility

Keep 256×256 autoencoder, add learned upsampling.

```bash
# Step 1: Train super-resolution network
python3 train_super_resolution.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --low_res 256 \
    --high_res 1024 \
    --batch_size 16 \
    --epochs 100 \
    --num_blocks 16

# Step 2: Generate with super-resolution
python3 generate_with_sr.py \
    --decoder_path decoder_zoom23.pth \
    --sr_model_path super_resolution_best.pth \
    --target_size 1024
```

**Pros:**
✅ Reuse existing 256×256 model  
✅ Can train different SR models for different scales  
✅ Faster than retraining full autoencoder  
✅ Can use on existing generated images

**Cons:**
❌ Two-stage process  
❌ Extra model to maintain  
❌ May introduce upsampling artifacts

**Expected Results:**
- Better than bicubic/Lanczos upscaling
- Sharp edges and textures
- 1024×1024 from 256×256 source

---

### Method 3: **Progressive Growing** 

Train in stages: 128→256→512→1024

```python
# Stage 1: Train at 128x128
# Stage 2: Upscale model, train at 256x256
# Stage 3: Upscale model, train at 512x512
# etc.
```

**Pros:**
✅ Stable training  
✅ Can achieve very high resolutions  
✅ Used by StyleGAN successfully

**Cons:**
❌ Complex training pipeline  
❌ Takes longest time  
❌ Requires careful implementation

---

### Method 4: **Diffusion Model Post-Processing**

Add diffusion refinement to generated samples.

**Pros:**
✅ State-of-art quality  
✅ Can fix artifacts  
✅ Adds fine details

**Cons:**
❌ Very slow generation  
❌ Complex to implement  
❌ Requires additional training

---

### Method 5: **Multi-Scale Architecture**

Train encoder/decoder at multiple resolutions simultaneously.

**Architecture:**
```
Input 512×512
├─ Encoder level 1 (512×512)
├─ Encoder level 2 (256×256)
├─ Encoder level 3 (128×128)
└─ Latent (512D)
    ├─ Decoder level 3 → 128×128
    ├─ Decoder level 2 → 256×256 (+ skip from encoder)
    └─ Decoder level 1 → 512×512 (+ skip from encoder)
```

**Pros:**
✅ Best of both worlds  
✅ Multi-resolution learning  
✅ Skip connections preserve details

**Cons:**
❌ Complex architecture  
❌ More VRAM needed  
❌ Longer training

---

## Recommended Workflow

### For Your Titan RTX Setup:

#### **Option A: Quick Win (1 day)**
Use existing model + learned super-resolution

```bash
# 1. Train SR network (2-3 hours)
python3 train_super_resolution.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --low_res 256 --high_res 1024 \
    --batch_size 16 --epochs 50

# 2. Generate high-res images
python3 generate_synthetic_zoom23_hires.py \
    --decoder_path decoder_zoom23.pth \
    --target_size 1024 \
    --upscale_method learned \
    --sr_model_path super_resolution_best.pth
```

#### **Option B: Best Quality (3-4 days)**
Retrain at 512×512 native

```bash
# 1. Train at 512x512 (2-3 days)
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --batch_size 16 \
    --embed_dim 768 \
    --epochs 50 \
    --multi_gpu

# 2. Extract latents
python3 extract_latents_zoom23.py \
    --encoder_path encoder_zoom23_512.pth

# 3. Generate natively at 512x512
python3 generate_synthetic_zoom23.py \
    --decoder_path decoder_zoom23_512.pth \
    --num_samples 100
```

#### **Option C: Maximum Quality (1 week)**
Combine both approaches

```bash
# 1. Train at 512x512
# 2. Train SR network 512→2048
# 3. Generate at 2048×2048
```

---

## Performance Comparison

| Method | Resolution | Quality (PSNR) | Time | VRAM |
|--------|-----------|----------------|------|------|
| Current (bicubic) | 256→1024 | ~18 dB | Fast | 4GB |
| Lanczos | 256→1024 | ~19 dB | Medium | 4GB |
| Learned SR | 256→1024 | ~22-24 dB | Medium | 6GB |
| Native 512 | 512 | ~23-25 dB | Slow train | 12GB |
| Native 1024 | 1024 | ~25-27 dB | Very slow | 20GB+ |

---

## Comparison: Upscaling Methods

### Current Methods (in generate_synthetic_zoom23_hires.py):

1. **Bicubic** (Fast)
   - Quality: ⭐⭐⭐
   - Speed: ⚡⚡⚡
   - Blurry edges

2. **Lanczos** (Best classical)
   - Quality: ⭐⭐⭐⭐
   - Speed: ⚡⚡
   - Sharp but can have ringing artifacts

3. **Learned** (Currently simple)
   - Quality: ⭐⭐⭐⭐
   - Speed: ⚡⚡
   - Better than classical but can be improved

### With Super-Resolution Training:

4. **EDSR-style SR** (Recommended)
   - Quality: ⭐⭐⭐⭐⭐
   - Speed: ⚡⚡
   - Best learned upscaling

5. **GAN-based SR** (Sharpest)
   - Quality: ⭐⭐⭐⭐⭐
   - Speed: ⚡
   - Most realistic textures

---

## Practical Steps for You

### Immediate (Today): Use Better Upscaling

```bash
# Use Lanczos for best quality with existing model
python3 generate_synthetic_zoom23_hires.py \
    --target_size 1024 \
    --upscale_method lanczos \
    --num_samples 50
```

### Short-term (This Week): Train SR Network

```bash
# Train super-resolution model
python3 train_super_resolution.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/23 \
    --low_res 256 \
    --high_res 1024 \
    --batch_size 16 \
    --epochs 50
```

### Long-term (Next Week): Retrain at 512×512

```bash
# Retrain entire autoencoder at higher resolution
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --batch_size 16 \
    --embed_dim 768 \
    --num_layers 8 \
    --epochs 50 \
    --multi_gpu
```

---

## Technical Details

### Why Native Training is Best:

1. **Direct Learning**: Model learns fine details directly
2. **No Interpolation**: No upscaling artifacts
3. **Better PSNR**: 23-25 dB vs 18-19 dB
4. **Texture Quality**: Realistic high-frequency details

### Why SR Networks Work:

1. **Learned Priors**: Learns what rock textures should look like
2. **Edge Enhancement**: Adds missing high-frequency details
3. **Hallucination**: Can invent plausible details
4. **Efficient**: Can be applied to any 256×256 output

### Architecture Changes for Higher Resolution:

```python
# Current (256x256)
img_size = 256
patch_size = 16
num_patches = 256  # (256/16)^2
embed_dim = 512

# Upgraded (512x512)
img_size = 512
patch_size = 16
num_patches = 1024  # (512/16)^2
embed_dim = 768  # Increased capacity
num_layers = 8  # More layers for complexity
```

---

## Summary

**Best approach for you:**

1. ✅ **Immediate**: Use Lanczos upscaling (already works)
2. ✅ **Short-term**: Train SR network (2-3 hours, significant improvement)
3. ✅ **Long-term**: Retrain at 512×512 (2-3 days, best quality)

Your Titan RTX has the power to do all of this!

