# Diffusion Model Quick Start

## What is a Diffusion Model?

A diffusion model learns to generate images by:
1. **Training**: Learning to remove noise from noisy images
2. **Generation**: Starting with pure noise and gradually denoising it into a realistic image

Think of it like an artist who learns to "clean up" a sketch into a finished artwork!

---

## Quick Commands

### 1. Train Diffusion Model (256×256)

```bash
cd /home/jdas/dreams-lab-portal/dreams_laboratory/scripts

# Train on all GPUs (recommended)
python3 train_diffusion_zoom23.py \
    --img_size 256 \
    --batch_size 32 \
    --epochs 100 \
    --multi_gpu \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw
```

**Expected time**: 3-5 days on Titan RTX  
**Output**: `diffusion_model_zoom23_ema.pth` (EMA model, best quality)

---

### 2. Generate Samples (After Training)

#### Option A: High Quality (Slow)
```bash
# DDPM: 1000 steps, best quality, ~2-3 minutes per image
python3 generate_diffusion_zoom23.py \
    --model_path diffusion_model_zoom23_ema.pth \
    --num_samples 20 \
    --batch_size 8
```

#### Option B: Fast (Recommended)
```bash
# DDIM: 50 steps, excellent quality, ~15-30 seconds per image
python3 generate_diffusion_zoom23.py \
    --model_path diffusion_model_zoom23_ema.pth \
    --num_samples 50 \
    --batch_size 8 \
    --use_ddim \
    --ddim_steps 50
```

#### Option C: Very Fast
```bash
# DDIM: 25 steps, good quality, ~10-15 seconds per image
python3 generate_diffusion_zoom23.py \
    --model_path diffusion_model_zoom23_ema.pth \
    --num_samples 100 \
    --batch_size 16 \
    --use_ddim \
    --ddim_steps 25
```

---

## Training in Parallel with Autoencoder

Your Titan RTX can train both at once! Run them on different GPUs or alternate:

### Terminal 1: Autoencoder (512×512)
```bash
# Already running!
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --multi_gpu
```

### Terminal 2: Diffusion (256×256)
```bash
# Start diffusion training
python3 train_diffusion_zoom23.py \
    --img_size 256 \
    --batch_size 32 \
    --epochs 100 \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw
```

---

## Training Progress

Monitor training:

```bash
# In another terminal
watch -n 1 nvidia-smi

# Check checkpoint directory
ls -lh checkpoints_diffusion_zoom23/
```

You'll see:
- `latest_checkpoint.pth` - Updated every 5 epochs
- `best_checkpoint.pth` - Best loss so far

---

## Generate During Training

You can generate samples even during training to see progress:

```bash
# Load latest checkpoint
python3 generate_diffusion_zoom23.py \
    --model_path checkpoints_diffusion_zoom23/latest_checkpoint.pth \
    --num_samples 10 \
    --use_ddim --ddim_steps 25
```

Early epochs will look noisy, but you'll see improvement!

---

## Expected Results Timeline

### After 10 epochs (~3-4 hours):
- Images will be blurry/noisy
- Basic structures emerging
- Not usable yet

### After 30 epochs (~12 hours):
- Recognizable rock textures
- Still some artifacts
- Getting better!

### After 50 epochs (~2 days):
- Good quality images
- Realistic textures
- Usable for most purposes

### After 100 epochs (~4 days):
- Excellent quality
- Sharp details
- State-of-the-art results

---

## Output Files

### Training outputs:
```
checkpoints_diffusion_zoom23/
├── latest_checkpoint.pth          # Latest model
├── best_checkpoint.pth            # Best model
diffusion_model_zoom23.pth         # Final main model
diffusion_model_zoom23_ema.pth     # Final EMA model (use this!)
```

### Generation outputs:
```
diffusion_synthetic_zoom23/
├── diffusion_samples_ddim50.png   # Grid visualization
├── diffusion_000.png              # Individual images
├── diffusion_001.png
├── ...
├── diffusion_000.npy              # NumPy arrays
└── ...
```

---

## Troubleshooting

### Out of Memory During Training
```bash
# Reduce batch size
python3 train_diffusion_zoom23.py --batch_size 16

# Or reduce image size
python3 train_diffusion_zoom23.py --img_size 128
```

### Out of Memory During Generation
```bash
# Reduce batch size
python3 generate_diffusion_zoom23.py --batch_size 4

# Generate fewer samples at once
python3 generate_diffusion_zoom23.py --num_samples 10
```

### Training is Slow
This is normal! Diffusion models take time:
- Each epoch: ~2-3 minutes on Titan RTX
- 100 epochs: ~4 days
- Be patient, results are worth it!

### Generated Images Look Bad
- Need more training epochs
- Try using EMA model: `--model_path diffusion_model_zoom23_ema.pth`
- Use DDIM with fewer steps: `--use_ddim --ddim_steps 50`

---

## Comparison: Autoencoder vs Diffusion

| Aspect | Autoencoder | Diffusion |
|--------|-------------|-----------|
| Quality | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| Speed | ⚡⚡⚡ Instant | ⚡ Slow (1-3 min) |
| Training | 2-3 days | 4-5 days |
| Control | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Moderate |

**Recommendation**: Use both!
- Autoencoder for rapid generation
- Diffusion for high-quality final outputs

---

## Advanced: Higher Resolution

After training at 256×256, you can train at 512×512:

```bash
python3 train_diffusion_zoom23.py \
    --img_size 512 \
    --batch_size 16 \
    --base_channels 128 \
    --epochs 100 \
    --multi_gpu
```

**Note**: Takes 2-3x longer to train!

---

## Tips for Best Results

### 1. Use EMA Model
```bash
--model_path diffusion_model_zoom23_ema.pth
```
EMA (Exponential Moving Average) produces better, more stable images.

### 2. Use DDIM for Speed
```bash
--use_ddim --ddim_steps 50
```
50 steps gives excellent quality, 20x faster than 1000 steps!

### 3. Generate in Batches
```bash
--batch_size 16
```
Larger batches are more efficient on Titan RTX.

### 4. Monitor Training
Check samples every 10-20 epochs to see progress.

---

## Next Steps

1. ✅ Start training: `python3 train_diffusion_zoom23.py --multi_gpu`
2. ⏳ Wait 3-5 days for full training
3. ✅ Generate samples: `python3 generate_diffusion_zoom23.py --use_ddim`
4. 🎨 Compare with autoencoder outputs
5. 📊 Use best model for your needs

---

## Help & References

- `python3 train_diffusion_zoom23.py --help`
- `python3 generate_diffusion_zoom23.py --help`
- See `AUTOENCODER_VS_DIFFUSION_GUIDE.md` for detailed comparison

Good luck! Diffusion models produce amazing results! 🚀

