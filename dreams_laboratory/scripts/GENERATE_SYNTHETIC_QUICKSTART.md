# Generate Synthetic Multispectral Images - Quick Start

## Step 1: Generate Random Synthetic Images

```bash
cd ~/dreams-lab-website-server/dreams_laboratory/scripts

python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 20 \
    --sample_method gaussian
```

**What this does:**
- Samples 20 random points in latent space
- Uses decoder to generate synthetic multispectral images
- Creates visualization and saves individual images

**Output:**
- `synthetic_samples_gaussian.png` - Grid visualization
- `synthetic_images/` directory with:
  - `synthetic_000.npy` through `synthetic_019.npy` - Multispectral data
  - `synthetic_000_rgb.png` through `synthetic_019_rgb.png` - RGB visualizations

---

## Step 2: Try Different Sampling Methods

### Gaussian (default - matches your data distribution):
```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 20 \
    --sample_method gaussian
```

### Uniform (explore entire latent space):
```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 20 \
    --sample_method uniform
```

### Sphere (samples from sphere matching data norm):
```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 20 \
    --sample_method sphere
```

---

## Step 3: Generate More Samples

```bash
# Generate 50 synthetic images
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 50
```

---

## Step 4: View Results

```bash
# View the visualization
eog synthetic_samples_gaussian.png  # or use your image viewer

# Check generated images
ls -lh synthetic_images/

# View individual RGB images
eog synthetic_images/synthetic_000_rgb.png
```

---

## Quick Test Command

Run this first to test everything works:

```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 5
```

This generates just 5 samples quickly to verify everything is working.

---

## Expected Output

You should see:
```
Loading encoder...
Loaded model from multispectral_vit.pth
Config: {'img_size': 960, 'patch_size': 16, ...}
Loading decoder from decoder.pth
Loading latent distribution from multispectral_latents.npy
Loaded 6820 latent representations

Generating 20 synthetic images...
Saved visualization to: synthetic_samples_gaussian.png

Saved 20 synthetic images to synthetic_images/
Files:
  - synthetic_XXX.npy: Multispectral data (5 bands)
  - synthetic_XXX_rgb.png: RGB visualization
```

---

## Troubleshooting

### If you get "decoder not found" error:
- Make sure `decoder.pth` exists in the current directory
- Check file path is correct

### If images look like noise:
- Decoder may need more training
- Try different sampling methods
- Check if reconstruction quality was good during training

### If out of memory:
- Reduce `--num_samples` to smaller number
- Use `--device cpu` if GPU memory is full

