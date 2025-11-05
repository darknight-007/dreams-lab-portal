# Next Steps After Training Autoencoder

## Step 1: Verify Training Completed Successfully

After training finishes, check that model files were saved:

```bash
ls -lh *.pth
```

You should see:
- `multispectral_vit.pth` - Updated encoder
- `decoder.pth` - Trained decoder
- `multispectral_autoencoder.pth` - Combined model

---

## Step 2: Generate Synthetic Images

Now you can generate synthetic multispectral images:

```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 20 \
    --sample_method gaussian
```

**Output:**
- `synthetic_samples_gaussian.png` - Visualization grid
- `synthetic_images/` directory with:
  - `synthetic_XXX.npy` - Multispectral data (5 bands)
  - `synthetic_XXX_rgb.png` - RGB visualizations

---

## Step 3: Evaluate Reconstruction Quality

Test how well the decoder reconstructs images:

```python
# Quick test script
python3 -c "
from multispectral_vit import MultispectralViT, MultispectralTileDataset
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

# Load models
encoder = MultispectralViT(...)
decoder = ReconstructionDecoder(...)
autoencoder = MultispectralAutoencoder(encoder, decoder)

# Load a few test images
dataset = MultispectralTileDataset('/mnt/22tb-hdd/.../bishop')
dataloader = DataLoader(dataset, batch_size=2)

# Test reconstruction
for images, _ in dataloader:
    reconstructed = autoencoder(images)
    mse = F.mse_loss(reconstructed, images)
    print(f'Reconstruction MSE: {mse.item():.6f}')
    break
"
```

---

## Step 4: Visualize Reconstruction Examples

Create comparison visualizations:

```python
# visualize_reconstruction.py
import torch
import matplotlib.pyplot as plt
from multispectral_vit import MultispectralViT, MultispectralTileDataset
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder
from torch.utils.data import DataLoader

# Load model
encoder = MultispectralViT(...)
decoder = ReconstructionDecoder(...)
autoencoder = MultispectralAutoencoder(encoder, decoder)

# Load some images
dataset = MultispectralTileDataset('/mnt/22tb-hdd/.../bishop')
dataloader = DataLoader(dataset, batch_size=4)

# Get batch
images, paths = next(iter(dataloader))

# Reconstruct
reconstructed = autoencoder(images)

# Visualize
fig, axes = plt.subplots(4, 2, figsize=(10, 20))
for i in range(4):
    # Original
    rgb_orig = visualize_rgb_band(images[i])
    axes[i, 0].imshow(rgb_orig)
    axes[i, 0].set_title('Original')
    
    # Reconstructed
    rgb_recon = visualize_rgb_band(reconstructed[i])
    axes[i, 1].imshow(rgb_recon)
    axes[i, 1].set_title('Reconstructed')

plt.savefig('reconstruction_comparison.png')
```

---

## Step 5: Generate Different Types of Synthetic Samples

### Random Samples:
```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --num_samples 50 \
    --sample_method gaussian
```

### Interpolated Samples (between two images):
```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --interpolate 0 100 \
    --interp_steps 15
```

### Uniform Distribution Samples:
```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --num_samples 20 \
    --sample_method uniform
```

---

## Step 6: Use Synthetic Images for Data Augmentation

Use generated images to augment your training data:

```python
# augment_dataset.py
import numpy as np
from pathlib import Path

# Load synthetic images
synthetic_dir = Path('synthetic_images')
synthetic_files = list(synthetic_dir.glob('synthetic_*.npy'))

# Load original dataset
original_dataset = MultispectralTileDataset('/mnt/22tb-hdd/.../bishop')

# Combine for training
augmented_dataset = ConcatDataset([original_dataset, synthetic_dataset])
```

---

## Step 7: Analyze Latent Space Quality

Check if the latent space learned meaningful representations:

```bash
python3 analyze_latents.py \
    --latent_file multispectral_latents.npy \
    --paths_file multispectral_tile_paths.txt \
    --visualize \
    --cluster \
    --n_clusters 10
```

---

## Step 8: Fine-tune for Specific Tasks

### For Classification:
```python
# Add classification head
classifier = nn.Linear(512, num_classes)
# Use encoder as feature extractor
features = encoder(images)
predictions = classifier(features)
```

### For Segmentation:
```python
# Use segmentation decoder
from multispectral_decoder import SegmentationDecoder
seg_decoder = SegmentationDecoder(num_classes=10)
segmentation = seg_decoder(encoder_output)
```

---

## Step 9: Save Generated Images to Disk

Export synthetic images for use in other tools:

```python
import rasterio
import numpy as np

# Load synthetic image
synthetic = np.load('synthetic_images/synthetic_000.npy')  # (5, H, W)

# Save as GeoTIFF (if you have georeferencing info)
with rasterio.open(
    'synthetic_000.tif',
    'w',
    driver='GTiff',
    height=960,
    width=960,
    count=5,
    dtype=synthetic.dtype
) as dst:
    for i in range(5):
        dst.write(synthetic[i], i+1)
```

---

## Step 10: Document Results

Create a summary of what you've learned:

```bash
# Check reconstruction quality
python3 -c "
import torch
from multispectral_vit import MultispectralViT, MultispectralTileDataset
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Load model
encoder = MultispectralViT(...)
decoder = ReconstructionDecoder(...)
autoencoder = MultispectralAutoencoder(encoder, decoder)

# Test on validation set
dataset = MultispectralTileDataset('/mnt/22tb-hdd/.../bishop')
dataloader = DataLoader(dataset, batch_size=8)

total_mse = 0
count = 0
for images, _ in dataloader:
    reconstructed = autoencoder(images)
    mse = F.mse_loss(reconstructed, images, reduction='sum')
    total_mse += mse.item()
    count += len(images)
    if count >= 100:  # Sample 100 images
        break

print(f'Average Reconstruction MSE: {total_mse/count:.6f}')
"
```

---

## Quick Reference Commands

### After Training:

```bash
# 1. Generate synthetic images
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --num_samples 20

# 2. Analyze what was learned
python3 analyze_latents.py --all

# 3. Test reconstruction quality
# (use Python script above)

# 4. Use for downstream tasks
# - Classification
# - Segmentation  
# - Anomaly detection
```

---

## Troubleshooting

### If reconstruction quality is poor:
- Train for more epochs: `--epochs 50`
- Adjust learning rate: `--lr 5e-5`
- Use larger batch size if memory allows
- Check if decoder is too small (increase hidden_dim)

### If synthetic images look random:
- Decoder may need more training
- Try different sampling methods (uniform vs gaussian)
- Check latent space distribution

### If out of memory:
- Reduce batch size: `--batch_size 2` or `1`
- Remove `--multi_gpu` flag
- Use gradient checkpointing

---

## Expected Outcomes

After successful training, you should be able to:

✅ **Generate synthetic multispectral images**  
✅ **Reconstruct images from latents**  
✅ **Interpolate between images**  
✅ **Use for data augmentation**  
✅ **Explore learned latent space**  

The decoder learns to map latent representations back to images, enabling you to create novel multispectral imagery!

