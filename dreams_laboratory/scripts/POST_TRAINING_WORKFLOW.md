# Post-Training Workflow: Zoom Level 23 Autoencoder

After training completes, follow these steps to evaluate, use, and generate synthetic samples from your trained model.

## Step 1: Verify Training Completed Successfully

Check that training completed and models were saved:

```bash
cd /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts

# Check if model files exist
ls -lh encoder_zoom23.pth decoder_zoom23.pth autoencoder_zoom23.pth

# Check training checkpoints
ls -lh checkpoints_zoom23/
```

**Expected Output Files:**
- `encoder_zoom23.pth` - Trained encoder (~50-100 MB)
- `decoder_zoom23.pth` - Trained decoder (~20-50 MB)
- `autoencoder_zoom23.pth` - Combined model
- `checkpoints_zoom23/latest_checkpoint.pth` - Latest checkpoint
- `checkpoints_zoom23/best_checkpoint.pth` - Best checkpoint (lowest loss)

## Step 2: Test Reconstruction Quality

Verify the model can reconstruct images accurately:

```bash
python3 test_reconstruction_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23 \
    --num_samples 10
```

**What to look for:**
- Low MSE loss (< 0.01 is good)
- Visually similar reconstructions
- No obvious artifacts or blurring

## Step 3: Extract Latent Representations

Extract latent representations from all training images (needed for synthetic generation):

```bash
python3 extract_latents_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23 \
    --output latents_zoom23.npy
```

**Output:**
- `latents_zoom23.npy` - Array of shape (8083, 512) containing all latent representations
- `latent_paths_zoom23.txt` - Paths corresponding to each latent

## Step 4: Generate Synthetic Samples

Generate synthetic RGB images from random latents:

```bash
python3 generate_synthetic_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 20 \
    --sample_method gaussian \
    --output_dir synthetic_zoom23
```

**Sampling Methods:**
- `gaussian` - Sample from Gaussian distribution (recommended)
- `uniform` - Uniform sampling within data bounds
- `sphere` - Sample from sphere with matching radius
- `pca` - PCA-based sampling (more structured)
- `real` - Use actual latents from dataset (baseline)

**Output:**
- `synthetic_samples_gaussian.png` - Visualization grid
- `synthetic_zoom23/synthetic_XXX.npy` - RGB image arrays (3, 256, 256)
- `synthetic_zoom23/synthetic_XXX_rgb.png` - PNG visualizations

## Step 5: Visualize Results

Create comprehensive visualizations:

```bash
python3 visualize_results_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23 \
    --output_dir results_zoom23
```

**Creates:**
- Reconstruction comparisons (original vs reconstructed)
- Latent space visualization (PCA/t-SNE)
- Synthetic sample gallery
- Interpolation examples
- Quality metrics report

## Step 6: Evaluate Model Metrics

Calculate quantitative metrics:

```bash
python3 evaluate_model_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23
```

**Metrics Calculated:**
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Perceptual Loss (LPIPS)
- Latent space statistics

## Step 7: Use Model for Downstream Tasks

### A. Feature Extraction

Use the encoder as a feature extractor:

```python
from multispectral_vit import MultispectralViT
import torch

# Load encoder
encoder = MultispectralViT(
    img_size=256,
    patch_size=16,
    in_channels=3,
    embed_dim=512
)
encoder.load_state_dict(torch.load('encoder_zoom23.pth')['model_state_dict'])
encoder.eval()

# Extract features
with torch.no_grad():
    features = encoder(image_batch)  # (B, 512)
```

### B. Image Similarity Search

Find similar images using latent representations:

```bash
python3 similarity_search_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --query_image path/to/query.png \
    --latent_file latents_zoom23.npy \
    --top_k 10
```

### C. Image Interpolation

Create smooth transitions between images:

```bash
python3 interpolate_images_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --image1 path/to/image1.png \
    --image2 path/to/image2.png \
    --num_steps 10 \
    --output interpolation.gif
```

## Step 8: Fine-Tuning (Optional)

If reconstruction quality is poor, consider:

1. **Train Longer**: Resume from checkpoint
   ```bash
   python3 train_zoom23_autoencoder.py \
       --encoder_path checkpoints_zoom23/best_checkpoint.pth \
       --epochs 100  # Continue training
   ```

2. **Adjust Hyperparameters**:
   - Increase `embed_dim` (512 → 768)
   - Increase `num_layers` (6 → 12)
   - Decrease learning rate (1e-4 → 5e-5)

3. **Data Augmentation**: Add augmentation during training

## Step 9: Export for Production

Export model for deployment:

```bash
# Export to ONNX format
python3 export_to_onnx_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --output encoder_zoom23.onnx decoder_zoom23.onnx
```

## Step 10: Documentation

Document your model:

```bash
python3 create_model_card_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --output model_card_zoom23.md
```

**Model Card Includes:**
- Architecture details
- Training hyperparameters
- Performance metrics
- Dataset information
- Usage examples
- Limitations

## Quick Reference Commands

```bash
# 1. Test reconstruction
python3 test_reconstruction_zoom23.py --encoder_path encoder_zoom23.pth --decoder_path decoder_zoom23.pth --zoom_level 23

# 2. Extract latents
python3 extract_latents_zoom23.py --encoder_path encoder_zoom23.pth --zoom_level 23

# 3. Generate synthetic samples
python3 generate_synthetic_zoom23.py --encoder_path encoder_zoom23.pth --decoder_path decoder_zoom23.pth --latent_file latents_zoom23.npy --num_samples 20

# 4. Visualize results
python3 visualize_results_zoom23.py --encoder_path encoder_zoom23.pth --decoder_path decoder_zoom23.pth
```

## Troubleshooting

**Poor Reconstruction Quality:**
- Check training loss converged
- Verify model files loaded correctly
- Test with training images first
- Consider training longer

**Synthetic Images Look Wrong:**
- Verify latent file was created correctly
- Try different sampling methods
- Check decoder was trained properly
- Use 'real' method first to verify decoder works

**Out of Memory:**
- Reduce batch size
- Use CPU for inference
- Process images in smaller batches

## Next Steps

1. **Experiment with sampling**: Try different methods to find best results
2. **Create evaluation dataset**: Separate test set for quantitative evaluation
3. **Fine-tune on specific features**: Train on filtered subset for specific applications
4. **Build application**: Use model for GIS tile generation, data augmentation, etc.

