# How to Generate Synthetic Multispectral Images

## Step 1: Train the Decoder

You need to train an autoencoder (encoder + decoder) first:

```bash
# Train encoder-decoder for reconstruction
python3 train_autoencoder.py \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --encoder_path multispectral_vit.pth \
    --multi_gpu \
    --batch_size 4 \
    --epochs 20
```

**OR** create a training script:

```python
from multispectral_vit import MultispectralViT
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder

# Create model
encoder = MultispectralViT(...)
decoder = ReconstructionDecoder(...)
autoencoder = MultispectralAutoencoder(encoder, decoder)

# Train on reconstruction task
# (MSE loss between original and reconstructed)
```

## Step 2: Generate Synthetic Images

Once you have a trained decoder:

```bash
python3 generate_synthetic.py \
    --encoder_path multispectral_vit.pth \
    --decoder_path decoder.pth \
    --latent_file multispectral_latents.npy \
    --num_samples 20 \
    --sample_method gaussian
```

## What Gets Generated:

1. **Random Synthetic Images**: Samples random points in latent space, decodes to images
2. **Interpolated Images**: Smooth transitions between two real images
3. **Novel Combinations**: Creates images that don't exist in your dataset

## Output:

- `synthetic_samples_gaussian.png`: Visualization grid
- `synthetic_images/synthetic_XXX.npy`: Multispectral data (5 bands)
- `synthetic_images/synthetic_XXX_rgb.png`: RGB visualizations

## Important Notes:

⚠️ **The decoder needs to be trained first!** 

Without training, generated images will be random noise.

The decoder learns to map latent representations back to images, so it needs to see many examples during training.

