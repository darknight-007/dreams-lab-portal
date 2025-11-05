# Training Transformer Autoencoder on Zoom Level 23 GIS Tiles

## Quick Start

Train a transformer-based autoencoder on Zoom Level 23 GIS tiles (8,083 RGB images at 256x256):

```bash
cd /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts

python3 train_zoom23_autoencoder.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 50 \
    --device cuda
```

## Training Parameters

- **Dataset**: 8,083 images from Zoom Level 23
- **Image Size**: 256×256 pixels (matches tile size)
- **Channels**: RGB (3 channels)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Epochs**: 50 (recommended for good quality)
- **Learning Rate**: 1e-4 (default)

## Model Architecture

- **Encoder**: Vision Transformer (ViT)
  - Embedding dimension: 512
  - Patch size: 16×16
  - Transformer layers: 6
  - Attention heads: 8
  - Cross-band attention: Enabled

- **Decoder**: Memory-efficient reconstruction decoder
  - Progressive upsampling
  - Hidden dimension: 512

## Output Files

After training, the following files will be saved:

- `encoder_zoom23.pth` - Trained encoder
- `decoder_zoom23.pth` - Trained decoder
- `autoencoder_zoom23.pth` - Combined model
- `checkpoints_zoom23/` - Training checkpoints
  - `latest_checkpoint.pth` - Most recent checkpoint
  - `best_checkpoint.pth` - Best checkpoint (lowest loss)

## Monitoring Training

Training progress is printed to console:
- Batch-level loss every 50 batches
- Epoch-level average loss
- Best checkpoint saves automatically

## Expected Training Time

- GPU (RTX 3090/4090): ~2-4 hours for 50 epochs
- GPU (RTX 2080): ~4-6 hours for 50 epochs
- CPU: Not recommended (very slow)

## Next Steps After Training

1. **Generate Synthetic Samples**:
   ```bash
   python3 generate_synthetic_zoom23.py \
       --encoder_path encoder_zoom23.pth \
       --decoder_path decoder_zoom23.pth \
       --num_samples 20
   ```

2. **Test Reconstruction Quality**:
   ```bash
   python3 test_reconstruction.py \
       --encoder_path encoder_zoom23.pth \
       --decoder_path decoder_zoom23.pth \
       --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
       --zoom_level 23
   ```

## Tips

- **Smaller GPU Memory**: Reduce batch size to 8 or 4
- **Faster Training**: Reduce epochs to 20-30 for initial testing
- **Better Quality**: Increase epochs to 100+ for production
- **Multi-GPU**: Add `--multi_gpu` flag if you have multiple GPUs

## Troubleshooting

- **Out of Memory**: Reduce batch size or image size
- **Slow Training**: Check GPU utilization, ensure CUDA is working
- **Poor Reconstruction**: Increase epochs, check learning rate

