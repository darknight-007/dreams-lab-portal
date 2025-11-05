# How Encoder-Decoder Works: Complete Guide

## Overview: The Encoder-Decoder Architecture

```
Input Image
    ↓
[ENCODER] ──────────→ Latent Representation
                         ↓
                    [DECODER]
                         ↓
                    Output (Task-specific)
```

The encoder compresses the input into a latent representation, and the decoder expands it back into the desired output format.

---

## How the Encoder Outputs are Used

### Current Encoder Output Options:

```python
# Option 1: CLS token only (single vector)
latent = encoder(x)  # (B, 512)

# Option 2: All patch embeddings
all_embeddings = encoder(x, return_all=True)  # (B, 3601, 512)
```

---

## Decoder Types

### 1. **Reconstruction Decoder** (Autoencoder)

**Purpose**: Reconstruct images from latent representations

**Flow**:
```
CLS Token (512-D) → MLP → Patches → Image Reconstruction
```

**Architecture**:
```
Encoder CLS Token (512-D)
    ↓
[Linear(512 → 1024)]
    ↓
[LayerNorm + GELU]
    ↓
[Linear(1024 → 2048)]
    ↓
[LayerNorm + GELU]
    ↓
[Linear(2048 → 3600×16×16×5)]
    ↓
Reshape to (B, 5, 960, 960)
```

**Use Cases**:
- Self-supervised learning
- Denoising
- Compression
- Anomaly detection (reconstruction error)

**Training**:
```python
# Loss: MSE between original and reconstructed
reconstructed = decoder(encoder(x))
loss = mse_loss(reconstructed, x)
```

---

### 2. **Segmentation Decoder** (Dense Prediction)

**Purpose**: Pixel-level classification

**Flow**:
```
All Patch Embeddings → Upsampling → Classification Head → Segmentation Mask
```

**Architecture**:
```
Patch Embeddings (B, 3600, 512)
    ↓
Reshape to (B, 512, 60, 60)  # Spatial grid
    ↓
[Upsample 2x] → (B, 256, 120, 120)
    ↓
[Conv + BatchNorm + ReLU]
    ↓
[Upsample 2x] → (B, 128, 240, 240)
    ↓
[Conv + BatchNorm + ReLU]
    ↓
[Upsample 2x] → (B, 64, 480, 480)
    ↓
[Conv + BatchNorm + ReLU]
    ↓
[Upsample 2x] → (B, 64, 960, 960)
    ↓
[Classification Head] → (B, num_classes, 960, 960)
```

**Use Cases**:
- Geological feature segmentation
- Rock type classification
- Vegetation mapping
- Land cover classification

**Training**:
```python
# Loss: Cross-entropy per pixel
segmentation = decoder(encoder(x, return_all=True))
loss = cross_entropy(segmentation, ground_truth_masks)
```

---

### 3. **Transformer Decoder** (Sequence Generation)

**Purpose**: Generate sequences or translate between formats

**Flow**:
```
Encoder Output → Transformer Decoder → Generated Sequence
```

**Architecture**:
```
Encoder Patches (B, 3600, 512)
    ↓
[Transformer Decoder Layers]
    ├─ Self-Attention (within decoder)
    ├─ Cross-Attention (to encoder)
    └─ Feed-Forward
    ↓
Decoded Patches (B, 3600, 512)
    ↓
[Project to Pixels] → (B, 5, 960, 960)
```

**Use Cases**:
- Image-to-image translation
- Super-resolution
- Style transfer
- Domain adaptation

---

## Complete Training Example

### Autoencoder Training:

```python
from multispectral_vit import MultispectralViT
from multispectral_decoder import ReconstructionDecoder, MultispectralAutoencoder

# Create model
encoder = MultispectralViT(...)
decoder = ReconstructionDecoder(...)
autoencoder = MultispectralAutoencoder(encoder, decoder)

# Training loop
for images, _ in dataloader:
    # Forward pass
    reconstructed = autoencoder(images)
    
    # Loss
    loss = F.mse_loss(reconstructed, images)
    
    # Backward
    loss.backward()
    optimizer.step()
```

### Segmentation Training:

```python
from multispectral_decoder import SegmentationDecoder, MultispectralSegmentationModel

# Create model
encoder = MultispectralViT(...)
decoder = SegmentationDecoder(num_classes=10)
model = MultispectralSegmentationModel(encoder, decoder)

# Training loop
for images, masks in dataloader:
    # Forward pass
    segmentation = model(images)
    
    # Loss
    loss = F.cross_entropy(segmentation, masks)
    
    # Backward
    loss.backward()
    optimizer.step()
```

---

## Information Flow

### Encoder → Decoder:

```
┌─────────────────────────────────────────┐
│           ENCODER                       │
│                                         │
│  Image (5×960×960)                      │
│      ↓                                  │
│  [Patch Embedding]                     │
│      ↓                                  │
│  [Cross-Band Attention]                │
│      ↓                                  │
│  [Transformer Encoder]                 │
│      ↓                                  │
│  ┌─────────────────────┐              │
│  │ CLS Token (512-D)   │ ←──────────┐ │
│  │ Patch 1 (512-D)     │            │ │
│  │ Patch 2 (512-D)     │            │ │
│  │ ...                 │            │ │
│  │ Patch 3600 (512-D)  │            │ │
│  └─────────────────────┘            │ │
│         │                            │ │
└─────────┼────────────────────────────┼─┘
          │                            │
          ↓                            ↓
┌─────────────────────────────────────────┐
│           DECODER                       │
│                                         │
│  For Reconstruction:                    │
│  CLS Token → MLP → Patches → Image     │
│                                         │
│  For Segmentation:                      │
│  All Patches → Upsample → Classes     │
│                                         │
│  Output: Task-specific prediction       │
└─────────────────────────────────────────┘
```

---

## Key Differences

| Decoder Type | Input | Output | Task |
|--------------|-------|--------|------|
| **Reconstruction** | CLS token (512-D) | Image (5×960×960) | Reconstruct input |
| **Segmentation** | All patches (3600×512) | Mask (classes×960×960) | Pixel classification |
| **Transformer** | All patches (3600×512) | Sequence/Image | Generation/Translation |

---

## Why This Architecture Works

1. **Encoder learns rich representations**: Captures spatial-spectral relationships
2. **Decoder specializes**: Each decoder type optimized for specific task
3. **Transfer learning**: Pre-trained encoder can be reused with different decoders
4. **Modular**: Encoder and decoder can be swapped independently

---

## Advanced: Multi-Task Learning

You can use the same encoder with multiple decoders:

```python
encoder = MultispectralViT(...)
reconstruction_decoder = ReconstructionDecoder(...)
segmentation_decoder = SegmentationDecoder(...)

# Encode once
encoder_output = encoder(x, return_all=True)
latent = encoder(x)

# Use for different tasks
reconstructed = reconstruction_decoder(latent)
segmentation = segmentation_decoder(encoder_output)
```

---

## Summary

**Encoder**: Compresses image → Latent representation
- CLS token: Global image representation
- Patch embeddings: Local spatial-spectral features

**Decoder**: Expands latent → Task-specific output
- Reconstruction: CLS → Image
- Segmentation: Patches → Mask
- Generation: Patches → Sequence

The encoder provides rich features, and the decoder adapts them to the specific task!


