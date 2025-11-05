# Transformer Encoding Process: Detailed Explanation

## Overview: The Complete Flow

```
Multispectral Image (5 bands, 960×960)
    ↓
[1] Patch Embedding (divide into patches)
    ↓
[2] Cross-Band Attention (learn spectral relationships)
    ↓
[3] Add CLS Token + Positional Encoding
    ↓
[4] Transformer Encoder (6 layers of self-attention)
    ↓
[5] Final Layer Norm
    ↓
Latent Representation (512-dimensional vector)
```

---

## Step 1: Patch Embedding 🔲

**Input**: `(B, 5, 960, 960)` - Batch of multispectral images  
**Output**: `(B, 3600, 512)` - Sequence of patch embeddings

### What Happens:

```python
# Image is divided into non-overlapping patches
patch_size = 16
num_patches = (960 // 16)² = 60 × 60 = 3600 patches

# Each patch contains:
# - 16 × 16 pixels
# - 5 spectral bands (Blue, Green, Red, RedEdge, NIR)
# - Total: 16 × 16 × 5 = 1,280 values per patch

# Convolution operation:
conv2d(in_channels=5, out_channels=512, kernel=16, stride=16)
# Projects each patch to a 512-dimensional embedding
```

### Example:
- **Before**: Image with 960×960 pixels, 5 bands
- **After**: 3,600 patches, each represented as a 512-D vector

**Why patches?**
- Transformers work on sequences, not images
- Patches = "words" in the image "sentence"
- Each patch is a token in the sequence

---

## Step 2: Cross-Band Attention 🔗

**Input**: `(B, 3600, 512)` - Patch embeddings  
**Output**: `(B, 3600, 512)` - Spectrally-aware embeddings

### What Happens:

```python
# Multi-head self-attention between patches
# Each patch "attends" to all other patches

Query (Q) = Linear(patch_embeddings)
Key (K)   = Linear(patch_embeddings)
Value (V) = Linear(patch_embeddings)

# Attention scores: How much each patch relates to others
attention_scores = softmax(Q @ K^T / √d_k)

# Weighted combination of values
output = attention_scores @ V
```

### Why This Matters:

1. **Spectral Relationships**: Patches learn which spectral bands are important
2. **Spatial Context**: Each patch sees information from all patches
3. **Feature Discovery**: Automatically discovers relationships (e.g., NIR-RedEdge correlation)

### Example:
- Patch containing vegetation → High attention to NIR/RedEdge bands
- Patch containing rock → High attention to visible bands
- Patches learn to weight spectral information appropriately

---

## Step 3: CLS Token + Positional Encoding 📍

**Input**: `(B, 3600, 512)` - Patch embeddings  
**Output**: `(B, 3601, 512)` - Sequence with CLS token

### CLS Token:

```python
# Learnable token added at the beginning
CLS token: [512-dimensional vector]
# Position 0: [CLS], Position 1-N: [patch1, patch2, ..., patch3600]

# Purpose: Aggregate information from all patches
# After encoding, CLS token contains global image representation
```

### Positional Encoding:

```python
# Learnable positional embeddings
# Each patch position gets a unique 512-D vector
pos_embed[0] = [0, 0, ..., 0]        # CLS token (no position)
pos_embed[1] = learned_vector_1      # Top-left patch
pos_embed[2] = learned_vector_2      # Next patch
...
pos_embed[3601] = learned_vector_3600 # Bottom-right patch

# Added to embeddings: x = patches + positional_encoding
```

**Why needed?**
- Transformers have no inherent notion of position
- Positional encoding tells the model where each patch is spatially located
- Learns spatial relationships (adjacent patches, corners, etc.)

---

## Step 4: Transformer Encoder Layers 🔄

**Input**: `(B, 3601, 512)` - Sequence with CLS token  
**Output**: `(B, 3601, 512)` - Refined embeddings

### Architecture (6 layers):

Each `TransformerEncoderLayer` contains:

```
Input (x)
    ↓
[1] Multi-Head Self-Attention
    ├─ Query, Key, Value projections
    ├─ Scaled dot-product attention
    ├─ Attention weights: softmax(QK^T / √d)
    └─ Output: Attention(Q, K, V)
    ↓
[2] Residual Connection + Layer Norm
    x = LayerNorm(x + attention_output)
    ↓
[3] Feed-Forward Network (MLP)
    ├─ Linear(512 → 2048)  [mlp_ratio = 4.0]
    ├─ GELU activation
    └─ Linear(2048 → 512)
    ↓
[4] Residual Connection + Layer Norm
    x = LayerNorm(x + mlp_output)
    ↓
Output
```

### Layer-by-Layer Processing:

**Layer 1**: Learns basic spatial relationships
- Patches learn to attend to nearby patches
- Low-level features emerge

**Layer 2-3**: Intermediate features
- More complex relationships
- Patterns across larger regions

**Layer 4-5**: High-level features
- Complex spatial-spectral patterns
- Geological structures emerge

**Layer 6**: Final refinement
- Global context integration
- CLS token aggregates all information

### Multi-Head Attention Details:

```python
num_heads = 8
head_dim = 512 // 8 = 64

# Each head attends to different aspects:
Head 1: Spatial relationships (nearby patches)
Head 2: Spectral relationships (band correlations)
Head 3: Edge detection
Head 4: Texture patterns
Head 5: Large-scale structures
Head 6: Small-scale details
Head 7: Cross-scale relationships
Head 8: Global context

# All heads concatenated → 512-D output
```

---

## Step 5: Final Layer Normalization 📏

**Input**: `(B, 3601, 512)` - From transformer encoder  
**Output**: `(B, 3601, 512)` - Normalized embeddings

```python
# Normalizes across the embedding dimension
# Stabilizes training
# Ensures consistent scale
```

---

## Step 6: Extract CLS Token 🎯

**Input**: `(B, 3601, 512)` - All embeddings  
**Output**: `(B, 512)` - Global image representation

```python
# CLS token is at position 0
latent = embeddings[:, 0]  # Extract first token

# This 512-D vector represents:
# - Entire image content
# - All spatial-spectral relationships
# - Geological features
# - Global context
```

---

## Complete Example Flow:

### Input Image:
```
Image: 960×960 pixels, 5 bands
- Band 1: Blue (450-515nm)
- Band 2: Green (515-570nm)
- Band 3: Red (615-680nm)
- Band 4: RedEdge (705-745nm)
- Band 5: NIR (780-900nm)
```

### Step 1: Patch Embedding
```
960×960 image → 60×60 = 3600 patches
Each patch: 16×16 pixels × 5 bands
Each patch → 512-D embedding vector
Result: [3600, 512] sequence
```

### Step 2: Cross-Band Attention
```
Each patch attends to all patches
Learns: "This patch has high NIR, relates to vegetation patches"
Result: [3600, 512] with spectral awareness
```

### Step 3: Add CLS + Position
```
Sequence: [CLS, patch1, patch2, ..., patch3600]
Position: [0, 1, 2, ..., 3600]
Result: [3601, 512]
```

### Step 4: Transformer Encoder (6 layers)
```
Layer 1: Basic patterns
Layer 2: Local structures
Layer 3: Regional features
Layer 4: Complex relationships
Layer 5: Geological structures
Layer 6: Global integration
Result: [3601, 512] refined embeddings
```

### Step 5: Extract CLS
```
CLS token aggregates all information
Result: [512] latent representation
```

---

## Key Concepts:

### Self-Attention Mechanism:

```python
# For each patch, compute attention to all patches:
Attention(Patch_i) = Σ_j (attention_weight_ij × Patch_j)

# Attention weights show relationships:
# - High weight = patches are related
# - Low weight = patches are unrelated
```

### Residual Connections:

```python
# Help gradient flow
# Allow identity mapping
output = LayerNorm(input + transformation(input))
```

### Layer Normalization:

```python
# Normalizes across embedding dimension
# Stabilizes training
# Ensures consistent scale across batches
```

---

## Mathematical Formulation:

### Attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q, K, V: Query, Key, Value matrices
- d_k: dimension of key (64 in this case)
- √d_k: scaling factor
```

### Multi-Head Attention:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

Where:
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Feed-Forward Network:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
# In this case: GELU activation instead of ReLU
```

---

## Why This Works for Multispectral Images:

1. **Spectral Information**: Cross-band attention learns band relationships
2. **Spatial Context**: Self-attention captures spatial patterns
3. **Multi-Scale**: Different layers capture different scales
4. **Global Context**: CLS token aggregates everything
5. **Learnable**: Adapts to geological features automatically

---

## Computational Complexity:

- **Patches**: O(N) where N = 3600 patches
- **Attention**: O(N²) - each patch attends to all patches
- **Total**: O(L × N²) where L = 6 layers
- **Memory**: Stores attention matrices for all heads

---

## Visualization of Attention Flow:

```
Patch Embeddings
    ↓
[Cross-Band Attention]
    ├─ Patch 1 ←→ Patch 2 (spectral similarity)
    ├─ Patch 1 ←→ Patch 100 (spatial proximity)
    └─ Patch 1 ←→ Patch 2000 (global context)
    ↓
[Transformer Layer 1]
    └─ Local relationships
    ↓
[Transformer Layer 2]
    └─ Regional patterns
    ↓
...
    ↓
[Transformer Layer 6]
    └─ Global integration
    ↓
CLS Token (contains everything)
```

---

## Training Dynamics:

1. **Early epochs**: Learns basic patterns
2. **Mid epochs**: Develops spatial-spectral relationships
3. **Late epochs**: Refines geological feature representations
4. **Final**: CLS token encodes rich semantic information

The transformer encoder progressively builds more complex representations by combining information across patches, bands, and scales!


