# Execution Time Estimate for sample_from_latent.py

## Command Breakdown:
```bash
python3 sample_from_latent.py \
    --model_path multispectral_vit.pth \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --sample_random 20 \
    --sample_method gaussian
```

## What Happens:

### Step 1: Load Model (~5-10 seconds)
- Loads multispectral_vit.pth
- Moves model to GPU (CUDA available)
- Model initialization

### Step 2: Load Latents (~1-2 seconds)
- Loads multispectral_latents.npy (14MB, 6820 tiles × 512-dim)
- Loads paths file
- **FAST** - Already pre-computed!

### Step 3: Load Dataset (~2-5 seconds)
- Creates dataset object
- Scans tile directory (doesn't load images yet)

### Step 4: Generate Random Samples (~0.1 seconds)
- Samples 20 random points from Gaussian distribution
- Very fast numpy operation

### Step 5: Find Nearest Neighbors (~5-15 seconds)
- Builds NearestNeighbors index on 6820 vectors
- Finds k=5 neighbors for each of 20 samples
- Most time-consuming step (but still fast)

### Step 6: Load 20 Images (~10-30 seconds)
- Loads 20 TIFF files (960×960×5 bands each)
- Uses rasterio to read multispectral data
- **SLOWEST STEP** - I/O bound

### Step 7: Create Visualization (~2-5 seconds)
- Converts multispectral to RGB
- Creates matplotlib figure
- Saves PNG file

---

## Total Time Estimate:

### **Best Case (GPU, fast storage)**: 
- **30-60 seconds**

### **Typical Case**:
- **45-90 seconds** (1-1.5 minutes)

### **Worst Case (slow I/O)**:
- **2-3 minutes** (if loading from slow external drive)

---

## Breakdown by Component:

| Step | Time | Notes |
|------|------|-------|
| Load Model | 5-10s | GPU transfer |
| Load Latents | 1-2s | Pre-computed |
| Load Dataset | 2-5s | Directory scan |
| Random Sampling | <1s | Fast numpy |
| Nearest Neighbors | 5-15s | sklearn indexing |
| **Load 20 Images** | **10-30s** | **I/O bound** |
| Visualization | 2-5s | Matplotlib |
| **TOTAL** | **30-90s** | **Mostly I/O** |

---

## Factors Affecting Speed:

### ✅ Fast Factors:
- ✅ Latents already pre-computed (saves ~5-10 minutes!)
- ✅ CUDA available (model loading faster)
- ✅ Only 20 samples (not 100+)

### ⚠️ Slow Factors:
- ⚠️ Loading TIFF files from external drive (`/mnt/22tb-hdd`)
- ⚠️ Multispectral TIFFs are large (5 bands × 960×960)
- ⚠️ Each TIFF needs rasterio processing

---

## Optimization Tips:

### If Running Slowly:

1. **Use pre-computed latents** (you're already doing this! ✅)
   ```bash
   --latent_file multispectral_latents.npy \
   --paths_file multispectral_tile_paths.txt
   ```

2. **Reduce batch size** (if loading from network):
   ```bash
   --batch_size 1  # Smaller batches
   ```

3. **Load fewer samples**:
   ```bash
   --sample_random 10  # Instead of 20
   ```

4. **Use SSD instead of HDD** (if possible)

---

## What to Expect:

### Progress Indicators:
```
Loading model...                    [5-10s]
Loading pre-computed latents...      [1-2s]
Loaded 6820 latent representations
Generating 20 random samples...      [<1s]
Finding nearest neighbors...        [5-15s]
Loading images...                    [10-30s]
  - Loading IMG_0188_5.tif
  - Loading IMG_0125_1.tif
  ...
Creating visualization...            [2-5s]
Saved visualization to: random_samples_gaussian.png
```

---

## Expected Output:

- **File**: `random_samples_gaussian.png`
- **Content**: 4×5 grid of images (20 samples)
- **Size**: ~2-5 MB PNG file

---

## If It Takes Longer Than Expected:

**Check**:
1. **I/O speed**: `iostat -x 1` (check disk utilization)
2. **Network mount**: If `/mnt/22tb-hdd` is network mounted, it will be slower
3. **Concurrent processes**: Other processes reading from the drive

**If > 5 minutes**: Something is wrong, check:
- Drive mount issues
- Corrupted TIFF files
- Memory issues

---

## Quick Test:

Run this to test just the slow parts:
```bash
time python3 sample_from_latent.py \
    --model_path multispectral_vit.pth \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --latent_file multispectral_latents.npy \
    --paths_file multispectral_tile_paths.txt \
    --sample_random 5  # Smaller test
```

---

## Summary:

**Expected Time**: **1-2 minutes** typically

**If > 3 minutes**: Check I/O performance

**If < 30 seconds**: Great! Everything is working optimally

The bottleneck is loading 20 multispectral TIFF files from the external drive, not the computation.

