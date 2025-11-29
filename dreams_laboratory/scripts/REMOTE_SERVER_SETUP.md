# Setting Up Training on Remote Server (192.168.0.232)

## Prerequisites

**Remote Server:**
- IP: 192.168.0.232
- GPUs: 2× Titan RTX (24GB each)
- Network: Same local network

## Step 1: Transfer Files to Remote Server

### Option A: Using rsync (Recommended)

```bash
# From your current machine
cd /home/jdas/dreams-lab-website-server

# Transfer all necessary scripts
rsync -avz --progress \
    dreams_laboratory/scripts/multispectral_vit.py \
    dreams_laboratory/scripts/multispectral_decoder.py \
    dreams_laboratory/scripts/train_zoom23_autoencoder.py \
    dreams_laboratory/scripts/test_reconstruction_zoom23.py \
    dreams_laboratory/scripts/extract_latents_zoom23.py \
    dreams_laboratory/scripts/generate_synthetic_zoom23.py \
    dreams_laboratory/scripts/analyze_tile_folder.py \
    dreams_laboratory/scripts/ZOOM23_TRAINING_GUIDE.md \
    dreams_laboratory/scripts/POST_TRAINING_WORKFLOW.md \
    dreams_laboratory/scripts/TITAN_RTX_TRAINING_GUIDE.md \
    user@192.168.0.232:/path/to/training/directory/
```

### Option B: Using scp

```bash
# Create a directory for scripts on remote server
ssh user@192.168.0.232 "mkdir -p ~/zoom23_training/scripts"

# Transfer files
scp dreams_laboratory/scripts/multispectral_vit.py \
    dreams_laboratory/scripts/multispectral_decoder.py \
    dreams_laboratory/scripts/train_zoom23_autoencoder.py \
    dreams_laboratory/scripts/test_reconstruction_zoom23.py \
    dreams_laboratory/scripts/extract_latents_zoom23.py \
    dreams_laboratory/scripts/generate_synthetic_zoom23.py \
    dreams_laboratory/scripts/analyze_tile_folder.py \
    user@192.168.0.232:~/zoom23_training/scripts/
```

### Option C: Using Git (If repo is accessible)

```bash
# On remote server
ssh user@192.168.0.232
cd ~/zoom23_training
git clone https://github.com/darknight-007/dreams-lab-portal.git
cd dreams-lab-portal/dreams_laboratory/scripts
```

## Step 2: Set Up Environment on Remote Server

### SSH into Remote Server

```bash
ssh user@192.168.0.232
```

### Check GPU Availability

```bash
nvidia-smi
# Should show 2× Titan RTX GPUs
```

### Create Python Environment

```bash
# Create virtual environment
python3 -m venv ~/zoom23_training/venv
source ~/zoom23_training/venv/bin/activate

# Install dependencies
pip install torch torchvision numpy pillow matplotlib rasterio scikit-learn
```

### Verify PyTorch GPU Support

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

## Step 3: Transfer Dataset (If Needed)

### Option A: Dataset Already on Remote Server

If your tiles are already accessible:
```bash
# Just verify the path
ls /path/to/tiles/raw/23/
```

### Option B: Transfer Dataset via Network Mount

```bash
# Mount remote directory if accessible
# Or use rsync for one-time transfer
rsync -avz --progress \
    /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw/ \
    user@192.168.0.232:/path/to/tiles/raw/
```

### Option C: Use Network Path

If the dataset is on a shared network drive:
```bash
# Mount the network share
ssh user@192.168.0.232
sudo mount -t nfs /path/to/shared/drive /mnt/tiles
```

## Step 4: Run Training on Remote Server

### Basic Training (512×512 with 2 GPUs)

```bash
ssh user@192.168.0.232
cd ~/zoom23_training/scripts
source ~/zoom23_training/venv/bin/activate

python3 train_zoom23_autoencoder.py \
    --tile_dir /path/to/tiles/raw \
    --zoom_level 23 \
    --img_size 512 \
    --patch_size 16 \
    --embed_dim 512 \
    --num_heads 8 \
    --num_layers 6 \
    --batch_size 16 \
    --epochs 50 \
    --device cuda \
    --multi_gpu \
    --checkpoint_dir checkpoints_zoom23_512
```

### Run in Screen/Tmux (Recommended)

**Using Screen:**
```bash
ssh user@192.168.0.232
screen -S zoom23_training

# Run training command
python3 train_zoom23_autoencoder.py ...

# Detach: Ctrl+A then D
# Reattach: screen -r zoom23_training
```

**Using Tmux:**
```bash
ssh user@192.168.0.232
tmux new -s zoom23_training

# Run training command
python3 train_zoom23_autoencoder.py ...

# Detach: Ctrl+B then D
# Reattach: tmux attach -t zoom23_training
```

## Step 5: Monitor Training Remotely

### Check GPU Usage

```bash
# On remote server
watch -n 1 nvidia-smi

# Or from your local machine
ssh user@192.168.0.232 "nvidia-smi"
```

### Check Training Progress

```bash
# View latest checkpoint
ssh user@192.168.0.232 "tail -f ~/zoom23_training/scripts/training.log"

# Or check checkpoint directory
ssh user@192.168.0.232 "ls -lh ~/zoom23_training/scripts/checkpoints_zoom23_512/"
```

### Monitor Logs

```bash
# If training outputs to log file
ssh user@192.168.0.232 "tail -f ~/zoom23_training/scripts/training.log"
```

## Step 6: Transfer Results Back (Optional)

After training completes:

```bash
# From your local machine
rsync -avz --progress \
    user@192.168.0.232:~/zoom23_training/scripts/encoder_zoom23.pth \
    user@192.168.0.232:~/zoom23_training/scripts/decoder_zoom23.pth \
    user@192.168.0.232:~/zoom23_training/scripts/checkpoints_zoom23_512/ \
    ~/local/results/
```

## Quick Setup Script

Create this script on remote server:

```bash
#!/bin/bash
# setup_remote_training.sh

# Create directory structure
mkdir -p ~/zoom23_training/{scripts,results,checkpoints}

# Create virtual environment
python3 -m venv ~/zoom23_training/venv
source ~/zoom23_training/venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision numpy pillow matplotlib rasterio scikit-learn

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "Setup complete! Ready for training."
```

## Network Configuration

### Ensure Network Access

```bash
# Test connectivity
ping 192.168.0.232

# Test SSH
ssh user@192.168.0.232 "echo 'Connection successful'"

# Test GPU access over SSH
ssh user@192.168.0.232 "nvidia-smi"
```

### Firewall (if needed)

```bash
# On remote server, ensure SSH is open
sudo ufw allow 22/tcp
```

## Recommended Workflow

### 1. Initial Setup

```bash
# On your local machine
rsync -avz dreams_laboratory/scripts/*.py user@192.168.0.232:~/zoom23_training/scripts/

# SSH and setup
ssh user@192.168.0.232
cd ~/zoom23_training
source venv/bin/activate
pip install torch torchvision numpy pillow matplotlib rasterio
```

### 2. Start Training

```bash
# SSH and start screen/tmux
ssh user@192.168.0.232
screen -S training
cd ~/zoom23_training/scripts

# Run training
python3 train_zoom23_autoencoder.py \
    --img_size 512 \
    --batch_size 16 \
    --multi_gpu \
    --epochs 50

# Detach (Ctrl+A, D)
```

### 3. Monitor Progress

```bash
# From local machine
ssh user@192.168.0.232 "nvidia-smi"
ssh user@192.168.0.232 "tail -20 ~/zoom23_training/scripts/checkpoints_zoom23_512/training.log"
```

### 4. Retrieve Results

```bash
# After training completes
rsync -avz user@192.168.0.232:~/zoom23_training/scripts/*.pth ~/local/results/
```

## Troubleshooting

### Connection Issues

```bash
# Test SSH
ssh -v user@192.168.0.232

# Check if server is reachable
ping 192.168.0.232
```

### GPU Not Detected

```bash
# On remote server
nvidia-smi
# Check CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Permission Issues

```bash
# Ensure scripts are executable
chmod +x ~/zoom23_training/scripts/*.py
```

### Path Issues

```bash
# Verify tile directory exists
ssh user@192.168.0.232 "ls /path/to/tiles/raw/23/"
```

## Expected Timeline

- **File Transfer**: 5-10 minutes (depending on network speed)
- **Environment Setup**: 10-15 minutes
- **Training (512×512)**: 6-8 hours
- **Total**: ~7-9 hours

## Summary Commands

```bash
# 1. Transfer files
rsync -avz dreams_laboratory/scripts/*.py user@192.168.0.232:~/zoom23_training/scripts/

# 2. SSH and setup
ssh user@192.168.0.232
cd ~/zoom23_training && source venv/bin/activate

# 3. Start training
screen -S training
python3 scripts/train_zoom23_autoencoder.py --img_size 512 --batch_size 16 --multi_gpu

# 4. Monitor (from local)
ssh user@192.168.0.232 "nvidia-smi"
```



