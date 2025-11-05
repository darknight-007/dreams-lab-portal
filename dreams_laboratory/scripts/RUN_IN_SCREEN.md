# Running Multispectral ViT in Screen Terminal

## Quick Start:

```bash
# Start a new screen session
screen -S multispectral_vit

# Navigate to script directory
cd /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts

# Run your script
python3 multispectral_vit.py \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --multi_gpu \
    --extract_latents \
    --batch_size 4 \
    --epochs 10
```

## Screen Commands:

### Starting Screen:
```bash
screen -S multispectral_vit    # Start new session with name
screen                        # Start anonymous session
```

### Detaching (leaves process running):
```
Press: Ctrl+A then D
```

### Reattaching:
```bash
screen -r multispectral_vit    # Reattach to named session
screen -r                      # List and attach to session
screen -ls                     # List all sessions
```

### Other Useful Commands:
```
Ctrl+A then C    # Create new window
Ctrl+A then N    # Next window
Ctrl+A then P    # Previous window
Ctrl+A then "    # List windows
Ctrl+A then X    # Close current window
exit              # Exit screen (or Ctrl+D)
```

## Recommended Workflow:

```bash
# 1. Start screen session
screen -S multispectral_training

# 2. Run your training script
cd /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts

python3 multispectral_vit.py \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --multi_gpu \
    --extract_latents \
    --batch_size 4 \
    --epochs 10 \
    --lr 1e-4

# 3. Detach: Press Ctrl+A then D

# 4. Later, reattach to check progress:
screen -r multispectral_training

# 5. Detach again: Ctrl+A then D
```

## Full Example with Logging:

```bash
# Start screen
screen -S multispectral_vit

# Run with output logging
cd /home/jdas/dreams-lab-website-server/dreams_laboratory/scripts

python3 multispectral_vit.py \
    --tile_dir /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019/bishop \
    --multi_gpu \
    --extract_latents \
    --batch_size 4 \
    --epochs 10 2>&1 | tee training.log

# Detach: Ctrl+A then D
```

## Tips:

1. **Name your sessions**: Use `-S <name>` so you can easily find them later
2. **Check sessions**: Use `screen -ls` to see all running sessions
3. **Multiple sessions**: You can run multiple training jobs in separate screen sessions
4. **Logging**: Redirect output to a file for easy monitoring

## Alternative: tmux (if you prefer)

```bash
tmux new -s multispectral_vit
# Run your script
# Detach: Ctrl+B then D
# Reattach: tmux attach -t multispectral_vit
```


