#!/usr/bin/env python3
"""
Pre-download VLM models for offline use.

Downloads models to ~/.cache/huggingface/ so they're ready when needed.
"""

from transformers import CLIPModel, CLIPProcessor
import argparse


def download_clip_models():
    """Download CLIP models."""
    
    models = {
        'base': 'openai/clip-vit-base-patch32',
        'large': 'openai/clip-vit-large-patch14',
        'huge': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    }
    
    print("Downloading CLIP models...")
    print("Models will be cached in: ~/.cache/huggingface/")
    print()
    
    for name, model_id in models.items():
        print(f"Downloading {name}: {model_id}")
        try:
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id)
            size = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✓ Downloaded {size:.0f}M parameters")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        print()


def download_llava_models():
    """Download LLaVA models (larger, vision-language LLMs)."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    
    models = {
        'llava-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-13b': 'llava-hf/llava-1.5-13b-hf'
    }
    
    print("Downloading LLaVA models (this will take a while)...")
    print()
    
    for name, model_id in models.items():
        print(f"Downloading {name}: {model_id}")
        print("  Warning: This is a large model (7-26 GB)")
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            # Just download, don't load to memory
            print(f"  ✓ Downloaded processor")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        print()


def check_disk_space():
    """Check available disk space."""
    import shutil
    from pathlib import Path
    
    home = Path.home()
    total, used, free = shutil.disk_usage(home)
    
    print("Disk Space Check:")
    print(f"  Total: {total / (2**30):.1f} GB")
    print(f"  Used:  {used / (2**30):.1f} GB")
    print(f"  Free:  {free / (2**30):.1f} GB")
    print()
    
    if free < 10 * (2**30):  # Less than 10GB free
        print("⚠️  Warning: Low disk space. Models may not download.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Download VLM models')
    parser.add_argument('--clip', action='store_true',
                       help='Download CLIP models')
    parser.add_argument('--llava', action='store_true',
                       help='Download LLaVA models (large!)')
    parser.add_argument('--all', action='store_true',
                       help='Download all models')
    parser.add_argument('--check', action='store_true',
                       help='Just check disk space')
    
    args = parser.parse_args()
    
    # Check disk space
    if not check_disk_space():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    if args.check:
        return
    
    if args.all or args.clip:
        download_clip_models()
    
    if args.all or args.llava:
        response = input("Download LLaVA models? They are 7-26 GB each. (y/n): ")
        if response.lower() == 'y':
            download_llava_models()
    
    if not (args.clip or args.llava or args.all):
        print("No models specified. Use --clip, --llava, or --all")
        print("\nRecommended: Start with CLIP")
        print("  python3 download_vlm_models.py --clip")


if __name__ == '__main__':
    main()

