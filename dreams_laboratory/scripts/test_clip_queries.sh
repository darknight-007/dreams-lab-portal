#!/bin/bash
# Test different types of CLIP queries to see what works best

cd /home/jdas/dreams-lab-portal/dreams_laboratory/scripts

echo "=== Testing Color-based Queries ==="
python3 vlm_clip_simple.py --mode search --query "white rock" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "dark grey stone" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "orange weathered surface" --output_dir clip_embeddings_zoom23

echo -e "\n=== Testing Texture-based Queries ==="
python3 vlm_clip_simple.py --mode search --query "smooth texture" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "rough fractured surface" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "crystalline sparkly texture" --output_dir clip_embeddings_zoom23

echo -e "\n=== Testing Rock Type Queries ==="
python3 vlm_clip_simple.py --mode search --query "granite" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "sandstone" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "volcanic rock" --output_dir clip_embeddings_zoom23

echo -e "\n=== Testing Feature-based Queries ==="
python3 vlm_clip_simple.py --mode search --query "visible crystals" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "layered structure" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "porous surface" --output_dir clip_embeddings_zoom23

echo -e "\n=== Testing Combined Queries ==="
python3 vlm_clip_simple.py --mode search --query "grey rock with white crystals" --output_dir clip_embeddings_zoom23
python3 vlm_clip_simple.py --mode search --query "rough orange weathered stone" --output_dir clip_embeddings_zoom23

echo -e "\n✓ All test queries complete!"
echo "Check clip_embeddings_zoom23/search_results/ for results"

