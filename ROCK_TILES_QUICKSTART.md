# Rock Tiles Quick Start Guide

> **Note**: `$PROJECT_ROOT` = your project directory, `/path/to/rock-tiles/raw` = your tile storage

## 🚀 Quick Access

### Direct Tile Access
```
https://rocks.deepgis.org/{z}/{x}/{y}.png
```

**Example**: https://rocks.deepgis.org/23/5892/12745.png

### TileServer GL (with viewer)
```
https://mbtiles.deepgis.org/data/rock_tiles_deepgis/
```

## 📋 Setup Checklist

- [x] Direct XYZ tile serving via `rocks.deepgis.org`
- [ ] Convert tiles to MBTiles format
- [ ] Add to TileServer GL
- [ ] Test access

## 🔧 Complete Setup (First Time)

### 1. Convert Tiles to MBTiles

```bash
cd $PROJECT_ROOT/dreams_laboratory/scripts

python3 convert_rocks_to_mbtiles.py \
    --tiles_dir /path/to/rock-tiles/raw \
    --output /tmp/rock_tiles_deepgis.mbtiles \
    --name "DeepGIS Rock Tiles" \
    --description "High-resolution rock surface imagery (zoom 15-23)"
```

**Time**: 5-10 minutes  
**Output**: ~1.2 GB MBTiles file

### 2. Install to TileServer

```bash
# Move to tileserver directory
mv /tmp/rock_tiles_deepgis.mbtiles \
   $PROJECT_ROOT/deepgis-xr/data/

# Set permissions
chmod 644 $PROJECT_ROOT/deepgis-xr/data/rock_tiles_deepgis.mbtiles
```

### 3. Restart TileServer

```bash
docker restart deepgis-xr_tileserver_1

# Verify it's running
docker ps | grep tileserver
```

### 4. Verify Access

Visit these URLs:
- Main TileServer: https://mbtiles.deepgis.org/
- Rock Tiles: https://mbtiles.deepgis.org/data/rock_tiles_deepgis/
- Interactive Viewer: https://mbtiles.deepgis.org/data/rock_tiles_deepgis/#18/33.78215/-111.26513

## 💻 Usage in Code

### Leaflet
```javascript
L.tileLayer('https://rocks.deepgis.org/{z}/{x}/{y}.png', {
    minZoom: 15,
    maxZoom: 23
}).addTo(map);
```

### Python
```python
import requests

# Fetch tile
z, x, y = 23, 5892, 12745
url = f'https://rocks.deepgis.org/{z}/{x}/{y}.png'
tile = requests.get(url).content
```

### CLIP/VLM Analysis
```python
from dreams_laboratory.scripts.vlm_clip_simple import SimpleCLIPEmbedder

clip = SimpleCLIPEmbedder()
embeddings = clip.encode_images(tile_paths)
results = clip.text_to_image_search("granite", embeddings, tile_paths)
```

## 📚 Full Documentation

See [deepgis-xr/ROCK_TILES_SETUP.md](deepgis-xr/ROCK_TILES_SETUP.md) for:
- Architecture details
- Advanced configuration
- Troubleshooting
- VLM/ML integration examples

## ✅ Status Check

```bash
# Check direct tiles
curl -I https://rocks.deepgis.org/23/5892/12745.png

# Check TileServer
curl https://mbtiles.deepgis.org/data/rock_tiles_deepgis.json

# Check container
docker ps | grep tileserver
```

## 🆘 Quick Troubleshooting

**Tiles not loading?**
```bash
# Check nginx
systemctl status nginx

# Check TileServer
docker logs deepgis-xr_tileserver_1 --tail 50

# Check file exists
ls -lh $PROJECT_ROOT/deepgis-xr/data/rock_tiles_deepgis.mbtiles
```

**Need to update tiles?**
Just re-run the conversion script (Step 1) and restart TileServer (Step 3).

---

*For detailed documentation, see [deepgis-xr/ROCK_TILES_SETUP.md](deepgis-xr/ROCK_TILES_SETUP.md)*

