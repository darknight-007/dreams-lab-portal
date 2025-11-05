# Dreams Lab Portal

A comprehensive web portal for the Dreams Laboratory, integrating multiple research platforms including OpenUAV, DeepGIS XR, and ROS2 applications. The portal provides a unified interface for robotics simulation, geospatial analysis, and extended reality applications.

## Project Overview

The Dreams Lab Portal serves as a central hub for various research platforms and tools:

- **OpenUAV Platform**: Multi-drone simulation environment with GPU acceleration
- **DeepGIS XR**: Geospatial analysis and visualization platform with XR capabilities
- **ROS2 Integration**: Robot Operating System 2 workspace for robotics applications
- **Blender Integration**: 3D modeling and rendering capabilities
- **Container Management**: Docker-based deployment and resource management

## Architecture

The system is built on a microservices architecture with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dreams Lab Portal                         │
├─────────────┬─────────────┬────────────────┬───────────────┤
│  OpenUAV    │  DeepGIS XR │    ROS2 WS     │    Blender    │
│  Platform   │  Services   │    Services    │    Render     │
├─────────────┴─────────────┴────────────────┴───────────────┤
│                     Nginx Reverse Proxy                      │
├─────────────────────────────────────────────────────────────┤
│                     Docker Networking                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Web Service**
   - Django-based web application
   - User authentication and management
   - Project coordination and monitoring
   - Resource allocation and scheduling

2. **OpenUAV Integration**
   - Multi-drone simulation environment
   - GPU-accelerated rendering
   - VNC-based remote access
   - PX4 SITL integration

3. **DeepGIS XR Service**
   - Geospatial data processing
   - XR visualization
   - Tile server integration
   - Machine learning capabilities

4. **ROS2 Workspace**
   - Robot control and simulation
   - Sensor data processing
   - Navigation and planning
   - Multi-robot coordination

## Prerequisites

- NVIDIA GPU with drivers 470+
- Docker 20.10+ and Docker Compose
- Python 3.9+
- ROS2 Humble (for robotics features)
- 16GB+ RAM recommended
- Ubuntu 20.04 or later

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/dreams-lab-portal.git
   cd dreams-lab-portal
   ```

2. Create and configure the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up Docker network:
   ```bash
   docker network create --subnet=172.20.0.0/16 dreamslab
   ```

4. Build and start services:
   ```bash
   docker-compose up --build
   ```

## Project Structure

```
dreams-lab-portal/
├── openuav2/           # OpenUAV simulation platform
├── deepgis-xr/         # Geospatial XR platform
├── ros2_ws/           # ROS2 workspace
├── dreams_laboratory/ # Django project directory
├── templates/         # HTML templates
├── static/           # Static assets
├── media/           # User uploaded files
├── nginx.conf       # Nginx configuration
├── Dockerfile       # Main service container
└── docker-compose.yml # Service orchestration
```

## Configuration

### Environment Variables
- `SECRET_KEY`: Django secret key
- `DEBUG`: Debug mode flag
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts
- `TWILIO_*`: Twilio integration settings
- `NVIDIA_*`: GPU configuration

### Docker Settings
- Runtime: NVIDIA Container Runtime
- Network: Custom bridge network (172.20.0.0/16)
- Volumes: Static, media, and database persistence
- Ports: Main service on 8080, subsidiary services on various ports

## Development

1. Start development server:
   ```bash
   docker-compose up
   ```

2. Access services:
   - Main Portal: http://localhost:8080
   - OpenUAV: http://localhost:8060
   - DeepGIS XR: http://localhost:8090
   - ROS2 Bridge: http://localhost:9090

## Features

- **User Management**
  - Authentication and authorization
  - Project access control
  - Resource allocation

- **Simulation Environment**
  - Multi-drone scenarios
  - GPU-accelerated rendering
  - Real-time visualization

- **Geospatial Analysis**
  - Data processing and visualization
  - XR integration
  - Tile server capabilities

- **Transformer-Based Autoencoder for GIS Tiles**
  - Vision Transformer (ViT) encoder-decoder architecture
  - Support for RGB (3 channels) and multispectral (5+ channels) imagery
  - Generative synthesis of GIS tile images
  - Latent space exploration and synthetic image generation
  - See [Transformer Autoencoder Documentation](#transformer-autoencoder) below

- **Robotics Integration**
  - ROS2 node management
  - Sensor data processing
  - Navigation and control

## Production Deployment

1. Update environment variables:
   ```bash
   DEBUG=False
   ALLOWED_HOSTS=your-domain.com
   SECRET_KEY=your-secure-key
   ```

2. Configure SSL/TLS:
   - Update nginx.conf with SSL settings
   - Add SSL certificates
   - Enable HTTPS redirects

3. Set up monitoring:
   - Configure logging
   - Set up system monitoring
   - Enable backup systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Follow coding standards and documentation guidelines

## License

This project is licensed under Creative Commons Legal Code CC0 1.0 Universal - see the LICENSE file for details. This means you can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

## Transformer Autoencoder

A Vision Transformer-based autoencoder for generative synthesis of GIS tile images. This system supports both RGB (3 channels) and multispectral (5+ channels) imagery, enabling the generation of synthetic GIS tiles from learned latent representations.

### Location

All transformer autoencoder scripts are located in `dreams_laboratory/scripts/`:

- `multispectral_vit.py` - Vision Transformer encoder with variable channel support
- `multispectral_decoder.py` - Reconstruction decoder with multiple architectures
- `train_zoom23_autoencoder.py` - Training script for zoom level 23 GIS tiles
- `test_reconstruction_zoom23.py` - Test reconstruction quality
- `extract_latents_zoom23.py` - Extract latent representations
- `generate_synthetic_zoom23.py` - Generate synthetic RGB images
- `analyze_tile_folder.py` - Analyze GIS tile folder structures

### Quick Start

#### 1. Analyze Your Tile Dataset

```bash
cd dreams_laboratory/scripts
python3 analyze_tile_folder.py /path/to/tile/directory
```

This will show:
- Total number of image files
- Files by zoom level
- File size statistics
- Directory structure analysis

#### 2. Train Autoencoder on Zoom Level 23 Tiles

```bash
python3 train_zoom23_autoencoder.py \
    --tile_dir /mnt/dreamslab-store/deepgis/deepgis_rocks2/static-root/rock-tiles/raw \
    --zoom_level 23 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 50 \
    --device cuda \
    --multi_gpu
```

**Dataset Requirements:**
- Minimum: 500-1,000 images (use smaller image size: 256×256)
- Recommended: 2,000-5,000 images (512×512 or 256×256)
- Ideal: 10,000+ images (960×960)

**Training Output:**
- `encoder_zoom23.pth` - Trained encoder
- `decoder_zoom23.pth` - Trained decoder
- `autoencoder_zoom23.pth` - Combined model
- `checkpoints_zoom23/` - Training checkpoints

#### 3. Test Reconstruction Quality

```bash
python3 test_reconstruction_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --zoom_level 23 \
    --num_samples 10
```

#### 4. Extract Latent Representations

```bash
python3 extract_latents_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --zoom_level 23 \
    --output latents_zoom23.npy
```

#### 5. Generate Synthetic Samples

```bash
python3 generate_synthetic_zoom23.py \
    --encoder_path encoder_zoom23.pth \
    --decoder_path decoder_zoom23.pth \
    --latent_file latents_zoom23.npy \
    --num_samples 20 \
    --sample_method gaussian
```

**Sampling Methods:**
- `gaussian` - Sample from Gaussian distribution (recommended)
- `uniform` - Uniform sampling within data bounds
- `real` - Use actual latents from dataset

### Model Architecture

- **Encoder**: Vision Transformer (ViT)
  - Embedding dimension: 512
  - Patch size: 16×16
  - Transformer layers: 6
  - Attention heads: 8
  - Cross-band attention: Enabled

- **Decoder**: Memory-efficient reconstruction decoder
  - Progressive upsampling
  - Hidden dimension: 512

### Supported Image Formats

- **RGB Images**: PNG, JPG, JPEG (3 channels)
- **Multispectral Images**: TIFF, GeoTIFF (5+ channels)
- **Image Sizes**: 256×256, 512×512, 960×960 (configurable)

### Documentation

For detailed documentation, see:
- `ZOOM23_TRAINING_GUIDE.md` - Complete training guide
- `POST_TRAINING_WORKFLOW.md` - Post-training workflow and usage
- `analyze_tile_folder.py` - Tile folder analysis tool

### Example Workflow

```bash
# 1. Analyze dataset
python3 analyze_tile_folder.py /path/to/tiles

# 2. Train model
python3 train_zoom23_autoencoder.py --zoom_level 23 --multi_gpu

# 3. Test reconstruction
python3 test_reconstruction_zoom23.py

# 4. Extract latents
python3 extract_latents_zoom23.py

# 5. Generate synthetic images
python3 generate_synthetic_zoom23.py --num_samples 20
```

### Requirements

- PyTorch 1.9+
- torchvision
- numpy
- PIL/Pillow
- rasterio (for multispectral TIFF support)
- matplotlib (for visualization)

## Support

For support and questions:
1. Check the documentation in respective project directories
2. Check transformer autoencoder guides in `dreams_laboratory/scripts/`
3. Open an issue in the repository
4. Contact the Dreams Laboratory team

## Acknowledgments

- Dreams Laboratory Research Team
- OpenUAV Project Contributors
- DeepGIS Community
- ROS2 Community 