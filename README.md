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

## Support

For support and questions:
1. Check the documentation in respective project directories
2. Open an issue in the repository
3. Contact the Dreams Laboratory team

## Acknowledgments

- Dreams Laboratory Research Team
- OpenUAV Project Contributors
- DeepGIS Community
- ROS2 Community 