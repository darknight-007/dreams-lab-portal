"""OpenUAV Manager Configuration"""

import os

# Base domain for VNC connections
VNC_BASE_DOMAIN = "digital-twin.deepgis.org"

# Container configuration
CONTAINER_PREFIX = "openuav"
CONTAINER_IMAGE = "openuav:px4-sitl"
CONTAINER_NETWORK = "dreamslab"

# Container ports (internal only)
CONTAINER_PORTS = {
    "vnc": 5901,
    "novnc": 6080
}

# Container status choices
CONTAINER_STATUS = {
    'RUNNING': 'running',
    'STOPPED': 'stopped',
    'REMOVED': 'removed',
    'ERROR': 'error'
}

# Session configuration
SESSION_TYPES = {
    'GUEST': 'guest',
    'USER': 'user'
}

# Security settings
CSRF_EXEMPT_PATHS = [
    'launch',
    'stop',
]

# Resource limits
MAX_CONTAINERS_PER_USER = 1
MAX_GUEST_CONTAINERS = 5
CONTAINER_TIMEOUT = 3600  # 1 hour

# Cleanup settings
CLEANUP_RESOURCES = [
    '/tmp/.X11-unix/X1',
    '/tmp/.X1-lock',
] 