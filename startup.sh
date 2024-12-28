#!/bin/bash

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check NVIDIA GPU
check_nvidia() {
    if command_exists nvidia-smi; then
        echo "NVIDIA GPU detected"
        nvidia-smi
        return 0
    else
        echo "No NVIDIA GPU detected"
        return 1
    fi
}

# Function to cleanup existing VNC sessions
cleanup_vnc() {
    echo "Cleaning up existing VNC sessions..."
    rm -f /tmp/.X11-unix/X1
    rm -f /root/.vnc/*.pid
    rm -f /root/.vnc/*.log
    pkill -f "Xvnc :1"
    pkill -f "websockify"
    sleep 2  # Wait for processes to terminate
}

# Function to setup Xvfb as fallback
setup_xvfb() {
    echo "Setting up Xvfb..."
    Xvfb :1 -screen 0 1920x1080x24 &
    export DISPLAY=:1
    sleep 2  # Wait for Xvfb to start
}

# Function to verify VNC server
verify_vnc() {
    local max_attempts=5
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if netstat -tuln | grep -q ":5901.*LISTEN"; then
            echo "VNC server is running and listening on port 5901"
            return 0
        fi
        echo "Waiting for VNC server to start (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "Failed to verify VNC server after $max_attempts attempts"
    return 1
}

# Function to start TurboVNC with GPU support
start_turbovnc() {
    echo "Starting TurboVNC server..."
    if check_nvidia; then
        # Set NVIDIA specific environment variables
        export VGL_DISPLAY=:1
        export CUDA_VISIBLE_DEVICES=0
        # Start TurboVNC with GPU support
        /opt/TurboVNC/bin/vncserver :1 \
            -geometry 1920x1080 \
            -depth 24 \
            -rfbport 5901 \
            -SecurityTypes None \
            -wm xfce4-session
    else
        # Fallback to Xvfb if no GPU
        setup_xvfb
        /opt/TurboVNC/bin/vncserver :1 \
            -geometry 1920x1080 \
            -depth 24 \
            -rfbport 5901 \
            -SecurityTypes None \
            -wm xfce4-session
    fi
}

# Function to start noVNC
start_novnc() {
    echo "Starting noVNC proxy..."
    /opt/novnc/utils/novnc_proxy --vnc localhost:5901 --listen 6080 &
    sleep 2
}

# Main execution
cleanup_vnc
start_turbovnc

# Verify VNC server
if ! verify_vnc; then
    echo "Failed to start VNC server, falling back to Xvfb..."
    cleanup_vnc
    setup_xvfb
    start_turbovnc
    if ! verify_vnc; then
        echo "Fatal: Could not start VNC server even with fallback"
        exit 1
    fi
fi

start_novnc

# Start PX4 SITL simulation
cd ~/src/PX4-Autopilot
HEADLESS=1 make px4_sitl gazebo

# Keep container running and log any errors
exec 2>&1
tail -f /root/.vnc/*log 