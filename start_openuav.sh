#!/bin/bash

# Function to check NVIDIA GPU availability
check_nvidia() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA GPU detected"
        return 0
    else
        echo "No NVIDIA GPU detected"
        return 1
    fi
}

# Function to wait for container readiness
wait_for_container() {
    local container_name=$1
    local max_attempts=30
    local attempt=1

    echo "Waiting for container services to start..."
    while [ $attempt -le $max_attempts ]; do
        if docker exec $container_name netstat -tuln | grep -q ":5901.*LISTEN"; then
            echo "VNC server is ready"
            if docker exec $container_name netstat -tuln | grep -q ":6080.*LISTEN"; then
                echo "noVNC proxy is ready"
                return 0
            fi
        fi
        echo "Waiting for services (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "Timeout waiting for container services"
    return 1
}

# Stop and remove any existing container with the same name
docker stop hungry_williams 2>/dev/null
docker rm hungry_williams 2>/dev/null

# Base docker run command
CMD="docker run -it \
    --name hungry_williams \
    --privileged \
    -p 6080:6080 \
    -p 5901:5901 \
    -e DISPLAY=:1 \
    -e PULSE_SERVER=unix:/tmp/pulseaudio.socket \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/dri:/dev/dri \
    -v /dev/shm:/dev/shm"

# Add NVIDIA specific configuration if GPU is available
if check_nvidia; then
    CMD="$CMD \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    --runtime=nvidia"
fi

# Add VNC password and execute startup script
CMD="$CMD \
    -e VNC_PASSWORD=liftoff \
    openuav:px4-sitl"

# Execute the command
echo "Starting OpenUAV container..."
eval $CMD &

# Wait for container to be ready
if ! wait_for_container hungry_williams; then
    echo "Failed to start container services"
    docker logs hungry_williams
    docker stop hungry_williams
    docker rm hungry_williams
    exit 1
fi

echo "Container is ready!"
echo "Access noVNC web interface at: http://localhost:6080/vnc.html"
echo "Or connect VNC client to: localhost:5901"