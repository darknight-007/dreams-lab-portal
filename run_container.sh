#!/bin/bash

# Default container name if not provided
CONTAINER_NAME=${CONTAINER_NAME:-digital-twin-default}

# Default ports
VNC_PORT=${VNC_PORT:-5901}
NOVNC_PORT=${NOVNC_PORT:-6080}

# Validate container name
if [[ ! $CONTAINER_NAME =~ ^digital-twin-[a-zA-Z0-9-]+$ ]]; then
    echo "Error: Invalid container name format. Must start with 'digital-twin-'"
    exit 1
fi

# Check if ports are available
if ! netstat -tuln | grep -q ":$VNC_PORT\|:$NOVNC_PORT"; then
    echo "Checking port availability..."
else
    echo "Error: Required ports ($VNC_PORT, $NOVNC_PORT) are already in use"
    exit 1
fi

# Launch the container with proper settings
docker run -d \
    --name ${CONTAINER_NAME} \
    --rm \
    --gpus all \
    --privileged \
    -p ${VNC_PORT}:5901 \
    -p ${NOVNC_PORT}:6080 \
    -e DISPLAY=:1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e CONTAINER_NAME=${CONTAINER_NAME} \
    openuav:px4-sitl

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "Container $CONTAINER_NAME started successfully"
    exit 0
else
    echo "Failed to start container $CONTAINER_NAME"
    exit 1
fi 