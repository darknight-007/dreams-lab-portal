#!/bin/bash

# Stop and remove any existing container with the same name
docker stop openuav-test 2>/dev/null
docker rm openuav-test 2>/dev/null

# Start new container
docker run -it \
    --name openuav-test \
    --privileged \
    --gpus all \
    -p 6080:6080 \
    -p 5901:5901 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -e VNC_PASSWORD=liftoff \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --runtime=nvidia \
    openuav:px4-sitl \
    /root/startup.sh 