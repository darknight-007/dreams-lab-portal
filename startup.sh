#!/bin/bash

# Clean up any existing VNC server files
rm -f /tmp/.X11-unix/X1
rm -f /root/.vnc/*.pid
rm -f /root/.vnc/*.log

# Start TurboVNC server with specified geometry
/opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 -rfbport 5901 -SecurityTypes None

# Wait a moment for VNC server to fully start
sleep 2

# Start noVNC proxy
/opt/novnc/utils/novnc_proxy --vnc localhost:5901 --listen 6080 &

# Wait for noVNC to start
sleep 2

# Start PX4 SITL simulation
cd ~/src/PX4-Autopilot
HEADLESS=1 make px4_sitl gazebo

# Keep container running
tail -f /dev/null 