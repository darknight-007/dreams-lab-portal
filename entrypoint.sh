#!/bin/bash

# Dependencies are installed into the image at build time. Re-running pip here
# on every start made the image layer pointless, added minutes to startup and
# required network access to boot. If you change requirements.txt, rebuild.

# Create required directories
mkdir -p /app/static
mkdir -p /app/staticfiles
mkdir -p /app/openuav2
mkdir -p /root/.vnc

# VNC password, from the environment rather than baked into the image.
if [ -z "${VNC_PASSWORD:-}" ]; then
    echo "entrypoint: VNC_PASSWORD is not set; refusing to start a VNC server with no password" >&2
    exit 1
fi
x11vnc -storepasswd "$VNC_PASSWORD" /root/.vnc/passwd

# Start Xvfb
Xvfb :1 -screen 0 1920x1080x24 &
export DISPLAY=:1

# Start VNC server
x11vnc -display :1 -forever -usepw -create &

# Start noVNC
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &

# Start XFCE4
DISPLAY=:1 startxfce4 &

# Copy run_container.sh to openuav2 directory if it is present. The host path
# /home/jdas/openuav2 is an empty root-owned directory, so this was failing
# silently on every start. openuav_manager builds its docker run command inline
# and does not call this script, so a miss here is not fatal.
if [ -f /home/jdas/openuav2/run_container.sh ]; then
    cp /home/jdas/openuav2/run_container.sh /app/openuav2/
    chmod +x /app/openuav2/run_container.sh
else
    echo "entrypoint: no run_container.sh at /home/jdas/openuav2, skipping" >&2
fi

# Ensure proper permissions
chown -R www-data:www-data /app/static /app/staticfiles

# Remove existing staticfiles to ensure clean collection
rm -rf /app/staticfiles/*

# Collect static files
echo "Collecting static files..."
python3 manage.py collectstatic --noinput --clear

# Apply migrations
echo "Applying migrations..."
python3 manage.py migrate

# Start Gunicorn
echo "Starting Gunicorn..."
exec gunicorn dreams_laboratory.wsgi:application \
    --bind 0.0.0.0:80 \
    --workers 3 \
    --timeout 120 \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance