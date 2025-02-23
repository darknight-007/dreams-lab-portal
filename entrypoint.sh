#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Create required directories
mkdir -p /app/static
mkdir -p /app/staticfiles
mkdir -p /app/openuav2
mkdir -p /root/.vnc

# Start Xvfb
Xvfb :1 -screen 0 1920x1080x24 &
export DISPLAY=:1

# Start VNC server
x11vnc -display :1 -forever -usepw -create &

# Start noVNC
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &

# Start XFCE4
DISPLAY=:1 startxfce4 &

# Copy run_container.sh to openuav2 directory
cp /home/jdas/openuav2/run_container.sh /app/openuav2/
chmod +x /app/openuav2/run_container.sh

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