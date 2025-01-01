#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Create required directories
mkdir -p /app/static
mkdir -p /app/staticfiles

# Collect static files
echo "Collecting static files..."
python3 manage.py collectstatic --noinput

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