#!/bin/bash

# Create static directory if it doesn't exist
mkdir -p /app/static
mkdir -p /app/staticfiles

echo "Starting collectstatic..."
python3 manage.py collectstatic --noinput
echo "Finished collectstatic"

echo "Starting gunicorn on port 80..."
gunicorn dreams_laboratory.wsgi:application \
    --bind 0.0.0.0:80 \
    --workers 3 \
    --timeout 120 \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    --reload