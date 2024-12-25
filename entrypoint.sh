#!/bin/bash

# Create required directories
mkdir -p /app/static
mkdir -p /app/staticfiles

# Collect static files
echo "Collecting static files..."
python3 manage.py collectstatic --noinput

# Check database tables
echo "Checking database tables..."
python3 manage.py shell << EOF
from django.db import connection
from openuav_manager.models import Container

# Get list of tables
tables = connection.introspection.table_names()
print("\nExisting tables:", tables)

# Check if Container table exists
container_table = 'openuav_manager_container'
if container_table not in tables:
    print(f"\nWARNING: {container_table} does not exist!")
    print("Creating migrations for openuav_manager...")
    from django.core.management import call_command
    call_command('makemigrations', 'openuav_manager')
    call_command('migrate', 'openuav_manager')
else:
    print(f"\n{container_table} exists!")
    # Show table schema
    cursor = connection.cursor()
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{container_table}';")
    print("\nTable schema:", cursor.fetchone()[0])
EOF

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