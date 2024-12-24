FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV SECRET_KEY "temporary-secret-key"

# Set the working directory
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
    python3.10 libopencv-dev  \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libsm6 \
    libx11-6 \
    libxi6 \
    libxxf86vm1 \
    libopenal1 \
    curl \
sqlite \
    python3-pip \
    apt-transport-https \
    ca-certificates \
    gnupg-agent \
    software-properties-common && \
    apt-get clean

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
    apt-get update && \
    apt-get install -y docker-ce-cli

COPY blender-4.2.2-linux-x64 /opt/blender
RUN chmod +x /opt/blender/blender

# Add Blender to PATH
ENV PATH="/opt/blender:$PATH"

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Create docker group with same GID as host
RUN groupadd -g 999 docker && \
    usermod -aG docker root

# Collect static files
# RUN python3 manage.py collectstatic --noinput

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "dreams_laboratory.wsgi:application"]

