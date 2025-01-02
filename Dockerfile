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

# Remove symlink and create openuav2 directory
RUN rm -f /app/openuav2 && mkdir -p /app/openuav2

# Copy run_container.sh to openuav2
COPY run_container.sh /app/openuav2/
RUN chmod +x /app/openuav2/run_container.sh

# Create docker group with same GID as host
RUN groupadd -g 999 docker && \
    usermod -aG docker root

# Collect static files
# RUN python3 manage.py collectstatic --noinput

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "dreams_laboratory.wsgi:application"]

# Install VNC and X11 dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    xvfb \
    x11vnc \
    xfce4 \
    xfce4-terminal \
    novnc \
    websockify \
    net-tools \
    netcat \
    && apt-get clean

# Download and install TurboVNC
RUN curl -fsSL -o turbovnc.deb https://sourceforge.net/projects/turbovnc/files/3.0.3/turbovnc_3.0.3_amd64.deb/download \
    && dpkg -i turbovnc.deb \
    && rm turbovnc.deb

# Create required directories
RUN mkdir -p /root/.vnc /tmp/.X11-unix

# Set VNC password
RUN mkdir -p /root/.vnc && x11vnc -storepasswd liftoff /root/.vnc/passwd

# Expose VNC and noVNC ports
EXPOSE 5901 6080

