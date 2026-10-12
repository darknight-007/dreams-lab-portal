FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# No SECRET_KEY default here. settings.py raises if it is unset, which is the
# behaviour we want; baking one in meant any run that missed the env var came
# up with a key that is public in this image's layers.

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

#COPY blender-4.2.2-linux-x64 /opt/blender
#RUN chmod +x /opt/blender/blender

# Add Blender to PATH
#ENV PATH="/opt/blender:$PATH"

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Replace whatever openuav2 came in from the build context with an empty dir.
# This was `rm -f`, which worked while openuav2 was a symlink but fails with
# "Is a directory" now that it is a real directory in the repo, taking the
# whole && chain down with it.
RUN rm -rf /app/openuav2 && mkdir -p /app/openuav2

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

# The VNC password is set at container start from $VNC_PASSWORD - see
# entrypoint.sh. It used to be baked in here as a literal, which meant the
# real password lived in an image layer and setting VNC_PASSWORD in .env had
# no effect on the actual x11vnc server.

# Expose VNC and noVNC ports
EXPOSE 5901 6080

# docker-compose.yml overrides this with entrypoint.sh, which is the real
# start path and binds port 80. This CMD is the fallback for a bare
# `docker run` and is kept last so it is not shadowed by later layers.
CMD ["bash", "/app/entrypoint.sh"]

