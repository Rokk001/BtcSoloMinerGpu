# For GPU support with NVIDIA, we use an NVIDIA CUDA base image.
# We use the devel image to get CUDA headers needed for PyCUDA compilation.
# The runtime libraries are included in the devel image.
# Note: Using --runtime=nvidia requires NVIDIA Container Toolkit to be installed on the host.
# SatoshiRig - Bitcoin solo mining client
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    make \
    libc6-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip (remove existing if they exist)
RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU dependencies (PyCUDA and PyOpenCL)
# These will be available when running with --runtime=nvidia or --gpus
RUN pip install --no-cache-dir pycuda>=2023.1 pyopencl>=2023.1.2

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

# Optional: set WALLET_ADDRESS at runtime or pass via CLI
ENV WALLET_ADDRESS=""
ENV CONFIG_FILE="/app/config/config.toml"
ENV COMPUTE_BACKEND="cpu"
ENV GPU_DEVICE="0"
ENV WEB_PORT="5000"

# Docker Labels for WebUI Navigation (Docker Desktop / Portainer)
LABEL org.opencontainers.image.title="SatoshiRig"
LABEL org.opencontainers.image.description="Bitcoin Solo Mining Client WebUI"
LABEL org.opencontainers.image.url="http://localhost:5000"
LABEL io.portainer.container.http.port="5000"
LABEL io.portainer.container.http.url="http://localhost:5000"
LABEL io.portainer.accesscontrol.public="true"
LABEL com.docker.compose.service.webui="http://localhost:5000"

CMD ["python", "-m", "SatoshiRig"]


