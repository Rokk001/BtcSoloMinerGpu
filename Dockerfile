# SatoshiRig - Bitcoin solo mining client
# Multi-stage build to keep final image small while enabling GPU via NVIDIA Container Toolkit

# -------- Builder stage: use CUDA devel only to compile/build Python deps (e.g., PyCUDA) --------
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    build-essential \
    ocl-icd-opencl-dev opencl-headers \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

WORKDIR /wheelhouse

COPY requirements.txt ./
# Build wheels WITH dependencies so runtime can install offline from /wheelhouse
# Note: Building with dependencies ensures all transitive dependencies are included
RUN pip wheel -r requirements.txt -w /wheelhouse
RUN pip wheel "pycuda>=2023.1" "pyopencl>=2023.1.2" "nvidia-ml-py>=12.0.0" -w /wheelhouse

# -------- Runtime stage: lightweight CUDA runtime image --------
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY --from=builder /wheelhouse /wheelhouse
COPY requirements.txt ./
# Install dependencies from wheels (PyCUDA, PyOpenCL, and nvidia-ml-py for GPU monitoring)
# Use --find-links to prefer wheels from /wheelhouse, but allow PyPI fallback for missing dependencies
# These will be available when running with --runtime=nvidia or --gpus
RUN pip install --no-cache-dir --find-links=/wheelhouse -r requirements.txt && \
    pip install --no-cache-dir --find-links=/wheelhouse pycuda pyopencl nvidia-ml-py && \
    rm -rf /wheelhouse

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

# Runtime configuration envs
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


