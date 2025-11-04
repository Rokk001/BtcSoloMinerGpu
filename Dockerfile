# For GPU support with NVIDIA, you can switch to an NVIDIA CUDA base image.
# Default remains slim Python for portability; to use GPU, run container with --gpus.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config

ENV PYTHONPATH=/app/src

# Optional: set WALLET_ADDRESS at runtime or pass via CLI
ENV WALLET_ADDRESS=""
ENV CONFIG_FILE="/app/config/config.toml"
ENV COMPUTE_BACKEND="cpu"
ENV GPU_DEVICE="0"

CMD ["python", "-m", "SatoshiRig"]


