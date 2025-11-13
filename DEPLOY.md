# Deploy SatoshiRig

This project uses a small CUDA runtime image and the NVIDIA Container Toolkit for GPU access. CI workflows run only when manually triggered.

## 1) Publish Docker image (manual)

1. GitHub → Actions → "Build and Publish Docker Image".
2. Click "Run workflow" → Run.
   - Builds the multi-stage image and pushes `ghcr.io/<owner>/satoshirig:latest`.
3. Optional: For a GitHub release, run "Create GitHub Release" (you may provide a `tag`, e.g., `v2.16.0`).

Notes:
- Workflows are manual only; they do not run on push/tag automatically.
- If GHCR package is private, see `MAKE_PACKAGE_PUBLIC.md`.

## 2) Run with NVIDIA GPU (recommended)

Prerequisites: Install NVIDIA drivers and the NVIDIA Container Toolkit on the host.

- NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-container-toolkit

Docker CLI:
```bash
# Pull latest image
docker pull ghcr.io/<owner>/satoshirig:latest

# Run with GPU access
docker run -d --name satoshirig --restart unless-stopped \
  --gpus all \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  -p 5000:5000 \
  ghcr.io/<owner>/satoshirig:latest
```

> Before starting the container, edit the host-side `config/config.toml` and set `[wallet].address` (or use the web UI and restart the miner).

Docker Compose:
```yaml
services:
  satoshirig:
    image: ghcr.io/<owner>/satoshirig:latest
    container_name: satoshirig
    restart: unless-stopped
    runtime: nvidia
    gpus: all
    environment:
      - COMPUTE_BACKEND=cuda
      - GPU_DEVICE=0
      - WEB_PORT=5000
      - STATS_FILE=/app/data/statistics.json
    ports:
      - "5000:5000"
    volumes:
      - ./config:/app/config
      - ./data:/app/data  # Persistent statistics storage
```
> Configure `[wallet].address` in `config/config.toml` (or via the web UI once the container is up) and restart the service to begin mining.

## 3) Local build (optional)

```bash
docker build -t satoshirig .
docker run -d --name satoshirig --restart unless-stopped \
  --gpus all \
  -p 5000:5000 \
  satoshirig
```
> If you launch without a wallet address in `config/config.toml`, the container will stay idle until you set it via the web UI and restart.

## 4) Troubleshooting

- Ensure `nvidia-smi` works on the host and `--gpus all` maps devices.
- For package visibility, follow `MAKE_PACKAGE_PUBLIC.md`.
- Web UI: http://localhost:5000
