# Project Status

Updated: 2025-01-04

## Overview
- Repository name: `SatoshiRig`
- Purpose: Neutral Bitcoin solo-mining client with configurable compute backend, Docker/Compose support, CI/CD, and web dashboard.

## Architecture
- Package: `src/SatoshiRig`
  - `core/miner.py`: `Miner` class (mining loop, logging).
  - `core/state.py`: `MinerState` dataclass (runtime state).
  - `clients/pool_client.py`: CKPool TCP JSON client (subscribe/authorize/notify/submit).
  - `cli.py`: argparse CLI; loads config, sets up logging, builds dependencies, starts `Miner`.
  - `config.py`: loads TOML config (`CONFIG_FILE` override supported).
  - `miner.py`: thin compatibility facade.
  - `web/server.py`: Flask web server with SocketIO for real-time mining status dashboard.
- Config: `config/config.toml` (pool, network, logging, miner, compute).
- Containerization: `Dockerfile`, `.dockerignore`, `docker-compose.yml` (Unraid-ready).
- CI: `.github/workflows/ci.yml` (install, format, test), `.github/workflows/release.yml` (releases from tags).
- Packaging: `pyproject.toml` with console script `satoshirig`.

## Usage
- Local: `python -m SatoshiRig --wallet <ADDR> [--config ./config/config.toml] [--backend cpu|cuda|opencl] [--gpu 0] [--web-port 5000]`
- Docker: `docker run --rm -e WALLET_ADDRESS=<ADDR> -p 5000:5000 satoshirig`
- Compose/Unraid: `docker compose up -d` with env vars in Unraid UI or `.env`.
- Web Dashboard: Access via `http://localhost:5000` (or configured port) when running.

## Recent Work
- Refactor to class-based miner and client separation.
- Neutral logging (no color banners, clean messages).
- Externalized configuration (TOML) and CLI flags.
- GPU backend selection prepared (cpu/cuda/opencl), with device index.
- Dockerfile and Compose added; README updated.
- CI hardening; package install in CI; auto-release workflow with correct permissions.
- Web dashboard added: Flask + SocketIO for real-time mining status monitoring.
- Configurable block source: web service or local Bitcoin Core RPC.
- Tags pushed: `v0.1.0`, `v0.1.1`, `v0.1.2`, `v1.0.0`.

## Open Items / Next Steps
- Implement real GPU hashing (CUDA/OpenCL kernels) and auto-detection.
- Add structured logging (JSON) option.
- Add integration tests with a mocked pool server.
- Optional metrics endpoint (Prometheus) for hashrate and connectivity.
- Publish Docker image to a registry and switch compose to use the image tag.

## How to Resume
- For releases: push a new tag `vX.Y.Z` to trigger the GitHub release workflow.
- For config changes: edit `config/config.toml` or pass via `--config`/env.
- For Unraid: adjust env in the UI or `.env`, then `docker compose up -d`.
- For web dashboard: access via `http://<host>:5000` (default port) when running.
