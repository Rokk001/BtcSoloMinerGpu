# Project Status

Updated: 2025-01-27

## Overview
- Repository name: `SatoshiRig`
- Purpose: Neutral Bitcoin solo-mining client with configurable compute backend, Docker/Compose support, CI/CD, and web dashboard.

## Architecture
- Package: `src/SatoshiRig`
  - `core/miner.py`: `Miner` class (mining loop, logging, GPU support with sequential nonce counter).
  - `core/state.py`: `MinerState` dataclass (runtime state).
  - `core/gpu_compute.py`: GPU compute module (CUDA/OpenCL support for GPU mining with improved initialization and error handling).
  - `clients/pool_client.py`: CKPool TCP JSON client (subscribe/authorize/notify/submit).
  - `cli.py`: argparse CLI; loads config, sets up logging, builds dependencies, starts `Miner`.
  - `config.py`: loads TOML config (`CONFIG_FILE` override supported).
  - `miner.py`: thin compatibility facade.
  - `web/server.py`: Flask web server with SocketIO for real-time mining status dashboard with tabs (Overview, Performance, Analytics, Intelligence, History, Settings).
- Config: `config/config.toml` (pool, network, logging, miner, compute).
- Containerization: `Dockerfile` (NVIDIA CUDA base image `nvidia/cuda:11.8.0-runtime-ubuntu22.04`), `.dockerignore`, `docker-compose.yml` (Unraid-ready, NVIDIA GPU support via `--runtime=nvidia` or `--gpus all`).
- CI: `.github/workflows/ci.yml` (install, format, test), `.github/workflows/release.yml` (releases from tags), `.github/workflows/docker-publish.yml` (builds and publishes Docker image to GHCR).
- Packaging: `pyproject.toml` with console script `satoshirig`.

## Usage
- Local: `python -m SatoshiRig --wallet <ADDR> [--config ./config/config.toml] [--backend cpu|cuda|opencl] [--gpu 0] [--web-port 5000]`
- Docker (local build): `docker build -t satoshirig . && docker run --rm -e WALLET_ADDRESS=<ADDR> -p 5000:5000 satoshirig`
- Docker (from GHCR): `docker run --rm -e WALLET_ADDRESS=<ADDR> -p 5000:5000 ghcr.io/rokk001/satoshirig:latest`
- Docker (NVIDIA GPU): `docker run --rm --gpus all -e WALLET_ADDRESS=<ADDR> -e COMPUTE_BACKEND=cuda ghcr.io/rokk001/satoshirig:latest` or `docker run --rm --runtime=nvidia -e WALLET_ADDRESS=<ADDR> -e COMPUTE_BACKEND=cuda ghcr.io/rokk001/satoshirig:latest`
- Compose/Unraid: `docker compose up -d` with env vars in Unraid UI or `.env`. Can use published image from GHCR: `ghcr.io/rokk001/satoshirig:latest`
- Web Dashboard: Access via `http://localhost:5000` (or configured port) when running.

## Recent Work
- Refactor to class-based miner and client separation.
- Neutral logging (no color banners, clean messages).
- Externalized configuration (TOML) and CLI flags.
- GPU mining support implemented (CUDA/OpenCL with parallel batch hashing for multiple nonces).
- Dockerfile and Compose added; README updated.
- CI hardening; package install in CI; auto-release workflow with correct permissions.
- Web dashboard added: Flask + SocketIO for real-time mining status monitoring.
- Configurable block source: web service or local Bitcoin Core RPC.
- Docker image published to GitHub Container Registry (GHCR): `ghcr.io/rokk001/satoshirig:latest` (public, automatically set on publish).
- Performance & Monitoring (Feature 1): CPU, Memory, GPU (NVIDIA) monitoring with real-time metrics.
- Mining Intelligence (Feature 2): Estimated time to block, block found probability, profitability calculator, difficulty trend analysis.
- Advanced Visualizations (Feature 3): Hash Rate vs Difficulty comparison chart, Performance Metrics Dashboard.
- WebGUI Navigation: Docker labels for Docker Desktop and Portainer WebUI integration.
- WebApp Restructured: Complete reorganization with tabbed interface (Overview, Performance, Analytics, Intelligence, History) for better UX and logical grouping.
- Uptime Calculation Fix: Fixed timezone issue by using Unix timestamps instead of ISO strings.
- Mining Control: Pause button now stops mining via API endpoints (`/api/stop`, `/api/start`).
- UI Improvements: Removed redundant "Connected" button, improved visual hierarchy with section headers.
- GPU Mining Support: Implemented CUDA/OpenCL support with parallel batch hashing (1024 nonces per iteration), automatic GPU initialization, fallback to CPU if GPU unavailable.
- GPU Mining Improvements (v2.5.0): Dockerfile now uses NVIDIA CUDA base image for proper GPU support, improved GPU initialization with better error handling, sequential nonce counter for complete coverage, removed pycuda.autoinit for flexible initialization, enhanced GPU device detection and validation.
- GPU Mining Fixes (v2.5.1-v2.5.2): Fixed Dockerfile python symlink issue, switched to CUDA devel image for PyCUDA compilation (includes CUDA headers).
- Time Formatting: "Estimated Time to Block" now displays in years, months, and days in English (e.g., "145883385836 years, 0 months, 26.5 days" instead of "53247435828136.5d").
- Hash Value Formatting: Hash values now display with magnitude units (K, M, G, T, P, E) for better readability (e.g., "145.79 KH/s" instead of "145788.53 H/s", "82.33 MH" instead of "82332425 H").
- Workflow Improvements: Docker publish workflow no longer fails if package visibility change fails (image is still built and pushed successfully).
- Favicon Added: Modern SVG favicon with Bitcoin symbol and dashboard gradient design.
- GPU Monitoring: Added nvidia-ml-py support for GPU metrics (usage, temperature, memory) in dashboard.
- Docker Build Fix: Allow PyPI fallback for missing dependencies during Docker build.
- Workflow Fix: Added `set +e` to prevent workflow failure when package visibility change fails.
- Favicon Fix (v2.7.0): Fixed favicon display by implementing static routes (`/favicon.ico`, `/favicon.svg`) instead of data URI, improving browser compatibility and caching.
- Web-based Configuration UI: Added Settings tab in web dashboard for managing all configuration options (pool, network, compute, database). Configuration values are loaded from Docker environment variables and config.toml, with sensitive data (wallet address, RPC passwords) left empty for security. Features include CPU/GPU mining toggles and database retention settings.
- GPU Utilization Control (v2.9.0): Added configurable GPU utilization percentage (1-100%) with time-slicing support. When set below 100%, the miner automatically pauses between GPU batches to free up GPU resources for other tasks (e.g., video transcoding). Implemented with dynamic pause calculation based on actual batch duration for precise control.
- Tags pushed: `v0.1.0`, `v0.1.1`, `v0.1.2`, `v1.0.0`, `v2.0.0` (project renamed to SatoshiRig), `v2.0.1` (NVIDIA GPU runtime support documentation), `v2.0.6-v2.0.10` (Docker image build and publish workflow fixes), `v2.1.0` (Complete WebUI overhaul with charts, stats, history, theme toggle, and Docker WebUI labels), `v2.2.0` (Performance & Monitoring, Mining Intelligence, Advanced Visualizations, WebGUI Navigation fixes), `v2.3.0` (WebApp restructured with tabs, Uptime fix, Pause button functionality, redundant Connected button removed), `v2.4.0` (Time formatting and hash value magnitude units), `v2.5.0` (GPU mining improvements with NVIDIA CUDA base image, enhanced GPU initialization, sequential nonce counter), `v2.5.1` (Dockerfile python symlink fix), `v2.5.2` (CUDA devel image for PyCUDA compilation), `v2.5.3` (Time formatting to English, documentation update), `v2.5.4` (Workflow fix: don't fail if package visibility change fails), `v2.5.5` (Favicon added, GPU monitoring support, Docker build fix, workflow improvements), `v2.5.6` (Code cleanup: remove trailing whitespace), `v2.6.0` (Security fixes, exception handling, encoding, GPU monitoring, workflow triggers, socket handling, hardcoded values, threading, random number generation, missing features), `v2.7.0` (Favicon fix: static route implementation for better browser compatibility), `v2.8.0` (Modern UI redesign, light theme re-implementation, config persistence and validation, full GPU kernel implementation with SHA256), `v2.9.0` (GPU utilization control with time-slicing support).
 - GPU Nonce Fix: Corrected nonce endianness to little-endian in GPU batch hashing.
 - Pool Client Robustness: Improved line-buffered parsing of `mining.notify` to handle partial TCP frames.
 - Config Validation: Enforced defaults and type casting across sections (pool, network, logging, miner, compute).
 - Smaller Image: Switched to multi-stage Docker build (devel builder -> runtime final) to reduce size; final base `nvidia/cuda:11.8.0-runtime-ubuntu22.04`.
 - Manual Workflows: All GitHub Actions now `workflow_dispatch` only; add `DEPLOY.md` with publish/run instructions.

## Open Items / Next Steps
- ✅ GPU mining support implemented (CUDA/OpenCL with parallel batch hashing)
- Optimize GPU kernels for better performance (currently uses parallel CPU threads)
- Add structured logging (JSON) option.
- Add integration tests with a mocked pool server.
- Optional metrics endpoint (Prometheus) for hashrate and connectivity.
- ✅ Docker image published to GHCR: `ghcr.io/rokk001/satoshirig:latest` (public, automatically set on publish).

## How to Resume
- For releases: push a new tag `vX.Y.Z` to trigger the GitHub release workflow.
- For config changes: edit `config/config.toml` or pass via `--config`/env.
- For Unraid: adjust env in the UI or `.env`, then `docker compose up -d`.
- For web dashboard: access via `http://<host>:5000` (default port) when running.
