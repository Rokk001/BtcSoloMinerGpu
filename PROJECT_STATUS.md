# Project Status

Updated: 2025-01-27

## Latest Changes (v2.25.0)
- **Critical Race Condition Fix**: Fixed race condition between pool connection and running miner
  - `connect_to_pool_only()` was closing socket while miner was using it, causing timeouts
  - Added thread-safety with `_socket_lock` in `PoolClient` for all socket operations
  - `connect_to_pool_only()` now checks if miner is running before attempting reconnection
  - Prevents "Subscribe failed: timed out" errors when config is changed during mining
- **Thread-Safety Improvements**: All pool socket operations are now thread-safe
  - Added `_socket_lock` to `PoolClient` class
  - All methods (`connect`, `subscribe`, `authorize`, `read_notify`, `submit`, `close`) now use lock
  - Prevents concurrent socket access conflicts
- **Pool Connection Logic**: Improved pool connection handling
  - `connect_to_pool_only()` skips reconnection if miner is already running
  - Web UI no longer calls `connect_to_pool_only()` when miner is running
  - Better error handling and logging for connection state checks

## Previous Changes (v2.24.0)
- **Critical Pool Subscribe Fix**: Fixed issue where pool sends multiple messages in same buffer
  - Pool may send `mining.notify` message immediately after `mining.subscribe` response
  - Code now searches through all received lines to find the correct subscribe response
  - Looks for response with `result` field and `id: 1` (matching our request)
  - Ignores `mining.notify` messages that may come first
  - Pool connection should now work reliably even when pool sends multiple messages

## Previous Changes (v2.23.0)
- **Critical CUDA Fix**: Fixed CUDA initialization error
  - Replaced `cuda.is_initialized()` with try/except pattern (function doesn't exist in all PyCUDA versions)
  - CUDA now initializes correctly on all PyCUDA versions
- **Critical Pool Connection Fix**: Fixed JSON decode error when pool response is > 1024 bytes
  - Changed from single `recv(1024)` to loop with multiple `recv(4096)` calls
  - Reads until complete line is received (handles responses up to 64KB)
  - Pool connection should now work reliably even with large responses

## Previous Changes (v2.22.0)
- **UI Improvements**: Removed export button from web dashboard
  - Export functionality was not needed, removed to simplify UI
- **Pool Connection Independence**: Pool connection is now established independently of mining
  - New `connect_to_pool_only()` method allows connecting to pool without starting mining
  - Pool connection is automatically established:
    - On application startup (if wallet is configured)
    - When miner is initialized via web UI
    - After pool configuration changes
  - Pool status is now visible in dashboard even when mining is not active
  - Better user experience: users can verify pool connectivity without starting mining

## Previous Changes (v2.21.0)
- **Critical Share Submission Fixes**: Fixed multiple critical bugs that prevented valid shares from being accepted
  - **Syntax Error Fix**: Fixed Python syntax error in GPU mining try-except block (corrected indentation)
  - **Bug #73 - extranonce2**: Fixed incorrect `extranonce2` value in share submission - now uses value from current iteration instead of stale state value
  - **Bug #74 - Race Condition**: Fixed race condition where `job_id`, `ntime`, `prev_hash` could change between solution discovery and submission
  - **Bug #75 - ntime Consistency**: Fixed inconsistent `ntime` usage - now uses single atomic value
  - All solution parameters are now captured atomically when solution is found, preventing state changes from causing share rejection

## Previous Changes (v2.20.0)
- **Critical Bitcoin Protocol Fixes**: Fixed multiple critical issues preventing mining from working
  - **Byte-Order Corrections**: All block header fields (version, prev_hash, merkle_root, ntime, nbits, nonce) are now correctly converted to little-endian format (Bitcoin standard)
  - **Hash Byte-Order**: CPU and GPU hashes are now converted to little-endian before target comparison
  - **Merkle Root Fixes**: Removed incorrect byte-reversal, fixed Merkle branch concatenation order, Merkle root is recalculated dynamically in the loop
  - **Dynamic State Updates**: `merkle_root`, `target`, `target_int`, `target_difficulty` are recalculated in every loop iteration
  - **extranonce2 Sequential Generation**: Changed from random to sequential generation for better coverage, regenerated for each iteration
  - **Nonce Formatting**: Nonces are now formatted in little-endian hex format for Bitcoin block headers
  - **clean_jobs Flag**: Implemented proper handling to reset all nonce counters when pool signals new job
  - **hash_count Updates**: Fixed hash count not being updated when mining is disabled
- **Technical Improvements**:
  - Added `_hex_to_little_endian()` and `_int_to_little_endian_hex()` helper functions
  - Complete refactoring of `_build_block_header()` method with proper byte-order conversion
  - Major refactoring of main mining loop for better state management and error handling

## Previous Changes (v2.19.3)
- **Logging to Docker Logs**: All logging now goes to stdout/stderr instead of separate log files
  - Logs are visible in `docker logs` command
  - No separate log files are created
  - Removed all FileHandler usage in favor of StreamHandler

## Overview
- Repository name: `SatoshiRig`
- Purpose: Neutral Bitcoin solo-mining client with configurable compute backend, Docker/Compose support, CI/CD, and web dashboard.

## Architecture
- Package: `src/SatoshiRig`
  - `core/`: Core mining logic
    - `__init__.py`: Module exports (Miner, MinerState, GPU compute)
    - `miner.py`: `Miner` class (mining loop, logging, GPU support with sequential nonce counter).
    - `state.py`: `MinerState` dataclass (runtime state).
    - `gpu_compute.py`: GPU compute module (CUDA/OpenCL support for GPU mining with improved initialization and error handling).
  - `clients/`: Pool communication
    - `__init__.py`: Module exports (PoolClient)
    - `pool_client.py`: CKPool TCP JSON client (subscribe/authorize/notify/submit).
  - `utils/`: Utility functions
    - `__init__.py`: Module exports (formatting functions)
    - `formatting.py`: Formatting utilities (hash numbers, time to block).
  - `web/`: Web dashboard
    - `__init__.py`: Module exports (start_web_server, status functions)
    - `server.py`: Flask web server with SocketIO for real-time mining status dashboard with tabs (Overview, Performance, Analytics, Intelligence, History, Settings).
    - `status.py`: Status management (STATUS dict, update_status, get_status, etc.).
  - `cli.py`: argparse CLI; loads config, sets up logging, builds dependencies, starts `Miner`.
  - `config.py`: loads TOML config (`CONFIG_FILE` override supported).
  - `miner.py`: thin compatibility facade (DEPRECATED - use `core.miner.Miner` instead).
- Config: `config/config.toml` (pool, network, logging, miner, compute).
- Containerization: `Dockerfile` (NVIDIA CUDA base image `nvidia/cuda:11.8.0-runtime-ubuntu22.04`), `.dockerignore`, `docker-compose.yml` (Unraid-ready, NVIDIA GPU support via `--runtime=nvidia` or `--gpus all`).
- CI: `.github/workflows/ci.yml` (install, format, test), `.github/workflows/release.yml` (releases from tags), `.github/workflows/docker-publish.yml` (builds and publishes Docker image to GHCR).
- Packaging: `pyproject.toml` with console script `satoshirig`.

## Usage
- Local: `python -m SatoshiRig --wallet <ADDR> [--config ./config/config.toml] [--backend cpu|cuda|opencl] [--gpu 0] [--web-port 5000]`
- Docker (local build): `docker build -t satoshirig . && docker run --rm -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data -p 5000:5000 satoshirig`
- Docker (from GHCR): `docker run --rm -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data -p 5000:5000 ghcr.io/rokk001/satoshirig:latest`
- Docker (NVIDIA GPU): `docker run --rm --gpus all -e COMPUTE_BACKEND=cuda -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data ghcr.io/rokk001/satoshirig:latest` or `docker run --rm --runtime=nvidia -e COMPUTE_BACKEND=cuda -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data ghcr.io/rokk001/satoshirig:latest`
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
- Critical Fixes (v2.10.0): Fixed multiple critical issues including: read_notify blocking with max buffer size and iteration limits, MinerState thread-safety with locks, target calculation validation for nbits, submit() timeout handling, _get_current_block_height() error handling in mining loop, deep_merge recursion depth limits, save_config filesystem error handling, and prevention of multiple start() calls.
- Project Structure Optimization (v2.10.0): Improved project organization with proper `__init__.py` files, created `utils/` module for formatting functions, split `web/` module into `server.py` and `status.py`, removed old `BtcSoloMinerGpu/` directory, cleaned up deprecated `miner.py` facade, and improved test structure with `unit/` and `integration/` directories.
- Critical Validation Fixes (v2.11.0): Fixed syntax error (elif indentation), added comprehensive None validation for coinbase fields and block header parameters, added merkle_root hex length validation, added tuple validation before GPU result unpacking, replaced direct config dict access with .get() to prevent KeyError exceptions. All critical validation issues resolved.
- Additional Critical Fixes (v2.12.0): Fixed undefined current_height in exception handler, added ValueError handling for int(hash_hex, 16), added KeyError prevention for pool subscribe response, added None check for last_error in pool connect, added binascii.Error handling for all block_header unhexlify operations in GPU/CPU fallback paths. All identified edge cases and error conditions now properly handled.
- UI Fixes (v2.13.0): Fixed CSRF validation to support `CORS_ORIGINS=*` and same-origin requests, fixed tab navigation by adding CSS rules to hide inactive tabs (`.tab-content { display: none; }` and `.tab-content.active { display: block; }`), improved tab restoration on page load to use `showTab()` directly instead of button clicks.
- Tags pushed: `v0.1.0`, `v0.1.1`, `v0.1.2`, `v1.0.0`, `v2.0.0` (project renamed to SatoshiRig), `v2.0.1` (NVIDIA GPU runtime support documentation), `v2.0.6-v2.0.10` (Docker image build and publish workflow fixes), `v2.1.0` (Complete WebUI overhaul with charts, stats, history, theme toggle, and Docker WebUI labels), `v2.2.0` (Performance & Monitoring, Mining Intelligence, Advanced Visualizations, WebGUI Navigation fixes), `v2.3.0` (WebApp restructured with tabs, Uptime fix, Pause button functionality, redundant Connected button removed), `v2.4.0` (Time formatting and hash value magnitude units), `v2.5.0` (GPU mining improvements with NVIDIA CUDA base image, enhanced GPU initialization, sequential nonce counter), `v2.5.1` (Dockerfile python symlink fix), `v2.5.2` (CUDA devel image for PyCUDA compilation), `v2.5.3` (Time formatting to English, documentation update), `v2.5.4` (Workflow fix: don't fail if package visibility change fails), `v2.5.5` (Favicon added, GPU monitoring support, Docker build fix, workflow improvements), `v2.5.6` (Code cleanup: remove trailing whitespace), `v2.6.0` (Security fixes, exception handling, encoding, GPU monitoring, workflow triggers, socket handling, hardcoded values, threading, random number generation, missing features), `v2.7.0` (Favicon fix: static route implementation for better browser compatibility), `v2.8.0` (Modern UI redesign, light theme re-implementation, config persistence and validation, full GPU kernel implementation with SHA256), `v2.9.0` (GPU utilization control with time-slicing support), `v2.10.0` (Critical fixes: thread-safety, blocking prevention, error handling, validation), `v2.11.0` (Critical validation fixes: syntax error, None validation, KeyError prevention, tuple unpacking safety), `v2.12.0` (Additional critical fixes: undefined variable handling, ValueError/KeyError prevention, binascii.Error handling), `v2.13.0` (UI fixes: CSRF validation with `*` support, tab navigation CSS, tab restoration).
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
