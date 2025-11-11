# SatoshiRig

<div align="center">

![SatoshiRig Logo](https://img.shields.io/badge/SatoshiRig-Bitcoin%20Solo%20Miner-orange?style=for-the-badge)

**A minimal, neutral Bitcoin solo-mining client with clean architecture, GPU support, and comprehensive web dashboard.**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Docker](#-docker) • [GPU Mining](#-gpu-mining)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Docker](#-docker)
- [GPU Mining](#-gpu-mining)
- [Web Dashboard](#-web-dashboard)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
 - [Deploy](#-deploy)

---

## 🎯 Overview

SatoshiRig is a professional Bitcoin solo-mining client designed for simplicity, reliability, and performance. It features:

- **Clean Architecture**: Modular design with clear separation of concerns
- **GPU Support**: CUDA and OpenCL support for accelerated mining
- **Web Dashboard**: Real-time monitoring and control interface
- **Docker Ready**: Fully containerized with NVIDIA GPU support
- **Configuration**: Flexible TOML-based configuration
- **Production Ready**: Built for long-running mining operations

---

## ✨ Features

### Core Functionality
- ✅ **Solo Mining**: Direct connection to CKPool for solo Bitcoin mining
- ✅ **GPU Support**: CUDA and OpenCL backends with automatic fallback to CPU
- ✅ **Parallel Batch Hashing**: Optimized batch processing (1024 nonces per iteration)
- ✅ **Sequential Nonce Counter**: Complete coverage of the 32-bit nonce space
- ✅ **Block Source**: Configurable source (Blockchain Explorer or local Bitcoin Core RPC)

### Web Dashboard
- 📊 **Real-time Monitoring**: Live hash rate, CPU, memory, and GPU metrics
- 📈 **Performance Analytics**: Historical charts and trend analysis
- 🧠 **Mining Intelligence**: Estimated time to block, probability calculations, profitability estimates
- ⚙️ **Configuration UI**: Web-based settings for all mining parameters (pool, network, compute, database)
- 🎛️ **GPU Utilization Control**: Configure GPU usage percentage (1-100%) to allow other GPU tasks to run simultaneously
- 🎨 **Modern UI**: Tabbed interface with dark/light theme support
- 📱 **Responsive Design**: Works on desktop and mobile devices

### Developer Experience
- 🔧 **TOML Configuration**: Human-readable configuration files
- 🐳 **Docker Support**: Pre-built images on GitHub Container Registry
- 📝 **Comprehensive Logging**: Configurable log levels and file output
- 🚀 **CI/CD Ready**: Automated builds and releases

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Pull and run the latest image
docker run -d \
  --name satoshirig \
  --restart unless-stopped \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -p 5000:5000 \
  ghcr.io/rokk001/satoshirig:latest
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/Rokk001/SatoshiRig.git
cd SatoshiRig

# Install dependencies
pip install -r requirements.txt

# Run miner
python -m SatoshiRig --wallet YOUR_BTC_ADDRESS
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Docker**: 19.03+ (for containerized deployment)
- **NVIDIA Container Toolkit**: (for GPU mining with Docker)

### Option 1: Docker (Recommended)

The easiest way to run SatoshiRig is using the pre-built Docker image:

```bash
docker pull ghcr.io/rokk001/satoshirig:latest
```

### Option 2: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rokk001/SatoshiRig.git
   cd SatoshiRig
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install GPU dependencies (optional):**
   ```bash
   pip install pycuda>=2023.1 pyopencl>=2023.1.2
   ```

---

## ⚙️ Configuration

### Environment Variables

All configuration can be done via environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WALLET_ADDRESS` | ✅ **Yes** | - | Your Bitcoin wallet address (REQUIRED) |
| `CONFIG_FILE` | No | `./config/config.toml` | Path to TOML config file |
| `COMPUTE_BACKEND` | No | `cpu` | Compute backend: `cpu`, `cuda`, or `opencl` |
| `GPU_DEVICE` | No | `0` | GPU device index (for CUDA/OpenCL backends) |
| `GPU_UTILIZATION_PERCENT` | No | `100` | GPU utilization percentage (1-100%) for time-slicing support |
| `WEB_PORT` | No | `5000` | Web dashboard port (set to `0` to disable) |
| `NVIDIA_VISIBLE_DEVICES` | No* | `all` | NVIDIA GPU visibility (*only for NVIDIA GPU) |
| `NVIDIA_DRIVER_CAPABILITIES` | No* | `compute,utility` | NVIDIA driver capabilities (*only for NVIDIA GPU) |

### Configuration File (`config/config.toml`)

The configuration file supports the following sections:

```toml
[pool]
host = "solo.ckpool.org"
port = 3333

[network]
# Block height source: "web" (Blockchain Explorer) or "local" (own Bitcoin Core Node)
source = "web"            # web | local
latest_block_url = "https://blockchain.info/latestblock"
request_timeout_secs = 15

# For source = "local": Standard Bitcoin Core JSON-RPC
rpc_url = "http://127.0.0.1:8332"
rpc_user = ""
rpc_password = ""

[logging]
file = "miner.log"
level = "INFO"            # DEBUG, INFO, WARNING, ERROR

[miner]
restart_delay_secs = 2
subscribe_thread_start_delay_secs = 4
hash_log_prefix_zeros = 7

[compute]
backend = "cpu"           # cpu | cuda | opencl
gpu_device = 0
```

### Command-Line Options

```bash
python -m SatoshiRig --help
```

Available options:
- `--wallet`, `-w`: Bitcoin wallet address (required)
- `--config`: Path to configuration file
- `--backend`: Compute backend (`cpu`, `cuda`, `opencl`)
- `--gpu`: GPU device index
- `--web-port`: Web dashboard port (default: 5000)
- `--no-web`: Disable web dashboard

---

## 🐳 Docker

### Quick Start

**Option 1: Use published image from GHCR (recommended):**

```bash
docker run -d \
  --name satoshirig \
  --restart unless-stopped \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -p 5000:5000 \
  ghcr.io/rokk001/satoshirig:latest
```

**Option 2: Build locally:**

```bash
docker build -t satoshirig .
docker run -d \
  --name satoshirig \
  --restart unless-stopped \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -p 5000:5000 \
  satoshirig
```

### Docker Image Details

- **Published Image**: `ghcr.io/rokk001/satoshirig:latest` (public, automatically updated)
- **Base Image**: Multi-stage build. Final image uses `nvidia/cuda:11.8.0-runtime-ubuntu22.04` (smaller). Build stage briefly uses `devel` to compile GPU wheels.
- **Working Directory**: `/app`
- **Default Config**: `/app/config/config.toml`
- **Default Web Port**: `5000`

### Docker Compose

Example `docker-compose.yml`:

```yaml
services:
  satoshirig:
    image: ghcr.io/rokk001/satoshirig:latest
    container_name: satoshirig
    restart: unless-stopped
    runtime: nvidia
    gpus: all
    environment:
      - WALLET_ADDRESS=YOUR_BTC_ADDRESS
      - COMPUTE_BACKEND=cuda
      - GPU_DEVICE=0
      - WEB_PORT=5000
    ports:
      - "5000:5000"
    volumes:
      - ./logs:/app/logs

### NVIDIA Container Toolkit

To enable GPU access, install the NVIDIA Container Toolkit on the host and run with `--gpus all` (or `runtime: nvidia`). The toolkit passes the NVIDIA driver and devices into the container; the image only needs CUDA runtime libraries.

- Docs: [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

Notes:
- The image is now significantly smaller because the CUDA developer toolchain is not shipped in the final layer. GPU Python deps are compiled in a build stage and only the wheels are copied into the runtime image.
```

Start with:
```bash
docker compose up -d
```

---

## 🎮 GPU Mining

### NVIDIA CUDA

**Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker with GPU support enabled

**Method 1: Using --gpus flag (recommended for Docker 19.03+):**

```bash
docker run -d \
  --name satoshirig \
  --restart unless-stopped \
  --gpus all \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  -p 5000:5000 \
  ghcr.io/rokk001/satoshirig:latest
```

**Method 2: Using --runtime=nvidia (for older Docker versions):**

```bash
docker run -d \
  --name satoshirig \
  --restart unless-stopped \
  --runtime=nvidia \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -p 5000:5000 \
  ghcr.io/rokk001/satoshirig:latest
```

**Specific GPU:**
```bash
docker run -d \
  --name satoshirig \
  --gpus device=0 \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  -p 5000:5000 \
  ghcr.io/rokk001/satoshirig:latest
```

### AMD/OpenCL

For AMD GPUs or integrated GPUs using OpenCL:

```bash
docker run -d \
  --name satoshirig \
  --device=/dev/dri:/dev/dri \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=opencl \
  -e GPU_DEVICE=0 \
  -p 5000:5000 \
  ghcr.io/rokk001/satoshirig:latest
```

### GPU Mining Notes

- **Automatic Fallback**: The miner automatically falls back to CPU if GPU initialization fails
- **Batch Processing**: GPU mining uses parallel batch hashing (1024 nonces per iteration)
- **Sequential Nonce Counter**: Complete coverage of the 32-bit nonce space
- **GPU Utilization Control**: Configure GPU usage percentage (1-100%) via web UI or config to allow other GPU tasks (e.g., video transcoding) to run simultaneously
- **Time-Slicing**: When GPU utilization is set below 100%, the miner automatically pauses between batches to free up GPU resources

---

## 📊 Web Dashboard

The web dashboard provides real-time monitoring and control of your mining operation.

### Access

Once running, access the dashboard at:
- **Local**: `http://localhost:5000`
- **Docker**: `http://<container-ip>:5000`
- **Remote**: `http://<host-ip>:5000`

### Features

#### Overview Tab
- Mining status and basic metrics
- Hash rate, uptime, and shares
- Wallet address with blockchain explorer link
- Pool connection status

#### Performance Tab
- **System Resources**: CPU, Memory, GPU usage and temperature
- **Performance Dashboard**: Real-time metrics visualization
- **GPU Monitoring**: NVIDIA GPU metrics (if available)

#### Analytics Tab
- **Hash Rate History**: Historical hash rate trends
- **Difficulty History**: Network difficulty over time
- **Comparison Charts**: Hash rate vs difficulty analysis

#### Intelligence Tab
- **Estimated Time to Block**: Formatted in years, months, and days
- **Block Found Probability**: Probability of finding a block in the next hour
- **Estimated Profitability**: BTC per day estimate
- **Difficulty Trend**: Network difficulty trend analysis (increasing/decreasing/stable)

#### History Tab
- **Share History**: Recent share submissions
- **Statistics Table**: Comprehensive mining statistics

#### Settings Tab
- **Pool Configuration**: Host and port settings
- **Network Configuration**: Block source (web/local), RPC settings
- **Compute Configuration**: Backend (CPU/CUDA/OpenCL), GPU device, batch size, workers, GPU utilization percentage
- **Mining Toggles**: Enable/disable CPU and GPU mining independently
- **Database Configuration**: Retention period in days

**Security Note**: Sensitive data (wallet address, RPC passwords) are not pre-filled from Docker environment variables for security reasons. You must enter these manually in the web UI.

Configuration values are loaded from:
1. Docker environment variables (e.g., `COMPUTE_BACKEND`, `GPU_DEVICE`)
2. `config.toml` file
3. Default values if not specified

Changes made in the web UI are saved to the database and can be applied to the running miner (requires restart for some settings).

### Formatting

- **Hash Values**: Automatically formatted with magnitude units (K, M, G, T, P, E)
  - Example: `145.79 KH/s` instead of `145788.53 H/s`
- **Time Estimates**: Displayed in years, months, and days
  - Example: `143640979699 years, 10 months, 8.5 days`

### Controls

- **Pause/Resume**: Control mining via the web interface
- **Theme Toggle**: Switch between dark and light themes
- **Auto-refresh**: Real-time updates via WebSocket

---

## 📁 Project Structure

```
SatoshiRig/
├── src/
│   └── SatoshiRig/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py                 # Command-line interface
│       ├── config.py              # TOML configuration loader
│       ├── miner.py               # Backward-compatible facade
│       ├── clients/
│       │   └── pool_client.py     # CKPool TCP JSON client
│       ├── core/
│       │   ├── miner.py           # Core mining logic
│       │   ├── state.py            # Miner state management
│       │   └── gpu_compute.py      # GPU compute (CUDA/OpenCL)
│       └── web/
│           └── server.py           # Flask web server with SocketIO
├── config/
│   └── config.toml                 # Default configuration
├── .github/
│   └── workflows/                  # CI/CD workflows
├── Dockerfile                       # Docker image definition
├── docker-compose.yml               # Docker Compose configuration
├── pyproject.toml                   # Python project metadata
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔧 Troubleshooting

### Common Issues

#### Docker Image Build Fails

**Problem**: Build fails with CUDA header errors

**Solution**: The Docker image uses `nvidia/cuda:11.8.0-devel-ubuntu22.04` which includes CUDA development tools. Ensure you're using the latest version of the Dockerfile.

#### GPU Mining Not Working

**Problem**: GPU mining falls back to CPU

**Checklist**:
1. Verify GPU is available: `nvidia-smi` (for NVIDIA)
2. Ensure `--gpus all` or `--runtime=nvidia` is set
3. Check `COMPUTE_BACKEND=cuda` or `COMPUTE_BACKEND=opencl` is set
4. Review logs for GPU initialization errors
5. Verify NVIDIA Container Toolkit is installed

#### Web Dashboard Not Accessible

**Problem**: Cannot access dashboard at `http://localhost:5000`

**Solution**:
1. Check if `WEB_PORT` is set to `0` (disabled)
2. Verify port mapping: `-p 5000:5000` or `ports: ["5000:5000"]`
3. Check firewall settings
4. Review container logs for errors

#### Package Visibility Issues

**Problem**: Docker image not accessible from GHCR

**Solution**: The package should be automatically set to public. If not:
1. Go to GitHub Packages: https://github.com/Rokk001?tab=packages
2. Select the `satoshirig` package
3. Go to "Package settings" → "Change visibility" → "Make public"

#### Pool notifications arrive partially or parsing fails

**Solution**: The TCP client now uses line-buffered reads and tolerates partial frames. If issues persist, check network stability and pool availability.

#### Manual-only workflows

**Note**: CI, release, and Docker publish workflows run only via manual trigger (`workflow_dispatch`). See `DEPLOY.md` for how to publish the image and create a release.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes

---

## 📄 License

This project is licensed under the MIT License.

---

## 🔗 Links

- **GitHub Repository**: https://github.com/Rokk001/SatoshiRig
- **Docker Image**: `ghcr.io/rokk001/satoshirig:latest`
- **GitHub Packages**: https://github.com/Rokk001?tab=packages&package_name=satoshirig

---

## 🚀 Deploy

See `DEPLOY.md` for manual publish via GitHub Actions and GPU run instructions. Workflows are manual-only; trigger them from Actions when you want to publish.

## 📝 Notes

### GPU Mining

GPU mining support is implemented with CUDA/OpenCL backends. The Docker image uses NVIDIA CUDA base image (`nvidia/cuda:11.8.0-devel-ubuntu22.04`) for proper GPU support. The miner automatically uses GPU if available and configured, otherwise falls back to CPU. GPU mining uses parallel batch hashing (1024 nonces per iteration) with sequential nonce counter for complete coverage. Enhanced GPU initialization with better error handling and device validation. For optimal performance, GPU kernels can be further optimized.

### Web Dashboard Formatting

- Hash values (Hash Rate, Peak Hash Rate, Average Hash Rate, Total Hashes) are automatically formatted with magnitude units (K, M, G, T, P, E) for better readability.
- Estimated time to block is displayed in years, months, and days for easier comprehension of very large time periods.

---

<div align="center">

**Made with ❤️ for the Bitcoin community**

⭐ Star this repo if you find it useful!

</div>
