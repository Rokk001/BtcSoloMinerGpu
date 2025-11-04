## SatoshiRig

Minimal, neutral Bitcoin Solo-Mining Client with clean architecture, config via TOML, and Docker support.

### Project Structure

```
src/
  SatoshiRig/
    __init__.py
    __main__.py
    cli.py                 # argparse CLI, Dependency Injection
    config.py              # TOML-Loader (CONFIG_FILE override supported)
    miner.py               # Backward-compatible facade (import convenience)
    clients/
      pool_client.py       # CKPool TCP JSON client
    core/
      miner.py             # Miner class (core logic)
      state.py             # MinerState dataclass
    web/
      server.py            # Flask web server with SocketIO for live dashboard
config/
  config.toml              # Default configuration
```

### Installation

```
pip install -r requirements.txt
```

### Run Locally

```
python -m SatoshiRig --wallet YOUR_BTC_ADDRESS --config ./config/config.toml --backend cpu --gpu 0
```

Alternatively via environment variables:

```
set WALLET_ADDRESS=YOUR_BTC_ADDRESS  # Windows
set CONFIG_FILE=./config/config.toml
set COMPUTE_BACKEND=cpu
set GPU_DEVICE=0
python -m SatoshiRig
```

### Docker

**Option 1: Use published image from GHCR (recommended):**

```
docker run --rm \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  ghcr.io/rokk001/satoshirig:latest
```

**Option 2: Build locally:**

```
docker build -t satoshirig .
```

Run (CPU):

```
docker run --rm \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  satoshirig
```

Run (NVIDIA GPU) - Using published image from GHCR:

**Method 1: Using --gpus flag (recommended for Docker 19.03+):**
```
docker run --rm --gpus all \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  ghcr.io/rokk001/satoshirig:latest
```

**Method 2: Using --runtime=nvidia (for older Docker versions or when --gpus is not available):**
```
docker run --rm --runtime=nvidia \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  ghcr.io/rokk001/satoshirig:latest
```

**Note:** The `--runtime=nvidia` parameter is required for Docker containers that need NVIDIA GPU access. Ensure you have the NVIDIA Container Toolkit installed on your system.

### Docker Compose (Unraid compatible)

A pre-configured compose file is included (`docker-compose.yml`).

**Important:** The compose file uses `build: .` to build from the local Dockerfile by default. 

**Using the published Docker image (recommended):**
The Docker image is now available on GitHub Container Registry (GHCR) and automatically set to public:
- Image: `ghcr.io/rokk001/satoshirig:latest`
- Status: Public (automatically set on publish)
- Package: https://github.com/Rokk001?tab=packages&package_name=satoshirig

To use the pre-built public image from GitHub Container Registry instead of building locally:
1. In `docker-compose.yml`, comment out the `build:` section
2. Uncomment the `image: ghcr.io/rokk001/satoshirig:latest` line
3. The image will be pulled automatically from GHCR

**Note:** The package is automatically set to public when published via the workflow. No manual steps required.

1) Set environment variables (Unraid UI or `.env` in project directory):

```
WALLET_ADDRESS=YOUR_BTC_ADDRESS
COMPUTE_BACKEND=cpu
GPU_DEVICE=0
NVIDIA_VISIBLE_DEVICES=all
WEB_PORT=5000
```

2) Start:

```
docker compose up -d
```

**For Unraid - IMPORTANT:**

If you see "access denied" errors for `ghcr.io/rokk001/satoshirig:latest`, Unraid is trying to pull an image instead of building locally.

**Solution 1: Fix in Unraid UI**
1. Go to Docker Compose Manager or container settings
2. Make sure "Build" is selected, NOT "Image"
3. If you see an "Image" field with `ghcr.io/rokk001/satoshirig:latest`, DELETE it or clear it completely
4. Make sure "Build Context" points to the project directory (where Dockerfile is located)
5. Make sure "Dockerfile" is set to `Dockerfile` or `./Dockerfile`
6. Save and restart the stack

**Solution 2: Use alternative compose file**
If Unraid keeps overriding the settings, use the build-only compose file:
```bash
docker-compose -f docker-compose.build.yml up -d
```

**Solution 3: Manual fix**
1. Stop the container in Unraid
2. Edit the `docker-compose.yml` file directly in Unraid (via terminal or file manager)
3. Search for any line starting with `image:` (even if commented)
4. Remove or comment out ALL `image:` lines completely
5. Make sure ONLY `build:` section exists (with `context: .` and `dockerfile: Dockerfile`)
6. Save the file
7. Restart the stack in Unraid

**The compose file can use either `build:` for local builds or `image: ghcr.io/rokk001/satoshirig:latest` for the published image from GHCR.**

**NVIDIA GPU Setup:**

For NVIDIA GPU support in Docker Compose, you have two options:

1. **Enable `runtime: nvidia` in docker-compose.yml** (uncomment the line):
   ```yaml
   runtime: nvidia
   ```
   This requires the NVIDIA Container Toolkit to be installed on your system.

2. **Use `deploy.resources.reservations.devices`** (for Docker Compose v3.8+):
   Uncomment the `deploy` section in `docker-compose.yml` instead of `runtime: nvidia`.

**For Unraid:**
- Install the "Nvidia Driver" plugin from Community Applications
- Enable `runtime: nvidia` in the compose file (uncomment the line)
- Set `COMPUTE_BACKEND=cuda` in your environment variables

**For AMD/iGPU:**
- You may need to pass through `/dev/dri` (see commented `devices` block in `docker-compose.yml`)

### Web Dashboard

The web dashboard is available on port 5000 (default) when running. Access via:

- Local: `http://localhost:5000`
- Docker: `http://<container-ip>:5000`
- Compose: `http://<host-ip>:5000`

Disable web dashboard with `--no-web` flag or set `WEB_PORT` to 0.

### Configuration

`config/config.toml` controls pool, network, logging, and mining parameters.

```
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
level = "INFO"

[miner]
restart_delay_secs = 2
subscribe_thread_start_delay_secs = 4
hash_log_prefix_zeros = 7

[compute]
backend = "cpu"  # cpu | cuda | opencl
gpu_device = 0
```

Override paths via `--config` or `CONFIG_FILE` environment variable.

Switching network source:
- Webservice (default): no action needed; uses `https://blockchain.info/latestblock`.
- Local node: set `source = "local"` in config and provide `rpc_url`, `rpc_user`, `rpc_password`.

### Logging

- Neutral, unobtrusive log messages (no colors/banners/branding)
- Log level and file configurable via `[logging]` section

### Notes

- GPU backend parameters are prepared; hash calculation currently uses CPU. A CUDA/OpenCL implementation can be added modularly via `compute` section.

### Releases

Stable versions are published as tags. See Releases/Tags on GitHub. CI automatically creates releases when tagging.
