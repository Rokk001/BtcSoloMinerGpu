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

Build:

```
docker build -t satoshirig .
```

Run (CPU):

```
docker run --rm \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  satoshirig
```

Run (NVIDIA GPU):

```
docker run --rm --gpus all \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  satoshirig
```

### Docker Compose (Unraid compatible)

A pre-configured compose file is included (`docker-compose.yml`).

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

Notes:
- For NVIDIA GPU in Unraid, install the Nvidia Driver plugin and optionally enable `runtime: nvidia` in compose.
- For iGPU/AMD, you may need to pass through `/dev/dri` (see commented `devices` block in `docker-compose.yml`).

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
