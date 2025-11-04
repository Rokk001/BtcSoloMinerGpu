## BtcSoloMinerGpu

Minimal, neutral Bitcoin Solo-Mining Client with clean architecture, config via TOML, and Docker support.

### Projektstruktur

```
src/
  BtcSoloMinerGpu/
    __init__.py
    __main__.py
    cli.py                 # argparse CLI, Dependency Injection
    config.py              # TOML-Loader (CONFIG_FILE override möglich)
    miner.py               # Abwärtskompatible Fassade (import convenience)
    clients/
      pool_client.py       # CKPool TCP JSON client
    core/
      miner.py             # Miner-Klasse (Kernlogik)
      state.py             # MinerState-Datenklasse
config/
  config.toml              # Standard-Konfiguration
```

### Installation

```
pip install -r requirements.txt
```

### Ausführen (lokal)

```
python -m BtcSoloMinerGpu --wallet YOUR_BTC_ADDRESS --config ./config/config.toml --backend cpu --gpu 0
```

Alternativ über Umgebungsvariablen:

```
set WALLET_ADDRESS=YOUR_BTC_ADDRESS
set CONFIG_FILE=./config/config.toml
set COMPUTE_BACKEND=cpu
set GPU_DEVICE=0
python -m BtcSoloMinerGpu
```

### Docker

Build:

```
docker build -t btcsominer-gpu .
```

Run (CPU):

```
docker run --rm \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  btcsominer-gpu
```

Run (NVIDIA GPU):

```
docker run --rm --gpus all \
  -e WALLET_ADDRESS=YOUR_BTC_ADDRESS \
  -e COMPUTE_BACKEND=cuda \
  -e GPU_DEVICE=0 \
  btcsominer-gpu

### Docker Compose (Unraid geeignet)

Eine vorgefertigte Compose-Datei ist enthalten (`docker-compose.yml`).

1) Umgebungsvariablen setzen (Unraid UI oder `.env` im Projektverzeichnis):

```
WALLET_ADDRESS=YOUR_BTC_ADDRESS
COMPUTE_BACKEND=cpu
GPU_DEVICE=0
NVIDIA_VISIBLE_DEVICES=all
```

2) Starten:

```
docker compose up -d
```

Hinweise:
- Für NVIDIA-GPU in Unraid den Nvidia-Driver installieren und in der Compose optional `runtime: nvidia` aktivieren.
- Für iGPU/AMD kann ggf. `/dev/dri` durchreichen (siehe auskommentierter `devices`-Block in `docker-compose.yml`).

### Releases

Stabile Versionen werden als Tags veröffentlicht. Siehe Releases/Tags auf GitHub. Die CI erstellt Releases automatisch beim Taggen.
```

### Konfiguration

`config/config.toml` steuert Pool, Netzwerk, Logging und Mining-Parameter.

```
[pool]
host = "solo.ckpool.org"
port = 3333

[network]
# Datenquelle für aktuelle Blockhöhe: "web" (Blockchain Explorer) oder "local" (eigener Bitcoin Core Node)
source = "web"            # web | local
latest_block_url = "https://blockchain.info/latestblock"
request_timeout_secs = 15
# Für source = "local": Standard Bitcoin Core JSON-RPC
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

Override-Pfade per `--config` oder `CONFIG_FILE` möglich.

Netzwerkquelle umschalten:
- Webservice (Standard): keine weitere Aktion nötig; nutzt `https://blockchain.info/latestblock`.
- Lokaler Node: setze in der Config `source = "local"` und trage `rpc_url`, `rpc_user`, `rpc_password` ein.

### Logging

- Neutrale, unauffällige Logmeldungen (keine Farben/Banner/Branding)
- Log-Level und Datei via `[logging]` konfigurierbar

### Hinweise

- GPU-Backend-Parameter sind vorbereitet; die Hash-Berechnung nutzt derzeit CPU. Eine CUDA/OpenCL-Implementierung kann modular über `compute` ergänzt werden.

Clean refactor with professional layout, standard naming, and Docker support.

### Structure

```
src/
  BtcSoloMinerGpu/
    __init__.py
    __main__.py
    cli.py
    context.py
    miner.py
tests/
requirements.txt
Dockerfile
.dockerignore
.gitignore
```

### Run locally

1) Install dependencies:
```
pip install -r requirements.txt
```

2) Run with wallet address (optional: custom config path):
```
python -m BtcSoloMinerGpu --wallet YOUR_BTC_ADDRESS --config ./config/config.toml --backend cpu --gpu 0
```

or via environment variable:
```
set WALLET_ADDRESS=YOUR_BTC_ADDRESS  # Windows
set CONFIG_FILE=./config/config.toml
set COMPUTE_BACKEND=cpu
set GPU_DEVICE=0
python -m BtcSoloMinerGpu
```

### Docker

Build image:
```
docker build -t btcsominer-gpu .
```

Run (pass wallet via env):
```
docker run --rm --gpus all -e WALLET_ADDRESS=YOUR_BTC_ADDRESS -e COMPUTE_BACKEND=cuda -e GPU_DEVICE=0 btcsominer-gpu
```

Config file is TOML at `config/config.toml`. Override with `--config` or `CONFIG_FILE`.

Press Ctrl+C to stop gracefully.
