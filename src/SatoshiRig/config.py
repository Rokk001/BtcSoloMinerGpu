import os
import tomllib
import tomli_w
from typing import Any, Dict


DEFAULT_CONFIG_PATHS = [
    os.environ.get("CONFIG_FILE") ,
    os.path.join(os.getcwd() , "config" , "config.toml") ,
    os.path.join(os.path.dirname(os.path.dirname(__file__)) , ".." , "config" , "config.toml") ,
]


def _first_existing_path(paths) :
    for p in paths :
        if p and os.path.isfile(os.path.abspath(p)) :
            return os.path.abspath(p)
    return None


def _validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required sections and types exist; apply sane defaults."""
    # Wallet section (optional, but preserve if present)
    cfg.setdefault("wallet", {})
    cfg["wallet"]["address"] = str(cfg["wallet"].get("address", "") or "")
    
    cfg.setdefault("pool", {})
    cfg["pool"].setdefault("host", "solo.ckpool.org")
    cfg["pool"].setdefault("port", 3333)
    try:
        cfg["pool"]["port"] = int(cfg["pool"]["port"])
    except Exception:
        cfg["pool"]["port"] = 3333

    cfg.setdefault("network", {})
    net = cfg["network"]
    net.setdefault("source", os.environ.get("BLOCK_SOURCE", "web"))
    net.setdefault("latest_block_url", "https://blockchain.info/latestblock")
    net.setdefault("request_timeout_secs", 15)
    net.setdefault("rpc_url", os.environ.get("BITCOIN_RPC_URL", "http://127.0.0.1:8332"))
    net.setdefault("rpc_user", os.environ.get("BITCOIN_RPC_USER", ""))
    net.setdefault("rpc_password", os.environ.get("BITCOIN_RPC_PASSWORD", ""))
    try:
        net["request_timeout_secs"] = int(net["request_timeout_secs"])
    except Exception:
        net["request_timeout_secs"] = 15

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("file", "miner.log")
    cfg["logging"].setdefault("level", "INFO")

    cfg.setdefault("miner", {})
    cfg["miner"].setdefault("restart_delay_secs", 2)
    cfg["miner"].setdefault("subscribe_thread_start_delay_secs", 4)
    cfg["miner"].setdefault("hash_log_prefix_zeros", 7)
    for k in ("restart_delay_secs", "subscribe_thread_start_delay_secs", "hash_log_prefix_zeros"):
        try:
            cfg["miner"][k] = int(cfg["miner"][k])
        except Exception:
            pass

    # Compute section already enriched below; just cast types here
    cfg.setdefault("compute", {})
    cfg["compute"].setdefault("backend", os.environ.get("COMPUTE_BACKEND", "cpu"))
    try:
        cfg["compute"]["gpu_device"] = int(cfg["compute"].get("gpu_device", os.environ.get("GPU_DEVICE", "0")))
    except Exception:
        cfg["compute"]["gpu_device"] = 0
    try:
        cfg["compute"]["batch_size"] = int(cfg["compute"].get("batch_size", os.environ.get("GPU_BATCH_SIZE", "256")))
    except Exception:
        cfg["compute"]["batch_size"] = 256
    try:
        cfg["compute"]["max_workers"] = int(cfg["compute"].get("max_workers", os.environ.get("GPU_MAX_WORKERS", "8")))
    except Exception:
        cfg["compute"]["max_workers"] = 8
    try:
        gpu_util = int(cfg["compute"].get("gpu_utilization_percent", os.environ.get("GPU_UTILIZATION_PERCENT", "100")))
        # Clamp between 1 and 100
        cfg["compute"]["gpu_utilization_percent"] = max(1, min(100, gpu_util))
    except Exception:
        cfg["compute"]["gpu_utilization_percent"] = 100

    return cfg


def load_config() :
    cfg_path = _first_existing_path(DEFAULT_CONFIG_PATHS)
    if not cfg_path :
        return {
            "wallet": {"address": ""},
            "pool": {"host": "solo.ckpool.org" , "port": 3333} ,
            "network": {
                "source": os.environ.get("BLOCK_SOURCE" , "web") ,  # web | local
                "latest_block_url": "https://blockchain.info/latestblock" ,
                "request_timeout_secs": 15 ,
                "rpc_url": os.environ.get("BITCOIN_RPC_URL" , "http://127.0.0.1:8332") ,
                "rpc_user": os.environ.get("BITCOIN_RPC_USER" , "") ,
                "rpc_password": os.environ.get("BITCOIN_RPC_PASSWORD" , "")
            } ,
            "logging": {"file": "miner.log" , "level": "INFO"} ,
            "miner": {
                "restart_delay_secs": 2 ,
                "subscribe_thread_start_delay_secs": 4 ,
                "hash_log_prefix_zeros": 7
            } ,
            "compute": {"backend": os.environ.get("COMPUTE_BACKEND" , "cpu") , "gpu_device": int(os.environ.get("GPU_DEVICE" , "0"))}
        }
    with open(cfg_path , "rb") as f :
        cfg = tomllib.load(f)
    compute = cfg.get("compute" , {})
    if "backend" not in compute :
        compute["backend"] = os.environ.get("COMPUTE_BACKEND" , "cpu")
    if "gpu_device" not in compute :
        compute["gpu_device"] = int(os.environ.get("GPU_DEVICE" , "0"))
    cfg["compute"] = compute
    return _validate_config(cfg)


def save_config(cfg: Dict[str, Any], config_path: str = None) -> str:
    """
    Save configuration to TOML file
    
    Args:
        cfg: Configuration dictionary to save
        config_path: Optional path to config file. If None, uses first existing path or default.
    
    Returns:
        Path to saved config file
    """
    if config_path is None:
        config_path = _first_existing_path(DEFAULT_CONFIG_PATHS)
        if config_path is None:
            # Use default path
            config_path = os.path.join(os.getcwd(), "config", "config.toml")
    
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Failed to create config directory: {e}")
    
    # Validate config before saving
    validated_cfg = _validate_config(cfg.copy())
    
    # Write to file with error handling for filesystem errors
    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(validated_cfg, f)
    except (OSError, PermissionError, IOError) as e:
        raise RuntimeError(f"Failed to write config file '{config_path}': {e}. Check file permissions and disk space.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error saving config file '{config_path}': {e}")
    
    return config_path


