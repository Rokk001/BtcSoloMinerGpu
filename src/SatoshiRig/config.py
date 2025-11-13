import os
import tomllib
import tomli_w
from typing import Any, Dict

from .db import get_value, set_value


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


def _bool_from_str(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_db_overrides(cfg: Dict[str, Any]) -> None:
    # Wallet - ALWAYS from DB, never from config.toml
    wallet_addr = get_value("settings", "wallet_address")
    cfg.setdefault("wallet", {})
    cfg["wallet"]["address"] = wallet_addr.strip() if wallet_addr is not None else ""

    # Pool
    pool_cfg = cfg.setdefault("pool", {})
    host = get_value("pool", "host")
    if host is not None:
        pool_cfg["host"] = host
    port = get_value("pool", "port")
    if port is not None:
        try:
            pool_cfg["port"] = int(port)
        except ValueError:
            pass

    # Network
    net_cfg = cfg.setdefault("network", {})
    for key in ("source", "latest_block_url", "rpc_url", "rpc_user", "rpc_password"):
        db_val = get_value("network", key)
        if db_val is not None:
            net_cfg[key] = db_val
    timeout_val = get_value("network", "request_timeout_secs")
    if timeout_val is not None:
        try:
            net_cfg["request_timeout_secs"] = int(timeout_val)
        except ValueError:
            pass

    # Compute
    comp_cfg = cfg.setdefault("compute", {})
    backend_val = get_value("compute", "backend")
    if backend_val is not None:
        comp_cfg["backend"] = backend_val
    for key in ("gpu_device", "batch_size", "max_workers", "gpu_utilization_percent"):
        db_val = get_value("compute", key)
        if db_val is not None:
            try:
                comp_cfg[key] = int(db_val)
            except ValueError:
                comp_cfg[key] = db_val
    cpu_enabled = get_value("compute", "cpu_mining_enabled")
    if cpu_enabled is not None:
        comp_cfg["cpu_mining_enabled"] = _bool_from_str(cpu_enabled, True)
    gpu_enabled = get_value("compute", "gpu_mining_enabled")
    if gpu_enabled is not None:
        comp_cfg["gpu_mining_enabled"] = _bool_from_str(gpu_enabled, False)

    # Database retention
    db_cfg = cfg.setdefault("database", {})
    retention = get_value("database", "retention_days")
    if retention is not None:
        try:
            db_cfg["retention_days"] = int(retention)
        except ValueError:
            pass


def persist_config_to_db(cfg: Dict[str, Any]) -> None:
    # Wallet
    set_value("settings", "wallet_address", cfg.get("wallet", {}).get("address", ""))

    # Pool
    pool_cfg = cfg.get("pool", {})
    set_value("pool", "host", str(pool_cfg.get("host", "")))
    set_value("pool", "port", str(pool_cfg.get("port", 0)))

    # Network
    net_cfg = cfg.get("network", {})
    for key in ("source", "latest_block_url", "rpc_url", "rpc_user", "rpc_password"):
        set_value("network", key, str(net_cfg.get(key, "")))
    set_value(
        "network",
        "request_timeout_secs",
        str(net_cfg.get("request_timeout_secs", 15)),
    )

    # Compute
    comp_cfg = cfg.get("compute", {})
    set_value("compute", "backend", str(comp_cfg.get("backend", "cuda")))
    for key in ("gpu_device", "batch_size", "max_workers", "gpu_utilization_percent"):
        set_value("compute", key, str(comp_cfg.get(key, 0)))
    set_value(
        "compute",
        "cpu_mining_enabled",
        "1" if comp_cfg.get("cpu_mining_enabled", True) else "0",
    )
    set_value(
        "compute",
        "gpu_mining_enabled",
        "1" if comp_cfg.get("gpu_mining_enabled", False) else "0",
    )

    # Database retention
    db_cfg = cfg.get("database", {})
    set_value("database", "retention_days", str(db_cfg.get("retention_days", 30)))


def _validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required sections and types exist; apply sane defaults."""
    # Wallet section - address comes from DB only, preserve what's set
    cfg.setdefault("wallet", {})
    # Do NOT set default address - it comes from DB only via _apply_db_overrides
    if "address" not in cfg["wallet"]:
        cfg["wallet"]["address"] = ""
    
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
    
    # Set defaults for mining toggles if not present
    if "cpu_mining_enabled" not in cfg["compute"]:
        cfg["compute"]["cpu_mining_enabled"] = True  # Default: CPU mining enabled
    if "gpu_mining_enabled" not in cfg["compute"]:
        cfg["compute"]["gpu_mining_enabled"] = False  # Default: GPU mining disabled

    return cfg


def load_config() :
    cfg_path = _first_existing_path(DEFAULT_CONFIG_PATHS)
    if not cfg_path :
        cfg = {
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
        _apply_db_overrides(cfg)
        return _validate_config(cfg)
    with open(cfg_path , "rb") as f :
        cfg = tomllib.load(f)
    # Remove wallet address from loaded config (it's DB-only)
    if "wallet" in cfg:
        cfg["wallet"] = cfg["wallet"].copy()
        cfg["wallet"].pop("address", None)
    compute = cfg.get("compute" , {})
    if "backend" not in compute :
        compute["backend"] = os.environ.get("COMPUTE_BACKEND" , "cpu")
    if "gpu_device" not in compute :
        compute["gpu_device"] = int(os.environ.get("GPU_DEVICE" , "0"))
    cfg["compute"] = compute
    _apply_db_overrides(cfg)  # Wallet comes from DB
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
    
    # Remove wallet address before saving to file (it's DB-only)
    file_cfg = validated_cfg.copy()
    if "wallet" in file_cfg:
        file_cfg["wallet"] = file_cfg["wallet"].copy()
        file_cfg["wallet"].pop("address", None)
    
    persist_config_to_db(validated_cfg)  # Save to DB with wallet
    
    # Write to file WITHOUT wallet address
    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(file_cfg, f)
    except (OSError, PermissionError, IOError) as e:
        raise RuntimeError(f"Failed to write config file '{config_path}': {e}. Check file permissions and disk space.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error saving config file '{config_path}': {e}")
    
    return config_path


