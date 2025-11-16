import os
import logging
from typing import Any, Dict

from .db import get_value, set_value
from .utils.logging_utils import _vlog

_logger = logging.getLogger("SatoshiRig.config")
_verbose_logging = True  # Always enable verbose logging for config


def _bool_from_str(value: str, default: bool = False) -> bool:
    _vlog(_logger, _verbose_logging, f"config._bool_from_str: START value={value}, default={default}")
    _vlog(_logger, _verbose_logging, f"config._bool_from_str: checking value is None: {value is None}")
    if value is None:
        _vlog(_logger, _verbose_logging, f"config._bool_from_str: value is None, returning default={default}")
        return default
    _vlog(_logger, _verbose_logging, f"config._bool_from_str: converting value to string and stripping")
    value_str = str(value).strip().lower()
    _vlog(_logger, _verbose_logging, f"config._bool_from_str: value_str={value_str}")
    result = value_str in {"1", "true", "yes", "on"}
    _vlog(_logger, _verbose_logging, f"config._bool_from_str: result={result}, END")
    return result


def load_config() -> Dict[str, Any]:
    """
    Load configuration from database only (no TOML files).
    Falls back to environment variables and defaults.
    """
    _vlog(_logger, _verbose_logging, "config.load_config: START")
    cfg = {}
    _vlog(_logger, _verbose_logging, f"config.load_config: cfg dict created, keys={list(cfg.keys())}")
    
    # Wallet - ALWAYS from DB only
    _vlog(_logger, _verbose_logging, "config.load_config: getting wallet address from DB")
    wallet_addr = get_value("settings", "wallet_address")
    _vlog(_logger, _verbose_logging, f"config.load_config: wallet_addr={'present' if wallet_addr else 'None'}, length={len(wallet_addr) if wallet_addr else 0}")
    _vlog(_logger, _verbose_logging, "config.load_config: setting cfg['wallet']")
    cfg["wallet"] = {"address": wallet_addr.strip() if wallet_addr else ""}
    _vlog(_logger, _verbose_logging, f"config.load_config: cfg['wallet'] set, address length={len(cfg['wallet']['address'])}")
    
    # Pool - from DB, fallback to env/defaults
    pool_host = get_value("pool", "host")
    pool_port = get_value("pool", "port")
    cfg["pool"] = {
        "host": pool_host if pool_host else os.environ.get("POOL_HOST", "solo.ckpool.org"),
        "port": int(pool_port) if pool_port else int(os.environ.get("POOL_PORT", "3333"))
    }
    
    # Network - from DB, fallback to env/defaults
    cfg["network"] = {
        "source": get_value("network", "source") or os.environ.get("BLOCK_SOURCE", "web"),
        "latest_block_url": get_value("network", "latest_block_url") or "https://blockchain.info/latestblock",
        "request_timeout_secs": int(get_value("network", "request_timeout_secs") or os.environ.get("REQUEST_TIMEOUT", "15")),
        "rpc_url": get_value("network", "rpc_url") or os.environ.get("BITCOIN_RPC_URL", "http://127.0.0.1:8332"),
        "rpc_user": get_value("network", "rpc_user") or os.environ.get("BITCOIN_RPC_USER", ""),
        "rpc_password": get_value("network", "rpc_password") or os.environ.get("BITCOIN_RPC_PASSWORD", ""),
    }
    
    # Logging - from DB, fallback to env/defaults
    cfg["logging"] = {
        "file": get_value("logging", "file") or os.environ.get("LOG_FILE", "miner.log"),
        "level": get_value("logging", "level") or os.environ.get("LOG_LEVEL", "INFO"),
        "verbose": _bool_from_str(get_value("logging", "verbose") or os.environ.get("VERBOSE_LOGGING", "false"), False),
    }
    
    # Miner - from DB, fallback to defaults
    cfg["miner"] = {
        "restart_delay_secs": int(get_value("miner", "restart_delay_secs") or "2"),
        "subscribe_thread_start_delay_secs": int(get_value("miner", "subscribe_thread_start_delay_secs") or "4"),
        "hash_log_prefix_zeros": int(get_value("miner", "hash_log_prefix_zeros") or "7"),
    }
    
    # Compute - from DB, fallback to env/defaults
    backend = get_value("compute", "backend") or os.environ.get("COMPUTE_BACKEND", "cpu")
    cfg["compute"] = {
        "backend": backend,
        "gpu_device": int(get_value("compute", "gpu_device") or os.environ.get("GPU_DEVICE", "0")),
        "batch_size": int(get_value("compute", "batch_size") or os.environ.get("GPU_BATCH_SIZE", "256")),
        "max_workers": int(get_value("compute", "max_workers") or os.environ.get("GPU_MAX_WORKERS", "8")),
        "gpu_utilization_percent": int(get_value("compute", "gpu_utilization_percent") or os.environ.get("GPU_UTILIZATION_PERCENT", "100")),
        "cpu_mining_enabled": _bool_from_str(get_value("compute", "cpu_mining_enabled"), True),
        "gpu_mining_enabled": _bool_from_str(get_value("compute", "gpu_mining_enabled"), False),
    }
    
    # Database retention
    retention = get_value("database", "retention_days")
    cfg["database"] = {
        "retention_days": int(retention) if retention else 30
    }
    
    return _validate_config(cfg)


def _validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required sections and types exist; apply sane defaults."""
    # Wallet
    cfg.setdefault("wallet", {})
    if "address" not in cfg["wallet"]:
        cfg["wallet"]["address"] = ""
    
    # Pool
    cfg.setdefault("pool", {})
    cfg["pool"].setdefault("host", "solo.ckpool.org")
    cfg["pool"].setdefault("port", 3333)
    try:
        cfg["pool"]["port"] = int(cfg["pool"]["port"])
    except Exception:
        cfg["pool"]["port"] = 3333

    # Network
    cfg.setdefault("network", {})
    net = cfg["network"]
    net.setdefault("source", "web")
    net.setdefault("latest_block_url", "https://blockchain.info/latestblock")
    net.setdefault("request_timeout_secs", 15)
    net.setdefault("rpc_url", "http://127.0.0.1:8332")
    net.setdefault("rpc_user", "")
    net.setdefault("rpc_password", "")
    try:
        net["request_timeout_secs"] = int(net["request_timeout_secs"])
    except Exception:
        net["request_timeout_secs"] = 15

    # Logging
    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("file", "miner.log")
    cfg["logging"].setdefault("level", "INFO")
    cfg["logging"].setdefault("verbose", False)

    # Miner
    cfg.setdefault("miner", {})
    cfg["miner"].setdefault("restart_delay_secs", 2)
    cfg["miner"].setdefault("subscribe_thread_start_delay_secs", 4)
    cfg["miner"].setdefault("hash_log_prefix_zeros", 7)
    for k in ("restart_delay_secs", "subscribe_thread_start_delay_secs", "hash_log_prefix_zeros"):
        try:
            cfg["miner"][k] = int(cfg["miner"][k])
        except Exception:
            pass

    # Compute
    cfg.setdefault("compute", {})
    cfg["compute"].setdefault("backend", "cpu")
    try:
        cfg["compute"]["gpu_device"] = int(cfg["compute"].get("gpu_device", 0))
    except Exception:
        cfg["compute"]["gpu_device"] = 0
    try:
        cfg["compute"]["batch_size"] = int(cfg["compute"].get("batch_size", 256))
    except Exception:
        cfg["compute"]["batch_size"] = 256
    try:
        cfg["compute"]["max_workers"] = int(cfg["compute"].get("max_workers", 8))
    except Exception:
        cfg["compute"]["max_workers"] = 8
    try:
        gpu_util = int(cfg["compute"].get("gpu_utilization_percent", 100))
        cfg["compute"]["gpu_utilization_percent"] = max(1, min(100, gpu_util))
    except Exception:
        cfg["compute"]["gpu_utilization_percent"] = 100
    
    if "cpu_mining_enabled" not in cfg["compute"]:
        cfg["compute"]["cpu_mining_enabled"] = True
    if "gpu_mining_enabled" not in cfg["compute"]:
        cfg["compute"]["gpu_mining_enabled"] = False

    # Database
    cfg.setdefault("database", {})
    cfg["database"].setdefault("retention_days", 30)
    try:
        cfg["database"]["retention_days"] = int(cfg["database"]["retention_days"])
    except Exception:
        cfg["database"]["retention_days"] = 30

    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """
    Save configuration to database only (no TOML files).
    """
    persist_config_to_db(cfg)


def persist_config_to_db(cfg: Dict[str, Any]) -> None:
    """Persist configuration to database"""
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

    # Logging
    log_cfg = cfg.get("logging", {})
    set_value("logging", "level", str(log_cfg.get("level", "INFO")))
    set_value("logging", "file", str(log_cfg.get("file", "miner.log")))

    # Miner
    miner_cfg = cfg.get("miner", {})
    set_value("miner", "restart_delay_secs", str(miner_cfg.get("restart_delay_secs", 2)))
    set_value("miner", "subscribe_thread_start_delay_secs", str(miner_cfg.get("subscribe_thread_start_delay_secs", 4)))
    set_value("miner", "hash_log_prefix_zeros", str(miner_cfg.get("hash_log_prefix_zeros", 7)))

    # Database retention
    db_cfg = cfg.get("database", {})
    set_value("database", "retention_days", str(db_cfg.get("retention_days", 30)))


CPU_MINING_ENABLED = True
