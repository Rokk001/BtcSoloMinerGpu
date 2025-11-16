import json
import logging
import os
import platform
import psutil
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from flask import Flask, Response, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# Fix for "Too many packets in payload" error - increase max_decode_packets
try:
    from engineio.payload import Payload

    Payload.max_decode_packets = (
        500  # Increase from default (likely 16) to handle large payloads
    )
except ImportError:
    pass  # Fallback if engineio is not available

from ..core.state import MinerState
from ..utils.formatting import format_hash_number, format_time_to_block
from ..config import persist_config_to_db
from ..logging_config import configure_logging

# Try to import GPU monitoring libraries
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import pyopencl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

# Check CUDA availability
try:
    import pycuda.driver as cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
except Exception:
    CUDA_AVAILABLE = False


# Import status management from separate module
from .status import (
    STATUS,
    STATUS_LOCK,
    STATS,
    STATS_LOCK,
    update_status,
    get_status,
    add_share,
    update_pool_status,
    set_socketio_instance,
)


# Performance & Monitoring Functions (Feature 1)
def update_performance_metrics():
    """Update CPU, memory, and GPU metrics"""
    try:
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory Usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU Monitoring (NVIDIA)
        gpu_usage = 0.0
        gpu_temperature = 0.0
        gpu_memory = 0.0

        if PYNVML_AVAILABLE:
            try:
                if not hasattr(update_performance_metrics, "nvml_initialized"):
                    try:
                        pynvml.nvmlInit()
                        update_performance_metrics.nvml_initialized = True
                        logging.debug("NVML initialized successfully")
                    except pynvml.NVMLError as e:
                        logging.warning(f"NVML initialization failed: {e}")
                        update_performance_metrics.nvml_initialized = False

                if update_performance_metrics.nvml_initialized:
                    try:
                        device_count = pynvml.nvmlDeviceGetCount()
                        if device_count == 0:
                            logging.debug("No NVIDIA GPUs found")
                        else:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_usage = util.gpu

                            temp = pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                            gpu_temperature = temp

                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            # Prevent division by zero
                            if mem_info.total > 0:
                                gpu_memory = (mem_info.used / mem_info.total) * 100
                            else:
                                gpu_memory = 0.0
                    except (pynvml.NVMLError, AttributeError, IndexError) as e:
                        logging.debug(f"GPU monitoring error: {e}")
            except Exception as e:
                logging.debug(f"Unexpected GPU monitoring error: {e}")

        # Update STATUS with lock protection
        with STATUS_LOCK:
            STATUS["cpu_usage"] = cpu_percent
            STATUS["memory_usage"] = memory_percent
            STATUS["gpu_usage"] = gpu_usage
            STATUS["gpu_temperature"] = gpu_temperature
            STATUS["gpu_memory"] = gpu_memory
    except Exception as e:
        logging.debug(f"Performance metrics error: {e}")
        # Update STATUS with lock protection even on error
        with STATUS_LOCK:
            STATUS["cpu_usage"] = 0.0
            STATUS["memory_usage"] = 0.0
            STATUS["gpu_usage"] = 0.0
            STATUS["gpu_temperature"] = 0.0
            STATUS["gpu_memory"] = 0.0


# Formatting Functions
# Mining Intelligence Functions (Feature 2)
# Formatting functions moved to utils.formatting


def calculate_mining_intelligence():
    """Calculate estimated time to block, probability, and profitability"""
    with STATUS_LOCK:
        hash_rate = STATUS.get("hash_rate", 0)
        best_difficulty = STATUS.get("best_difficulty", 0)
        target_difficulty = STATUS.get("target_difficulty", 0)
        network_difficulty = STATUS.get("network_difficulty", 0)

        if not hash_rate or hash_rate <= 0:
            STATUS["estimated_time_to_block"] = None
            STATUS["block_found_probability"] = 0.0
            STATUS["estimated_profitability"] = 0.0
            return

        # Calculate target difficulty from nbits (simplified)
        # For Bitcoin, difficulty = 65535 * 256^(exponent-3) / mantissa
        if target_difficulty > 0:
            difficulty = target_difficulty
        elif network_difficulty > 0:
            difficulty = network_difficulty
        elif best_difficulty > 0:
            # Use best difficulty as approximation
            difficulty = best_difficulty * 2  # Conservative estimate
        else:
            # Default Bitcoin network difficulty (approximate)
            difficulty = 50_000_000_000  # ~50 trillion

        # Estimated time to block (in seconds)
        # Expected hashes = 2^32 * difficulty
        expected_hashes = (2**32) * difficulty
        if hash_rate > 0:
            estimated_seconds = expected_hashes / hash_rate
            STATUS["estimated_time_to_block"] = estimated_seconds

            # Convert to human-readable format (years, months, days)
            STATUS["estimated_time_to_block_formatted"] = format_time_to_block(
                estimated_seconds
            )

            # Block found probability (simplified - probability of finding block in next hour)
            # P = 1 - e^(-hash_rate * 3600 / expected_hashes)
            import math

            if expected_hashes > 0:
                prob = 1 - math.exp(-(hash_rate * 3600) / expected_hashes)
                STATUS["block_found_probability"] = prob * 100
            else:
                STATUS["block_found_probability"] = 0.0

            # Estimated profitability (BTC per day) - very simplified
            # Assumes block reward = 3.125 BTC (current halving)
            block_reward = 3.125
            blocks_per_day = (86400 / estimated_seconds) if estimated_seconds > 0 else 0
            btc_per_day = blocks_per_day * block_reward
            STATUS["estimated_profitability"] = btc_per_day

            # Difficulty trend analysis
            if len(STATUS["difficulty_history"]) >= 2:
                recent = list(STATUS["difficulty_history"])[-5:]
                if len(recent) >= 2:
                    avg_recent = sum(recent[-3:]) / len(recent[-3:])
                    # Prevent division by zero: check if recent[:-3] is not empty
                    if len(recent) > 3 and len(recent[:-3]) > 0:
                        avg_older = sum(recent[:-3]) / len(recent[:-3])
                    else:
                        avg_older = recent[0] if len(recent) > 0 else 0
                    if avg_older > 0 and avg_recent > avg_older * 1.05:
                        STATUS["difficulty_trend"] = "increasing"
                    elif avg_older > 0 and avg_recent < avg_older * 0.95:
                        STATUS["difficulty_trend"] = "decreasing"
                    else:
                        STATUS["difficulty_trend"] = "stable"
        else:
            STATUS["estimated_time_to_block"] = None
            STATUS["block_found_probability"] = 0.0
            STATUS["estimated_profitability"] = 0.0


# Background thread for performance monitoring
def performance_monitor_thread():
    """Background thread to continuously update performance metrics"""
    global _performance_running
    while _performance_running:
        try:
            update_performance_metrics()
            calculate_mining_intelligence()
            time.sleep(2)  # Update every 2 seconds
        except Exception as e:
            logging.debug(f"Performance monitor error: {e}")
            time.sleep(5)


# Start performance monitoring thread
_performance_thread = None
_performance_running = False


def start_performance_monitoring():
    """Start the performance monitoring background thread"""
    global _performance_thread, _performance_running
    if _performance_thread is None or not _performance_thread.is_alive():
        _performance_running = True
        _performance_thread = threading.Thread(
            target=performance_monitor_thread, daemon=True
        )
        _performance_thread.start()


def stop_performance_monitoring():
    """Stop the performance monitoring background thread"""
    global _performance_running
    _performance_running = False


# Configure logging from environment (can be overridden at runtime via API)
configure_logging()
app = Flask(__name__, static_url_path="/static")
# Use environment variable for SECRET_KEY or generate a random one
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
# CORS: Allow specific origins or localhost by default for security
cors_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
).split(",")
# Configure SocketIO with increased buffer size to handle larger payloads
# This fixes the "Too many packets in payload" error
socketio = SocketIO(
    app,
    cors_allowed_origins=cors_origins,
    max_http_buffer_size=10e6,  # 10MB buffer size (default is 100KB) - increased to handle large payloads
    max_packet_size=10e6,  # 10MB max packet size
    ping_timeout=60,
    ping_interval=25,
    logger=True,  # Enable SocketIO logging to aid debugging of handshake/transport
    engineio_logger=True,
    async_mode="threading",  # Use threading mode for better compatibility
)

# Register SocketIO instance with status module for immediate updates
set_socketio_instance(socketio)


@app.route("/favicon.ico")
def favicon():
    """Serve favicon as SVG"""
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<defs>
<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#1e3c72;stop-opacity:1" />
<stop offset="100%" style="stop-color:#2a5298;stop-opacity:1" />
</linearGradient>
<linearGradient id="btc" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#F7931A;stop-opacity:1" />
<stop offset="100%" style="stop-color:#FFA500;stop-opacity:1" />
</linearGradient>
</defs>
<rect width="100" height="100" rx="20" fill="url(#bg)"/>
<circle cx="50" cy="50" r="35" fill="url(#btc)" stroke="#fff" stroke-width="2"/>
<path d="M50 30 L58 42 L50 50 L42 42 Z M50 50 L58 62 L50 70 L42 62 Z" fill="#1a1a1a"/>
<circle cx="50" cy="50" r="6" fill="#F7931A"/>
</svg>"""
    return Response(
        svg_content,
        mimetype="image/svg+xml",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@app.route("/favicon.svg")
def favicon_svg():
    """Serve favicon as SVG"""
    return favicon()


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/stop", methods=["POST"])
def stop_mining():
    """Stop mining by setting shutdown flag"""
    from flask import request

    # CSRF protection
    if not _check_csrf_protection(request):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "CSRF validation failed",
                    "message": "Request origin not allowed. CSRF protection enabled.",
                }
            ),
            403,
        )

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Maximum {_api_rate_limit_max_requests} requests per {_api_rate_limit_window} seconds.",
                }
            ),
            429,
        )

    try:
        global _miner_state
        if not _miner_state:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "MinerStateNotAvailable",
                        "message": "Miner state not available. Miner may not be running.",
                    }
                ),
                503,
            )

        # Thread-safe shutdown flag update
        with _miner_state._lock:
            _miner_state.shutdown_flag = True
        update_status("running", False)
        return jsonify({"success": True, "message": "Mining stopped"})
    except Exception as e:
        logging.error(f"Error stopping mining: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "InternalError",
                    "message": f"Failed to stop mining: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/start", methods=["POST"])
def start_mining():
    """Start or resume mining"""
    from flask import request
    import threading

    # CSRF protection
    if not _check_csrf_protection(request):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "CSRF validation failed",
                    "message": "Request origin not allowed. CSRF protection enabled.",
                }
            ),
            403,
        )

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Maximum {_api_rate_limit_max_requests} requests per {_api_rate_limit_window} seconds.",
                }
            ),
            429,
        )

    try:
        global _miner_state, _miner

        # Check if miner instance exists
        if not _miner:
            # Try to load wallet from config and start miner
            from ..config import load_config

            cfg = load_config()
            wallet = cfg.get("wallet", {}).get("address", "").strip()

            if not wallet:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "NoWalletAddress",
                            "message": "Wallet address not configured. Please set it in Settings first.",
                        }
                    ),
                    400,
                )

            # Initialize and start miner
            try:
                from ..clients.pool_client import PoolClient
                from ..core.miner import Miner
                from ..core.state import MinerState
                import logging

                logger = logging.getLogger("SatoshiRig")
                pool = PoolClient(cfg["pool"]["host"], int(cfg["pool"]["port"]))
                miner_state = MinerState() if not _miner_state else _miner_state
                miner = Miner(wallet, cfg, pool, miner_state, logger)

                set_miner(miner)
                set_miner_state(miner_state)

                # Connect to pool first (without starting mining)
                try:
                    connect_thread = threading.Thread(
                        target=miner.connect_to_pool_only, daemon=True
                    )
                    connect_thread.start()
                    # Wait a bit for connection to establish
                    connect_thread.join(timeout=5)
                except Exception as connect_error:
                    logger.warning(f"Failed to connect to pool: {connect_error}")

                # Start miner in background thread
                miner_thread = threading.Thread(target=miner.start, daemon=True)
                miner_thread.start()

                update_status("running", True)
                update_status("wallet_address", wallet)
                return jsonify(
                    {"success": True, "message": "Miner started successfully"}
                )
            except Exception as e:
                logger = logging.getLogger("SatoshiRig.web")
                logger.error(f"Failed to start miner: {e}", exc_info=True)
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "StartFailed",
                            "message": f"Failed to start miner: {str(e)}",
                        }
                    ),
                    500,
                )

        # Miner exists, just clear shutdown flag
        if not _miner_state:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "MinerStateNotAvailable",
                        "message": "Miner state not available. Miner may not be running.",
                    }
                ),
                503,
            )

        # Thread-safe shutdown flag update
        with _miner_state._lock:
            _miner_state.shutdown_flag = False
        update_status("running", True)
        return jsonify({"success": True, "message": "Mining resumed"})
    except Exception as e:
        logging.error(f"Error starting mining: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "InternalError",
                    "message": f"Failed to start mining: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/config", methods=["GET"])
def get_config_api():
    """Get current configuration (sanitized)"""
    try:
        config = get_config_for_ui()
        return jsonify({"success": True, "config": config}), 200
    except Exception as e:
        logger = logging.getLogger("SatoshiRig.web")
        logger.error(f"Error getting config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def save_config_api():
    """Save configuration (validates and stores)"""
    from flask import request

    # CSRF protection
    if not _check_csrf_protection(request):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "CSRF validation failed",
                    "message": "Request origin not allowed. CSRF protection enabled.",
                }
            ),
            403,
        )

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Maximum {_api_rate_limit_max_requests} requests per {_api_rate_limit_window} seconds.",
                }
            ),
            429,
        )

    try:
        data = request.get_json()
        if not data or "config" not in data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid request: missing 'config' field",
                    }
                ),
                400,
            )

        config = data["config"]

        # Validate configuration
        validation_errors = []

        # Validate wallet address
        if "wallet" in config and "address" in config["wallet"]:
            wallet_address = config["wallet"]["address"].strip()
            if wallet_address:
                if len(wallet_address) < 26 or len(wallet_address) > 62:
                    validation_errors.append(
                        "Invalid wallet address length (must be 26-62 characters)"
                    )
                if not all(
                    c
                    in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    for c in wallet_address
                ):
                    validation_errors.append(
                        "Invalid wallet address format (contains invalid characters)"
                    )

        # Validate pool configuration
        if "pool" in config:
            if "host" in config["pool"]:
                host = config["pool"]["host"].strip()
                if not host:
                    validation_errors.append("Pool host cannot be empty")
            if "port" in config["pool"]:
                try:
                    port = int(config["pool"]["port"])
                    if port < 1 or port > 65535:
                        validation_errors.append(
                            "Pool port must be between 1 and 65535"
                        )
                except (ValueError, TypeError):
                    validation_errors.append("Pool port must be a valid number")

        # Validate network configuration
        if "network" in config:
            if "source" in config["network"]:
                source = config["network"]["source"]
                if source not in ["web", "local"]:
                    validation_errors.append("Network source must be 'web' or 'local'")
            if "request_timeout_secs" in config["network"]:
                try:
                    timeout = int(config["network"]["request_timeout_secs"])
                    if timeout < 1 or timeout > 300:
                        validation_errors.append(
                            "Request timeout must be between 1 and 300 seconds"
                        )
                except (ValueError, TypeError):
                    validation_errors.append("Request timeout must be a valid number")

        # Validate compute configuration
        if "compute" in config:
            if "backend" in config["compute"]:
                backend = config["compute"]["backend"]
                if backend not in ["cpu", "cuda", "opencl"]:
                    validation_errors.append(
                        "Compute backend must be 'cpu', 'cuda', or 'opencl'"
                    )
            if "gpu_device" in config["compute"]:
                try:
                    gpu_device = int(config["compute"]["gpu_device"])
                    if gpu_device < 0:
                        validation_errors.append("GPU device must be >= 0")
                except (ValueError, TypeError):
                    validation_errors.append("GPU device must be a valid number")
            if "batch_size" in config["compute"]:
                try:
                    batch_size = int(config["compute"]["batch_size"])
                    if batch_size < 1 or batch_size > 100000:
                        validation_errors.append(
                            "Batch size must be between 1 and 100000"
                        )
                except (ValueError, TypeError):
                    validation_errors.append("Batch size must be a valid number")
            if "max_workers" in config["compute"]:
                try:
                    max_workers = int(config["compute"]["max_workers"])
                    if max_workers < 1 or max_workers > 128:
                        validation_errors.append(
                            "Max workers must be between 1 and 128"
                        )
                except (ValueError, TypeError):
                    validation_errors.append("Max workers must be a valid number")
            if "gpu_utilization_percent" in config["compute"]:
                try:
                    gpu_util = int(config["compute"]["gpu_utilization_percent"])
                    if gpu_util < 1 or gpu_util > 100:
                        validation_errors.append(
                            "GPU utilization must be between 1 and 100 percent"
                        )
                except (ValueError, TypeError):
                    validation_errors.append("GPU utilization must be a valid number")

        # Validate database configuration
        if "database" in config and "retention_days" in config["database"]:
            try:
                retention = int(config["database"]["retention_days"])
                if retention < 1 or retention > 3650:
                    validation_errors.append(
                        "Database retention must be between 1 and 3650 days"
                    )
            except (ValueError, TypeError):
                validation_errors.append("Database retention must be a valid number")

        # Validate logging configuration
        if "logging" in config:
            if "level" in config["logging"]:
                log_level = config["logging"]["level"].upper()
                if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    validation_errors.append(
                        "Logging level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
                    )

        if validation_errors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Validation failed",
                        "errors": validation_errors,
                    }
                ),
                400,
            )

        # Save to database only (no TOML files)
        try:
            from ..config import persist_config_to_db, load_config

            # Load existing config from database
            try:
                existing_config = load_config()
            except Exception:
                # If loading fails, use sanitized config as fallback
                existing_config = get_config_for_ui()

            # Deep merge: update existing config with new values (with deep copy to avoid reference issues)
            # Add recursion depth limit to prevent stack overflow
            import copy

            def deep_merge(base, update, depth=0, max_depth=50):
                if depth > max_depth:
                    raise RuntimeError(
                        f"deep_merge recursion depth exceeded {max_depth}, possible circular reference"
                    )
                for key, value in update.items():
                    if (
                        key in base
                        and isinstance(base[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_merge(base[key], value, depth + 1, max_depth)
                    else:
                        # Deep copy to avoid reference issues
                        base[key] = (
                            copy.deepcopy(value)
                            if isinstance(value, (dict, list))
                            else value
                        )

            # Create full config for saving (merge with existing)
            full_config = existing_config.copy()
            deep_merge(full_config, config)

            # Save to database only
            persist_config_to_db(full_config)
            logger = logging.getLogger("SatoshiRig.web")
            logger.info("Configuration saved to database")

            # Update logging level if changed
            if "logging" in config:
                log_level = config["logging"].get("level", "INFO")
                try:
                    update_logging_level(log_level)
                    logger.info("Logging level updated to %s", log_level)
                except Exception:
                    logger.warning("Failed to update logging level during config save")

            # Reload config from database
            try:
                saved_config = load_config()
                set_config(saved_config)

                # Check if wallet address was set and miner is not running - auto-start miner
                wallet = saved_config.get("wallet", {}).get("address", "").strip()
                if wallet and not _miner:
                    logger.info(
                        f"Wallet address configured, attempting to auto-start miner..."
                    )
                    # Try to start miner (will be handled by /api/start if user clicks start button)
                    # We don't auto-start here to avoid starting miner without user consent

                # If miner exists and pool config changed, reconnect to pool
                if _miner and wallet:
                    try:
                        # Only reconnect if miner is not running (to avoid race condition)
                        if not _miner._running:
                            # Update pool client with new config
                            _miner.pool.host = saved_config["pool"]["host"]
                            _miner.pool.port = int(saved_config["pool"]["port"])
                            # Connect to pool (without starting mining) in background thread
                            connect_thread = threading.Thread(
                                target=_miner.connect_to_pool_only, daemon=True
                            )
                            connect_thread.start()
                            logger.info(
                                "Pool connection re-established after config change"
                            )
                        else:
                            logger.debug(
                                "Miner is running, pool connection is already established - skipping reconnect to avoid interference"
                            )
                    except Exception as pool_connect_error:
                        logger.warning(
                            f"Failed to reconnect to pool after config change: {pool_connect_error}"
                        )
            except Exception as e:
                logger.warning(f"Could not reload config after save: {e}")
                set_config(full_config)
        except Exception as e:
            logger = logging.getLogger("SatoshiRig.web")
            logger.error(f"Error saving config to database: {e}")
            # Continue anyway - at least update in-memory config
            set_config(config)

        return (
            jsonify({"success": True, "message": "Configuration saved successfully"}),
            200,
        )
    except Exception as e:
        logger = logging.getLogger("SatoshiRig.web")
        logger.error(f"Error saving config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config/cpu-mining", methods=["POST"])
def toggle_cpu_mining():
    """Toggle CPU mining on/off"""
    from flask import request

    # CSRF protection
    if not _check_csrf_protection(request):
        return jsonify({"success": False, "error": "CSRF validation failed"}), 403

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"success": False, "error": "Rate limit exceeded"}), 429

    try:
        data = request.get_json()
        enabled = data.get("enabled", True)

        config = get_config_for_ui()
        config["compute"]["cpu_mining_enabled"] = enabled
        set_config(config)

        # Apply to running miner
        global _miner
        if _miner:
            try:
                _miner.update_config({"compute": config["compute"]})
            except Exception as e:
                logger = logging.getLogger("SatoshiRig.web")
                logger.error(f"Error updating miner config: {e}")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Failed to apply config to miner: {str(e)}",
                        }
                    ),
                    500,
                )

        return jsonify({"success": True, "cpu_mining_enabled": enabled}), 200
    except Exception as e:
        logger = logging.getLogger("SatoshiRig.web")
        logger.error(f"Error toggling CPU mining: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config/gpu-mining", methods=["POST"])
def toggle_gpu_mining():
    """Toggle GPU mining on/off"""
    from flask import request

    # CSRF protection
    if not _check_csrf_protection(request):
        return jsonify({"success": False, "error": "CSRF validation failed"}), 403

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"success": False, "error": "Rate limit exceeded"}), 429

    try:
        data = request.get_json()
        enabled = data.get("enabled", True)

        config = get_config_for_ui()
        config["compute"]["gpu_mining_enabled"] = enabled

        # Update backend based on GPU availability
        if enabled:
            current_backend = config["compute"].get("backend")
            if current_backend == "cuda" and not CUDA_AVAILABLE:
                current_backend = None
            if current_backend == "opencl" and not OPENCL_AVAILABLE:
                current_backend = None

            if current_backend in ("cuda", "opencl"):
                config["compute"]["backend"] = current_backend
            elif CUDA_AVAILABLE:
                config["compute"]["backend"] = "cuda"
            elif OPENCL_AVAILABLE:
                config["compute"]["backend"] = "opencl"
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "No GPU backend available. Install PyCUDA or PyOpenCL.",
                        }
                    ),
                    400,
                )
        set_config(config)

        # Apply to running miner
        global _miner
        if _miner:
            try:
                _miner.update_config({"compute": config["compute"]})
            except Exception as e:
                logger = logging.getLogger("SatoshiRig.web")
                logger.error(f"Error updating miner config: {e}")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Failed to apply config to miner: {str(e)}",
                        }
                    ),
                    500,
                )

        return (
            jsonify(
                {
                    "success": True,
                    "gpu_mining_enabled": enabled,
                    "backend": config["compute"]["backend"],
                }
            ),
            200,
        )
    except Exception as e:
        logger = logging.getLogger("SatoshiRig.web")
        logger.error(f"Error toggling GPU mining: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/config/db-retention", methods=["POST"])
def set_db_retention():
    """Set database retention period in days"""
    from flask import request

    # CSRF protection
    if not _check_csrf_protection(request):
        return jsonify({"success": False, "error": "CSRF validation failed"}), 403

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({"success": False, "error": "Rate limit exceeded"}), 429

    try:
        data = request.get_json()
        days = int(data.get("days", 30))

        if days < 1 or days > 3650:  # Max 10 years
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Retention days must be between 1 and 3650",
                    }
                ),
                400,
            )

        config = get_config_for_ui()
        config["database"]["retention_days"] = days
        set_config(config)

        # TODO: Save to database

        return jsonify({"success": True, "retention_days": days}), 200
    except (ValueError, TypeError) as e:
        return jsonify({"success": False, "error": "Invalid days value"}), 400
    except Exception as e:
        logger = logging.getLogger("SatoshiRig.web")
        logger.error(f"Error setting DB retention: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@socketio.on_error_default
def default_error_handler(e):
    """Handle SocketIO errors gracefully"""
    logger = logging.getLogger("SatoshiRig.web")
    logger.warning(f"SocketIO error: {e}")
    return False  # Don't propagate the error


@socketio.on("connect")
def handle_connect():
    emit("status", get_status())


@socketio.on("get_status")
def handle_get_status():
    emit("status", get_status())


# Global flag to control broadcast thread
_broadcast_running = False


def broadcast_status():
    global _broadcast_running
    _broadcast_running = True
    while _broadcast_running:
        try:
            with STATUS_LOCK:
                # Add current values to history
                if STATUS["hash_rate"] > 0:
                    STATUS["hash_rate_history"].append(STATUS["hash_rate"])
                if STATUS["best_difficulty"] > 0:
                    STATUS["difficulty_history"].append(STATUS["best_difficulty"])
            socketio.emit("status", get_status())
        except Exception as e:
            logger = logging.getLogger("SatoshiRig.web")
            logger.error(f"Error in broadcast_status: {e}")
        time.sleep(2)


def stop_web_server():
    """Stop background threads for web server"""
    global _broadcast_running, _performance_running
    _broadcast_running = False
    stop_performance_monitoring()


def start_web_server(host: str = "0.0.0.0", port: int = 5000):
    logger = logging.getLogger("SatoshiRig.web")
    logger.info("Starting web server on %s:%s", host, port)
    # Use Unix timestamp (seconds since epoch) for accurate uptime calculation
    start_timestamp = time.time()
    update_status("start_time", start_timestamp)
    update_status("running", True)
    with STATS_LOCK:
        STATS["start_time"] = start_timestamp
    # Start background threads
    threading.Thread(target=broadcast_status, daemon=True).start()
    start_performance_monitoring()
    try:
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
    finally:
        # Cleanup on shutdown
        stop_web_server()


# Global reference to miner state for controlling mining
_miner_state = None

# Global reference to configuration (sanitized for web UI)
_config = None

# Rate limiting for API endpoints
_api_rate_limit = {}
_api_rate_limit_lock = threading.Lock()
_api_rate_limit_window = 60  # seconds
_api_rate_limit_max_requests = 10  # max requests per window


def _check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit"""
    with _api_rate_limit_lock:
        now = time.time()
        if client_ip not in _api_rate_limit:
            _api_rate_limit[client_ip] = []

        # Remove old requests outside the window
        _api_rate_limit[client_ip] = [
            req_time
            for req_time in _api_rate_limit[client_ip]
            if now - req_time < _api_rate_limit_window
        ]

        # Check if limit exceeded
        if len(_api_rate_limit[client_ip]) >= _api_rate_limit_max_requests:
            return False

        # Add current request
        _api_rate_limit[client_ip].append(now)
        return True


def _check_csrf_protection(request) -> bool:
    """Check CSRF protection via Origin/Referer header validation"""
    # Get allowed origins from CORS configuration (use same as CORS settings)
    cors_origins_str = os.environ.get(
        "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
    )

    # If CORS_ORIGINS is "*", allow all origins (less secure but convenient)
    if cors_origins_str.strip() == "*":
        return True

    allowed_origins = cors_origins_str.split(",")

    # Get current request host (same-origin requests should always be allowed)
    current_host = request.host  # e.g., "satoshirig.zhome.ch:5000" or "localhost:5000"
    current_host_base = current_host.split(":")[0]  # Remove port if present

    # Get Origin or Referer header
    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")

    # If no Origin/Referer, allow only if from same origin (local requests)
    if not origin and not referer:
        # Allow requests without Origin/Referer for localhost/127.0.0.1
        # This is acceptable for a local mining application
        return True

    # Always allow same-origin requests (most secure - request comes from same domain)
    if origin:
        try:
            origin_host = (
                origin.split("://")[1].split("/")[0]
                if "://" in origin
                else origin.split("/")[0]
            )
            origin_host_base = origin_host.split(":")[0]  # Remove port if present
            if origin_host_base == current_host_base:
                return True
        except (IndexError, AttributeError):
            pass

    if referer:
        try:
            referer_host = (
                referer.split("://")[1].split("/")[0]
                if "://" in referer
                else referer.split("/")[0]
            )
            referer_host_base = referer_host.split(":")[0]  # Remove port if present
            if referer_host_base == current_host_base:
                return True
        except (IndexError, AttributeError):
            pass

    # Check Origin header against allowed origins
    if origin:
        # Remove protocol and path, keep only origin
        try:
            origin_base = (
                origin.split("://")[1].split("/")[0]
                if "://" in origin
                else origin.split("/")[0]
            )
            origin_base_no_port = origin_base.split(":")[0]
            for allowed in allowed_origins:
                allowed = allowed.strip()
                if not allowed:
                    continue
                allowed_base = (
                    allowed.split("://")[1].split("/")[0]
                    if "://" in allowed
                    else allowed.split("/")[0]
                )
                allowed_base_no_port = allowed_base.split(":")[0]
                if (
                    origin_base_no_port == allowed_base_no_port
                    or origin_base_no_port in allowed_base_no_port
                    or allowed_base_no_port in origin_base_no_port
                ):
                    return True
        except (IndexError, AttributeError):
            pass

    # Fallback to Referer header
    if referer:
        try:
            referer_base = (
                referer.split("://")[1].split("/")[0]
                if "://" in referer
                else referer.split("/")[0]
            )
            referer_base_no_port = referer_base.split(":")[0]
            for allowed in allowed_origins:
                allowed = allowed.strip()
                if not allowed:
                    continue
                allowed_base = (
                    allowed.split("://")[1].split("/")[0]
                    if "://" in allowed
                    else allowed.split("/")[0]
                )
                allowed_base_no_port = allowed_base.split(":")[0]
                if (
                    referer_base_no_port == allowed_base_no_port
                    or referer_base_no_port in allowed_base_no_port
                    or allowed_base_no_port in referer_base_no_port
                ):
                    return True
        except (IndexError, AttributeError):
            pass

    # If Origin/Referer doesn't match allowed origins or current host, reject
    return False


def set_miner_state(miner_state):
    """Set the miner state reference for controlling mining"""
    global _miner_state
    _miner_state = miner_state


def set_miner(miner):
    """Set the miner instance reference for dynamic config updates"""
    global _miner
    _miner = miner


def update_logging_level(log_level: str, log_file: str = None):
    """Update logging level dynamically at runtime

    Note: log_file parameter is ignored - all logs go to stdout/stderr for Docker logs
    """
    # Reconfigure logging via central helper which will recreate handlers
    try:
        from ..logging_config import configure_logging

        # configure_logging will set root level and handlers
        configure_logging(level=log_level, log_file=log_file)
    except Exception:
        # Fallback: do a minimal level update if helper fails
        import logging

        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(log_level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        web_logger = logging.getLogger("SatoshiRig.web")
        web_logger.setLevel(level)
        global _miner
        if _miner and hasattr(_miner, "log"):
            _miner.log.setLevel(level)


def set_config(config: dict):
    """Set the configuration reference for web UI (sanitized - no sensitive data)"""
    global _config
    # Create sanitized copy without sensitive data
    # Preserve wallet address if it exists in the config being set
    wallet_address = config.get("wallet", {}).get("address", "")
    sanitized = {
        "wallet": {"address": wallet_address},  # Preserve wallet address if provided
        "pool": {
            "host": config.get("pool", {}).get("host", "solo.ckpool.org"),
            "port": config.get("pool", {}).get("port", 3333),
        },
        "network": {
            "source": config.get("network", {}).get("source", "web"),
            "latest_block_url": config.get("network", {}).get(
                "latest_block_url", "https://blockchain.info/latestblock"
            ),
            "request_timeout_secs": config.get("network", {}).get(
                "request_timeout_secs", 15
            ),
            "rpc_url": config.get("network", {}).get(
                "rpc_url", "http://127.0.0.1:8332"
            ),
            "rpc_user": "",  # Empty - user must enter
            "rpc_password": "",  # Empty - user must enter
        },
        "logging": {
            "level": config.get("logging", {}).get("level", "INFO"),
        },
        "miner": {
            "restart_delay_secs": config.get("miner", {}).get("restart_delay_secs", 2),
            "subscribe_thread_start_delay_secs": config.get("miner", {}).get(
                "subscribe_thread_start_delay_secs", 4
            ),
            "hash_log_prefix_zeros": config.get("miner", {}).get(
                "hash_log_prefix_zeros", 7
            ),
        },
        "compute": {
            "backend": config.get("compute", {}).get("backend", "cuda"),
            "gpu_device": config.get("compute", {}).get("gpu_device", 0),
            "batch_size": config.get("compute", {}).get("batch_size", 256),
            "max_workers": config.get("compute", {}).get("max_workers", 8),
            "gpu_utilization_percent": config.get("compute", {}).get(
                "gpu_utilization_percent", 100
            ),
            "cpu_mining_enabled": config.get("compute", {}).get(
                "cpu_mining_enabled", True
            ),
            "gpu_mining_enabled": config.get("compute", {}).get(
                "gpu_mining_enabled",
                config.get("compute", {}).get("backend") in ["cuda", "opencl"],
            ),
        },
        "database": {
            "retention_days": int(
                os.environ.get("DB_RETENTION_DAYS", "30")
            )  # Default: 30 days
        },
    }
    _config = sanitized
    persist_config_to_db(sanitized)


def get_config_for_ui() -> dict:
    """Get current configuration (sanitized) for web UI"""
    global _config
    if _config is None:
        # Try to load from file to get saved wallet address
        try:
            from ..config import load_config

            file_config = load_config()
            wallet_address = file_config.get("wallet", {}).get("address", "")
        except Exception:
            wallet_address = ""

        # Return defaults if not set
        return {
            "wallet": {"address": wallet_address},
            "pool": {"host": "solo.ckpool.org", "port": 3333},
            "network": {
                "source": "web",
                "latest_block_url": "https://blockchain.info/latestblock",
                "request_timeout_secs": 15,
                "rpc_url": "http://127.0.0.1:8332",
                "rpc_user": "",
                "rpc_password": "",
            },
            "logging": {"file": "miner.log", "level": "INFO"},
            "miner": {
                "restart_delay_secs": 2,
                "subscribe_thread_start_delay_secs": 4,
                "hash_log_prefix_zeros": 7,
            },
            "compute": {
                "backend": "cuda",
                "gpu_device": 0,
                "batch_size": 256,
                "max_workers": 8,
                "gpu_utilization_percent": 100,
                "cpu_mining_enabled": True,
                "gpu_mining_enabled": False,
            },
            "database": {"retention_days": 30},
        }
    return _config.copy()


# Export functions for use by miner
__all__ = [
    "start_web_server",
    "update_status",
    "get_status",
    "add_share",
    "update_pool_status",
    "set_miner_state",
    "set_miner",
    "set_config",
    "get_config_for_ui",
]


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatoshiRig - Status Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <link rel="apple-touch-icon" href="/favicon.svg">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        :root {
            --bg-gradient-start: #0f0c29;
            --bg-gradient-mid: #302b63;
            --bg-gradient-end: #24243e;
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #b8b8d4;
            --accent-primary: #00d4ff;
            --accent-secondary: #4ade80;
            --accent-danger: #ff6b6b;
            --accent-warning: #ffd93d;
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.3);
            --shadow-glow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        /* Light Theme Variables */
        body.light-theme {
            --bg-gradient-start: #f0f4f8;
            --bg-gradient-mid: #e2e8f0;
            --bg-gradient-end: #cbd5e1;
            --card-bg: rgba(255, 255, 255, 0.9);
            --card-border: rgba(0, 0, 0, 0.1);
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --accent-primary: #0ea5e9;
            --accent-secondary: #10b981;
            --accent-danger: #ef4444;
            --accent-warning: #f59e0b;
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.15);
            --shadow-glow: 0 0 20px rgba(14, 165, 233, 0.2);
        }
        
        body.light-theme::before {
            background: 
                radial-gradient(circle at 20% 50%, rgba(14, 165, 233, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-mid) 50%, var(--bg-gradient-end) 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            padding: 1rem;
            position: relative;
            overflow-x: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(74, 222, 128, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        /* Modern Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1.5rem 2rem;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow-lg);
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        h1 {
            font-size: clamp(1.5rem, 4vw, 2.5rem);
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }
        
        .controls {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 600;
            background: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(10px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .btn:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md), var(--shadow-glow);
            border-color: var(--accent-primary);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        /* Modern Tabs */
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            padding: 0.5rem;
            border-radius: 16px;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow-md);
        }
        
        .tab {
            padding: 0.875rem 1.5rem;
            border: none;
            border-radius: 12px;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            white-space: nowrap;
        }
        
        .tab:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .tab.active {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: var(--text-primary);
            box-shadow: var(--shadow-sm), 0 0 20px rgba(0, 212, 255, 0.4);
        }
        
        /* Tab Content - Hide inactive tabs */
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Status Cards */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 1.75rem;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow-md);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .status-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            transform: scaleX(0);
            transition: transform 0.4s;
        }
        
        .status-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-lg), var(--shadow-glow);
            border-color: var(--accent-primary);
        }
        
        .status-card:hover::before {
            transform: scaleX(1);
        }
        
        .status-card h2 {
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.75rem;
        }
        
        .status-value {
            font-size: clamp(1.5rem, 4vw, 2.5rem);
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .status-value.running {
            color: var(--accent-secondary);
            -webkit-text-fill-color: var(--accent-secondary);
        }
        
        .status-value.stopped {
            color: var(--accent-danger);
            -webkit-text-fill-color: var(--accent-danger);
        }
        
        .status-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 400;
        }
        /* Hash Display */
        .hash-display {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            word-break: break-all;
            margin-top: 1rem;
            border: 1px solid var(--card-border);
        }
        /* Chart Container */
        .chart-container {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow-lg);
            margin-bottom: 2rem;
        }
        
        .chart-container h2 {
            margin-bottom: 1.5rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Settings */
        .settings-section {
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow-md);
        }
        
        .settings-section h3 {
            margin-top: 0;
            margin-bottom: 1.5rem;
            font-size: 1.125rem;
            font-weight: 700;
            color: var(--accent-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }
        
        .setting-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .setting-item label {
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .setting-item input[type="text"],
        .setting-item input[type="number"],
        .setting-item input[type="password"],
        .setting-item select {
            padding: 0.875rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.9375rem;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        .setting-item input:focus,
        .setting-item select:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .setting-item select option {
            background: #1a1a2e;
            color: var(--text-primary);
            padding: 0.5rem;
        }
        /* Toggle Switch Styles */
        .toggle-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        
        .toggle-label {
            margin: 0;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 56px;
            height: 32px;
            flex-shrink: 0;
        }
        
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid var(--card-border);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 32px;
        }
        
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 4px;
            bottom: 4px;
            background: var(--text-primary);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 50%;
            box-shadow: var(--shadow-sm);
        }
        
        .toggle-switch input:checked + .toggle-slider {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            border-color: var(--accent-primary);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        }
        
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(24px);
            background: var(--text-primary);
        }
        
        .toggle-switch:hover .toggle-slider {
            border-color: var(--accent-primary);
        }
        .settings-actions {
            margin-top: 2rem;
            display: flex;
            gap: 1rem;
        }
        /* Buttons */
        .settings-actions {
            margin-top: 2rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: var(--text-primary);
            font-weight: 600;
            padding: 1rem 2rem;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
            box-shadow: var(--shadow-md);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg), var(--shadow-glow);
        }
        
        .btn-primary:active {
            transform: translateY(0);
        }
        
        .btn-secondary {
            background: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid var(--card-border);
            font-weight: 600;
            padding: 1rem 2rem;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent-primary);
            transform: translateY(-2px);
        }
        .status-card a {
            transition: opacity 0.3s;
            color: var(--accent);
            text-decoration: none;
        }
        .status-card a:hover {
            opacity: 0.8;
            text-decoration: underline;
        }
        .errors {
            background: rgba(255, 0, 0, 0.1);
            border-left: 4px solid var(--error);
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .errors h3 {
            margin-bottom: 10px;
            color: var(--error);
        }
        .error-item {
            padding: 8px;
            margin: 5px 0;
            background: rgba(255, 0, 0, 0.1);
            border-radius: 4px;
            font-size: 0.9em;
        }
        /* Pool Status Badge */
        .pool-status {
            display: inline-block;
            padding: 0.375rem 0.875rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
        }
        
        .pool-connected {
            background: rgba(74, 222, 128, 0.2);
            color: var(--accent-secondary);
            border: 1px solid var(--accent-secondary);
        }
        
        .pool-disconnected {
            background: rgba(255, 107, 107, 0.2);
            color: var(--accent-danger);
            border: 1px solid var(--accent-danger);
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .stats-table td {
            padding: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stats-table td:first-child {
            color: var(--text-secondary);
        }
        .share-history {
            max-height: 200px;
            overflow-y: auto;
            margin-top: 10px;
        }
        .share-item {
            padding: 8px;
            margin: 5px 0;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
            font-size: 0.85em;
        }
        .share-accepted { border-left: 3px solid var(--accent); }
        .share-rejected { border-left: 3px solid var(--error); }
        
        /* Section Headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .section-group {
            margin-bottom: 2rem;
        }
        
        /* Light Theme Specific Adjustments */
        body.light-theme .status-card {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(0, 0, 0, 0.1);
        }
        
        body.light-theme .status-card:hover {
            box-shadow: var(--shadow-lg), 0 0 20px rgba(14, 165, 233, 0.2);
        }
        
        body.light-theme .chart-container {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(0, 0, 0, 0.1);
        }
        
        body.light-theme .settings-section {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(0, 0, 0, 0.1);
        }
        
        body.light-theme .header {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(0, 0, 0, 0.1);
        }
        
        body.light-theme .tabs {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(0, 0, 0, 0.1);
        }
        
        body.light-theme .hash-display {
            background: rgba(0, 0, 0, 0.05);
            border-color: rgba(0, 0, 0, 0.1);
            color: var(--text-primary);
        }
        
        body.light-theme .setting-item input[type="text"],
        body.light-theme .setting-item input[type="number"],
        body.light-theme .setting-item input[type="password"],
        body.light-theme .setting-item select {
            background: rgba(255, 255, 255, 0.8);
            border-color: rgba(0, 0, 0, 0.15);
            color: var(--text-primary);
        }
        
        body.light-theme .setting-item select option {
            background: #ffffff;
            color: var(--text-primary);
        }
        
        body.light-theme .toggle-slider {
            background: rgba(0, 0, 0, 0.1);
            border-color: rgba(0, 0, 0, 0.2);
        }
        
        body.light-theme .toggle-slider:before {
            background: #ffffff;
        }
        
        body.light-theme .btn {
            background: rgba(255, 255, 255, 0.9);
            border-color: rgba(0, 0, 0, 0.1);
            color: var(--text-primary);
        }
        
        body.light-theme .btn:hover {
            background: rgba(255, 255, 255, 1);
            box-shadow: var(--shadow-md), 0 0 20px rgba(14, 165, 233, 0.2);
        }
        
        body.light-theme .pool-connected {
            background: rgba(16, 185, 129, 0.15);
            color: #059669;
            border-color: #10b981;
        }
        
        body.light-theme .pool-disconnected {
            background: rgba(239, 68, 68, 0.15);
            color: #dc2626;
            border-color: #ef4444;
        }
        
        body.light-theme .errors {
            background: rgba(239, 68, 68, 0.1);
            border-left-color: var(--accent-danger);
        }
        
        body.light-theme .stats-table td {
            border-bottom-color: rgba(0, 0, 0, 0.1);
        }
        
        body.light-theme .share-history {
            background: rgba(0, 0, 0, 0.02);
        }
        
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SatoshiRig Status Dashboard</h1>
            <div class="controls">
                <button class="btn" onclick="toggleTheme()" id="themeToggle">🌓 Theme</button>
                <button class="btn" onclick="toggleMining()">
                    <span id="autoRefreshText">⏸️ Pause</span>
                </button>
            </div>
        </div>
        
        <!-- Tabs Navigation -->
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab" onclick="showTab('performance')">⚡ Performance</button>
            <button class="tab" onclick="showTab('analytics')">📈 Analytics</button>
            <button class="tab" onclick="showTab('intelligence')">🧠 Intelligence</button>
            <button class="tab" onclick="showTab('history')">📜 History</button>
            <button class="tab" onclick="showTab('settings')">⚙️ Settings</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="section-group">
                <h2 class="section-header">Mining Status</h2>
                <div class="status-grid">
                    <div class="status-card">
                        <h2>Status</h2>
                        <div class="status-value running" id="runningStatus">-</div>
                        <div class="status-label">Mining Status</div>
                    </div>
                    <div class="status-card">
                        <h2>Pool Connection</h2>
                        <div class="status-value" id="poolStatus">-</div>
                        <div class="pool-status pool-disconnected" id="poolStatusBadge">Disconnected</div>
                        <div class="status-label" id="poolInfo">-</div>
                    </div>
                    <div class="status-card">
                        <h2>Current Block Height</h2>
                        <div class="status-value" id="currentHeight">-</div>
                        <div class="status-label">Block Height</div>
                    </div>
                    <div class="status-card">
                        <h2>Best Difficulty</h2>
                        <div class="status-value" id="bestDifficulty">-</div>
                        <div class="status-label">Difficulty</div>
                    </div>
                </div>
            </div>
            
            <div class="section-group">
                <h2 class="section-header">Performance Metrics</h2>
                <div class="status-grid">
                    <div class="status-card">
                        <h2>Hash Rate</h2>
                        <div class="status-value" id="hashRate">-</div>
                        <div class="status-label">Hashes/Second</div>
                    </div>
                    <div class="status-card">
                        <h2>Peak Hash Rate</h2>
                        <div class="status-value" id="peakHashRate">-</div>
                        <div class="status-label">Peak Performance</div>
                    </div>
                    <div class="status-card">
                        <h2>Average Hash Rate</h2>
                        <div class="status-value" id="averageHashRate">-</div>
                        <div class="status-label">Average Performance</div>
                    </div>
                    <div class="status-card">
                        <h2>Total Hashes</h2>
                        <div class="status-value" id="totalHashes">-</div>
                        <div class="status-label">Total Computed</div>
                    </div>
                    <div class="status-card">
                        <h2>Uptime</h2>
                        <div class="status-value" id="uptime">-</div>
                        <div class="status-label">Runtime</div>
                    </div>
                    <div class="status-card">
                        <h2>Shares</h2>
                        <div class="status-value" id="sharesSubmitted">0</div>
                        <div class="status-label">
                            <span id="sharesAccepted" style="color: var(--accent);">Accepted: 0</span> | 
                            <span id="sharesRejected" style="color: var(--error);">Rejected: 0</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section-group">
                <h2 class="section-header">Mining Information</h2>
                <div class="status-grid">
                    <div class="status-card">
                        <h2>Wallet Address</h2>
                        <div class="status-value" id="walletAddress" style="font-size: 1.2em; word-break: break-all;">-</div>
                        <div class="status-label">
                            <a id="walletLink" href="#" target="_blank">View on Blockchain Explorer</a>
                        </div>
                    </div>
                    <div class="status-card">
                        <h2>Job ID</h2>
                        <div class="status-value" id="jobId" style="font-size: 1.2em;">-</div>
                        <div class="status-label">Current Mining Job</div>
                    </div>
                    <div class="status-card">
                        <h2>Last Hash</h2>
                        <div class="hash-display" id="lastHash">-</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Performance Tab -->
        <div id="performance" class="tab-content">
            <div class="section-group">
                <h2 class="section-header">System Resources</h2>
                <div class="status-grid">
                    <div class="status-card">
                        <h2>CPU Usage</h2>
                        <div class="status-value" id="cpuUsage">-</div>
                        <div class="status-label">CPU Utilization</div>
                    </div>
                    <div class="status-card">
                        <h2>Memory Usage</h2>
                        <div class="status-value" id="memoryUsage">-</div>
                        <div class="status-label">RAM Utilization</div>
                    </div>
                    <div class="status-card">
                        <h2>GPU Usage</h2>
                        <div class="status-value" id="gpuUsage">-</div>
                        <div class="status-label">GPU Utilization</div>
                    </div>
                    <div class="status-card">
                        <h2>GPU Temperature</h2>
                        <div class="status-value" id="gpuTemperature">-</div>
                        <div class="status-label">GPU Temp (°C)</div>
                    </div>
                    <div class="status-card">
                        <h2>GPU Memory</h2>
                        <div class="status-value" id="gpuMemory">-</div>
                        <div class="status-label">GPU Memory Usage</div>
                    </div>
                </div>
            </div>
            
            <div class="section-group">
                <h2 class="section-header">Performance Dashboard</h2>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Analytics Tab -->
        <div id="analytics" class="tab-content">
            <div class="section-group">
                <h2 class="section-header">Hash Rate History</h2>
                <div class="chart-container">
                    <canvas id="hashRateChart"></canvas>
                </div>
            </div>
            
            <div class="section-group">
                <h2 class="section-header">Difficulty History</h2>
                <div class="chart-container">
                    <canvas id="difficultyChart"></canvas>
                </div>
            </div>
            
            <div class="section-group">
                <h2 class="section-header">Hash Rate vs Difficulty Comparison</h2>
                <div class="chart-container">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Intelligence Tab -->
        <div id="intelligence" class="tab-content">
            <div class="section-group">
                <h2 class="section-header">Mining Estimates</h2>
                <div class="status-grid">
                    <div class="status-card">
                        <h2>Estimated Time to Block</h2>
                        <div class="status-value" id="estimatedTimeToBlock" style="font-size: 1.8em;">-</div>
                        <div class="status-label">Expected Time</div>
                    </div>
                    <div class="status-card">
                        <h2>Block Found Probability</h2>
                        <div class="status-value" id="blockFoundProbability">-</div>
                        <div class="status-label">Probability (Next Hour)</div>
                    </div>
                    <div class="status-card">
                        <h2>Estimated Profitability</h2>
                        <div class="status-value" id="estimatedProfitability">-</div>
                        <div class="status-label">BTC per Day</div>
                    </div>
                    <div class="status-card">
                        <h2>Difficulty Trend</h2>
                        <div class="status-value" id="difficultyTrend" style="font-size: 1.5em;">-</div>
                        <div class="status-label">Network Trend</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- History Tab -->
        <div id="history" class="tab-content">
            <div class="section-group">
                <h2 class="section-header">Share History</h2>
                <div class="status-card">
                    <div class="share-history" id="shareHistory">
                        <div style="color: var(--text-secondary);">No shares submitted yet</div>
                    </div>
                </div>
            </div>
            
            <div class="section-group">
                <h2 class="section-header">Statistics</h2>
                <div class="status-card">
                    <table class="stats-table">
                <tr>
                    <td>Total Hashes:</td>
                    <td id="statTotalHashes">0</td>
                </tr>
                <tr>
                    <td>Peak Hash Rate:</td>
                    <td id="statPeakHashRate">0.00 H/s</td>
                </tr>
                <tr>
                    <td>Average Hash Rate:</td>
                    <td id="statAverageHashRate">0.00 H/s</td>
                </tr>
                <tr>
                    <td>Shares Submitted:</td>
                    <td id="statSharesSubmitted">0</td>
                </tr>
                <tr>
                    <td>Shares Accepted:</td>
                    <td id="statSharesAccepted">0</td>
                </tr>
                <tr>
                    <td>Shares Rejected:</td>
                    <td id="statSharesRejected">0</td>
                </tr>
                <tr>
                    <td>Success Rate:</td>
                    <td id="statSuccessRate">0.00%</td>
                </tr>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Settings Tab -->
        <div id="settings" class="tab-content">
            <div class="section-group">
                <h2 class="section-header">⚙️ Settings</h2>
                
                <!-- Wallet Configuration -->
                <div class="settings-section">
                    <h3>Wallet Configuration</h3>
                    <div class="settings-grid">
                        <div class="setting-item">
                            <label for="wallet-address">Wallet Address:</label>
                            <input type="text" id="wallet-address" placeholder="Enter Bitcoin wallet address">
                        </div>
                    </div>
                </div>
                
                <!-- Pool Configuration -->
                <div class="settings-section">
                    <h3>Pool Configuration</h3>
                    <div class="settings-grid">
                        <div class="setting-item">
                            <label for="pool-host">Pool Host:</label>
                            <input type="text" id="pool-host" placeholder="solo.ckpool.org">
                        </div>
                        <div class="setting-item">
                            <label for="pool-port">Pool Port:</label>
                            <input type="number" id="pool-port" placeholder="3333" min="1" max="65535">
                        </div>
                    </div>
                </div>
                
                <!-- Network Configuration -->
                <div class="settings-section">
                    <h3>Network Configuration</h3>
                    <div class="settings-grid">
                        <div class="setting-item">
                            <label for="network-source">Block Source:</label>
                            <select id="network-source">
                                <option value="web">Web (Blockchain.info)</option>
                                <option value="local">Local (Bitcoin Core RPC)</option>
                            </select>
                        </div>
                        <div class="setting-item">
                            <label for="network-url">Block URL:</label>
                            <input type="text" id="network-url" placeholder="https://blockchain.info/latestblock">
                        </div>
                        <div class="setting-item">
                            <label for="rpc-url">RPC URL:</label>
                            <input type="text" id="rpc-url" placeholder="http://127.0.0.1:8332">
                        </div>
                        <div class="setting-item">
                            <label for="rpc-user">RPC User:</label>
                            <input type="text" id="rpc-user" placeholder="(empty)">
                        </div>
                        <div class="setting-item">
                            <label for="rpc-password">RPC Password:</label>
                            <input type="password" id="rpc-password" placeholder="(empty)">
                        </div>
                    </div>
                </div>
                
                <!-- Compute Configuration -->
                <div class="settings-section">
                    <h3>Compute Configuration</h3>
                    <div class="settings-grid">
                        <div class="setting-item">
                            <label for="compute-backend">GPU Backend:</label>
                            <select id="compute-backend">
                                <option value="cuda">CUDA</option>
                                <option value="opencl">OpenCL</option>
                            </select>
                            <small style="display: block; margin-top: 0.25rem; color: var(--text-secondary); font-size: 0.875rem;">
                                Backend für GPU-Mining. CPU-Mining wird über den Toggle unten gesteuert.
                            </small>
                        </div>
                        <div class="setting-item">
                            <label for="gpu-device">GPU Device:</label>
                            <input type="number" id="gpu-device" placeholder="0" min="0">
                        </div>
                        <div class="setting-item">
                            <label for="batch-size">Batch Size:</label>
                            <input type="number" id="batch-size" placeholder="256" min="1">
                        </div>
                        <div class="setting-item">
                            <label for="max-workers">Max Workers:</label>
                            <input type="number" id="max-workers" placeholder="8" min="1">
                        </div>
                        <div class="setting-item">
                            <label for="gpu-utilization">GPU Utilization (%):</label>
                            <input type="number" id="gpu-utilization" placeholder="100" min="1" max="100">
                            <small style="display: block; margin-top: 0.25rem; color: var(--text-secondary); font-size: 0.875rem;">
                                Percentage of GPU time used for mining (1-100%). Lower values allow other GPU tasks to run.
                            </small>
                        </div>
                        <div class="setting-item">
                            <div class="toggle-container">
                                <label class="toggle-label">CPU Mining Enabled</label>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="cpu-mining-enabled">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                        </div>
                        <div class="setting-item">
                            <div class="toggle-container">
                                <label class="toggle-label">GPU Mining Enabled</label>
                                <label class="toggle-switch">
                                    <input type="checkbox" id="gpu-mining-enabled">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Database Configuration -->
                <div class="settings-section">
                    <h3>Database Configuration</h3>
                    <div class="settings-grid">
                        <div class="setting-item">
                            <label for="db-retention">Retention (Days):</label>
                            <input type="number" id="db-retention" placeholder="30" min="1" max="3650">
                        </div>
                    </div>
                </div>
                
                <!-- Logging Configuration -->
                <div class="settings-section">
                    <h3>Logging Configuration</h3>
                    <div class="settings-grid">
                        <div class="setting-item">
                            <label for="logging-level">Logging Level:</label>
                            <select id="logging-level">
                                <option value="DEBUG">DEBUG - Verbose (all messages)</option>
                                <option value="INFO">INFO - Informational messages</option>
                                <option value="WARNING">WARNING - Warnings only</option>
                                <option value="ERROR">ERROR - Errors only</option>
                            </select>
                            <small style="display: block; margin-top: 0.25rem; color: var(--text-secondary); font-size: 0.875rem;">
                                Controls the verbosity of log messages. DEBUG provides the most detailed logging.
                            </small>
                        </div>
                        <div class="setting-item">
                            <label for="logging-file">Log File:</label>
                            <input type="text" id="logging-file" placeholder="miner.log">
                            <small style="display: block; margin-top: 0.25rem; color: var(--text-secondary); font-size: 0.875rem;">
                                Path to the log file (relative to the working directory).
                            </small>
                        </div>
                    </div>
                </div>
                
                <!-- Save Button -->
                <div class="settings-actions">
                    <button onclick="saveConfig()" class="btn-primary">💾 Save Configuration</button>
                    <button onclick="loadConfig()" class="btn-secondary">🔄 Reload from Server</button>
                </div>
            </div>
        </div>

        <div class="errors" id="errorsContainer" style="display: none;">
            <h3>Errors</h3>
            <div id="errorsList"></div>
        </div>
    </div>
    <script>
        const socket = io();
        let startTime = null;
        let autoRefresh = true;
        let refreshInterval = null;

        // Chart.js configurations
        const hashRateChart = new Chart(document.getElementById('hashRateChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Hash Rate (H/s)',
                    data: [],
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: true, labels: { color: '#fff' } }
                },
                scales: {
                    y: { beginAtZero: true, ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });

        const difficultyChart = new Chart(document.getElementById('difficultyChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Difficulty',
                    data: [],
                    borderColor: '#a8d5ff',
                    backgroundColor: 'rgba(168, 213, 255, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: true, labels: { color: '#fff' } }
                },
                scales: {
                    y: { beginAtZero: true, ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });

        // Comparison Chart (Hash Rate vs Difficulty) - Feature 3
        const comparisonChart = new Chart(document.getElementById('comparisonChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Hash Rate (H/s)',
                    data: [],
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                }, {
                    label: 'Difficulty',
                    data: [],
                    borderColor: '#a8d5ff',
                    backgroundColor: 'rgba(168, 213, 255, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: true, labels: { color: '#fff' } }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        beginAtZero: true,
                        ticks: { color: '#4ade80' },
                        grid: { color: 'rgba(74, 222, 128, 0.1)' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        beginAtZero: true,
                        ticks: { color: '#a8d5ff' },
                        grid: { drawOnChartArea: false }
                    },
                    x: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });

        // Performance Metrics Chart - Feature 3
        const performanceChart = new Chart(document.getElementById('performanceChart'), {
            type: 'bar',
            data: {
                labels: ['CPU', 'Memory', 'GPU', 'GPU Temp'],
                datasets: [{
                    label: 'Usage (%)',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(74, 222, 128, 0.6)',
                        'rgba(168, 213, 255, 0.6)',
                        'rgba(251, 191, 36, 0.6)',
                        'rgba(239, 68, 68, 0.6)'
                    ],
                    borderColor: [
                        '#4ade80',
                        '#a8d5ff',
                        '#fbbf24',
                        '#ef4444'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                if (context.dataIndex === 3) {
                                    return 'GPU Temp: ' + context.parsed.y + '°C';
                                }
                                return 'Usage: ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#ccc', callback: function(value) {
                            if (this.dataIndex === 3) return value + '°C';
                            return value + '%';
                        }},
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    },
                    x: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });

        function updateCharts(data) {
            if (data.hash_rate_history && data.hash_rate_history.length > 0) {
                const labels = data.hash_rate_history.map((_, i) => i * 2 + 's');
                hashRateChart.data.labels = labels.slice(-60);
                hashRateChart.data.datasets[0].data = data.hash_rate_history.slice(-60);
                hashRateChart.update('none');
            }
            if (data.difficulty_history && data.difficulty_history.length > 0) {
                const labels = data.difficulty_history.map((_, i) => i * 2 + 's');
                difficultyChart.data.labels = labels.slice(-60);
                difficultyChart.data.datasets[0].data = data.difficulty_history.slice(-60);
                difficultyChart.update('none');
            }
            // Update comparison chart (Hash Rate vs Difficulty) - Feature 3
            if (data.hash_rate_history && data.difficulty_history && 
                data.hash_rate_history.length > 0 && data.difficulty_history.length > 0) {
                const minLen = Math.min(data.hash_rate_history.length, data.difficulty_history.length);
                const labels = data.hash_rate_history.slice(-minLen).map((_, i) => i * 2 + 's');
                comparisonChart.data.labels = labels;
                comparisonChart.data.datasets[0].data = data.hash_rate_history.slice(-minLen);
                comparisonChart.data.datasets[1].data = data.difficulty_history.slice(-minLen);
                comparisonChart.update('none');
            }
            // Update performance chart - Feature 1
            if (data.cpu_usage !== undefined || data.memory_usage !== undefined || 
                data.gpu_usage !== undefined || data.gpu_temperature !== undefined) {
                performanceChart.data.datasets[0].data = [
                    data.cpu_usage || 0,
                    data.memory_usage || 0,
                    data.gpu_usage || 0,
                    data.gpu_temperature || 0
                ];
                performanceChart.update('none');
            }
        }

        function toggleTheme() {
            document.body.classList.toggle('light-theme');
            const isLight = document.body.classList.contains('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            
            // Update button icon
            const themeToggle = document.getElementById('themeToggle');
            if (themeToggle) {
                themeToggle.textContent = isLight ? '🌙 Dark' : '☀️ Light';
            }
        }
        
        let miningPaused = false;

        function toggleMining() {
            if (miningPaused) {
                // Resume mining
                fetch('/api/start', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            miningPaused = false;
                            document.getElementById('autoRefreshText').textContent = '⏸️ Pause';
                        }
                    })
                    .catch(error => console.error('Error starting mining:', error));
            } else {
                // Pause/Stop mining
                fetch('/api/stop', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            miningPaused = true;
                            document.getElementById('autoRefreshText').textContent = '▶️ Resume';
                        }
                    })
                    .catch(error => console.error('Error stopping mining:', error));
            }
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const pauseText = miningPaused ? '▶️ Resume' : '⏸️ Pause';
            document.getElementById('autoRefreshText').textContent = autoRefresh ? pauseText : '⏸️ Pause';
            if (autoRefresh) {
                startRefresh();
            } else {
                clearInterval(refreshInterval);
            }
        }

        function startRefresh() {
            if (refreshInterval) clearInterval(refreshInterval);
            refreshInterval = setInterval(() => {
                if (autoRefresh) socket.emit('get_status');
            }, 3000);
        }

        // Tab Navigation
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            // Find and activate the clicked tab button
            document.querySelectorAll('.tab').forEach(btn => {
                if (btn.textContent.includes(tabName.charAt(0).toUpperCase() + tabName.slice(1)) || 
                    (tabName === 'overview' && btn.textContent.includes('Overview')) ||
                    (tabName === 'settings' && btn.textContent.includes('Settings'))) {
                    btn.classList.add('active');
                }
            });
            
            // Load config when settings tab is shown
            if (tabName === 'settings') {
                loadConfig();
            }
            
            // Save active tab
            localStorage.setItem('activeTab', tabName);
        }
        
        // Load configuration from server
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const data = await response.json();
                if (data.success) {
                    const config = data.config;
                    
                    // Wallet
                    document.getElementById('wallet-address').value = config.wallet?.address || '';
                    
                    // Pool
                    document.getElementById('pool-host').value = config.pool?.host || '';
                    document.getElementById('pool-port').value = config.pool?.port || '';
                    
                    // Network
                    document.getElementById('network-source').value = config.network?.source || 'web';
                    document.getElementById('network-url').value = config.network?.latest_block_url || '';
                    document.getElementById('rpc-url').value = config.network?.rpc_url || '';
                    document.getElementById('rpc-user').value = config.network?.rpc_user || '';
                    document.getElementById('rpc-password').value = config.network?.rpc_password || '';
                    
                    // Compute
                    const backendSelect = document.getElementById('compute-backend');
                    const storedBackend = localStorage.getItem('preferredGpuBackend') || 'cuda';
                    const backend = config.compute?.backend;
                    if (backend === 'cuda' || backend === 'opencl') {
                        backendSelect.value = backend;
                        localStorage.setItem('preferredGpuBackend', backend);
                    } else {
                        backendSelect.value = storedBackend;
                    }
                    document.getElementById('gpu-device').value = config.compute?.gpu_device || 0;
                    document.getElementById('batch-size').value = config.compute?.batch_size || 256;
                    document.getElementById('max-workers').value = config.compute?.max_workers || 8;
                    document.getElementById('gpu-utilization').value = config.compute?.gpu_utilization_percent || 100;
                    document.getElementById('cpu-mining-enabled').checked = config.compute?.cpu_mining_enabled !== false;
                    document.getElementById('gpu-mining-enabled').checked = config.compute?.gpu_mining_enabled === true;
                    
                    // Database
                    document.getElementById('db-retention').value = config.database?.retention_days || 30;
                    
                    // Logging
                    document.getElementById('logging-level').value = config.logging?.level || 'INFO';
                    document.getElementById('logging-file').value = config.logging?.file || 'miner.log';
                    
                    console.log('Configuration loaded successfully');
                } else {
                    console.error('Failed to load config:', data.error);
                    alert('Failed to load configuration: ' + data.error);
                }
            } catch (error) {
                console.error('Error loading config:', error);
                alert('Error loading configuration: ' + error.message);
            }
        }
        
        // Save configuration to server
        async function saveConfig() {
            const config = {
                wallet: {
                    address: document.getElementById('wallet-address').value
                },
                pool: {
                    host: document.getElementById('pool-host').value,
                    port: parseInt(document.getElementById('pool-port').value) || 3333
                },
                network: {
                    source: document.getElementById('network-source').value,
                    latest_block_url: document.getElementById('network-url').value,
                    request_timeout_secs: 15,
                    rpc_url: document.getElementById('rpc-url').value,
                    rpc_user: document.getElementById('rpc-user').value,
                    rpc_password: document.getElementById('rpc-password').value
                },
                logging: {
                    file: document.getElementById('logging-file').value || 'miner.log',
                    level: document.getElementById('logging-level').value || 'INFO'
                },
                miner: {
                    restart_delay_secs: 2,
                    subscribe_thread_start_delay_secs: 4,
                    hash_log_prefix_zeros: 7
                },
                compute: {
                    backend: document.getElementById('compute-backend').value,
                    gpu_device: parseInt(document.getElementById('gpu-device').value, 10) || 0,
                    batch_size: parseInt(document.getElementById('batch-size').value, 10) || 256,
                    max_workers: parseInt(document.getElementById('max-workers').value, 10) || 8,
                    gpu_utilization_percent: Math.max(1, Math.min(100, parseInt(document.getElementById('gpu-utilization').value, 10) || 100)),
                    cpu_mining_enabled: document.getElementById('cpu-mining-enabled').checked,
                    gpu_mining_enabled: document.getElementById('gpu-mining-enabled').checked
                },
                database: {
                    retention_days: parseInt(document.getElementById('db-retention').value) || 30
                }
            };
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Origin': window.location.origin,
                        'Referer': window.location.href
                    },
                    body: JSON.stringify({ config })
                });
                
                const data = await response.json();
                if (data.success) {
                    alert('Configuration saved successfully!');
                    // Reload config to show saved wallet address
                    await loadConfig();
                } else {
                    alert('Failed to save configuration: ' + data.error);
                }
            } catch (error) {
                console.error('Error saving config:', error);
                alert('Error saving configuration: ' + error.message);
            }
        }
        
        // Toggle CPU mining
        async function toggleCpuMining(enabled) {
            try {
                const response = await fetch('/api/config/cpu-mining', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Origin': window.location.origin,
                        'Referer': window.location.href
                    },
                    body: JSON.stringify({ enabled })
                });
                const data = await response.json();
                if (data.success) {
                    console.log('CPU mining toggled:', enabled);
                } else {
                    alert('Failed to toggle CPU mining: ' + data.error);
                }
            } catch (error) {
                console.error('Error toggling CPU mining:', error);
            }
        }
        
        // Toggle GPU mining
        async function toggleGpuMining(enabled) {
            try {
                const response = await fetch('/api/config/gpu-mining', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Origin': window.location.origin,
                        'Referer': window.location.href
                    },
                    body: JSON.stringify({ enabled })
                });
                const data = await response.json();
                if (data.success) {
                    console.log('GPU mining toggled:', enabled);
                } else {
                    alert('Failed to toggle GPU mining: ' + data.error);
                }
            } catch (error) {
                console.error('Error toggling GPU mining:', error);
            }
        }
        
        // Event listeners for checkboxes
        document.addEventListener('DOMContentLoaded', function() {
            const cpuCheckbox = document.getElementById('cpu-mining-enabled');
            const gpuCheckbox = document.getElementById('gpu-mining-enabled');
            const backendSelect = document.getElementById('compute-backend');
            
            if (backendSelect) {
                backendSelect.addEventListener('change', function() {
                    localStorage.setItem('preferredGpuBackend', this.value);
                });
            }
            
            if (cpuCheckbox) {
                cpuCheckbox.addEventListener('change', function() {
                    toggleCpuMining(this.checked);
                });
            }
            
            if (gpuCheckbox) {
                gpuCheckbox.addEventListener('change', function() {
                    if (this.checked && backendSelect) {
                        const preferred = localStorage.getItem('preferredGpuBackend') || 'cuda';
                        if (backendSelect.value !== preferred) {
                            backendSelect.value = preferred;
                        }
                    }
                    toggleGpuMining(this.checked);
                });
            }
        });
        
        // Load saved theme and tab
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            const themeToggle = document.getElementById('themeToggle');
            if (themeToggle) {
                themeToggle.textContent = '🌙 Dark';
            }
        }
        
        // Restore saved tab on page load
        const savedTab = localStorage.getItem('activeTab') || 'overview';
        showTab(savedTab);

        socket.on('connect', () => {
            startRefresh();
        });

        socket.on('disconnect', () => {
            // Connection status removed - redundant information
        });

        // Format hash numbers with magnitude units (K, M, G, T, P, E)
        function formatHashNumber(value, unit = 'H/s') {
            if (value === 0 || !value) return '0 ' + unit;
            
            const absValue = Math.abs(value);
            if (absValue < 1000) {
                return value.toFixed(2) + ' ' + unit;
            } else if (absValue < 1000000) {
                return (value / 1000).toFixed(2) + ' K' + unit;
            } else if (absValue < 1000000000) {
                return (value / 1000000).toFixed(2) + ' M' + unit;
            } else if (absValue < 1000000000000) {
                return (value / 1000000000).toFixed(2) + ' G' + unit;
            } else if (absValue < 1000000000000000) {
                return (value / 1000000000000).toFixed(2) + ' T' + unit;
            } else if (absValue < 1000000000000000000) {
                return (value / 1000000000000000).toFixed(2) + ' P' + unit;
            } else {
                return (value / 1000000000000000000).toFixed(2) + ' E' + unit;
            }
        }

        socket.on('status', (data) => {
            if (!startTime && data.start_time) {
                // Handle both Unix timestamp (number) and ISO string (backward compatibility)
                if (typeof data.start_time === 'number') {
                    // Unix timestamp in seconds, convert to milliseconds for Date
                    startTime = new Date(data.start_time * 1000);
                } else {
                    // ISO string (backward compatibility)
                    startTime = new Date(data.start_time);
                }
            }

            // Update status cards
            document.getElementById('runningStatus').textContent = data.running ? 'Running' : 'Stopped';
            document.getElementById('runningStatus').className = data.running ? 'status-value running' : 'status-value stopped';
            
            document.getElementById('currentHeight').textContent = data.current_height || 0;
            document.getElementById('bestDifficulty').textContent = data.best_difficulty ? data.best_difficulty.toFixed(2) : '0.00';
            document.getElementById('hashRate').textContent = formatHashNumber(data.hash_rate || 0, 'H/s');
            document.getElementById('peakHashRate').textContent = formatHashNumber(data.peak_hash_rate || 0, 'H/s');
            document.getElementById('averageHashRate').textContent = formatHashNumber(data.average_hash_rate || 0, 'H/s');
            document.getElementById('totalHashes').textContent = formatHashNumber(data.total_hashes || 0, 'H');
            
            // Pool status
            const poolConnected = data.pool_connected || false;
            document.getElementById('poolStatus').textContent = poolConnected ? 'Connected' : 'Disconnected';
            document.getElementById('poolStatusBadge').className = poolConnected ? 'pool-status pool-connected' : 'pool-status pool-disconnected';
            document.getElementById('poolStatusBadge').textContent = poolConnected ? 'Connected' : 'Disconnected';
            if (data.pool_host && data.pool_port) {
                document.getElementById('poolInfo').textContent = `${data.pool_host}:${data.pool_port}`;
            }
            
            // Job ID
            if (data.job_id) {
                document.getElementById('jobId').textContent = data.job_id;
            }
            
            // Shares
            document.getElementById('sharesSubmitted').textContent = data.shares_submitted || 0;
            document.getElementById('sharesAccepted').textContent = `Accepted: ${data.shares_accepted || 0}`;
            document.getElementById('sharesRejected').textContent = `Rejected: ${data.shares_rejected || 0}`;
            
            // Statistics table
            document.getElementById('statTotalHashes').textContent = formatHashNumber(data.total_hashes || 0, 'H');
            document.getElementById('statPeakHashRate').textContent = formatHashNumber(data.peak_hash_rate || 0, 'H/s');
            document.getElementById('statAverageHashRate').textContent = formatHashNumber(data.average_hash_rate || 0, 'H/s');
            document.getElementById('statSharesSubmitted').textContent = data.shares_submitted || 0;
            document.getElementById('statSharesAccepted').textContent = data.shares_accepted || 0;
            document.getElementById('statSharesRejected').textContent = data.shares_rejected || 0;
            const successRate = data.shares_submitted > 0 ? ((data.shares_accepted || 0) / data.shares_submitted * 100).toFixed(2) : 0;
            document.getElementById('statSuccessRate').textContent = successRate + '%';
            
            // Share history
            if (data.shares && data.shares.length > 0) {
                const historyHtml = data.shares.slice().reverse().map(share => {
                    const date = new Date(share.timestamp);
                    const timeStr = date.toLocaleTimeString();
                    return `<div class="share-item share-accepted">${timeStr} - Share Submitted</div>`;
                }).join('');
                document.getElementById('shareHistory').innerHTML = historyHtml;
            }
            
            // Uptime
            if (startTime) {
                const now = new Date();
                const uptime = Math.floor((now - startTime) / 1000);
                // Ensure uptime is not negative (shouldn't happen, but safety check)
                const safeUptime = Math.max(0, uptime);
                const hours = Math.floor(safeUptime / 3600);
                const minutes = Math.floor((safeUptime % 3600) / 60);
                const seconds = safeUptime % 60;
                document.getElementById('uptime').textContent = `${hours}h ${minutes}m ${seconds}s`;
            } else if (data.start_time) {
                // Fallback: calculate uptime from server-side timestamp if startTime not set
                const startTimeValue = typeof data.start_time === 'number' 
                    ? data.start_time * 1000 
                    : new Date(data.start_time).getTime();
                const now = Date.now();
                const uptime = Math.floor((now - startTimeValue) / 1000);
                const safeUptime = Math.max(0, uptime);
                const hours = Math.floor(safeUptime / 3600);
                const minutes = Math.floor((safeUptime % 3600) / 60);
                const seconds = safeUptime % 60;
                document.getElementById('uptime').textContent = `${hours}h ${minutes}m ${seconds}s`;
            }

            if (data.last_hash) {
                document.getElementById('lastHash').textContent = data.last_hash;
            }

            if (data.errors && data.errors.length > 0) {
                document.getElementById('errorsContainer').style.display = 'block';
                document.getElementById('errorsList').innerHTML = data.errors.map(e => 
                    `<div class="error-item">${e}</div>`
                ).join('');
            } else {
                document.getElementById('errorsContainer').style.display = 'none';
            }

            // Update wallet address and link
            if (data.wallet_address) {
                document.getElementById('walletAddress').textContent = data.wallet_address;
                if (data.explorer_url) {
                    const link = document.getElementById('walletLink');
                    link.href = data.explorer_url;
                    link.style.display = 'inline';
                } else {
                    document.getElementById('walletLink').style.display = 'none';
                }
            }
            
            // Performance & Monitoring (Feature 1)
            if (data.cpu_usage !== undefined) {
                document.getElementById('cpuUsage').textContent = data.cpu_usage.toFixed(1) + '%';
            }
            if (data.memory_usage !== undefined) {
                document.getElementById('memoryUsage').textContent = data.memory_usage.toFixed(1) + '%';
            }
            if (data.gpu_usage !== undefined && data.gpu_usage > 0) {
                document.getElementById('gpuUsage').textContent = data.gpu_usage.toFixed(1) + '%';
            } else {
                document.getElementById('gpuUsage').textContent = 'N/A';
            }
            if (data.gpu_temperature !== undefined && data.gpu_temperature > 0) {
                document.getElementById('gpuTemperature').textContent = data.gpu_temperature.toFixed(0) + '°C';
            } else {
                document.getElementById('gpuTemperature').textContent = 'N/A';
            }
            if (data.gpu_memory !== undefined && data.gpu_memory > 0) {
                document.getElementById('gpuMemory').textContent = data.gpu_memory.toFixed(1) + '%';
            } else {
                document.getElementById('gpuMemory').textContent = 'N/A';
            }
            
            // Mining Intelligence (Feature 2)
            if (data.estimated_time_to_block_formatted) {
                document.getElementById('estimatedTimeToBlock').textContent = data.estimated_time_to_block_formatted;
            } else if (data.estimated_time_to_block) {
                const seconds = data.estimated_time_to_block;
                // Format as years, months, days
                if (seconds < 60) {
                    document.getElementById('estimatedTimeToBlock').textContent = seconds.toFixed(1) + 's';
                } else if (seconds < 3600) {
                    document.getElementById('estimatedTimeToBlock').textContent = (seconds / 60).toFixed(1) + 'm';
                } else if (seconds < 86400) {
                    document.getElementById('estimatedTimeToBlock').textContent = (seconds / 3600).toFixed(1) + 'h';
                } else {
                    // Convert to years, months, days
                    const totalDays = seconds / 86400;
                    const years = Math.floor(totalDays / 365);
                    const remainingDays = totalDays - (years * 365);
                    const months = Math.floor(remainingDays / 30);
                    const days = remainingDays - (months * 30);
                    
                    const parts = [];
                    if (years > 0) {
                        parts.push(years + ' ' + (years === 1 ? 'year' : 'years'));
                    }
                    if (months > 0) {
                        parts.push(months + ' ' + (months === 1 ? 'month' : 'months'));
                    }
                    if (days > 0 || parts.length === 0) {
                        parts.push(days.toFixed(1) + ' ' + (days === 1 ? 'day' : 'days'));
                    }
                    
                    document.getElementById('estimatedTimeToBlock').textContent = parts.join(', ');
                }
            } else {
                document.getElementById('estimatedTimeToBlock').textContent = 'N/A';
            }
            
            if (data.block_found_probability !== undefined) {
                document.getElementById('blockFoundProbability').textContent = data.block_found_probability.toFixed(4) + '%';
            } else {
                document.getElementById('blockFoundProbability').textContent = '0.00%';
            }
            
            if (data.estimated_profitability !== undefined && data.estimated_profitability > 0) {
                document.getElementById('estimatedProfitability').textContent = data.estimated_profitability.toFixed(8) + ' BTC';
            } else {
                document.getElementById('estimatedProfitability').textContent = '0.00000000 BTC';
            }
            
            if (data.difficulty_trend) {
                const trend = data.difficulty_trend;
                const trendEmoji = trend === 'increasing' ? '📈' : trend === 'decreasing' ? '📉' : '➡️';
                const trendText = trend === 'increasing' ? 'Increasing' : trend === 'decreasing' ? 'Decreasing' : 'Stable';
                document.getElementById('difficultyTrend').textContent = trendEmoji + ' ' + trendText;
            } else {
                document.getElementById('difficultyTrend').textContent = '➡️ Stable';
            }
            
            // Update charts
            updateCharts(data);
        });

        startRefresh();
    </script>
</body>
</html>
"""
