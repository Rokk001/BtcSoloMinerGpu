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

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit

from ..core.state import MinerState

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


STATUS_LOCK = threading.Lock()
STATUS: Dict = {
    "running": False,
    "current_height": 0,
    "best_difficulty": 0.0,
    "hash_rate": 0.0,
    "last_hash": None,
    "uptime_seconds": 0,
    "start_time": None,
    "wallet_address": None,
    "explorer_url": None,
    "pool_connected": False,
    "pool_host": None,
    "pool_port": None,
    "job_id": None,
    "total_hashes": 0,
    "peak_hash_rate": 0.0,
    "average_hash_rate": 0.0,
    "shares_submitted": 0,
    "shares_accepted": 0,
    "shares_rejected": 0,
    "last_share_time": None,
    "hash_rate_history": deque(maxlen=60),  # Last 60 data points (2 minutes at 2s intervals)
    "difficulty_history": deque(maxlen=60),
    # Performance & Monitoring (Feature 1)
    "cpu_usage": 0.0,
    "memory_usage": 0.0,
    "gpu_usage": 0.0,
    "gpu_temperature": 0.0,
    "gpu_memory": 0.0,
    # Mining Intelligence (Feature 2)
    "estimated_time_to_block": None,
    "block_found_probability": 0.0,
    "estimated_profitability": 0.0,
    "difficulty_trend": "stable",  # increasing, decreasing, stable
    "network_difficulty": 0.0,
    "target_difficulty": 0.0,
    "errors": []
}

# Statistics tracking
STATS_LOCK = threading.Lock()
STATS = {
    "total_hashes": 0,
    "peak_hash_rate": 0.0,
    "hash_rate_samples": deque(maxlen=300),  # Last 300 samples for average
    "shares": deque(maxlen=100),  # Last 100 shares to prevent memory leak
    "start_time": None
}


def update_status(key: str, value):
    with STATUS_LOCK:
        STATUS[key] = value
        # Update statistics
        if key == "hash_rate" and value:
            with STATS_LOCK:
                STATS["hash_rate_samples"].append(value)
                if value > STATS["peak_hash_rate"]:
                    STATS["peak_hash_rate"] = value
                    STATUS["peak_hash_rate"] = value
                if len(STATS["hash_rate_samples"]) > 0:
                    avg = sum(STATS["hash_rate_samples"]) / len(STATS["hash_rate_samples"])
                    STATUS["average_hash_rate"] = avg
        if key == "total_hashes":
            with STATS_LOCK:
                STATS["total_hashes"] = value
        if key == "shares_submitted":
            with STATS_LOCK:
                STATS["shares"].append({
                    "timestamp": datetime.now().isoformat(),
                    "accepted": value
                })


def get_status() -> Dict:
    with STATUS_LOCK:
        status = STATUS.copy()
        # Add history arrays
        status["hash_rate_history"] = list(status["hash_rate_history"])
        status["difficulty_history"] = list(status["difficulty_history"])
        # Add statistics
        with STATS_LOCK:
            status["total_hashes"] = STATS["total_hashes"]
            # Convert deque to list and get last 10 shares
            shares_list = list(STATS["shares"])
            status["shares"] = shares_list[-10:]  # Last 10 shares
        return status


def add_share(accepted: bool, response: str = None):
    with STATUS_LOCK:
        if accepted:
            STATUS["shares_accepted"] += 1
        else:
            STATUS["shares_rejected"] += 1
        STATUS["shares_submitted"] += 1
        STATUS["last_share_time"] = datetime.now().isoformat()
        update_status("shares_submitted", STATUS["shares_submitted"])


def update_pool_status(connected: bool, host: str = None, port: int = None):
    with STATUS_LOCK:
        STATUS["pool_connected"] = connected
        if host:
            STATUS["pool_host"] = host
        if port:
            STATUS["pool_port"] = port


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
                if not hasattr(update_performance_metrics, 'nvml_initialized'):
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
                            
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            gpu_temperature = temp
                            
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_memory = (mem_info.used / mem_info.total) * 100
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
def format_hash_number(value: float, unit: str = "H/s") -> str:
    """Format hash numbers with magnitude units (K, M, G, T, P, E)"""
    if value == 0:
        return f"0 {unit}"
    
    abs_value = abs(value)
    if abs_value < 1000:
        return f"{value:.2f} {unit}"
    elif abs_value < 1_000_000:
        return f"{value / 1000:.2f} K{unit}"
    elif abs_value < 1_000_000_000:
        return f"{value / 1_000_000:.2f} M{unit}"
    elif abs_value < 1_000_000_000_000:
        return f"{value / 1_000_000_000:.2f} G{unit}"
    elif abs_value < 1_000_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} T{unit}"
    elif abs_value < 1_000_000_000_000_000_000:
        return f"{value / 1_000_000_000_000_000:.2f} P{unit}"
    else:
        return f"{value / 1_000_000_000_000_000_000:.2f} E{unit}"


# Mining Intelligence Functions (Feature 2)
def format_time_to_block(seconds: float) -> str:
    """Convert seconds to human-readable format: years, months, days"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        # Convert to years, months, days
        total_days = seconds / 86400
        years = int(total_days / 365)
        remaining_days = total_days - (years * 365)
        months = int(remaining_days / 30)
        days = remaining_days - (months * 30)
        
        parts = []
        if years > 0:
            parts.append(f"{years} {'year' if years == 1 else 'years'}")
        if months > 0:
            parts.append(f"{months} {'month' if months == 1 else 'months'}")
        if days > 0 or len(parts) == 0:
            parts.append(f"{days:.1f} {'day' if days == 1 else 'days'}")
        
        return ", ".join(parts)


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
        expected_hashes = (2 ** 32) * difficulty
        if hash_rate > 0:
            estimated_seconds = expected_hashes / hash_rate
            STATUS["estimated_time_to_block"] = estimated_seconds
            
            # Convert to human-readable format (years, months, days)
            STATUS["estimated_time_to_block_formatted"] = format_time_to_block(estimated_seconds)
            
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
                    avg_older = sum(recent[:-3]) / len(recent[:-3]) if len(recent) > 3 else recent[0]
                    if avg_recent > avg_older * 1.05:
                        STATUS["difficulty_trend"] = "increasing"
                    elif avg_recent < avg_older * 0.95:
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
    while True:
        try:
            update_performance_metrics()
            calculate_mining_intelligence()
            time.sleep(2)  # Update every 2 seconds
        except Exception as e:
            logging.debug(f"Performance monitor error: {e}")
            time.sleep(5)


# Start performance monitoring thread
_performance_thread = None
def start_performance_monitoring():
    """Start the performance monitoring background thread"""
    global _performance_thread
    if _performance_thread is None or not _performance_thread.is_alive():
        _performance_thread = threading.Thread(target=performance_monitor_thread, daemon=True)
        _performance_thread.start()


app = Flask(__name__, static_url_path="/static")
# Use environment variable for SECRET_KEY or generate a random one
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
# CORS: Allow specific origins or localhost by default for security
cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
socketio = SocketIO(app, cors_allowed_origins=cors_origins)


@app.route("/favicon.ico")
def favicon():
    """Serve favicon as SVG"""
    svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
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
</svg>'''
    return app.response_class(
        response=svg_content,
        mimetype="image/svg+xml",
        headers={"Cache-Control": "public, max-age=31536000"}
    )


@app.route("/favicon.svg")
def favicon_svg():
    """Serve favicon as SVG"""
    return favicon()


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/export")
def export_stats():
    """Export statistics as JSON"""
    stats = get_status()
    return app.response_class(
        response=json.dumps(stats, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=satoshirig-stats.json"}
    )


@app.route("/api/stop", methods=["POST"])
def stop_mining():
    """Stop mining by setting shutdown flag"""
    from flask import request
    
    # CSRF protection
    if not _check_csrf_protection(request):
        return jsonify({
            "success": False,
            "error": "CSRF validation failed",
            "message": "Request origin not allowed. CSRF protection enabled."
        }), 403
    
    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({
            "success": False,
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Maximum {_api_rate_limit_max_requests} requests per {_api_rate_limit_window} seconds."
        }), 429
    
    try:
        global _miner_state
        if not _miner_state:
            return jsonify({
                "success": False,
                "error": "MinerStateNotAvailable",
                "message": "Miner state not available. Miner may not be running."
            }), 503
        
        _miner_state.shutdown_flag = True
        update_status("running", False)
        return jsonify({
            "success": True,
            "message": "Mining stopped"
        })
    except Exception as e:
        logging.error(f"Error stopping mining: {e}")
        return jsonify({
            "success": False,
            "error": "InternalError",
            "message": f"Failed to stop mining: {str(e)}"
        }), 500


@app.route("/api/start", methods=["POST"])
def start_mining():
    """Resume mining by clearing shutdown flag (Note: Requires miner restart to actually resume)"""
    from flask import request
    
    # CSRF protection
    if not _check_csrf_protection(request):
        return jsonify({
            "success": False,
            "error": "CSRF validation failed",
            "message": "Request origin not allowed. CSRF protection enabled."
        }), 403
    
    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not _check_rate_limit(client_ip):
        return jsonify({
            "success": False,
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Maximum {_api_rate_limit_max_requests} requests per {_api_rate_limit_window} seconds."
        }), 429
    
    try:
        global _miner_state
        if not _miner_state:
            return jsonify({
                "success": False,
                "error": "MinerStateNotAvailable",
                "message": "Miner state not available. Miner may not be running."
            }), 503
        
        _miner_state.shutdown_flag = False
        update_status("running", True)
        return jsonify({
            "success": True,
            "message": "Mining resumed (may require restart)"
        })
    except Exception as e:
        logging.error(f"Error starting mining: {e}")
        return jsonify({
            "success": False,
            "error": "InternalError",
            "message": f"Failed to start mining: {str(e)}"
        }), 500


@socketio.on("connect")
def handle_connect():
    emit("status", get_status())


@socketio.on("get_status")
def handle_get_status():
    emit("status", get_status())


def broadcast_status():
    while True:
        with STATUS_LOCK:
            # Add current values to history
            if STATUS["hash_rate"] > 0:
                STATUS["hash_rate_history"].append(STATUS["hash_rate"])
            if STATUS["best_difficulty"] > 0:
                STATUS["difficulty_history"].append(STATUS["best_difficulty"])
        socketio.emit("status", get_status())
        time.sleep(2)


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
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)


# Global reference to miner state for controlling mining
_miner_state = None

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
            req_time for req_time in _api_rate_limit[client_ip]
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
    allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
    
    # Get Origin or Referer header
    origin = request.headers.get('Origin')
    referer = request.headers.get('Referer')
    
    # If no Origin/Referer, allow only if from same origin (local requests)
    if not origin and not referer:
        # Allow requests without Origin/Referer for localhost/127.0.0.1
        # This is acceptable for a local mining application
        return True
    
    # Check Origin header first (more reliable)
    if origin:
        # Remove protocol and path, keep only origin
        origin_base = origin.split('://')[1].split('/')[0] if '://' in origin else origin.split('/')[0]
        for allowed in allowed_origins:
            allowed_base = allowed.split('://')[1].split('/')[0] if '://' in allowed else allowed.split('/')[0]
            if origin_base == allowed_base or origin_base in allowed_base or allowed_base in origin_base:
                return True
    
    # Fallback to Referer header
    if referer:
        referer_base = referer.split('://')[1].split('/')[0] if '://' in referer else referer.split('/')[0]
        for allowed in allowed_origins:
            allowed_base = allowed.split('://')[1].split('/')[0] if '://' in allowed else allowed.split('/')[0]
            if referer_base == allowed_base or referer_base in allowed_base or allowed_base in referer_base:
                return True
    
    # If Origin/Referer doesn't match allowed origins, reject
    return False


def set_miner_state(miner_state):
    """Set the miner state reference for controlling mining"""
    global _miner_state
    _miner_state = miner_state


# Export functions for use by miner
__all__ = ["start_web_server", "update_status", "get_status", "add_share", "update_pool_status", "set_miner_state"]


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
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --bg-secondary: rgba(255, 255, 255, 0.1);
            --text-primary: #fff;
            --text-secondary: #ccc;
            --accent: #4ade80;
            --error: #f87171;
            --card-bg: rgba(255, 255, 255, 0.1);
        }
        body.light-theme {
            --bg-primary: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
            --bg-secondary: rgba(255, 255, 255, 0.9);
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --accent: #10b981;
            --error: #ef4444;
            --card-bg: rgba(255, 255, 255, 0.9);
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
            min-height: 100vh;
            transition: background 0.3s ease;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }
        h1 {
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            background: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s;
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: transform 0.3s;
        }
        .status-card:hover {
            transform: translateY(-5px);
        }
        .status-card h2 {
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #a8d5ff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .light-theme .status-card h2 {
            color: #3b82f6;
        }
        .status-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .status-label {
            font-size: 0.9em;
            color: var(--text-secondary);
            opacity: 0.8;
        }
        .running { color: var(--accent); }
        .stopped { color: var(--error); }
        .hash-display {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            word-break: break-all;
            margin-top: 10px;
        }
        .chart-container {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin-bottom: 30px;
        }
        .chart-container h2 {
            margin-bottom: 20px;
            color: #a8d5ff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .light-theme .chart-container h2 {
            color: #3b82f6;
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
        .pool-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-top: 5px;
        }
        .pool-connected { background: var(--accent); color: #000; }
        .pool-disconnected { background: var(--error); color: #fff; }
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
        
        /* Tabs Navigation */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            flex-wrap: wrap;
        }
        .tab {
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: all 0.3s;
            position: relative;
            top: 2px;
        }
        .tab:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }
        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            background: rgba(74, 222, 128, 0.1);
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Section Headers */
        .section-header {
            font-size: 1.5em;
            margin: 30px 0 20px 0;
            color: var(--accent);
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(74, 222, 128, 0.3);
        }
        .section-group {
            margin-bottom: 40px;
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .status-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; align-items: flex-start; }
            .tabs { flex-direction: column; gap: 5px; }
            .tab { padding: 10px 16px; }
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SatoshiRig Status Dashboard</h1>
            <div class="controls">
                <button class="btn" onclick="toggleTheme()">🌓 Theme</button>
                <button class="btn" onclick="exportStats()">📥 Export</button>
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
            localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
        }

        function exportStats() {
            window.location.href = '/export';
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
                    (tabName === 'overview' && btn.textContent.includes('Overview'))) {
                    btn.classList.add('active');
                }
            });
            
            // Save active tab
            localStorage.setItem('activeTab', tabName);
        }
        
        // Load saved theme and tab
        if (localStorage.getItem('theme') === 'light') {
            document.body.classList.add('light-theme');
        }
        
        const savedTab = localStorage.getItem('activeTab') || 'overview';
        if (savedTab !== 'overview') {
            document.querySelectorAll('.tab').forEach(btn => {
                if (btn.textContent.includes(savedTab.charAt(0).toUpperCase() + savedTab.slice(1))) {
                    btn.click();
                }
            });
        }

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

