"""Status management for the web dashboard."""

import threading
from collections import deque
from datetime import datetime
from typing import Dict, Optional

# Import persistent statistics
from .stats_persistence import load_statistics, save_statistics, merge_statistics

STATUS_LOCK = threading.Lock()

# Global reference to SocketIO instance for immediate status updates
_socketio_instance: Optional[object] = None

# Load persistent statistics on startup
_persistent_stats = load_statistics()

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
    # Initialize with persistent statistics
    "total_hashes": _persistent_stats["total_hashes"],
    "peak_hash_rate": _persistent_stats["peak_hash_rate"],
    "average_hash_rate": 0.0,
    "shares_submitted": _persistent_stats["shares_submitted"],
    "shares_accepted": _persistent_stats["shares_accepted"],
    "shares_rejected": _persistent_stats["shares_rejected"],
    "last_share_time": None,
    "hash_rate_history": deque(
        maxlen=60
    ),  # Last 60 data points (2 minutes at 2s intervals)
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
    "errors": [],
}

# Statistics tracking
STATS_LOCK = threading.Lock()
STATS = {
    "total_hashes": _persistent_stats["total_hashes"],  # Start with persistent value
    "peak_hash_rate": _persistent_stats[
        "peak_hash_rate"
    ],  # Start with persistent value
    "hash_rate_samples": deque(maxlen=300),  # Last 300 samples for average
    "shares": deque(maxlen=100),  # Last 100 shares to prevent memory leak
    "start_time": None,
    # Track session values (to merge with persistent on save)
    "session_total_hashes": 0,
    "session_shares_submitted": 0,
    "session_shares_accepted": 0,
    "session_shares_rejected": 0,
    "initial_total_hashes": _persistent_stats["total_hashes"],
    "initial_shares_submitted": _persistent_stats["shares_submitted"],
    "initial_shares_accepted": _persistent_stats["shares_accepted"],
    "initial_shares_rejected": _persistent_stats["shares_rejected"],
}

# Auto-save interval (save every N updates)
AUTO_SAVE_INTERVAL = 10  # Save every 10 status updates
_save_counter = 0


def update_status(key: str, value):
    """Update a status value (thread-safe)"""
    global _save_counter

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
                    avg = sum(STATS["hash_rate_samples"]) / len(
                        STATS["hash_rate_samples"]
                    )
                    STATUS["average_hash_rate"] = avg
        if key == "total_hashes":
            with STATS_LOCK:
                # Calculate session increment (current - previous value)
                old_value = STATS["total_hashes"]
                STATS["total_hashes"] = value
                # Track increment since last save (only if value increased)
                if value > old_value:
                    STATS["session_total_hashes"] += value - old_value
                # If value decreased (shouldn't happen, but handle gracefully), reset to current
                elif value < old_value:
                    # This might happen on restart - use the new value as baseline
                    STATS["session_total_hashes"] = max(
                        0, value - STATS["initial_total_hashes"]
                    )
        if key == "shares_submitted":
            with STATS_LOCK:
                STATS["shares"].append(
                    {"timestamp": datetime.now().isoformat(), "accepted": value}
                )
                # Track increment since last save
                old_value = STATUS.get("shares_submitted", 0)
                if value > old_value:
                    STATS["session_shares_submitted"] += value - old_value

        # Auto-save periodically
        _save_counter += 1
        if _save_counter >= AUTO_SAVE_INTERVAL:
            _save_counter = 0
            _auto_save_statistics()

    # Emit selective updates to frontend to avoid flooding SocketIO
    try:
        if _socketio_instance is not None and key in (
            "hash_rate",
            "total_hashes",
            "last_hash",
            "pool_connected",
            "shares_submitted",
        ):
            try:
                _socketio_instance.emit("status", get_status())
            except Exception:
                # Ignore emission errors (e.g., socket not ready)
                pass
    except Exception:
        # Defensive: any error here should not break mining
        pass


def _auto_save_statistics():
    """Auto-save statistics to persistent storage."""
    with STATUS_LOCK:
        with STATS_LOCK:
            # Get current persistent stats
            persistent = load_statistics()

            # Calculate new values (initial persistent + session increments)
            new_total_hashes = (
                STATS["initial_total_hashes"] + STATS["session_total_hashes"]
            )
            new_shares_submitted = (
                STATS["initial_shares_submitted"] + STATS["session_shares_submitted"]
            )
            new_shares_accepted = STATUS.get("shares_accepted", 0)
            new_shares_rejected = STATUS.get("shares_rejected", 0)

            # Merge with existing persistent (in case file was updated externally)
            merged = merge_statistics(
                persistent["total_hashes"],
                persistent["peak_hash_rate"],
                persistent["shares_submitted"],
                persistent["shares_accepted"],
                persistent["shares_rejected"],
                STATS["session_total_hashes"],  # Only save session increment
                STATS[
                    "peak_hash_rate"
                ],  # Use current peak (max of persistent and current)
                STATS["session_shares_submitted"],  # Only save session increment
                new_shares_accepted
                - STATS["initial_shares_accepted"],  # Session increment
                new_shares_rejected
                - STATS["initial_shares_rejected"],  # Session increment
            )

            save_statistics(
                merged["total_hashes"],
                merged["peak_hash_rate"],
                merged["shares_submitted"],
                merged["shares_accepted"],
                merged["shares_rejected"],
            )

            # Update initial values to current persistent values
            STATS["initial_total_hashes"] = merged["total_hashes"]
            STATS["initial_shares_submitted"] = merged["shares_submitted"]
            STATS["initial_shares_accepted"] = merged["shares_accepted"]
            STATS["initial_shares_rejected"] = merged["shares_rejected"]

            # Reset session counters (values are now in persistent storage)
            STATS["session_total_hashes"] = 0
            STATS["session_shares_submitted"] = 0


def save_statistics_now():
    """Manually save statistics to persistent storage (call on shutdown)."""
    _auto_save_statistics()


def get_status() -> Dict:
    """Get current status (thread-safe copy)"""
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
    """Add a share to statistics"""
    with STATUS_LOCK:
        if accepted:
            STATUS["shares_accepted"] += 1
            with STATS_LOCK:
                STATS["session_shares_accepted"] += 1
        else:
            STATUS["shares_rejected"] += 1
            with STATS_LOCK:
                STATS["session_shares_rejected"] += 1
        STATUS["shares_submitted"] += 1
        STATUS["last_share_time"] = datetime.now().isoformat()
        update_status("shares_submitted", STATUS["shares_submitted"])


def set_socketio_instance(socketio_instance):
    """Set the SocketIO instance for immediate status updates"""
    global _socketio_instance
    _socketio_instance = socketio_instance


def update_pool_status(connected: bool, host: str = None, port: int = None):
    """Update pool connection status and immediately notify frontend via SocketIO"""
    with STATUS_LOCK:
        STATUS["pool_connected"] = connected
        if host:
            STATUS["pool_host"] = host
        if port:
            STATUS["pool_port"] = port

    # Immediately notify frontend via SocketIO if available
    if _socketio_instance is not None:
        try:
            _socketio_instance.emit("status", get_status())
        except Exception:
            # Ignore errors if SocketIO is not ready or clients are not connected
            pass
