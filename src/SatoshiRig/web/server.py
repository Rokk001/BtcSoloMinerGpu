import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

from ..core.state import MinerState


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
    "errors": []
}

# Statistics tracking
STATS_LOCK = threading.Lock()
STATS = {
    "total_hashes": 0,
    "peak_hash_rate": 0.0,
    "hash_rate_samples": deque(maxlen=300),  # Last 300 samples for average
    "shares": [],
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
            status["shares"] = STATS["shares"][-10:]  # Last 10 shares
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


app = Flask(__name__, static_url_path="/static")
app.config["SECRET_KEY"] = "satoshirig-miner-status"
socketio = SocketIO(app, cors_allowed_origins="*")


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
    update_status("start_time", datetime.now().isoformat())
    update_status("running", True)
    with STATS_LOCK:
        STATS["start_time"] = datetime.now().isoformat()
    threading.Thread(target=broadcast_status, daemon=True).start()
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)


# Export functions for use by miner
__all__ = ["start_web_server", "update_status", "get_status", "add_share", "update_pool_status"]


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatoshiRig - Status Dashboard</title>
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
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            z-index: 1000;
        }
        .connected { background: var(--accent); color: #000; }
        .disconnected { background: var(--error); color: #fff; }
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
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .status-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; align-items: flex-start; }
        }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span id="connectionText">Connecting...</span>
    </div>
    <div class="container">
        <div class="header">
            <h1>SatoshiRig Status Dashboard</h1>
            <div class="controls">
                <button class="btn" onclick="toggleTheme()">🌓 Theme</button>
                <button class="btn" onclick="exportStats()">📥 Export</button>
                <button class="btn" onclick="toggleAutoRefresh()">
                    <span id="autoRefreshText">⏸️ Pause</span>
                </button>
            </div>
        </div>
        
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
            <div class="status-card">
                <h2>Job ID</h2>
                <div class="status-value" id="jobId" style="font-size: 1.2em;">-</div>
                <div class="status-label">Current Mining Job</div>
            </div>
            <div class="status-card">
                <h2>Last Hash</h2>
                <div class="hash-display" id="lastHash">-</div>
            </div>
            <div class="status-card">
                <h2>Wallet Address</h2>
                <div class="status-value" id="walletAddress" style="font-size: 1.2em; word-break: break-all;">-</div>
                <div class="status-label">
                    <a id="walletLink" href="#" target="_blank">View on Blockchain Explorer</a>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Hash Rate History</h2>
            <canvas id="hashRateChart"></canvas>
        </div>

        <div class="chart-container">
            <h2>Difficulty History</h2>
            <canvas id="difficultyChart"></canvas>
        </div>

        <div class="status-card">
            <h2>Share History</h2>
            <div class="share-history" id="shareHistory">
                <div style="color: var(--text-secondary);">No shares submitted yet</div>
            </div>
        </div>

        <div class="status-card">
            <h2>Statistics</h2>
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
        }

        function toggleTheme() {
            document.body.classList.toggle('light-theme');
            localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
        }

        function exportStats() {
            window.location.href = '/export';
        }

        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            document.getElementById('autoRefreshText').textContent = autoRefresh ? '⏸️ Pause' : '▶️ Resume';
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

        // Load saved theme
        if (localStorage.getItem('theme') === 'light') {
            document.body.classList.add('light-theme');
        }

        socket.on('connect', () => {
            document.getElementById('connectionStatus').className = 'connection-status connected';
            document.getElementById('connectionText').textContent = 'Connected';
            startRefresh();
        });

        socket.on('disconnect', () => {
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
            document.getElementById('connectionText').textContent = 'Disconnected';
        });

        socket.on('status', (data) => {
            if (!startTime && data.start_time) {
                startTime = new Date(data.start_time);
            }

            // Update status cards
            document.getElementById('runningStatus').textContent = data.running ? 'Running' : 'Stopped';
            document.getElementById('runningStatus').className = data.running ? 'status-value running' : 'status-value stopped';
            
            document.getElementById('currentHeight').textContent = data.current_height || 0;
            document.getElementById('bestDifficulty').textContent = data.best_difficulty ? data.best_difficulty.toFixed(2) : '0.00';
            document.getElementById('hashRate').textContent = data.hash_rate ? data.hash_rate.toFixed(2) : '0.00';
            document.getElementById('peakHashRate').textContent = data.peak_hash_rate ? data.peak_hash_rate.toFixed(2) : '0.00';
            document.getElementById('averageHashRate').textContent = data.average_hash_rate ? data.average_hash_rate.toFixed(2) : '0.00';
            document.getElementById('totalHashes').textContent = data.total_hashes ? data.total_hashes.toLocaleString() : '0';
            
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
            document.getElementById('statTotalHashes').textContent = (data.total_hashes || 0).toLocaleString();
            document.getElementById('statPeakHashRate').textContent = (data.peak_hash_rate || 0).toFixed(2) + ' H/s';
            document.getElementById('statAverageHashRate').textContent = (data.average_hash_rate || 0).toFixed(2) + ' H/s';
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
                const uptime = Math.floor((new Date() - startTime) / 1000);
                const hours = Math.floor(uptime / 3600);
                const minutes = Math.floor((uptime % 3600) / 60);
                const seconds = uptime % 60;
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
            
            // Update charts
            updateCharts(data);
        });

        startRefresh();
    </script>
</body>
</html>
"""

