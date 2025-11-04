import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict

from flask import Flask , render_template_string
from flask_socketio import SocketIO , emit

from ..core.state import MinerState


STATUS_LOCK = threading.Lock()
STATUS: Dict = {
    "running": False ,
    "current_height": 0 ,
    "best_difficulty": 0.0 ,
    "hash_rate": 0.0 ,
    "last_hash": None ,
    "uptime_seconds": 0 ,
    "start_time": None ,
    "errors": []
}


def update_status(key: str , value) :
    with STATUS_LOCK :
        STATUS[key] = value


def get_status() -> Dict :
    with STATUS_LOCK :
        return STATUS.copy()


app = Flask(__name__ , static_url_path = "/static")
app.config["SECRET_KEY"] = "btcsolo-miner-status"
socketio = SocketIO(app , cors_allowed_origins = "*")


@app.route("/")
def index() :
    return render_template_string(INDEX_HTML)


@socketio.on("connect")
def handle_connect() :
    emit("status" , get_status())


@socketio.on("get_status")
def handle_get_status() :
    emit("status" , get_status())


def broadcast_status() :
    while True :
        socketio.emit("status" , get_status())
        time.sleep(2)


def start_web_server(host: str = "0.0.0.0" , port: int = 5000) :
    logger = logging.getLogger("BtcSoloMinerGpu.web")
    logger.info("Starting web server on %s:%s" , host , port)
    update_status("start_time" , datetime.now().isoformat())
    update_status("running" , True)
    threading.Thread(target = broadcast_status , daemon = True).start()
    socketio.run(app , host = host , port = port , allow_unsafe_werkzeug = True)


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BtcSoloMinerGpu - Status Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .status-card h2 {
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #a8d5ff;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .status-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .status-label {
            font-size: 0.9em;
            color: #ccc;
            opacity: 0.8;
        }
        .running { color: #4ade80; }
        .stopped { color: #f87171; }
        .hash-display {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            word-break: break-all;
            margin-top: 10px;
        }
        .errors {
            background: rgba(255, 0, 0, 0.1);
            border-left: 4px solid #f87171;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .errors h3 {
            margin-bottom: 10px;
            color: #f87171;
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
        }
        .connected { background: #4ade80; color: #000; }
        .disconnected { background: #f87171; color: #fff; }
    </style>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span id="connectionText">Connecting...</span>
    </div>
    <div class="container">
        <h1>BtcSoloMinerGpu Status Dashboard</h1>
        <div class="status-grid">
            <div class="status-card">
                <h2>Status</h2>
                <div class="status-value running" id="runningStatus">-</div>
                <div class="status-label">Mining Status</div>
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
                <h2>Uptime</h2>
                <div class="status-value" id="uptime">-</div>
                <div class="status-label">Runtime</div>
            </div>
            <div class="status-card">
                <h2>Last Hash</h2>
                <div class="hash-display" id="lastHash">-</div>
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

        socket.on('connect', () => {
            document.getElementById('connectionStatus').className = 'connection-status connected';
            document.getElementById('connectionText').textContent = 'Connected';
        });

        socket.on('disconnect', () => {
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
            document.getElementById('connectionText').textContent = 'Disconnected';
        });

        socket.on('status', (data) => {
            if (!startTime && data.start_time) {
                startTime = new Date(data.start_time);
            }

            document.getElementById('runningStatus').textContent = data.running ? 'Running' : 'Stopped';
            document.getElementById('runningStatus').className = data.running ? 'status-value running' : 'status-value stopped';
            document.getElementById('currentHeight').textContent = data.current_height || 0;
            document.getElementById('bestDifficulty').textContent = data.best_difficulty ? data.best_difficulty.toFixed(2) : '0.00';
            document.getElementById('hashRate').textContent = data.hash_rate ? data.hash_rate.toFixed(2) : '0.00';
            
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
        });

        setInterval(() => {
            socket.emit('get_status');
        }, 3000);
    </script>
</body>
</html>
"""

