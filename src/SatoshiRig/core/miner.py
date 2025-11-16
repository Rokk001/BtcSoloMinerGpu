import binascii
import hashlib
import logging
import random
import select
import socket
import threading
import time
from datetime import datetime
from typing import Any, Optional

import requests

from ..clients.pool_client import PoolClient
from .state import MinerState

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Log to stdout/stderr for Docker logs
)

try:
    from ..web import update_status, update_pool_status
except ImportError:

    def update_status(key, value):
        pass

    def update_pool_status(connected, host=None, port=None):
        pass


def now_time():
    return datetime.now().time()


# Import _vlog from utils instead of defining it here
from ..utils.logging_utils import _vlog


class Miner:
    def __init__(
        self,
        wallet_address: str,
        config: dict,
        pool_client: PoolClient,
        state: MinerState,
        logger: logging.Logger,
    ):
        logger.debug("Miner.__init__: START")
        self.wallet = wallet_address
        logger.debug(f"Miner.__init__: wallet={wallet_address}")
        self.cfg = config
        logger.debug(f"Miner.__init__: config loaded, keys={list(config.keys())}")
        self.pool = pool_client
        logger.debug("Miner.__init__: pool_client assigned")
        self.state = state
        logger.debug("Miner.__init__: state assigned")
        self.log = logger
        self.log.debug("Miner.__init__: logger assigned")
        self.total_hash_count = 0  # Persistent total hash count across loops
        self.log.debug("Miner.__init__: total_hash_count=0")
        self.gpu_miner = None
        self.log.debug("Miner.__init__: gpu_miner=None")
        self.gpu_nonce_counter = (
            0  # Counter for sequential nonce generation in GPU mode
        )
        self.log.debug("Miner.__init__: gpu_nonce_counter=0")
        self.cpu_nonce_counter = (
            0  # Counter for sequential nonce generation in CPU mode
        )
        self.log.debug("Miner.__init__: cpu_nonce_counter=0")
        self._config_lock = threading.Lock()  # Lock for thread-safe config access
        self.log.debug("Miner.__init__: _config_lock created")
        self._running = False  # Flag to prevent multiple calls to start()
        self.log.debug("Miner.__init__: _running=False")
        self._notification_thread = None  # Thread for listening to pool notifications
        self.log.debug("Miner.__init__: _notification_thread=None")
        self._notification_thread_running = False  # Flag for notification thread
        self.log.debug("Miner.__init__: _notification_thread_running=False")
        # Verbose logging is enabled if explicitly set OR if logger is at DEBUG level
        self._verbose_logging = self.cfg.get("logging", {}).get(
            "verbose", False
        ) or self.log.isEnabledFor(logging.DEBUG)
        self.log.debug(
            f"Miner.__init__: _verbose_logging={self._verbose_logging} (from config: {self.cfg.get('logging', {}).get('verbose', False)}, DEBUG enabled: {self.log.isEnabledFor(logging.DEBUG)})"
        )

        # Initialize GPU miner if configured
        self.log.debug("Miner.__init__: calling _initialize_gpu_miner()")
        self._initialize_gpu_miner()
        self.log.debug("Miner.__init__: END")

    def _initialize_gpu_miner(self):
        """Initialize GPU miner based on configuration"""
        # Thread-safe config access
        with self._config_lock:
            compute_backend = self.cfg.get("compute", {}).get("backend", "cpu")
            gpu_mining_enabled = self.cfg.get("compute", {}).get(
                "gpu_mining_enabled", False
            )
            gpu_device = self.cfg.get("compute", {}).get("gpu_device", 0)
            batch_size = self.cfg.get("compute", {}).get("batch_size", 256)
            max_workers = self.cfg.get("compute", {}).get("max_workers", 8)

        # Cleanup existing GPU miner if any
        if self.gpu_miner:
            try:
                self.gpu_miner.cleanup()
            except Exception as e:
                self.log.debug(f"Error cleaning up GPU miner: {e}")
            self.gpu_miner = None

        # Only initialize GPU miner if gpu_mining_enabled is True and backend is cuda/opencl
        if gpu_mining_enabled and compute_backend in ["cuda", "opencl"]:
            if compute_backend == "cuda":
                try:
                    self.log.debug(
                        f"Initializing CUDA GPU miner: device={gpu_device}, batch_size={batch_size}, max_workers={max_workers}"
                    )
                    from .gpu_compute import create_gpu_miner

                    self.gpu_miner = create_gpu_miner(
                        "cuda",
                        device_id=gpu_device,
                        logger=self.log,
                        batch_size=batch_size,
                        max_workers=max_workers,
                    )
                    self.log.info(
                        f"CUDA GPU miner initialized on device {gpu_device} (batch_size={batch_size}, max_workers={max_workers})"
                    )
                    self.log.debug(
                        "CUDA GPU miner initialization completed successfully"
                    )
                except Exception as e:
                    self.log.error(f"Failed to initialize CUDA miner: {e}")
                    self.log.warning("Falling back to CPU mining")
                    self.gpu_miner = None
            elif compute_backend == "opencl":
                try:
                    self.log.debug(
                        f"Initializing OpenCL GPU miner: device={gpu_device}, batch_size={batch_size}, max_workers={max_workers}"
                    )
                    from .gpu_compute import create_gpu_miner

                    self.gpu_miner = create_gpu_miner(
                        "opencl",
                        device_id=gpu_device,
                        logger=self.log,
                        batch_size=batch_size,
                        max_workers=max_workers,
                    )
                    self.log.info(
                        f"OpenCL GPU miner initialized on device {gpu_device} (batch_size={batch_size}, max_workers={max_workers})"
                    )
                    self.log.debug(
                        "OpenCL GPU miner initialization completed successfully"
                    )
                except Exception as e:
                    self.log.error(f"Failed to initialize OpenCL miner: {e}")
                    self.log.warning("Falling back to CPU mining")
                    self.gpu_miner = None
        else:
            self.log.debug(
                f"GPU mining not enabled (gpu_mining_enabled={gpu_mining_enabled}, backend={compute_backend})"
            )

    def update_config(self, new_config: dict):
        """Update miner configuration at runtime"""
        import copy

        # Deep merge config with deep copy to avoid reference issues
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

        # Store old backend to detect changes (thread-safe access)
        with self._config_lock:
            old_backend = self.cfg.get("compute", {}).get("backend", "cpu")
            old_gpu_enabled = self.cfg.get("compute", {}).get(
                "gpu_mining_enabled", False
            )

            # Update config (with lock protection)
            deep_merge(self.cfg, new_config)

            # Check if GPU configuration changed
            new_backend = self.cfg.get("compute", {}).get("backend", "cpu")
            new_gpu_enabled = self.cfg.get("compute", {}).get(
                "gpu_mining_enabled", False
            )

        # Reinitialize GPU miner if backend or gpu_mining_enabled changed (outside lock to avoid deadlock)
        if old_backend != new_backend or old_gpu_enabled != new_gpu_enabled:
            self.log.info(
                f"GPU configuration changed (backend: {old_backend} -> {new_backend}, gpu_enabled: {old_gpu_enabled} -> {new_gpu_enabled})"
            )
            self.log.debug(f"Reinitializing GPU miner with new configuration")
            # Reset nonce counter when GPU is toggled
            if not new_gpu_enabled:
                self.gpu_nonce_counter = 0
                self.log.debug("GPU mining disabled, resetting nonce counter")
            self._initialize_gpu_miner()

    def _get_current_block_height(self) -> int:
        net = self.cfg.get("network", {})
        source = (net.get("source") or "web").lower()
        max_retries = 3
        retry_delay = 2  # seconds

        if source == "local":
            # Bitcoin Core JSON-RPC: getblockcount
            payload = {
                "jsonrpc": "1.0",
                "id": "satoshirig",
                "method": "getblockcount",
                "params": [],
            }
            auth = None
            if net.get("rpc_user") or net.get("rpc_password"):
                auth = (net.get("rpc_user", ""), net.get("rpc_password", ""))

            last_error = None
            for attempt in range(max_retries):
                try:
                    r = requests.post(
                        net.get("rpc_url"),
                        json=payload,
                        auth=auth,
                        timeout=net.get("request_timeout_secs", 15),
                    )
                    r.raise_for_status()
                    data = r.json()
                    return int(data["result"])  # returns block count (height)
                except (
                    requests.RequestException,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        self.log.warning(
                            f"Failed to get block height from RPC (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        self.log.error(
                            f"Failed to get block height from RPC after {max_retries} attempts: {e}"
                        )
                        raise
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to get block height: unknown error")
        else:
            last_error = None
            for attempt in range(max_retries):
                try:
                    r = requests.get(
                        net["latest_block_url"], timeout=net.get("request_timeout_secs")
                    )
                    r.raise_for_status()
                    return int(r.json()["height"])
                except (
                    requests.RequestException,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        self.log.warning(
                            f"Failed to get block height from web API (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        self.log.error(
                            f"Failed to get block height from web API after {max_retries} attempts: {e}"
                        )
                        raise
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to get block height: unknown error")

    def _hex_to_little_endian(self, hex_str: str, expected_length: int) -> str:
        """
        Convert hex string to little-endian format for Bitcoin block headers.
        Bitcoin uses little-endian byte order for all fields in the block header.

        Args:
            hex_str: Hex string (may be in big-endian or little-endian)
            expected_length: Expected length in hex characters (must be even)

        Returns:
            Hex string in little-endian format
        """
        if not hex_str:
            return "0" * expected_length

        hex_str = hex_str.strip()

        # Pad to expected length
        if len(hex_str) < expected_length:
            hex_str = hex_str.zfill(expected_length)
        elif len(hex_str) > expected_length:
            hex_str = hex_str[:expected_length]

        # Convert to little-endian: reverse byte order (not bit order)
        # Each byte is 2 hex characters
        if len(hex_str) % 2 != 0:
            raise ValueError(f"Hex string length must be even: {len(hex_str)}")

        # Reverse byte order: "ABCDEF" -> "EFCDAB" (each pair is a byte)
        bytes_list = [hex_str[i : i + 2] for i in range(0, len(hex_str), 2)]
        return "".join(reversed(bytes_list))

    def _int_to_little_endian_hex(self, value: int, byte_length: int) -> str:
        """
        Convert integer to little-endian hex string.

        Args:
            value: Integer value
            byte_length: Number of bytes (hex length will be byte_length * 2)

        Returns:
            Hex string in little-endian format
        """
        # Convert to hex and pad to correct length
        hex_str = f"{value:0{byte_length * 2}x}"
        # Reverse byte order for little-endian
        bytes_list = [hex_str[i : i + 2] for i in range(0, len(hex_str), 2)]
        return "".join(reversed(bytes_list))

    def _build_block_header(self, prev_hash, merkle_root, ntime, nbits, nonce_hex):
        # Validate all required fields
        with self.state._lock:
            version = self.state.version

        if (
            not version
            or not prev_hash
            or not merkle_root
            or not ntime
            or not nbits
            or not nonce_hex
        ):
            missing = []
            if not version:
                missing.append("version")
            if not prev_hash:
                missing.append("prev_hash")
            if not merkle_root:
                missing.append("merkle_root")
            if not ntime:
                missing.append("ntime")
            if not nbits:
                missing.append("nbits")
            if not nonce_hex:
                missing.append("nonce_hex")
            self.log.error(
                f"Missing required fields for block header: {', '.join(missing)}"
            )
            raise RuntimeError(
                f"Missing required fields for block header: {', '.join(missing)}"
            )

        # Bitcoin block header is exactly 80 bytes (160 hex chars):
        # - version: 4 bytes (8 hex chars) - little-endian
        # - prev_hash: 32 bytes (64 hex chars) - little-endian
        # - merkle_root: 32 bytes (64 hex chars) - little-endian
        # - ntime: 4 bytes (8 hex chars) - little-endian
        # - nbits: 4 bytes (8 hex chars) - little-endian
        # - nonce: 4 bytes (8 hex chars) - little-endian
        # Total: 80 bytes = 160 hex chars

        # Convert all fields to little-endian format (Bitcoin standard)
        # Pool may send fields in big-endian, so we need to convert
        version = self._hex_to_little_endian(version.strip() if version else "", 8)
        prev_hash = self._hex_to_little_endian(
            prev_hash.strip() if prev_hash else "", 64
        )
        merkle_root = self._hex_to_little_endian(
            merkle_root.strip() if merkle_root else "", 64
        )
        ntime = self._hex_to_little_endian(ntime.strip() if ntime else "", 8)
        nbits = self._hex_to_little_endian(nbits.strip() if nbits else "", 8)
        nonce_hex = self._hex_to_little_endian(
            nonce_hex.strip() if nonce_hex else "", 8
        )

        # Log field lengths for debugging
        self.log.debug(
            f"Block header field lengths: version={len(version)}, prev_hash={len(prev_hash)}, merkle_root={len(merkle_root)}, ntime={len(ntime)}, nbits={len(nbits)}, nonce_hex={len(nonce_hex)}"
        )

        # Final validation
        if (
            len(version) != 8
            or len(prev_hash) != 64
            or len(merkle_root) != 64
            or len(ntime) != 8
            or len(nbits) != 8
            or len(nonce_hex) != 8
        ):
            self.log.error(
                f"Invalid block header field lengths after conversion: version={len(version)}, prev_hash={len(prev_hash)}, merkle_root={len(merkle_root)}, ntime={len(ntime)}, nbits={len(nbits)}, nonce_hex={len(nonce_hex)}"
            )
            raise RuntimeError("Invalid block header field lengths after conversion")

        block_header = version + prev_hash + merkle_root + ntime + nbits + nonce_hex

        # Validate total length
        if len(block_header) != 160:
            self.log.error(
                f"Invalid block header total length: {len(block_header)} (expected 160 hex chars = 80 bytes)"
            )
            raise RuntimeError(f"Invalid block header length: {len(block_header)}")

        self.log.debug(f"Block header built successfully: length={len(block_header)}")
        return block_header

    def start(self):
        # Prevent multiple calls to start()
        if self._running:
            self.log.warning(
                "Miner.start() called multiple times, ignoring duplicate call"
            )
            return
        self._running = True

        try:
            # Check if pool is already connected (e.g., from connect_to_pool_only())
            with self.pool._socket_lock:
                pool_already_connected = (
                    self.pool.sock is not None and self.pool.sock.fileno() != -1
                )

            with self.state._lock:
                has_subscription = (
                    self.state.extranonce1 is not None
                    and self.state.extranonce2_size is not None
                )

            # If pool is connected but subscription is not yet available, wait a bit
            # This handles the case where connect_to_pool_only() is still running
            if pool_already_connected and not has_subscription:
                self.log.info(
                    "Pool connected but subscription not yet available, waiting for connect_to_pool_only() to complete..."
                )
                # Wait up to 5 seconds for subscription to become available
                for wait_iteration in range(10):
                    time.sleep(0.5)
                    with self.state._lock:
                        has_subscription = (
                            self.state.extranonce1 is not None
                            and self.state.extranonce2_size is not None
                        )
                    if has_subscription:
                        self.log.info(
                            "Subscription now available, proceeding with mining startup"
                        )
                        break
                    # Check if socket is still connected
                    with self.pool._socket_lock:
                        if self.pool.sock is None or self.pool.sock.fileno() == -1:
                            self.log.warning(
                                "Socket disconnected while waiting for subscription, will reconnect"
                            )
                            pool_already_connected = False
                            break

            if pool_already_connected and has_subscription:
                # Reuse existing connection
                self.log.info(
                    "Pool already connected and subscribed, reusing existing connection"
                )
                self.log.debug(
                    f"Existing extranonce1: {self.state.extranonce1}, extranonce2_size: {self.state.extranonce2_size}"
                )
                update_pool_status(True, self.pool.host, self.pool.port)
                # Just authorize if needed (authorize is idempotent)
                try:
                    self.pool.authorize(self.wallet)
                except Exception as auth_error:
                    self.log.warning(
                        f"Authorization failed (may already be authorized): {auth_error}"
                    )
                self.log.info("Authorized, waiting for mining notification...")
            else:
                # Need to connect and subscribe
                self.log.info(
                    "Connecting to pool %s:%s...", self.pool.host, self.pool.port
                )
                self.log.debug(
                    f"Pool connection parameters: host={self.pool.host}, port={self.pool.port}"
                )
                # Close existing connection if any (in case it's in a bad state)
                with self.pool._socket_lock:
                    if self.pool.sock:
                        try:
                            self.pool.sock.close()
                            self.pool.sock = None
                        except Exception:
                            pass
                self.pool.connect()
                self.log.info("Connected to pool, subscribing...")
                self.log.debug("Sending subscription request to pool")
                update_pool_status(True, self.pool.host, self.pool.port)
                sub_details, extranonce1, extranonce2_size = self.pool.subscribe()
                self.log.debug(
                    f"Subscription response: extranonce1={extranonce1}, extranonce2_size={extranonce2_size}"
                )
                with self.state._lock:
                    self.state.subscription_details = sub_details
                    self.state.extranonce1 = extranonce1
                    self.state.extranonce2_size = extranonce2_size
                self.log.info("Subscribed to pool, authorizing...")
                self.log.debug(f"Authorizing with wallet: {self.wallet[:10]}...")
                self.pool.authorize(self.wallet)
                self.log.info(
                    "Authorized, starting notification listener and mining loop..."
                )

            # Start background thread to continuously listen for pool notifications
            self._notification_thread_running = True
            self._notification_thread = threading.Thread(
                target=self._listen_for_notifications, daemon=True
            )
            self._notification_thread.start()
            self.log.debug("Notification listener thread started")

            # Try to get initial notification with short timeout
            # If we don't get one, the mining loop will wait for it
            initial_notification_received = False
            try:
                self.log.debug(
                    "Waiting for initial mining.notify message from pool (5s timeout)"
                )
                # Temporarily set shorter timeout for initial notification
                with self.pool._socket_lock:
                    if self.pool.sock:
                        original_timeout = self.pool.sock.gettimeout()
                        self.pool.sock.settimeout(5.0)  # 5 second timeout

                responses = self.pool.read_notify()

                # Restore original timeout
                with self.pool._socket_lock:
                    if self.pool.sock:
                        self.pool.sock.settimeout(
                            original_timeout if original_timeout else self.pool.timeout
                        )

                if (
                    responses
                    and len(responses) > 0
                    and "params" in responses[0]
                    and len(responses[0]["params"]) >= 9
                ):
                    with self.state._lock:
                        (
                            self.state.job_id,
                            self.state.prev_hash,
                            self.state.coinbase_part1,
                            self.state.coinbase_part2,
                            self.state.merkle_branch,
                            self.state.version,
                            self.state.nbits,
                            self.state.ntime,
                            self.state.clean_jobs,
                        ) = responses[0]["params"]
                        self.state.updated_prev_hash = self.state.prev_hash
                    self.log.info(
                        f"Received initial mining notification: job_id={self.state.job_id}"
                    )
                    update_status("job_id", self.state.job_id)
                    initial_notification_received = True
            except (socket.timeout, ConnectionError) as e:
                self.log.warning(
                    f"Timeout waiting for initial notification: {e}. Mining loop will wait for notification."
                )
                # Restore original timeout
                with self.pool._socket_lock:
                    if self.pool.sock:
                        try:
                            self.pool.sock.settimeout(self.pool.timeout)
                        except:
                            pass
            except Exception as e:
                self.log.warning(
                    f"Error reading initial notification: {e}. Mining loop will wait for notification."
                )

            # Check if we have valid state data before starting mining loop
            if not initial_notification_received:
                with self.state._lock:
                    has_valid_state = (
                        self.state.nbits
                        and self.state.prev_hash
                        and self.state.extranonce1
                        and self.state.extranonce2_size
                    )
                if not has_valid_state:
                    self.log.info(
                        "No valid mining state yet, waiting for notification from pool..."
                    )
                    # Wait a bit for notification thread to receive initial notification
                    for _ in range(10):  # Wait up to 10 seconds
                        time.sleep(1)
                        with self.state._lock:
                            if self.state.nbits and self.state.prev_hash:
                                self.log.info(
                                    "Received mining state from notification thread"
                                )
                                break
                    else:
                        self.log.warning(
                            "No mining state received after waiting, starting mining loop anyway"
                        )

            self.log.info("Starting mining loop...")
            self.log.debug("Entering main mining loop")
            return self._mine_loop()
        except Exception as e:
            self.log.error(f"Failed to start mining: {e}", exc_info=True)
            self._running = False
            update_status("running", False)
            update_pool_status(False)
            raise

    def _listen_for_notifications(self):
        """Continuously listen for pool notifications in background thread"""
        self.log.info("Notification listener thread started")
        iteration_count = 0
        while self._notification_thread_running and self._running:
            try:
                iteration_count += 1
                with self.pool._socket_lock:
                    if self.pool.sock is None:
                        self.log.warning(
                            f"Notification thread: Socket is None (iteration {iteration_count}), waiting 1s..."
                        )
                        time.sleep(1)
                        continue
                    if self.pool.sock.fileno() == -1:
                        self.log.warning(
                            f"Notification thread: Socket fileno is -1 (iteration {iteration_count}), waiting 1s..."
                        )
                        time.sleep(1)
                        continue

                    # Use select to check if data is available (non-blocking check)
                    readable, _, _ = select.select([self.pool.sock], [], [], 1.0)
                    if not readable:
                        # No data available, continue loop (log every 10 iterations to avoid spam)
                        if iteration_count % 10 == 0:
                            self.log.debug(
                                f"Notification thread: No data available (iteration {iteration_count})"
                            )
                        continue

                    # Data available, set short timeout for reading
                    original_timeout = self.pool.sock.gettimeout()
                    self.pool.sock.settimeout(
                        2.0
                    )  # 2 second timeout for non-blocking read
                    self.log.debug(
                        f"Notification thread: Data available, reading (iteration {iteration_count})"
                    )

                # Read notification (outside lock to avoid blocking)
                try:
                    responses = self.pool.read_notify()
                    self.log.debug(
                        f"Notification thread: read_notify() returned {len(responses) if responses else 0} responses"
                    )

                    # Restore original timeout (inside lock)
                    with self.pool._socket_lock:
                        if self.pool.sock:
                            self.pool.sock.settimeout(
                                original_timeout
                                if original_timeout
                                else self.pool.timeout
                            )

                    if responses and len(responses) > 0:
                        for idx, response in enumerate(responses):
                            self.log.debug(
                                f"Notification thread: Processing response {idx+1}/{len(responses)}, keys={list(response.keys()) if isinstance(response, dict) else 'not a dict'}"
                            )
                            if "params" in response and len(response["params"]) >= 9:
                                # Process mining.notify message
                                with self.state._lock:
                                    (
                                        self.state.job_id,
                                        self.state.prev_hash,
                                        self.state.coinbase_part1,
                                        self.state.coinbase_part2,
                                        self.state.merkle_branch,
                                        self.state.version,
                                        self.state.nbits,
                                        self.state.ntime,
                                        self.state.clean_jobs,
                                    ) = response["params"]
                                    self.state.updated_prev_hash = self.state.prev_hash

                                self.log.info(
                                    f"Notification thread: Received mining.notify - job_id={self.state.job_id}, prev_hash={self.state.prev_hash[:16]}..., nbits={self.state.nbits}, ntime={self.state.ntime}"
                                )
                                self.log.debug(
                                    f"Notification thread: State updated - version={self.state.version}, clean_jobs={self.state.clean_jobs}, extranonce1={self.state.extranonce1[:8] if self.state.extranonce1 else None}..."
                                )
                                update_status("job_id", self.state.job_id)
                            else:
                                self.log.warning(
                                    f"Notification thread: Response {idx+1} is not a valid mining.notify (has 'params': {'params' in response}, params length: {len(response.get('params', [])) if 'params' in response else 0})"
                                )
                except (socket.timeout, ConnectionError) as e:
                    # Timeout or connection error - this is OK, just continue listening
                    self.log.warning(
                        f"Notification thread: read_notify timeout/error (iteration {iteration_count}): {e}"
                    )
                    # Restore original timeout
                    with self.pool._socket_lock:
                        if self.pool.sock:
                            try:
                                self.pool.sock.settimeout(self.pool.timeout)
                            except:
                                pass
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    self.log.error(
                        f"Notification thread: Error reading notification (iteration {iteration_count}): {e}",
                        exc_info=True,
                    )
                    # Restore original timeout
                    with self.pool._socket_lock:
                        if self.pool.sock:
                            try:
                                self.pool.sock.settimeout(self.pool.timeout)
                            except:
                                pass
                    time.sleep(1)
                    continue

            except Exception as e:
                self.log.error(
                    f"Notification thread: Unexpected error (iteration {iteration_count}): {e}",
                    exc_info=True,
                )
                time.sleep(1)

        self.log.info("Notification listener thread stopped")

    def connect_to_pool_only(self):
        """Connect to pool, subscribe, and authorize without starting mining loop"""
        try:
            # Check if miner is already running - if so, don't interfere with existing connection
            if self._running:
                self.log.debug(
                    "Miner is already running, pool connection is already established - skipping connect_to_pool_only"
                )
                # Check if socket is still connected
                with self.pool._socket_lock:
                    if self.pool.sock:
                        try:
                            # Quick check if socket is still alive
                            self.pool.sock.settimeout(0.1)
                            # Try to peek at socket (non-blocking)
                            readable, _, _ = select.select(
                                [self.pool.sock], [], [], 0.1
                            )
                            if readable or self.pool.sock.fileno() != -1:
                                self.log.debug(
                                    "Pool connection is already established and active"
                                )
                                update_pool_status(True, self.pool.host, self.pool.port)
                                return True
                        except Exception:
                            # Socket might be dead, continue to reconnect
                            pass
                # If we get here, socket might be dead, but miner is running
                # Don't interfere - let the miner handle reconnection
                self.log.warning(
                    "Miner is running but socket check failed - not reconnecting to avoid interference"
                )
                return False

            self.log.info("Connecting to pool %s:%s...", self.pool.host, self.pool.port)
            self.log.debug(
                f"Pool connection parameters: host={self.pool.host}, port={self.pool.port}"
            )

            # Close existing connection if any (only if miner is not running)
            with self.pool._socket_lock:
                if self.pool.sock:
                    try:
                        self.pool.sock.close()
                        self.pool.sock = None
                    except Exception:
                        pass

            self.pool.connect()
            self.log.info("Connected to pool, subscribing...")
            self.log.debug("Sending subscription request to pool")
            update_pool_status(True, self.pool.host, self.pool.port)

            sub_details, extranonce1, extranonce2_size = self.pool.subscribe()
            self.log.debug(
                f"Subscription response: extranonce1={extranonce1}, extranonce2_size={extranonce2_size}"
            )
            with self.state._lock:
                self.state.subscription_details = sub_details
                self.state.extranonce1 = extranonce1
                self.state.extranonce2_size = extranonce2_size

            self.log.info("Subscribed to pool, authorizing...")
            self.log.debug(f"Authorizing with wallet: {self.wallet[:10]}...")
            self.pool.authorize(self.wallet)
            self.log.info(
                "Authorized with pool (connection established, mining not started)"
            )

            return True
        except Exception as e:
            self.log.error(f"Failed to connect to pool: {e}", exc_info=True)
            update_pool_status(False)
            return False

    def _mine_loop(self):
        """Main mining loop - wrapped in retry logic to handle transient errors"""
        restart_delay = self.cfg.get("miner", {}).get("restart_delay_secs", 2)
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                return self._mine_loop_internal()
            except (
                ConnectionError,
                RuntimeError,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                retry_count += 1
                error_msg = str(e).lower()
                is_connection_error = (
                    "connection" in error_msg
                    or "socket" in error_msg
                    or "not connected" in error_msg
                )

                if is_connection_error:
                    self.log.warning(
                        f"Pool connection error (attempt {retry_count}/{max_retries}): {e}. "
                        f"Reconnecting in {restart_delay}s..."
                    )
                    # Try to reconnect
                    try:
                        if self.pool.sock:
                            try:
                                self.pool.close()
                            except (Exception, OSError, ConnectionError) as close_error:
                                # Ignore errors when closing socket during reconnection
                                self.log.debug(
                                    f"Error closing socket during reconnection: {close_error}"
                                )
                                pass
                        self.pool.connect()
                        update_pool_status(True, self.pool.host, self.pool.port)
                        # Re-subscribe and re-authorize
                        sub_details, extranonce1, extranonce2_size = (
                            self.pool.subscribe()
                        )
                        with self.state._lock:
                            self.state.subscription_details = sub_details
                            self.state.extranonce1 = extranonce1
                            self.state.extranonce2_size = extranonce2_size
                        self.pool.authorize(self.wallet)
                        self.log.info("Successfully reconnected to pool")
                    except Exception as reconnect_error:
                        self.log.error(
                            f"Failed to reconnect to pool: {reconnect_error}"
                        )
                        update_pool_status(False)
                else:
                    self.log.warning(
                        f"Mining loop error (attempt {retry_count}/{max_retries}): {e}. "
                        f"Retrying in {restart_delay}s..."
                    )

                time.sleep(restart_delay)
                # Check if we should continue
                with self.state._lock:
                    if self.state.shutdown_flag:
                        self.log.info("Shutdown requested, stopping mining loop")
                        update_status("running", False)
                        update_pool_status(False)
                        self._running = False
                        self._notification_thread_running = (
                            False  # Stop notification thread
                        )
                        return
            except Exception as e:
                self.log.error(f"Unexpected error in mining loop: {e}", exc_info=True)
                retry_count += 1
                if retry_count >= max_retries:
                    self.log.error("Max retries reached, stopping mining loop")
                    update_status("running", False)
                    update_pool_status(False)
                    self._running = False
                    self._notification_thread_running = (
                        False  # Stop notification thread
                    )
                    raise
                time.sleep(restart_delay)

        # If we get here, all retries failed
        self.log.error("Mining loop failed after all retries")
        update_status("running", False)
        update_pool_status(False)
        self._running = False
        self._notification_thread_running = False  # Stop notification thread

    def _mine_loop_internal(self):
        """Internal mining loop implementation"""
        with self.state._lock:
            if self.state.height_to_best_difficulty.get(-1) == -1:
                restart_delay = self.cfg.get("miner", {}).get("restart_delay_secs", 2)
                if restart_delay:
                    time.sleep(restart_delay)

        # Validate nbits before target calculation to prevent crashes
        with self.state._lock:
            nbits = self.state.nbits
            extranonce2_size = self.state.extranonce2_size

        if not nbits or len(nbits) < 2:
            self.log.error(
                f"Invalid nbits: {nbits}, waiting for valid data from pool..."
            )
            raise RuntimeError(f"Invalid nbits value: {nbits}")

        try:
            exponent = int(nbits[:2], 16)
            if exponent < 3 or exponent > 255:
                self.log.error(f"Invalid nbits exponent: {exponent}")
                raise RuntimeError(f"Invalid nbits exponent: {exponent}")
            target = (nbits[2:] + "00" * (exponent - 3)).zfill(64)
        except (ValueError, IndexError) as e:
            self.log.error(f"Failed to parse nbits '{nbits}': {e}")
            raise RuntimeError(f"Failed to parse nbits: {e}")

        # Calculate target difficulty from nbits
        # Difficulty = 0x00000000FFFF0000000000000000000000000000000000000000000000000000 / target
        try:
            target_int = int(target, 16)
        except ValueError as e:
            self.log.error(f"Failed to convert target to int: {e}")
            raise RuntimeError(f"Invalid target value: {target}")

        # Bitcoin reference difficulty: 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        reference_diff = int(
            "00000000FFFF0000000000000000000000000000000000000000000000000000", 16
        )
        if target_int > 0:
            target_difficulty = reference_diff / target_int
            update_status("target_difficulty", target_difficulty)

        if not extranonce2_size or extranonce2_size <= 0:
            self.log.error(
                f"Invalid extranonce2_size: {extranonce2_size}, waiting for valid data from pool..."
            )
            raise RuntimeError(f"Invalid extranonce2_size: {extranonce2_size}")

        # Initialize extranonce2 counter if not exists (sequential generation)
        if not hasattr(self, "extranonce2_counter"):
            self.extranonce2_counter = 0

        # Generate extranonce2 sequentially (not randomly) for better coverage
        extranonce2 = hex(self.extranonce2_counter)[2:].zfill(2 * extranonce2_size)
        self.extranonce2_counter = (self.extranonce2_counter + 1) % (
            2 ** (8 * extranonce2_size)
        )
        self.log.debug(f"Generated extranonce2 (sequential): {extranonce2}")
        with self.state._lock:
            self.state.extranonce2 = extranonce2

        # Get current height with timeout handling to prevent blocking
        try:
            current_height = self._get_current_block_height()
        except Exception as e:
            self.log.error(
                f"Failed to get current block height: {e}, using previous height"
            )
            # Use previous height as fallback
            with self.state._lock:
                current_height = (
                    self.state.local_height if self.state.local_height > 0 else 0
                )

        with self.state._lock:
            self.state.height_to_best_difficulty[current_height + 1] = 0
            self.state.local_height = current_height

        self.log.info("Mining block height %s", current_height + 1)
        self.log.debug(
            f"Mining parameters: height={current_height + 1}, target_difficulty={target_difficulty if target_int > 0 else 'N/A'}, target={target[:16]}..."
        )
        update_status("current_height", current_height + 1)

        prefix_zeros = "0" * self.cfg.get("miner", {}).get("hash_log_prefix_zeros", 7)
        self.log.debug(
            f"Hash log prefix zeros: {len(prefix_zeros)} (will log hashes starting with {prefix_zeros})"
        )
        hash_count = 0
        start_time = time.time()

        # Log mining configuration
        with self._config_lock:
            cpu_mining_enabled = self.cfg.get("compute", {}).get(
                "cpu_mining_enabled", True
            )
            gpu_mining_enabled = self.cfg.get("compute", {}).get(
                "gpu_mining_enabled", False
            )
        self.log.info(
            f"Mining configuration: CPU enabled={cpu_mining_enabled}, GPU enabled={gpu_mining_enabled}, GPU miner available={self.gpu_miner is not None}"
        )

        # Log initial state values
        with self.state._lock:
            self.log.info(
                f"Initial mining state: nbits={self.state.nbits}, prev_hash={self.state.prev_hash[:16] if self.state.prev_hash else None}..., ntime={self.state.ntime}, extranonce1={self.state.extranonce1[:8] if self.state.extranonce1 else None}..., extranonce2_size={self.state.extranonce2_size}"
            )
            self.log.debug(
                f"Initial state details: job_id={self.state.job_id}, version={self.state.version}, coinbase_part1 length={len(self.state.coinbase_part1) if self.state.coinbase_part1 else 0}, coinbase_part2 length={len(self.state.coinbase_part2) if self.state.coinbase_part2 else 0}, merkle_branch length={len(self.state.merkle_branch) if self.state.merkle_branch else 0}"
            )

        self.log.info("Starting hash computation loop")

        while True:
            # CRITICAL: Wrap entire loop iteration in try-except to ensure loop always progresses
            # Store initial hash_count to detect if it was incremented during this iteration
            _vlog(
                self.log,
                self._verbose_logging,
                f"LOOP_START: hash_count={hash_count}, initial_hash_count={hash_count}",
            )
            initial_hash_count = hash_count
            _vlog(
                self.log,
                self._verbose_logging,
                f"LOOP: initial_hash_count set to {initial_hash_count}",
            )
            try:
                _vlog(self.log, self._verbose_logging, "LOOP: entering try block")
                _vlog(self.log, self._verbose_logging, "LOOP: acquiring state._lock")
                with self.state._lock:
                    _vlog(self.log, self._verbose_logging, "LOOP: inside state._lock")
                    shutdown_flag = self.state.shutdown_flag
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: shutdown_flag={shutdown_flag}",
                    )
                    prev_hash = self.state.prev_hash
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: prev_hash={'present' if prev_hash else 'None'} (length={len(prev_hash) if prev_hash else 0})",
                    )
                    updated_prev_hash = self.state.updated_prev_hash
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: updated_prev_hash={'present' if updated_prev_hash else 'None'} (length={len(updated_prev_hash) if updated_prev_hash else 0})",
                    )
                _vlog(self.log, self._verbose_logging, "LOOP: released state._lock")

                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking shutdown_flag={shutdown_flag}",
                )
                if shutdown_flag:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: shutdown_flag is True, shutting down",
                    )
                    update_status("running", False)
                    update_pool_status(False)
                    self._running = False
                    # Cleanup GPU miner before shutdown
                    if self.gpu_miner:
                        try:
                            self.gpu_miner.cleanup()
                            self.log.debug("GPU miner cleaned up on shutdown")
                        except Exception as e:
                            self.log.debug(
                                f"Error cleaning up GPU miner on shutdown: {e}"
                            )
                    break

                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: comparing prev_hash != updated_prev_hash: {prev_hash != updated_prev_hash}",
                )
                if prev_hash != updated_prev_hash:
                    _vlog(self.log, self._verbose_logging, "LOOP: new block detected")
                    self.log.info("New block detected: %s", prev_hash)
                    with self.state._lock:
                        best_diff = self.state.height_to_best_difficulty.get(
                            current_height + 1, 0
                        )
                    self.log.info(
                        "Best difficulty for height %s was %s",
                        current_height + 1,
                        best_diff,
                    )

                    with self.state._lock:
                        self.state.updated_prev_hash = prev_hash
                        self.state.height_to_best_difficulty[-1] = -1

                    # Update job_id from new block with timeout handling
                    try:
                        responses = self.pool.read_notify()
                        if (
                            responses
                            and len(responses) > 0
                            and "params" in responses[0]
                            and len(responses[0]["params"]) > 0
                        ):
                            # Update ALL fields from new block notification, not just job_id
                            with self.state._lock:
                                if len(responses[0]["params"]) >= 9:
                                    (
                                        self.state.job_id,
                                        self.state.prev_hash,
                                        self.state.coinbase_part1,
                                        self.state.coinbase_part2,
                                        self.state.merkle_branch,
                                        self.state.version,
                                        self.state.nbits,
                                        self.state.ntime,
                                        self.state.clean_jobs,
                                    ) = responses[0]["params"]
                                    self.state.updated_prev_hash = self.state.prev_hash
                                else:
                                    # Fallback: only update job_id if full params not available
                                    self.state.job_id = responses[0]["params"][0]
                        update_status("job_id", self.state.job_id)
                    except Exception as e:
                        self.log.error(
                            f"Failed to read notify for new block: {e}, continuing with existing job_id"
                        )

                    # Update current_height for new block (with timeout handling)
                    try:
                        current_height = self._get_current_block_height()
                        with self.state._lock:
                            self.state.height_to_best_difficulty[current_height + 1] = 0
                            self.state.local_height = current_height
                    except Exception as e:
                        self.log.error(
                            f"Failed to get current block height: {e}, using previous height"
                        )
                        # Use previous height as fallback
                        with self.state._lock:
                            current_height = (
                                self.state.local_height
                                if self.state.local_height > 0
                                else 0
                            )
                        # current_height is now always defined
                        with self.state._lock:
                            self.state.height_to_best_difficulty[current_height + 1] = 0

                    self.log.info("Mining block height %s", current_height + 1)
                    update_status("current_height", current_height + 1)

                    # Reset extranonce2 counter for new block (#67)
                    if hasattr(self, "extranonce2_counter"):
                        self.extranonce2_counter = 0
                        self.log.debug("Reset extranonce2_counter for new block")

                    # Handle clean_jobs flag (#57)
                    with self.state._lock:
                        clean_jobs = self.state.clean_jobs
                    if clean_jobs:
                        self.log.info(
                            "clean_jobs flag set - resetting mining state for new block"
                        )
                        # Reset nonce counters when clean_jobs is True
                        self.cpu_nonce_counter = 0
                        self.gpu_nonce_counter = 0
                        if hasattr(self, "extranonce2_counter"):
                            self.extranonce2_counter = 0

                    # Continue loop instead of recursive call to avoid stack overflow
                    # This will trigger recalculation of merkle_root, target, etc. in the loop
                    continue

                # Check CPU/GPU mining flags (thread-safe access)
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: checking CPU/GPU mining flags",
                )
                _vlog(self.log, self._verbose_logging, "LOOP: acquiring _config_lock")
                with self._config_lock:
                    _vlog(self.log, self._verbose_logging, "LOOP: inside _config_lock")
                    cpu_mining_enabled = self.cfg.get("compute", {}).get(
                        "cpu_mining_enabled", True
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: cpu_mining_enabled={cpu_mining_enabled}",
                    )
                    gpu_mining_enabled = self.cfg.get("compute", {}).get(
                        "gpu_mining_enabled", False
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: gpu_mining_enabled={gpu_mining_enabled}",
                    )
                    gpu_utilization_percent = self.cfg.get("compute", {}).get(
                        "gpu_utilization_percent", 100
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: gpu_utilization_percent={gpu_utilization_percent}",
                    )
                _vlog(self.log, self._verbose_logging, "LOOP: released _config_lock")

                # Recalculate target, target_int, target_difficulty dynamically (#44, #50-55)
                _vlog(self.log, self._verbose_logging, "LOOP: recalculating target")
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: acquiring state._lock for nbits",
                )
                with self.state._lock:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: inside state._lock for nbits",
                    )
                    nbits = self.state.nbits
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: nbits={'present' if nbits else 'None'} (length={len(nbits) if nbits else 0})",
                    )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: released state._lock for nbits",
                )

                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking nbits validity: not nbits={not nbits}, len(nbits)={len(nbits) if nbits else 0}",
                )
                if not nbits or len(nbits) < 2:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: invalid nbits, incrementing hash_count from {hash_count}",
                    )
                    self.log.warning(
                        f"Invalid nbits in loop: {nbits}, skipping iteration"
                    )
                    hash_count += 1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hash_count incremented to {hash_count}, continuing",
                    )
                    continue

                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: entering try block for target calculation",
                )
                try:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: converting nbits[:2]={nbits[:2] if nbits else 'None'} to int",
                    )
                    exponent = int(nbits[:2], 16)
                    _vlog(self.log, self._verbose_logging, f"LOOP: exponent={exponent}")
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: checking exponent validity: {exponent < 3 or exponent > 255}",
                    )
                    if exponent < 3 or exponent > 255:
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: invalid exponent, incrementing hash_count from {hash_count}",
                        )
                        self.log.warning(
                            f"Invalid nbits exponent in loop: {exponent}, skipping iteration"
                        )
                        hash_count += 1
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: hash_count incremented to {hash_count}, continuing",
                        )
                        continue
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: calculating target from nbits[2:]={nbits[2:] if nbits else 'None'}, exponent={exponent}",
                    )
                    target = (nbits[2:] + "00" * (exponent - 3)).zfill(64)
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: target calculated, length={len(target)}, target[:32]={target[:32] if target else 'None'}...",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: converting target to int",
                    )
                    target_int = int(target, 16)
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: target_int={target_int}",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: calculating reference_diff",
                    )
                    reference_diff = int(
                        "00000000FFFF0000000000000000000000000000000000000000000000000000",
                        16,
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: reference_diff={reference_diff}",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: calculating target_difficulty, target_int > 0={target_int > 0}",
                    )
                    target_difficulty = (
                        reference_diff / target_int if target_int > 0 else 0
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: target_difficulty={target_difficulty}",
                    )
                except (ValueError, IndexError, ZeroDivisionError) as e:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: exception in target calculation: {type(e).__name__}: {e}",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: incrementing hash_count from {hash_count}",
                    )
                    self.log.warning(
                        f"Failed to recalculate target in loop: {e}, skipping iteration"
                    )
                    hash_count += 1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hash_count incremented to {hash_count}, continuing",
                    )
                    continue

                # Recalculate merkle_root dynamically (#64)
                _vlog(
                    self.log, self._verbose_logging, "LOOP: recalculating merkle_root"
                )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: acquiring state._lock for merkle_root",
                )
                with self.state._lock:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: inside state._lock for merkle_root",
                    )
                    coinbase_part1 = self.state.coinbase_part1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: coinbase_part1={'present' if coinbase_part1 else 'None'} (length={len(coinbase_part1) if coinbase_part1 else 0})",
                    )
                    extranonce1 = self.state.extranonce1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: extranonce1={'present' if extranonce1 else 'None'} (length={len(extranonce1) if extranonce1 else 0})",
                    )
                    coinbase_part2 = self.state.coinbase_part2
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: coinbase_part2={'present' if coinbase_part2 else 'None'} (length={len(coinbase_part2) if coinbase_part2 else 0})",
                    )
                    merkle_branch = self.state.merkle_branch
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: merkle_branch={'present' if merkle_branch else 'None'} (length={len(merkle_branch) if merkle_branch else 0})",
                    )
                    extranonce2_size = self.state.extranonce2_size
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: extranonce2_size={extranonce2_size}",
                    )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: released state._lock for merkle_root",
                )

                # Log state values every 1000 iterations
                if hash_count % 1000 == 0:
                    self.log.debug(
                        f"Mining state check: coinbase_part1={'present' if coinbase_part1 else 'MISSING'}, extranonce1={'present' if extranonce1 else 'MISSING'}, coinbase_part2={'present' if coinbase_part2 else 'MISSING'}, extranonce2_size={extranonce2_size}, merkle_branch length={len(merkle_branch) if merkle_branch else 0}"
                    )

                # Generate new extranonce2 for this iteration (#67)
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: generating extranonce2, hasattr extranonce2_counter={hasattr(self, 'extranonce2_counter')}",
                )
                if not hasattr(self, "extranonce2_counter"):
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: extranonce2_counter not found, initializing to 0",
                    )
                    self.extranonce2_counter = 0
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: extranonce2_counter={self.extranonce2_counter}, extranonce2_size={extranonce2_size}",
                )
                extranonce2 = hex(self.extranonce2_counter)[2:].zfill(
                    2 * extranonce2_size
                )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: extranonce2 generated: {extranonce2} (length={len(extranonce2)})",
                )
                self.extranonce2_counter = (self.extranonce2_counter + 1) % (
                    2 ** (8 * extranonce2_size)
                )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: extranonce2_counter incremented to {self.extranonce2_counter}",
                )

                # Build coinbase and calculate merkle_root
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking coinbase fields: coinbase_part1={bool(coinbase_part1)}, extranonce1={bool(extranonce1)}, coinbase_part2={bool(coinbase_part2)}",
                )
                if not coinbase_part1 or not extranonce1 or not coinbase_part2:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: missing coinbase fields, incrementing hash_count from {hash_count}",
                    )
                    if hash_count % 100 == 0:  # Log every 100 iterations to avoid spam
                        self.log.warning(
                            f"Missing coinbase fields (iteration {hash_count}): coinbase_part1={'present' if coinbase_part1 else 'MISSING'}, extranonce1={'present' if extranonce1 else 'MISSING'}, coinbase_part2={'present' if coinbase_part2 else 'MISSING'}, skipping iteration"
                        )
                    hash_count += 1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hash_count incremented to {hash_count}, continuing",
                    )
                    continue

                # Initialize merkle_root to None before calculation
                merkle_root = None
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: merkle_root initialized to None",
                )

                _vlog(self.log, self._verbose_logging, "LOOP: building coinbase")
                coinbase = coinbase_part1 + extranonce1 + extranonce2 + coinbase_part2
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: coinbase built, length={len(coinbase)}, coinbase[:50]={coinbase[:50] if coinbase else 'None'}...",
                )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: entering try block for coinbase hash",
                )
                try:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: unhexlifying coinbase, length={len(coinbase)}",
                    )
                    coinbase_bytes = binascii.unhexlify(coinbase)
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: coinbase_bytes length={len(coinbase_bytes)}",
                    )
                    _vlog(self.log, self._verbose_logging, "LOOP: first SHA256")
                    coinbase_hash_intermediate = hashlib.sha256(coinbase_bytes).digest()
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: coinbase_hash_intermediate length={len(coinbase_hash_intermediate)}",
                    )
                    _vlog(self.log, self._verbose_logging, "LOOP: second SHA256")
                    coinbase_hash_bin = hashlib.sha256(
                        coinbase_hash_intermediate
                    ).digest()
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: coinbase_hash_bin length={len(coinbase_hash_bin)}",
                    )
                except (binascii.Error, ValueError) as e:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: exception in coinbase hash: {type(e).__name__}: {e}",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: incrementing hash_count from {hash_count}",
                    )
                    if hash_count % 100 == 0:  # Log every 100 iterations to avoid spam
                        self.log.error(
                            f"Failed to unhexlify coinbase (iteration {hash_count}): {e}, coinbase length={len(coinbase)}"
                        )
                    hash_count += 1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hash_count incremented to {hash_count}, continuing",
                    )
                    continue

                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: initializing merkle_root_bin from coinbase_hash_bin",
                )
                merkle_root_bin = coinbase_hash_bin
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: merkle_root_bin initialized, length={len(merkle_root_bin)}",
                )
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking merkle_branch: {bool(merkle_branch)}, length={len(merkle_branch) if merkle_branch else 0}",
                )
                if merkle_branch:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: processing {len(merkle_branch)} merkle branch hashes",
                    )
                    for idx, branch_hash in enumerate(merkle_branch):
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: processing merkle branch {idx+1}/{len(merkle_branch)}, branch_hash={'present' if branch_hash else 'None'} (length={len(branch_hash) if branch_hash else 0})",
                        )
                        try:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: unhexlifying branch_hash",
                            )
                            branch_hash_bytes = binascii.unhexlify(branch_hash)
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: branch_hash_bytes length={len(branch_hash_bytes)}",
                            )
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                "LOOP: combining merkle_root_bin + branch_hash_bytes",
                            )
                            combined = merkle_root_bin + branch_hash_bytes
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: combined length={len(combined)}",
                            )
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                "LOOP: first SHA256 on combined",
                            )
                            intermediate = hashlib.sha256(combined).digest()
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: intermediate length={len(intermediate)}",
                            )
                            _vlog(
                                self.log, self._verbose_logging, "LOOP: second SHA256"
                            )
                            merkle_root_bin = hashlib.sha256(intermediate).digest()
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: merkle_root_bin updated, length={len(merkle_root_bin)}",
                            )
                        except (binascii.Error, ValueError) as e:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: exception in merkle branch hash {idx+1}: {type(e).__name__}: {e}",
                            )
                            if (
                                hash_count % 100 == 0
                            ):  # Log every 100 iterations to avoid spam
                                self.log.error(
                                    f"Failed to unhexlify merkle branch hash (iteration {hash_count}): {e}, branch_hash={branch_hash[:50] if branch_hash else None}"
                                )
                            # Skip this branch hash and continue with next
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: skipping branch hash {idx+1}, continuing to next",
                            )
                            continue
                else:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: no merkle_branch, using coinbase_hash_bin as merkle_root_bin",
                    )

                # Convert merkle_root to little-endian hex for block header
                _vlog(
                    self.log,
                    self._verbose_logging,
                    "LOOP: converting merkle_root to little-endian hex",
                )
                try:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hexlifying merkle_root_bin, length={len(merkle_root_bin)}",
                    )
                    merkle_root_hex = binascii.hexlify(merkle_root_bin).decode()
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: merkle_root_hex length={len(merkle_root_hex)}, merkle_root_hex[:32]={merkle_root_hex[:32] if merkle_root_hex else 'None'}...",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: calling _hex_to_little_endian",
                    )
                    merkle_root = self._hex_to_little_endian(merkle_root_hex, 64)
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: merkle_root converted, length={len(merkle_root) if merkle_root else 0}, merkle_root[:32]={merkle_root[:32] if merkle_root else 'None'}...",
                    )
                except (ValueError, TypeError, AttributeError) as e:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: exception in merkle_root conversion: {type(e).__name__}: {e}",
                    )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: incrementing hash_count from {hash_count}",
                    )
                    if hash_count % 100 == 0:  # Log every 100 iterations to avoid spam
                        self.log.error(
                            f"Failed to convert merkle_root to little-endian (iteration {hash_count}): {e}"
                        )
                    hash_count += 1
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hash_count incremented to {hash_count}, continuing",
                    )
                    continue

                # Initialize hash_hex to None (will be set by GPU or CPU mining)
                _vlog(
                    self.log, self._verbose_logging, "LOOP: initializing hash variables"
                )
                hash_hex = None
                _vlog(self.log, self._verbose_logging, "LOOP: hash_hex=None")
                nonce_hex = None
                _vlog(self.log, self._verbose_logging, "LOOP: nonce_hex=None")
                cpu_hash_hex = None
                _vlog(self.log, self._verbose_logging, "LOOP: cpu_hash_hex=None")
                cpu_nonce_hex = None
                _vlog(self.log, self._verbose_logging, "LOOP: cpu_nonce_hex=None")
                block_header_hex = None
                _vlog(self.log, self._verbose_logging, "LOOP: block_header_hex=None")

                # Log mining iteration start (every 1000 iterations to avoid spam)
                if hash_count % 1000 == 0:
                    with self.state._lock:
                        current_nbits = self.state.nbits
                        current_prev_hash = self.state.prev_hash
                    self.log.info(
                        f"Mining iteration {hash_count}: CPU enabled={cpu_mining_enabled}, GPU enabled={gpu_mining_enabled}, GPU miner={self.gpu_miner is not None}, nbits={current_nbits}, prev_hash={current_prev_hash[:16] if current_prev_hash else None}..."
                    )
                    # Add INFO log immediately after iteration log to track progress
                    self.log.info(
                        f"After iteration log: hash_count={hash_count}, merkle_root={'defined' if 'merkle_root' in locals() else 'NOT DEFINED'}, entering GPU/CPU mining check"
                    )

                # Use GPU miner if enabled and available
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking GPU mining: gpu_mining_enabled={gpu_mining_enabled}, gpu_miner={self.gpu_miner is not None}",
                )
                if hash_count % 1000 == 0:
                    self.log.info(
                        f"Before GPU check: gpu_mining_enabled={gpu_mining_enabled}, gpu_miner={self.gpu_miner is not None}, merkle_root={'defined' if 'merkle_root' in locals() else 'NOT DEFINED'}"
                    )
                if gpu_mining_enabled and self.gpu_miner:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: GPU mining enabled and available, entering GPU mining block",
                    )
                    # GPU mining: test multiple nonces in parallel batch
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: acquiring state._lock for GPU mining",
                    )
                    with self.state._lock:
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            "LOOP: inside state._lock for GPU mining",
                        )
                        prev_hash = self.state.prev_hash
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: GPU prev_hash={'present' if prev_hash else 'None'} (length={len(prev_hash) if prev_hash else 0})",
                        )
                        ntime = self.state.ntime
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: GPU ntime={'present' if ntime else 'None'} (length={len(ntime) if ntime else 0})",
                        )
                        nbits = self.state.nbits
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: GPU nbits={'present' if nbits else 'None'} (length={len(nbits) if nbits else 0})",
                        )
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            "LOOP: entering try block for GPU block header build",
                        )
                        try:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: calling _build_block_header with prev_hash={prev_hash[:16] if prev_hash else None}..., merkle_root={merkle_root[:16] if merkle_root else None}..., ntime={ntime}, nbits={nbits}, nonce=00000000",
                            )
                            block_header_base = self._build_block_header(
                                prev_hash, merkle_root, ntime, nbits, "00000000"
                            )
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: _build_block_header returned, block_header_base length={len(block_header_base) if block_header_base else 0}",
                            )
                            block_header_hex = block_header_base
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: block_header_hex set to block_header_base, length={len(block_header_hex) if block_header_hex else 0}",
                            )
                            if hash_count % 1000 == 0:
                                self.log.debug(
                                    f"GPU mining: block_header_base built, length={len(block_header_base)}, start_nonce={self.gpu_nonce_counter}"
                                )
                        except (RuntimeError, ValueError) as e:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                f"LOOP: exception in GPU block header build: {type(e).__name__}: {e}",
                            )
                            if (
                                hash_count % 100 == 0
                            ):  # Log every 100 iterations to avoid spam
                                self.log.error(
                                    f"GPU mining: Failed to build block header (iteration {hash_count}): {e}"
                                )
                                self.log.error(
                                    f"GPU mining: Block header fields: prev_hash={'present' if prev_hash else 'MISSING'} (length={len(prev_hash) if prev_hash else 0}), merkle_root={'present' if merkle_root else 'MISSING'} (length={len(merkle_root) if merkle_root else 0}), ntime={'present' if ntime else 'MISSING'} (length={len(ntime) if ntime else 0}), nbits={'present' if nbits else 'MISSING'} (length={len(nbits) if nbits else 0})"
                                )
                            block_header_hex = None
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                "LOOP: block_header_hex set to None due to exception",
                            )
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: released state._lock for GPU mining",
                    )
                    hash_hex = None
                    _vlog(self.log, self._verbose_logging, "LOOP: GPU hash_hex=None")
                    nonce_hex = None
                    _vlog(self.log, self._verbose_logging, "LOOP: GPU nonce_hex=None")

                    # Use sequential nonce counter for better coverage (cycles through 2^32)
                    # Use batch_size from config instead of hardcoded value
                    num_nonces_per_batch = self.cfg.get("compute", {}).get(
                        "batch_size", 256
                    )
                    if hash_count % 1000 == 0:
                        self.log.debug(
                            f"GPU batch mining: num_nonces={num_nonces_per_batch}, start_nonce={self.gpu_nonce_counter}"
                        )

                    # Try GPU batch hashing (use sequential nonce counter for better coverage)
                    if block_header_hex is None:
                        # Block header build failed, skip GPU mining
                        if (
                            hash_count % 100 == 0
                        ):  # Log every 100 iterations to avoid spam
                            self.log.warning(
                                f"GPU mining: Skipping due to block header build failure (iteration {hash_count})"
                            )
                        hash_hex = None
                        nonce_hex = None
                        # CRITICAL: If CPU mining is also disabled, increment hash_count here
                        if not cpu_mining_enabled:
                            hash_count += 1
                            self.total_hash_count += 1
                            update_status("total_hashes", self.total_hash_count)
                    else:
                        try:
                            batch_start_time = time.time()
                            result = self.gpu_miner.hash_block_header(
                                block_header_hex,
                                num_nonces=num_nonces_per_batch,
                                start_nonce=self.gpu_nonce_counter,
                            )
                            batch_duration = time.time() - batch_start_time
                            if hash_count % 1000 == 0:
                                self.log.debug(
                                    f"GPU batch completed in {batch_duration:.4f}s, result={'found' if result else 'none'}, hash_count={hash_count}"
                                )

                            if (
                                result
                                and isinstance(result, tuple)
                                and len(result) == 2
                            ):
                                hash_hex, best_nonce = result
                                # Convert GPU hash to little-endian for Bitcoin (#69)
                                # GPU returns hash in big-endian (from binascii.hexlify), but Bitcoin uses little-endian
                                hash_hex = self._hex_to_little_endian(hash_hex, 64)
                                # Convert nonce to little-endian hex format for Bitcoin block header (#72)
                                nonce_hex = self._int_to_little_endian_hex(
                                    best_nonce, 4
                                )
                                # Update hash count based on actual batch size
                                hash_count += num_nonces_per_batch
                                self.total_hash_count += num_nonces_per_batch
                                update_status("total_hashes", self.total_hash_count)
                                # Increment nonce counter for next batch (FIX: use start_nonce + num_nonces, not best_nonce + 1)
                                # This ensures we don't skip any nonce ranges
                                self.gpu_nonce_counter = (
                                    self.gpu_nonce_counter + num_nonces_per_batch
                                ) % (2**32)

                                # Time-Slicing: Pause based on GPU utilization percentage
                                if gpu_utilization_percent < 100 and batch_duration > 0:
                                    # Calculate pause time: if 20% utilization, pause 80% of the time
                                    # Formula: pause_time = batch_duration * (100 - utilization) / utilization
                                    pause_ratio = (
                                        100 - gpu_utilization_percent
                                    ) / gpu_utilization_percent
                                    pause_time = batch_duration * pause_ratio
                                    if pause_time > 0:
                                        time.sleep(pause_time)
                            else:
                                # GPU returned None - CPU mining will handle it if enabled
                                hash_hex = None
                                nonce_hex = None
                                # CRITICAL: If CPU mining is also disabled, increment hash_count here
                                if not cpu_mining_enabled:
                                    hash_count += 1
                                    self.total_hash_count += 1
                                    update_status("total_hashes", self.total_hash_count)
                        except Exception as e:
                            # GPU error - CPU mining will handle it if enabled
                            if (
                                hash_count % 100 == 0
                            ):  # Log every 100 iterations to avoid spam
                                self.log.error(
                                    f"GPU mining error (iteration {hash_count}): {e}, CPU mining will continue if enabled",
                                    exc_info=True,
                                )
                            hash_hex = None
                            nonce_hex = None
                            # CRITICAL: If CPU mining is also disabled, increment hash_count here
                            if not cpu_mining_enabled:
                                hash_count += 1
                                self.total_hash_count += 1
                                update_status("total_hashes", self.total_hash_count)
                else:
                    # GPU mining disabled or not available
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: GPU mining disabled or not available: gpu_mining_enabled={gpu_mining_enabled}, gpu_miner={self.gpu_miner is not None}",
                    )
                    if hash_count % 1000 == 0:
                        self.log.debug(
                            f"GPU mining: DISABLED or NOT AVAILABLE (iteration {hash_count}, enabled={gpu_mining_enabled}, miner={self.gpu_miner is not None})"
                        )
                    hash_hex = None
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: GPU hash_hex=None (GPU disabled)",
                    )
                    nonce_hex = None
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: GPU nonce_hex=None (GPU disabled)",
                    )

                # CPU mining (runs independently if enabled, regardless of GPU status)
                # This should run even when GPU is disabled or not available
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking CPU mining: cpu_mining_enabled={cpu_mining_enabled}",
                )
                if hash_count % 1000 == 0:
                    self.log.info(
                        f"Before CPU check: cpu_mining_enabled={cpu_mining_enabled}, merkle_root={'defined' if 'merkle_root' in locals() else 'NOT DEFINED'}"
                    )
                if cpu_mining_enabled:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: CPU mining enabled, entering CPU mining block",
                    )
                    # Quick INFO-level instrumentation to verify CPU branch is entered
                    try:
                        self.log.info(
                            "CPU mining enabled: entering CPU mining block (hash_count=%s)",
                            hash_count,
                        )
                    except Exception:
                        pass
                    # Use a batched nonce loop for CPU mining to ensure we try many nonces per iteration
                    try:
                        _vlog(self.log, self._verbose_logging, "LOOP: entering try block for CPU mining (batched)")
                        _vlog(self.log, self._verbose_logging, "LOOP: acquiring state._lock for CPU mining")
                        with self.state._lock:
                            prev_hash = self.state.prev_hash
                            ntime = self.state.ntime
                            nbits = self.state.nbits
                        _vlog(self.log, self._verbose_logging, "LOOP: released state._lock for CPU mining")

                        # Build a block header base with nonce set to 00000000, then iterate a batch of nonces
                        block_header_base = self._build_block_header(prev_hash, merkle_root, ntime, nbits, "00000000")
                        block_header_hex = block_header_base

                        num_nonces_per_batch = self.cfg.get("compute", {}).get("batch_size", 256)
                        if not isinstance(num_nonces_per_batch, int) or num_nonces_per_batch <= 0:
                            num_nonces_per_batch = 1

                        if hash_count % 1000 == 0:
                            self.log.debug(f"CPU batch mining: num_nonces={num_nonces_per_batch}, start_nonce={self.cpu_nonce_counter}")

                        cpu_found = False
                        cpu_hash_hex = None
                        cpu_nonce_hex = None

                        for i_nonce in range(num_nonces_per_batch):
                            nonce_int = (self.cpu_nonce_counter + i_nonce) % (2 ** 32)
                            try:
                                trial_nonce_hex = self._int_to_little_endian_hex(nonce_int, 4)
                            except (ValueError, TypeError):
                                continue

                            trial_block_header_hex = block_header_hex[:-8] + trial_nonce_hex
                            try:
                                trial_block_header_bytes = binascii.unhexlify(trial_block_header_hex)
                            except binascii.Error:
                                hash_count += 1
                                self.total_hash_count += 1
                                update_status("total_hashes", self.total_hash_count)
                                continue

                            cpu_hash_intermediate = hashlib.sha256(trial_block_header_bytes).digest()
                            cpu_hash_bin = hashlib.sha256(cpu_hash_intermediate).digest()

                            try:
                                cpu_hash_big = binascii.hexlify(cpu_hash_bin).decode()
                                cpu_hash_little = self._hex_to_little_endian(cpu_hash_big, 64)
                            except Exception:
                                hash_count += 1
                                self.total_hash_count += 1
                                update_status("total_hashes", self.total_hash_count)
                                continue

                            hash_count += 1
                            self.total_hash_count += 1
                            update_status("total_hashes", self.total_hash_count)

                            try:
                                cpu_hash_int = int(cpu_hash_little, 16)
                            except ValueError:
                                continue

                            if cpu_hash_int < target_int:
                                cpu_found = True
                                cpu_hash_hex = cpu_hash_little
                                cpu_nonce_hex = trial_nonce_hex
                                self.cpu_nonce_counter = (nonce_int + 1) % (2 ** 32)
                                break

                        if not cpu_found:
                            self.cpu_nonce_counter = (self.cpu_nonce_counter + num_nonces_per_batch) % (2 ** 32)

                    except Exception as e:
                        # Catch ALL unexpected errors in CPU mining to ensure loop continues
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: unexpected exception in CPU mining: {type(e).__name__}: {e}",
                        )
                        if (
                            hash_count % 100 == 0
                        ):  # Log every 100 iterations to avoid spam
                            self.log.error(
                                f"CPU mining: Unexpected error (iteration {hash_count}): {e}",
                                exc_info=True,
                            )
                        cpu_hash_hex = None
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            "LOOP: CPU cpu_hash_hex=None due to unexpected exception",
                        )
                        cpu_nonce_hex = None
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            "LOOP: CPU cpu_nonce_hex=None due to unexpected exception",
                        )
                        # CRITICAL: Always increment hash_count even on unexpected errors
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: incrementing hash_count from {hash_count}",
                        )
                        hash_count += 1
                        self.total_hash_count += 1
                        update_status("total_hashes", self.total_hash_count)
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: hash_count incremented to {hash_count}",
                        )
                else:
                    # CPU mining disabled
                    _vlog(self.log, self._verbose_logging, "LOOP: CPU mining disabled")
                    if hash_count % 1000 == 0:
                        self.log.debug(f"CPU mining: DISABLED (iteration {hash_count})")
                    cpu_hash_hex = None
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: CPU cpu_hash_hex=None (CPU disabled)",
                    )
                    cpu_nonce_hex = None
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: CPU cpu_nonce_hex=None (CPU disabled)",
                    )
                    # CRITICAL: If CPU mining is disabled and GPU is also disabled/not available, increment hash_count
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: checking if GPU also disabled: gpu_mining_enabled={gpu_mining_enabled}, gpu_miner={self.gpu_miner is not None}",
                    )
                    if not (gpu_mining_enabled and self.gpu_miner):
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: both CPU and GPU disabled, incrementing hash_count from {hash_count}",
                        )
                        hash_count += 1
                        self.total_hash_count += 1
                        update_status("total_hashes", self.total_hash_count)
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: hash_count incremented to {hash_count}",
                        )

                # If GPU didn't produce a hash, use CPU hash
                # This code should run regardless of whether CPU mining is enabled or disabled
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: selecting best hash, hash_hex={'present' if hash_hex else 'None'}, cpu_hash_hex={'present' if cpu_hash_hex else 'None'}",
                )
                if hash_hex is None:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: hash_hex is None, using CPU hash",
                    )
                    hash_hex = cpu_hash_hex
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: hash_hex set to cpu_hash_hex, hash_hex={'present' if hash_hex else 'None'}",
                    )
                    nonce_hex = cpu_nonce_hex
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: nonce_hex set to cpu_nonce_hex, nonce_hex={'present' if nonce_hex else 'None'}",
                    )
                # If both produced hashes, use the better one (lower value = better)
                elif cpu_hash_hex is not None:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: both GPU and CPU hashes present, comparing",
                    )
                    try:
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: converting hashes to int for comparison",
                        )
                        cpu_hash_int = int(cpu_hash_hex, 16)
                        gpu_hash_int = int(hash_hex, 16)
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: cpu_hash_int={cpu_hash_int}, gpu_hash_int={gpu_hash_int}, cpu_hash_int < gpu_hash_int={cpu_hash_int < gpu_hash_int}",
                        )
                        if cpu_hash_int < gpu_hash_int:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                "LOOP: CPU hash is better, using CPU hash",
                            )
                            hash_hex = cpu_hash_hex
                            nonce_hex = cpu_nonce_hex
                        else:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                "LOOP: GPU hash is better, keeping GPU hash",
                            )
                    except ValueError as e:
                        _vlog(
                            self.log,
                            self._verbose_logging,
                            f"LOOP: exception in hash comparison: {type(e).__name__}: {e}",
                        )
                        # If comparison fails, use GPU hash (or CPU if GPU is None)
                        if hash_hex is None:
                            _vlog(
                                self.log,
                                self._verbose_logging,
                                "LOOP: hash_hex is None after comparison error, using CPU hash",
                            )
                            hash_hex = cpu_hash_hex
                            nonce_hex = cpu_nonce_hex

                # Calculate hash rate in every iteration (before checking hash_hex/nonce_hex)
                # This ensures hash rate is always updated, even if hash_hex is None
                _vlog(self.log, self._verbose_logging, "LOOP: calculating hash rate")
                elapsed = time.time() - start_time
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: elapsed={elapsed}, start_time={start_time}",
                )
                if elapsed > 0:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        f"LOOP: calculating hash_rate, hash_count={hash_count}, elapsed={elapsed}",
                    )
                    hash_rate = hash_count / elapsed
                    _vlog(
                        self.log, self._verbose_logging, f"LOOP: hash_rate={hash_rate}"
                    )
                    update_status("hash_rate", hash_rate)
                    # Log hash rate every 1000 hashes in DEBUG mode
                    if hash_count % 1000 == 0:
                        self.log.debug(
                            f"Hash rate: {hash_rate:.2f} H/s, total hashes: {hash_count}, elapsed: {elapsed:.2f}s"
                        )
                else:
                    _vlog(
                        self.log,
                        self._verbose_logging,
                        "LOOP: elapsed <= 0, skipping hash rate calculation",
                    )

                # Update hash_count even if no mining was performed (#51)
                _vlog(
                    self.log,
                    self._verbose_logging,
                    f"LOOP: checking hash_hex and nonce_hex: hash_hex={'present' if hash_hex else 'None'}, nonce_hex={'present' if nonce_hex else 'None'}",
                )
                if hash_hex is None or nonce_hex is None:
                    if not cpu_mining_enabled and not (
                        gpu_mining_enabled and self.gpu_miner
                    ):
                        # Both CPU and GPU mining disabled - pause briefly
                        self.log.warning(
                            "Both CPU and GPU mining are disabled. Pausing..."
                        )
                        hash_count += 1  # Still count iteration
                        time.sleep(0.1)
                        continue
                    else:
                        # Mining enabled but no hash produced (error case)
                        # Log detailed error with state information
                        self.log.error(
                            f"hash_hex or nonce_hex not defined - this should not happen! "
                            f"hash_hex={hash_hex}, nonce_hex={nonce_hex}, "
                            f"cpu_hash_hex={cpu_hash_hex}, cpu_nonce_hex={cpu_nonce_hex}, "
                            f"cpu_enabled={cpu_mining_enabled}, gpu_enabled={gpu_mining_enabled}, "
                            f"gpu_miner={self.gpu_miner is not None}"
                        )
                        hash_count += 1  # Still count iteration
                        continue

                if hash_hex.startswith(prefix_zeros):
                    self.log.debug(
                        "Candidate hash %s at height %s (nonce=%s)",
                        hash_hex,
                        current_height + 1,
                        nonce_hex,
                    )
                    update_status("last_hash", hash_hex)
                try:
                    this_hash_int = int(hash_hex, 16)
                except ValueError as e:
                    self.log.error(
                        f"Invalid hash_hex format: {hash_hex[:50]}... Error: {e}"
                    )
                    hash_count += 1
                    continue

                # Prevent division by zero (hash_hex could be all zeros)
                if this_hash_int == 0:
                    self.log.warning(
                        f"Hash is zero, skipping difficulty calculation: {hash_hex}"
                    )
                    hash_count += 1
                    continue

                # Bitcoin reference difficulty: 0x00000000FFFF0000000000000000000000000000000000000000000000000000
                reference_diff = int(
                    "00000000FFFF0000000000000000000000000000000000000000000000000000",
                    16,
                )
                difficulty = reference_diff / this_hash_int

                # Ensure height_to_best_difficulty key exists (may have changed if new block detected)
                with self.state._lock:
                    if (current_height + 1) not in self.state.height_to_best_difficulty:
                        self.state.height_to_best_difficulty[current_height + 1] = 0

                    if (
                        self.state.height_to_best_difficulty[current_height + 1]
                        < difficulty
                    ):
                        self.state.height_to_best_difficulty[current_height + 1] = (
                            difficulty
                        )
                    update_status("best_difficulty", difficulty)

                # Validate hash_hex and target lengths before comparison
                if len(hash_hex) != 64:
                    self.log.error(
                        f"Invalid hash_hex length: {len(hash_hex)} (expected 64 hex chars)"
                    )
                    hash_count += 1
                    continue
                if len(target) != 64:
                    self.log.error(
                        f"Invalid target length: {len(target)} (expected 64 hex chars)"
                    )
                    hash_count += 1
                    continue

                # Compare numerically (not as strings!)
                try:
                    hash_int = int(hash_hex, 16)
                    target_int = int(target, 16)
                except ValueError as e:
                    self.log.error(f"Failed to convert hash_hex or target to int: {e}")
                    hash_count += 1
                    continue

                if hash_int < target_int:
                    self.log.info("Block solved at height %s", current_height + 1)
                    self.log.info("Block hash %s", hash_hex)

                    # CRITICAL FIX (#73, #74, #75): Capture all values from current iteration BEFORE state might change
                    # These values are from the iteration where the solution was found
                    solution_extranonce2 = extranonce2  # From line 772 - current iteration (not from state!)
                    solution_merkle_root = (
                        merkle_root  # From line 798 - current iteration
                    )

                    # Get stable state values (these should not change during a job, but capture them atomically)
                    with self.state._lock:
                        solution_job_id = self.state.job_id
                        solution_ntime = self.state.ntime
                        solution_prev_hash = self.state.prev_hash
                        solution_nbits = self.state.nbits
                        solution_version = self.state.version

                    # Build block header for logging (reconstruct from solution)
                    solution_block_header = self._build_block_header(
                        solution_prev_hash,
                        solution_merkle_root,
                        solution_ntime,
                        solution_nbits,
                        nonce_hex,
                    )
                    self.log.debug("Blockheader %s", solution_block_header)
                    self.log.debug(
                        f"Solution details: nonce={nonce_hex}, extranonce2={solution_extranonce2}, ntime={solution_ntime}, job_id={solution_job_id}"
                    )
                    try:
                        self.log.debug(
                            f"Submitting solution to pool: wallet={self.wallet[:10]}..., job_id={solution_job_id}, nonce={nonce_hex}, extranonce2={solution_extranonce2}"
                        )
                        ret = self.pool.submit(
                            self.wallet,
                            solution_job_id,
                            solution_extranonce2,
                            solution_ntime,
                            nonce_hex,
                        )
                        self.log.info("Pool response %s", ret)
                        self.log.debug(f"Full pool response: {ret}")
                        try:
                            from ..web import add_share

                            response_str = (
                                ret.decode() if isinstance(ret, bytes) else str(ret)
                            )
                            # Parse pool response more carefully
                            # Valid responses: {"result":true} or {"result": true} or {"error":null,"result":true}
                            # Invalid: {"error":"...","result":false} or {"result":false}
                            import json

                            try:
                                response_json = json.loads(response_str)
                                accepted = (
                                    response_json.get("result") is True
                                    and response_json.get("error") is None
                                )
                            except (json.JSONDecodeError, AttributeError, TypeError):
                                # Fallback to string matching if JSON parsing fails
                                accepted = (
                                    '"result":true' in response_str
                                    or '"result": true' in response_str
                                ) and '"error"' not in response_str.lower()
                            add_share(accepted, response_str)
                        except (
                            ImportError,
                            AttributeError,
                            UnicodeDecodeError,
                            Exception,
                        ) as e:
                            # Log but don't fail if web module is not available or share tracking fails
                            self.log.debug(f"Could not track share: {e}")
                            pass
                        return True
                    except (ConnectionError, RuntimeError, Exception) as e:
                        self.log.error(f"Failed to submit share to pool: {e}")
                        # Continue mining even if submit fails - pool might recover
                        # Don't return True to avoid marking as successful
                        pass

                    # CRITICAL: Ensure hash_count is incremented at end of successful iteration
                    # This handles cases where hash_count was not incremented due to early continue/break
                    # Only increment if hash_count hasn't changed since start of iteration
                    if hash_count == initial_hash_count:
                        hash_count += 1
                        self.total_hash_count += 1
                        update_status("total_hashes", self.total_hash_count)

            except Exception as e:
                # CRITICAL: Catch ALL exceptions to ensure loop always progresses
                # Log error but continue mining
                if hash_count % 100 == 0:  # Log every 100 iterations to avoid spam
                    self.log.error(
                        f"Unexpected error in mining loop (iteration {hash_count}): {e}",
                        exc_info=True,
                    )

                # CRITICAL: Always increment hash_count even on unexpected errors
                hash_count += 1
                self.total_hash_count += 1
                update_status("total_hashes", self.total_hash_count)

                # Continue to next iteration
                continue
