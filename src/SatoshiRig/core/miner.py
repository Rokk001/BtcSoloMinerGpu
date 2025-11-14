import binascii
import hashlib
import logging
import random
import threading
import time
from datetime import datetime
from typing import Any, Optional

import requests

from ..clients.pool_client import PoolClient
from .state import MinerState

# Initialize logger
logging.basicConfig(
    filename='logs/mining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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


class Miner:
    def __init__(
        self,
        wallet_address: str,
        config: dict,
        pool_client: PoolClient,
        state: MinerState,
        logger: logging.Logger,
    ):
        self.wallet = wallet_address
        self.cfg = config
        self.pool = pool_client
        self.state = state
        self.log = logger
        self.total_hash_count = 0  # Persistent total hash count across loops
        self.gpu_miner = None
        self.gpu_nonce_counter = (
            0  # Counter for sequential nonce generation in GPU mode
        )
        self._config_lock = threading.Lock()  # Lock for thread-safe config access
        self._running = False  # Flag to prevent multiple calls to start()

        # Initialize GPU miner if configured
        self._initialize_gpu_miner()

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
                    self.log.debug(f"Initializing CUDA GPU miner: device={gpu_device}, batch_size={batch_size}, max_workers={max_workers}")
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
                    self.log.debug("CUDA GPU miner initialization completed successfully")
                except Exception as e:
                    self.log.error(f"Failed to initialize CUDA miner: {e}")
                    self.log.warning("Falling back to CPU mining")
                    self.gpu_miner = None
            elif compute_backend == "opencl":
                try:
                    self.log.debug(f"Initializing OpenCL GPU miner: device={gpu_device}, batch_size={batch_size}, max_workers={max_workers}")
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
                    self.log.debug("OpenCL GPU miner initialization completed successfully")
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

        return (
            version
            + prev_hash
            + merkle_root
            + ntime
            + nbits
            + nonce_hex
            + "000000800000000000000000000000000000000000000000000000000000000000000000000000000000000080020000"
        )

    def start(self):
        # Prevent multiple calls to start()
        if self._running:
            self.log.warning(
                "Miner.start() called multiple times, ignoring duplicate call"
            )
            return
        self._running = True

        try:
            self.log.info("Connecting to pool %s:%s...", self.pool.host, self.pool.port)
            self.log.debug(f"Pool connection parameters: host={self.pool.host}, port={self.pool.port}")
            self.pool.connect()
            self.log.info("Connected to pool, subscribing...")
            self.log.debug("Sending subscription request to pool")
            update_pool_status(True, self.pool.host, self.pool.port)
            sub_details, extranonce1, extranonce2_size = self.pool.subscribe()
            self.log.debug(f"Subscription response: extranonce1={extranonce1}, extranonce2_size={extranonce2_size}")
            with self.state._lock:
                self.state.subscription_details = sub_details
                self.state.extranonce1 = extranonce1
                self.state.extranonce2_size = extranonce2_size
            self.log.info("Subscribed to pool, authorizing...")
            self.log.debug(f"Authorizing with wallet: {self.wallet[:10]}...")
            self.pool.authorize(self.wallet)
            self.log.info("Authorized, waiting for mining notification...")
            self.log.debug("Waiting for mining.notify message from pool")
            responses = self.pool.read_notify()
            if not responses or len(responses) == 0:
                raise RuntimeError("No mining notification received from pool")
            if "params" not in responses[0] or len(responses[0]["params"]) < 9:
                raise RuntimeError(
                    f"Invalid mining notification format: {responses[0] if responses else 'empty'}"
                )
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
            self.log.debug(f"Mining notification received: job_id={self.state.job_id}, prev_hash={self.state.prev_hash[:16]}..., version={self.state.version}, nbits={self.state.nbits}, ntime={self.state.ntime}")
            update_status("job_id", self.state.job_id)
            self.log.info("Mining notification received, starting mining loop...")
            self.log.debug("Entering main mining loop")
            return self._mine_loop()
        except Exception as e:
            self.log.error(f"Failed to start mining: {e}", exc_info=True)
            self._running = False
            update_status("running", False)
            update_pool_status(False)
            raise

    def _mine_loop(self):
        """Main mining loop - wrapped in retry logic to handle transient errors"""
        restart_delay = self.cfg.get("miner", {}).get("restart_delay_secs", 2)
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return self._mine_loop_internal()
            except (ConnectionError, RuntimeError, ValueError, KeyError, IndexError) as e:
                retry_count += 1
                error_msg = str(e).lower()
                is_connection_error = (
                    "connection" in error_msg or 
                    "socket" in error_msg or
                    "not connected" in error_msg
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
                            except:
                                pass
                        self.pool.connect()
                        update_pool_status(True, self.pool.host, self.pool.port)
                        # Re-subscribe and re-authorize
                        sub_details, extranonce1, extranonce2_size = self.pool.subscribe()
                        with self.state._lock:
                            self.state.subscription_details = sub_details
                            self.state.extranonce1 = extranonce1
                            self.state.extranonce2_size = extranonce2_size
                        self.pool.authorize(self.wallet)
                        self.log.info("Successfully reconnected to pool")
                    except Exception as reconnect_error:
                        self.log.error(f"Failed to reconnect to pool: {reconnect_error}")
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
                        return
            except Exception as e:
                self.log.error(f"Unexpected error in mining loop: {e}", exc_info=True)
                retry_count += 1
                if retry_count >= max_retries:
                    self.log.error("Max retries reached, stopping mining loop")
                    update_status("running", False)
                    update_pool_status(False)
                    self._running = False
                    raise
                time.sleep(restart_delay)
        
        # If we get here, all retries failed
        self.log.error("Mining loop failed after all retries")
        update_status("running", False)
        update_pool_status(False)
        self._running = False

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
            self.log.error(f"Invalid nbits: {nbits}, waiting for valid data from pool...")
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

        reference_diff = int(
            "00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16
        )
        if target_int > 0:
            target_difficulty = reference_diff / target_int
            update_status("target_difficulty", target_difficulty)

        if not extranonce2_size or extranonce2_size <= 0:
            self.log.error(f"Invalid extranonce2_size: {extranonce2_size}, waiting for valid data from pool...")
            raise RuntimeError(f"Invalid extranonce2_size: {extranonce2_size}")

        extranonce2 = hex(random.getrandbits(32))[2:].zfill(2 * extranonce2_size)
        self.log.debug(f"Generated extranonce2: {extranonce2}")
        with self.state._lock:
            self.state.extranonce2 = extranonce2

        with self.state._lock:
            coinbase_part1 = self.state.coinbase_part1
            extranonce1 = self.state.extranonce1
            coinbase_part2 = self.state.coinbase_part2
            merkle_branch = self.state.merkle_branch

        # Validate required fields before building coinbase
        if not coinbase_part1 or not extranonce1 or not coinbase_part2:
            self.log.error(
                f"Missing required coinbase fields: coinbase_part1={coinbase_part1 is not None}, extranonce1={extranonce1 is not None}, coinbase_part2={coinbase_part2 is not None}. Waiting for pool notification..."
            )
            raise RuntimeError(
                "Missing required coinbase fields. Pool may not have sent complete mining notification."
            )

        coinbase = coinbase_part1 + extranonce1 + extranonce2 + coinbase_part2
        self.log.debug(f"Built coinbase: length={len(coinbase)}, extranonce1={extranonce1}, extranonce2={extranonce2}")
        coinbase_hash_bin = hashlib.sha256(
            hashlib.sha256(binascii.unhexlify(coinbase)).digest()
        ).digest()

        merkle_root = coinbase_hash_bin
        if merkle_branch:
            self.log.debug(f"Computing merkle root with {len(merkle_branch)} branch hashes")
            for branch_hash in merkle_branch:
                merkle_root = hashlib.sha256(
                    hashlib.sha256(
                        merkle_root + binascii.unhexlify(branch_hash)
                    ).digest()
                ).digest()

        merkle_root = binascii.hexlify(merkle_root).decode()
        self.log.debug(f"Computed merkle_root: {merkle_root[:32]}...")
        # Reverse byte order (little-endian to big-endian) - ensure even length
        if len(merkle_root) % 2 != 0:
            self.log.error(f"Invalid merkle_root hex length: {len(merkle_root)}")
            raise RuntimeError(f"Invalid merkle_root hex length: {len(merkle_root)}")
        merkle_root = "".join(
            [
                merkle_root[i] + merkle_root[i + 1]
                for i in range(0, len(merkle_root), 2)
            ][::-1]
        )

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
        self.log.debug(f"Mining parameters: height={current_height + 1}, target_difficulty={target_difficulty if target_int > 0 else 'N/A'}, target={target[:16]}...")
        update_status("current_height", current_height + 1)

        prefix_zeros = "0" * self.cfg.get("miner", {}).get("hash_log_prefix_zeros", 7)
        self.log.debug(f"Hash log prefix zeros: {len(prefix_zeros)} (will log hashes starting with {prefix_zeros})")
        hash_count = 0
        start_time = time.time()
        self.log.debug("Starting hash computation loop")

        while True:
            with self.state._lock:
                shutdown_flag = self.state.shutdown_flag
                prev_hash = self.state.prev_hash
                updated_prev_hash = self.state.updated_prev_hash

            if shutdown_flag:
                update_status("running", False)
                update_pool_status(False)
                self._running = False
                # Cleanup GPU miner before shutdown
                if self.gpu_miner:
                    try:
                        self.gpu_miner.cleanup()
                        self.log.debug("GPU miner cleaned up on shutdown")
                    except Exception as e:
                        self.log.debug(f"Error cleaning up GPU miner on shutdown: {e}")
                break

            if prev_hash != updated_prev_hash:
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
                        with self.state._lock:
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
                # Continue loop instead of recursive call to avoid stack overflow
                continue

            # Check CPU/GPU mining flags (thread-safe access)
            with self._config_lock:
                cpu_mining_enabled = self.cfg.get("compute", {}).get(
                    "cpu_mining_enabled", True
                )
                gpu_mining_enabled = self.cfg.get("compute", {}).get(
                    "gpu_mining_enabled", False
                )
                gpu_utilization_percent = self.cfg.get("compute", {}).get(
                    "gpu_utilization_percent", 100
                )

            # Initialize hash_hex to None (will be set by GPU or CPU mining)
            hash_hex = None
            nonce_hex = None
            cpu_hash_hex = None
            cpu_nonce_hex = None

            # Use GPU miner if enabled and available
            if gpu_mining_enabled and self.gpu_miner:
                # GPU mining: test multiple nonces in parallel batch
                with self.state._lock:
                    prev_hash = self.state.prev_hash
                    ntime = self.state.ntime
                    nbits = self.state.nbits
                block_header_base = self._build_block_header(
                    prev_hash, merkle_root, ntime, nbits, "00000000"
                )
                block_header_hex = block_header_base
                self.log.debug(f"GPU mining: block_header_base length={len(block_header_base)}, start_nonce={self.gpu_nonce_counter}")

                # Use sequential nonce counter for better coverage (cycles through 2^32)
                # Use batch_size from config instead of hardcoded value
                num_nonces_per_batch = self.cfg.get("compute", {}).get(
                    "batch_size", 256
                )
                self.log.debug(f"GPU batch mining: num_nonces={num_nonces_per_batch}, start_nonce={self.gpu_nonce_counter}")

                # Try GPU batch hashing (use sequential nonce counter for better coverage)
                try:
                    batch_start_time = time.time()
                    result = self.gpu_miner.hash_block_header(
                        block_header_hex,
                        num_nonces=num_nonces_per_batch,
                        start_nonce=self.gpu_nonce_counter,
                    )
                    batch_duration = time.time() - batch_start_time
                    self.log.debug(f"GPU batch completed in {batch_duration:.4f}s, result={'found' if result else 'none'}")

                    if result and isinstance(result, tuple) and len(result) == 2:
                        hash_hex, best_nonce = result
                        nonce_hex = f"{best_nonce:08x}"
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
                except Exception as e:
                    # GPU error - CPU mining will handle it if enabled
                    self.log.error(f"GPU mining error: {e}, CPU mining will continue if enabled")
                    hash_hex = None
                    nonce_hex = None

            # CPU mining (runs independently if enabled, regardless of GPU status)
            # This should run even when GPU is disabled or not available
            if cpu_mining_enabled:
                # CPU mining (original implementation)
                with self.state._lock:
                    prev_hash = self.state.prev_hash
                    ntime = self.state.ntime
                    nbits = self.state.nbits
                cpu_nonce_hex = hex(random.getrandbits(32))[2:].zfill(8)
                self.log.debug(f"CPU mining: generated nonce={cpu_nonce_hex}")
                block_header = self._build_block_header(
                    prev_hash, merkle_root, ntime, nbits, cpu_nonce_hex
                )
                try:
                    block_header_bytes = binascii.unhexlify(block_header)
                except binascii.Error as e:
                    self.log.error(
                        f"Invalid block_header hex in CPU mining: {block_header[:50]}... Error: {e}"
                    )
                    cpu_hash_hex = None
                    cpu_nonce_hex = None
                else:
                    cpu_hash_hex = hashlib.sha256(
                        hashlib.sha256(block_header_bytes).digest()
                    ).digest()
                    cpu_hash_hex = binascii.hexlify(cpu_hash_hex).decode()
                    self.log.debug(f"CPU hash computed: {cpu_hash_hex[:32]}...")
                    hash_count += 1
                    self.total_hash_count += 1
                    update_status("total_hashes", self.total_hash_count)
                
                # If GPU didn't produce a hash, use CPU hash
                if hash_hex is None:
                    hash_hex = cpu_hash_hex
                    nonce_hex = cpu_nonce_hex
                # If both produced hashes, use the better one (lower value = better)
                elif cpu_hash_hex is not None:
                    try:
                        if int(cpu_hash_hex, 16) < int(hash_hex, 16):
                            hash_hex = cpu_hash_hex
                            nonce_hex = cpu_nonce_hex
                    except ValueError:
                        # If comparison fails, use GPU hash (or CPU if GPU is None)
                        if hash_hex is None:
                            hash_hex = cpu_hash_hex
                            nonce_hex = cpu_nonce_hex

            # Check if we have a valid hash
            if hash_hex is None or nonce_hex is None:
                if not cpu_mining_enabled and not (gpu_mining_enabled and self.gpu_miner):
                    # Both CPU and GPU mining disabled - pause briefly
                    self.log.warning("Both CPU and GPU mining are disabled. Pausing...")
                    time.sleep(0.1)
                    continue
                else:
                    self.log.error(
                        "hash_hex or nonce_hex not defined - this should not happen!"
                    )
                    continue

            if hash_hex.startswith(prefix_zeros):
                self.log.debug(
                    "Candidate hash %s at height %s (nonce=%s)", hash_hex, current_height + 1, nonce_hex
                )
                update_status("last_hash", hash_hex)
            try:
                this_hash_int = int(hash_hex, 16)
            except ValueError as e:
                self.log.error(
                    f"Invalid hash_hex format: {hash_hex[:50]}... Error: {e}"
                )
                continue

            # Prevent division by zero (hash_hex could be all zeros)
            if this_hash_int == 0:
                self.log.warning(
                    f"Hash is zero, skipping difficulty calculation: {hash_hex}"
                )
                continue

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

            elapsed = time.time() - start_time
            if elapsed > 0:
                hash_rate = hash_count / elapsed
                update_status("hash_rate", hash_rate)
                # Log hash rate every 1000 hashes in DEBUG mode
                if hash_count % 1000 == 0:
                    self.log.debug(f"Hash rate: {hash_rate:.2f} H/s, total hashes: {hash_count}, elapsed: {elapsed:.2f}s")

            if hash_hex < target:
                self.log.info("Block solved at height %s", current_height + 1)
                self.log.info("Block hash %s", hash_hex)
                # Build block header for logging (reconstruct from solution)
                with self.state._lock:
                    prev_hash = self.state.prev_hash
                    ntime = self.state.ntime
                    nbits = self.state.nbits
                solution_block_header = self._build_block_header(
                    prev_hash, merkle_root, ntime, nbits, nonce_hex
                )
                self.log.debug("Blockheader %s", solution_block_header)
                self.log.debug(f"Solution details: nonce={nonce_hex}, extranonce2={extranonce2}, ntime={ntime}, job_id={self.state.job_id}")
                try:
                    with self.state._lock:
                        job_id = self.state.job_id
                        extranonce2 = self.state.extranonce2
                        ntime = self.state.ntime
                    self.log.debug(f"Submitting solution to pool: wallet={self.wallet[:10]}..., job_id={job_id}, nonce={nonce_hex}")
                    ret = self.pool.submit(
                        self.wallet, job_id, extranonce2, ntime, nonce_hex
                    )
                    self.log.info("Pool response %s", ret)
                    self.log.debug(f"Full pool response: {ret}")
                    try:
                        from ..web import add_share

                        response_str = (
                            ret.decode() if isinstance(ret, bytes) else str(ret)
                        )
                        accepted = (
                            '"result":true' in response_str
                            or '"result": true' in response_str
                            or "true" in response_str.lower()
                        )
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
