# Changelog

All notable changes to SatoshiRig will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.25.4] - 2025-01-27

### Fixed
- **Critical: Pool Connection Timeout and Mining Loop Blocking**: Fixed issue where mining loop was blocked waiting for pool notifications
  - `miner.start()` was blocking on `read_notify()` waiting for `mining.notify` message
  - After 30 seconds timeout, connection would break and mining would never start
  - Implemented background notification listener thread to continuously listen for pool notifications
  - Mining loop now starts immediately even if initial notification is not received
  - Notification thread updates state asynchronously when new blocks arrive
  - Prevents "read_notify failed: timed out" errors that prevented mining from starting

### Changed
- **Asynchronous Notification Handling**: Improved pool notification processing
  - Background thread continuously listens for `mining.notify` messages
  - Mining loop no longer blocks waiting for notifications
  - Initial notification wait reduced from 30 seconds to 5 seconds
  - State updates happen asynchronously without blocking mining operations
  - Better handling of pools that don't send immediate notifications after authorize

### Technical Details
- Added `_listen_for_notifications()` method that runs in background thread
- Uses `select.select()` for non-blocking socket data availability checks
- Short timeout (2 seconds) for notification reads to avoid blocking
- Notification thread properly stops when miner is stopped
- Thread-safe state updates with proper locking

## [2.25.3] - 2025-01-27

### Fixed
- **Critical: Race Condition in Pool Connection Startup**: Fixed timeout when `miner.start()` and `connect_to_pool_only()` run simultaneously
  - `miner.start()` now checks if pool is already connected before attempting new connection
  - Reuses existing connection and subscription if available (from `connect_to_pool_only()`)
  - Prevents "Subscribe failed: timed out" errors when starting miner after pool connection is established
  - Eliminates duplicate connection attempts that caused socket conflicts

### Changed
- **Pool Connection Reuse**: Improved connection handling in `miner.start()`
  - Checks for existing socket and subscription before connecting
  - Reuses existing connection when available, avoiding unnecessary reconnection
  - Better error handling and logging for connection state validation

## [2.25.2] - 2025-01-27

### Fixed
- **Pool Status Dashboard Update**: Fixed pool connection status not updating immediately in web dashboard
  - `update_pool_status()` now sends immediate SocketIO notification to frontend
  - Dashboard shows correct connection status in real-time instead of waiting 2-3 seconds
  - Added `set_socketio_instance()` function to register SocketIO instance for immediate updates
  - Pool status badge now updates instantly when connection state changes

## [2.25.1] - 2025-01-27

### Fixed
- **Critical: Syntax Error in Pool Client**: Fixed `SyntaxError: expected 'except' or 'finally' block` in `pool_client.py`
  - `subscribe()` method: Fixed incorrect indentation - all code within `try` block is now properly indented
  - `read_notify()` method: Fixed `else:` block and response processing to be within `try` block
  - Application can now start without syntax errors

## [2.25.0] - 2025-01-27

### Fixed
- **Critical: Race Condition in Pool Connection**: Fixed race condition between `connect_to_pool_only()` and running miner
  - `connect_to_pool_only()` was closing socket while miner was using it, causing timeouts
  - Added thread-safety with `_socket_lock` in `PoolClient` for all socket operations
  - `connect_to_pool_only()` now checks if miner is running before attempting reconnection
  - Prevents "Subscribe failed: timed out" errors when config is changed during mining
- **Thread-Safety for Socket Operations**: All pool socket operations are now thread-safe
  - Added `_socket_lock` to `PoolClient` class
  - All methods (`connect`, `subscribe`, `authorize`, `read_notify`, `submit`, `close`) now use lock
  - Prevents concurrent socket access conflicts

### Changed
- **Pool Connection Logic**: Improved pool connection handling
  - `connect_to_pool_only()` skips reconnection if miner is already running
  - Web UI no longer calls `connect_to_pool_only()` when miner is running
  - Better error handling and logging for connection state checks
  - Socket state validation before attempting reconnection

## [2.24.0] - 2025-01-27

### Fixed
- **Critical: Pool Subscribe Multiple Messages**: Fixed issue where pool sends multiple messages (subscribe response + mining.notify) in same buffer
  - Pool may send `mining.notify` message immediately after `mining.subscribe` response
  - Code now searches through all received lines to find the correct subscribe response
  - Looks for response with `result` field and `id: 1` (matching our request)
  - Ignores `mining.notify` messages that may come first
  - Prevents "Invalid subscribe response: missing 'result' field" error

### Changed
- **Pool Subscribe**: Improved message parsing to handle multiple messages from pool
  - Reads all lines from buffer until subscribe response is found
  - Continues reading if only `mining.notify` messages are received
  - Better error messages showing what was actually received

## [2.23.0] - 2025-01-27

### Fixed
- **Critical: CUDA Initialization**: Fixed CUDA initialization error `module 'pycuda.driver' has no attribute 'is_initialized'`
  - Replaced `cuda.is_initialized()` check with try/except pattern using `cuda.Device.count()`
  - `is_initialized()` doesn't exist in all PyCUDA versions
  - CUDA now initializes correctly on all PyCUDA versions
- **Critical: Pool Subscribe JSON Error**: Fixed "Unterminated string" JSON decode error when pool response > 1024 bytes
  - Changed from single `recv(1024)` to loop with multiple `recv(4096)` calls
  - Reads until complete line (ending with `\n`) is received
  - Handles large pool responses up to 64KB
  - Prevents JSON decode errors from truncated responses

### Changed
- **Pool Subscribe**: Improved robustness for handling large pool responses
  - Buffer-based reading with 64KB maximum size
  - Automatic timeout handling
  - Better error messages for connection issues

## [2.22.0] - 2025-01-27

### Removed
- **Export Button**: Removed export statistics button from web dashboard
  - Removed `/export` API endpoint
  - Removed `exportStats()` JavaScript function
  - Removed export button from UI header

### Added
- **Independent Pool Connection**: Pool connection is now established independently of mining
  - New `connect_to_pool_only()` method in Miner class
  - Connects to pool, subscribes, and authorizes without starting mining loop
  - Pool connection is automatically established when:
    - Application starts (if wallet is configured)
    - Miner is initialized via web UI
    - Pool configuration is changed

### Changed
- **Pool Connection Behavior**: Pool connection is now maintained even when mining is paused or not active
  - Pool status is visible in dashboard regardless of mining state
  - Connection is automatically re-established after pool configuration changes
  - Better user experience: users can see pool connection status without starting mining

## [2.21.0] - 2025-01-27

### Fixed
- **Critical: Syntax Error**: Fixed Python syntax error in GPU mining try-except block
  - Corrected indentation of lines 852-885 to be inside the try block
  - Prevents `SyntaxError: expected 'except' or 'finally' block` on startup
- **Critical: Share Submission Bug #73**: Fixed incorrect `extranonce2` value in share submission
  - `extranonce2` was being read from state instead of using the value from the current iteration
  - Now correctly uses the `extranonce2` value that was used to generate the solution
  - Prevents pool from rejecting valid shares due to mismatched `extranonce2`
- **Critical: Share Submission Bug #74**: Fixed race condition in share submission
  - All solution values (`job_id`, `ntime`, `prev_hash`, `nbits`, `version`) are now captured atomically when solution is found
  - Prevents state changes between solution discovery and submission from causing share rejection
  - Values are stored in `solution_*` variables before submission
- **Critical: Share Submission Bug #75**: Fixed inconsistent `ntime` usage
  - `ntime` was being read from state twice (once for logging, once for submission)
  - Now uses a single consistent `solution_ntime` value captured atomically

### Changed
- **Share Submission**: Refactored share submission logic to capture all solution parameters atomically
  - All values from the iteration where the solution was found are now preserved
  - Prevents race conditions when pool sends new block notifications during submission

## [2.20.0] - 2025-01-27

### Fixed
- **Critical: Bitcoin Block Header Byte-Order**: Fixed incorrect byte-order handling for all block header fields
  - All fields (version, prev_hash, merkle_root, ntime, nbits, nonce) are now correctly converted to little-endian format
  - Bitcoin block headers require little-endian byte order, but pool may send fields in big-endian
  - Added `_hex_to_little_endian()` and `_int_to_little_endian_hex()` helper functions for proper conversion
- **Critical: Hash Byte-Order for Target Comparison**: Fixed hash comparison by converting hashes to little-endian
  - CPU and GPU hashes are now converted to little-endian before target comparison
  - Bitcoin uses little-endian for hash comparison, but `binascii.hexlify()` returns big-endian
- **Critical: Merkle Root Calculation**: Fixed multiple issues with Merkle root computation
  - Removed incorrect byte-reversal that was corrupting the Merkle root
  - Merkle root is now correctly converted to little-endian for block header
  - Merkle root is recalculated dynamically in the mining loop (not just once)
  - Fixed Merkle branch concatenation order (merkle_root + branch_hash)
- **Critical: Dynamic State Updates**: Fixed stale state issues in mining loop
  - `merkle_root`, `target`, `target_int`, `target_difficulty` are now recalculated in every loop iteration
  - `extranonce2` is now generated sequentially (not randomly) and regenerated for each iteration
  - `extranonce2_counter` is reset when a new block is detected
- **Critical: Nonce Formatting**: Fixed nonce byte-order in block headers
  - Nonces are now formatted in little-endian hex format for Bitcoin block headers
  - CPU and GPU nonces are correctly converted using `_int_to_little_endian_hex()`
- **Critical: clean_jobs Flag**: Implemented proper handling of `clean_jobs` flag from pool
  - When `clean_jobs` is True, all nonce counters (CPU, GPU, extranonce2) are reset
  - Ensures proper state reset when pool signals a new job
- **Critical: hash_count Updates**: Fixed hash count not being updated when mining is disabled
  - `hash_count` is now updated in every loop iteration, even when no mining occurs
  - Prevents incorrect hash rate calculations

### Changed
- **Block Header Construction**: Complete refactoring of `_build_block_header()` method
  - All input fields are now converted to little-endian before concatenation
  - Removed incorrect padding/truncation logic that could corrupt fields
  - Added proper validation and conversion for all 6 block header fields
- **Mining Loop**: Major refactoring of the main mining loop
  - Merkle root calculation moved inside the loop for dynamic updates
  - Target calculation moved inside the loop for dynamic updates
  - extranonce2 generation moved inside the loop for sequential coverage
  - Better error handling and state management throughout

### Technical Details
- Added helper functions for Bitcoin byte-order conversion:
  - `_hex_to_little_endian(hex_str, expected_length)`: Converts hex string to little-endian
  - `_int_to_little_endian_hex(value, byte_length)`: Converts integer to little-endian hex
- All block header fields are now properly converted from pool format (may be big-endian) to Bitcoin format (little-endian)
- Hash values are converted to little-endian before comparison with target
- Nonce values are formatted in little-endian before being inserted into block headers

## [2.19.3] - 2025-01-XX

### Changed
- **Logging to Docker Logs**: All logging now goes to stdout/stderr instead of separate log files
  - Logs are now visible in `docker logs` command
  - No separate log files are created
  - Log level configuration in web UI still works, but only affects stdout/stderr output
  - Removed all `FileHandler` usage in favor of `StreamHandler` for better Docker integration

### Removed
- Separate log file output - all logs now go to Docker stdout/stderr

## [2.19.2] - 2025-01-XX

### Fixed
- **SocketIO "Too many packets in payload" error**: Fixed by increasing `Payload.max_decode_packets` from default (16) to 500
  - This is the root cause fix - EngineIO was limiting the number of packets per payload
  - Combined with increased buffer sizes from v2.19.1, this should completely resolve the issue

### Changed
- EngineIO Payload.max_decode_packets is now set to 500 to handle large status payloads with history arrays

## [2.19.1] - 2025-01-XX

### Fixed
- **SocketIO "Too many packets in payload" error**: Further improved by increasing HTTP buffer size to 10MB and adding max_packet_size configuration
- Added `async_mode='threading'` to SocketIO configuration for better compatibility
- Increased buffer size from 1MB to 10MB to handle very large payloads

### Changed
- SocketIO configuration now uses 10MB buffer size (increased from 1MB) to handle larger payloads
- Added explicit `max_packet_size` parameter to SocketIO configuration

## [2.19.0] - 2025-01-XX

### Changed
- **Removed TOML Configuration**: Complete removal of TOML-based configuration system
  - All configuration is now stored exclusively in the SQLite database
  - No more `config/config.toml` files needed or used
  - Configuration is managed entirely via the web UI
  - Environment variables still work as fallbacks for initial setup
- **Simplified CLI**: Removed `--config` parameter (no longer needed)
- **Improved Miner Startup**: Miner can now be started automatically via web UI when wallet address is configured
  - `/api/start` endpoint now initializes and starts the miner if it doesn't exist
  - No container restart required after setting wallet address
- **Enhanced Log Path Handling**: 
  - Log directories are automatically created if they don't exist
  - Relative log paths are converted to absolute paths
  - Better support for Docker volume mounts (e.g., `/app/logs/miner.log`)

### Removed
- `tomli-w` dependency (no longer needed)
- TOML file reading/writing functionality
- `--config` CLI parameter
- `CONFIG_FILE` environment variable support

### Fixed
- Log file path handling now properly creates directories and handles relative paths
- Miner startup now works correctly when wallet is configured via web UI

## [2.18.1] - 2025-01-XX

### Fixed
- **SocketIO "Too many packets in payload" error**: Fixed by increasing HTTP buffer size to 1MB and configuring SocketIO with proper timeout settings
- Added error handler for SocketIO to gracefully handle connection errors
- Fixed python-engineio version constraint to prevent compatibility issues

### Changed
- SocketIO configuration now uses increased buffer size (1MB instead of 100KB) to handle larger payloads
- Disabled SocketIO's own logging to reduce noise in container logs
- Improved SocketIO connection stability with optimized ping timeout and interval settings

## [2.18.0] - 2025-01-XX

### Added
- **Verbose Logging System**: Comprehensive logging system with configurable log levels
  - Logging level can be set in web UI under Settings (DEBUG, INFO, WARNING, ERROR)
  - Logging level persists to database and config file
  - Dynamic logging level updates at runtime without restart
  - Extensive DEBUG-level logging throughout the miner for detailed troubleshooting
  - Log file path configurable via web UI
- **Enhanced Debug Logging**: Added detailed DEBUG logs for:
  - GPU miner initialization (CUDA/OpenCL)
  - Pool connection and subscription details
  - Mining notification parameters
  - Coinbase and merkle root calculations
  - GPU and CPU mining operations
  - Hash computations and hash rate statistics
  - Block solutions and pool submissions

### Changed
- Logging configuration now persists to database alongside other settings
- Root logger, SatoshiRig logger, web logger, and miner logger all respect the configured log level

## [2.17.3] - 2025-11-13

### Fixed
- CPU mining now works independently of GPU mining status
- Changed `elif cpu_mining_enabled` to `if cpu_mining_enabled` to allow parallel CPU and GPU mining
- When both CPU and GPU mining are enabled, both run in parallel and the better hash is used
- Improved error handling in mining loop with better logging for debugging

## [2.17.2] - 2025-11-13

### Changed
- Wallet address is now stored and loaded exclusively from database, never from config.toml
- Wallet address is automatically removed from config.toml when saving configuration
- Wallet address from config.toml is ignored on load (only database value is used)

### Fixed
- Wallet address is now always loaded from database, even if empty
- Miner correctly uses wallet address set by user in web UI settings

## [2.17.1] - 2025-11-13

### Fixed
- Empty wallet address strings in database no longer overwrite valid addresses from config.toml
- Mining toggles (cpu_mining_enabled, gpu_mining_enabled) now have proper defaults when not present in database
- Miner now starts correctly when wallet address is configured in config.toml but database contains empty string

## [2.17.0] - 2025-11-13

### Added
- SQLite-backed state database (`data/state.db`, override via `STATE_DB`) storing wallet, pool/network/compute configuration, and mining statistics
- Web UI and CLI now load defaults from the database when present; configuration updates and statistics are written back automatically
- Legacy JSON statistics files are migrated transparently on first run

### Changed
- Removed usage of `STATS_FILE`; Docker examples mount the `data/` directory for persistent state
- Documentation updated to reflect the new persistence layer

## [2.16.1] - 2025-11-13

### Fixed
- CLI now keeps the web dashboard running even when no wallet address is configured, allowing the address to be entered via the UI before restarting
- Documentation updated to reflect the new configuration workflow and removal of wallet-related environment variables

## [2.16.0] - 2025-11-13

### Changed
- Wallet address is now stored in `config.toml` (`[wallet].address`) and surfaced via the web UI; environment variables / Docker overrides are no longer required
- CLI now reads the wallet from configuration and persists CLI-provided addresses back to the config file
- Docker Compose example no longer injects `WALLET_ADDRESS`

### Fixed
- Wallet address remains available after container restarts (no reset to empty field)

## [2.15.0] - 2025-11-13

### Changed
- GPU backend dropdown now strictly controls the GPU runtime (CUDA/OpenCL); CPU mining is governed solely by the CPU toggle
- Web UI remembers the preferred GPU backend and reapplies it automatically when GPU mining is re-enabled
- Configuration persistence now retains the previous GPU backend even when GPU mining is disabled

### Fixed
- CPU mining toggle now correctly persists its state instead of being forced back to enabled
- Prevented the UI from displaying `cpu` as backend and clarified that backend selection applies only to GPU mining

## [2.14.0] - 2024-12-XX

### Added
- **Persistent Statistics**: Statistics are now automatically saved to disk and persist across Docker container restarts
  - Statistics file location: `/app/data/statistics.json` (configurable via `STATS_FILE` environment variable)
  - Auto-saves every 10 status updates and on shutdown
  - Preserves: total_hashes, peak_hash_rate, shares_submitted, shares_accepted, shares_rejected
- **Improved CPU/GPU Mining Control**: 
  - CPU mining toggle now automatically manages backend selection
  - GPU backend dropdown only shows CUDA/OpenCL (CPU is controlled via toggle)
  - Both CPU and GPU mining can be enabled simultaneously
  - Backend selection is intelligently managed based on toggle states

### Changed
- **Compute Configuration UI**: 
  - Backend dropdown renamed to "GPU Backend" and only shows CUDA/OpenCL options
  - CPU mining is now exclusively controlled via the CPU Mining toggle
  - Improved tooltips and labels for better user understanding
- **Statistics Persistence**: Statistics are now loaded from persistent storage on startup and merged with new session data

### Fixed
- **Wallet Address Saving**: Wallet address is now properly saved when using the "Save Configuration" button
- **Backend Logic**: Fixed redundant CPU option in backend dropdown - CPU is now only controlled via toggle

### Technical Details
- Added `stats_persistence.py` module for handling persistent statistics storage
- Implemented thread-safe file operations with atomic writes
- Added shutdown handlers to ensure statistics are saved on container stop
- Updated Docker Compose configuration to include persistent data volume

## [2.13.0] - Previous Release

### Features
- GPU support with CUDA and OpenCL backends
- Web dashboard with real-time monitoring
- Docker support with NVIDIA Container Toolkit
- Configuration via TOML files and environment variables

---

[2.19.3]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.19.3
[2.19.2]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.19.2
[2.19.1]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.19.1
[2.19.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.19.0
[2.18.1]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.18.1
[2.18.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.18.0
[2.17.3]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.3
[2.17.2]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.2
[2.17.1]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.1
[2.17.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.0
[2.16.1]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.16.1
[2.16.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.16.0
[2.15.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.15.0
[2.14.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.14.0
[2.13.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.13.0

