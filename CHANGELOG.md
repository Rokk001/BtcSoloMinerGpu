# Changelog

All notable changes to SatoshiRig will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

