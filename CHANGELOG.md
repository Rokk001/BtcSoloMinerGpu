# Changelog

All notable changes to SatoshiRig will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[2.17.2]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.2
[2.17.1]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.1
[2.17.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.17.0
[2.16.1]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.16.1
[2.16.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.16.0
[2.15.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.15.0
[2.14.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.14.0
[2.13.0]: https://github.com/Rokk001/SatoshiRig/releases/tag/v2.13.0

