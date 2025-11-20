import os
import sys
import argparse
import logging
import threading
import time
import atexit
from signal import SIGINT , signal

from .config import load_config, save_config
from .clients.pool_client import PoolClient
from .core.state import MinerState
from .core.miner import Miner
from .utils.logging_utils import _vlog

try :
    from .web import start_web_server
    WEB_AVAILABLE = True
except ImportError :
    WEB_AVAILABLE = False


STATE = MinerState()
_logger = logging.getLogger("SatoshiRig.cli")
_verbose_logging = True  # Always enable verbose logging for CLI


def _handle_sigint(signal_received , frame) :
    _vlog(_logger, _verbose_logging, f"cli._handle_sigint: START signal_received={signal_received}, frame={frame is not None}")
    _vlog(_logger, _verbose_logging, "cli._handle_sigint: setting STATE.shutdown_flag=True")
    STATE.shutdown_flag = True
    _vlog(_logger, _verbose_logging, "cli._handle_sigint: printing termination message")
    print("Terminating miner, please wait…")
    # Save statistics on shutdown
    _vlog(_logger, _verbose_logging, "cli._handle_sigint: attempting to save statistics")
    try:
        from .web.status import save_statistics_now
        _vlog(_logger, _verbose_logging, "cli._handle_sigint: calling save_statistics_now()")
        save_statistics_now()
        _vlog(_logger, _verbose_logging, "cli._handle_sigint: save_statistics_now() completed")
    except Exception as e:
        _vlog(_logger, _verbose_logging, f"cli._handle_sigint: exception saving statistics: {type(e).__name__}: {e}")
        pass  # Ignore errors during shutdown
    _vlog(_logger, _verbose_logging, "cli._handle_sigint: END")


def main() :
    # Add INFO-level logging immediately (not just _vlog)
    import logging
    logger = logging.getLogger("SatoshiRig.cli")
    logger.info("=" * 80)
    logger.info("CLI main() called")
    logger.info("=" * 80)
    
    _vlog(_logger, _verbose_logging, "cli.main: START")
    _vlog(_logger, _verbose_logging, "cli.main: creating ArgumentParser")
    parser = argparse.ArgumentParser(prog = "satoshirig")
    _vlog(_logger, _verbose_logging, "cli.main: adding arguments")
    parser.add_argument("--wallet" , "-w" , required = False , help = "BTC wallet address")
    parser.add_argument("--backend" , required = False , choices = ["cpu" , "cuda" , "opencl"])
    parser.add_argument("--gpu" , type = int , required = False , help = "GPU device index")
    parser.add_argument("--web-port" , type = int , default = 5000 , help = "Web dashboard port (default: 5000)")
    parser.add_argument("--no-web" , action = "store_true" , help = "Disable web dashboard")
    _vlog(_logger, _verbose_logging, "cli.main: parsing arguments")
    args = parser.parse_args()
    
    # Log CLI arguments at INFO level
    logger.info(f"CLI arguments: wallet={'set' if args.wallet else 'not set'}, backend={args.backend}, gpu={args.gpu}, web_port={args.web_port}, no_web={args.no_web}")
    _vlog(_logger, _verbose_logging, f"cli.main: args parsed, wallet={args.wallet is not None}, backend={args.backend}, gpu={args.gpu}, web_port={args.web_port}, no_web={args.no_web}")

    _vlog(_logger, _verbose_logging, f"cli.main: checking args.backend: {args.backend is not None}")
    if args.backend :
        _vlog(_logger, _verbose_logging, f"cli.main: setting COMPUTE_BACKEND={args.backend}")
        os.environ["COMPUTE_BACKEND"] = args.backend
    _vlog(_logger, _verbose_logging, f"cli.main: checking args.gpu: {args.gpu is not None}")
    if args.gpu is not None :
        _vlog(_logger, _verbose_logging, f"cli.main: setting GPU_DEVICE={args.gpu}")
        os.environ["GPU_DEVICE"] = str(args.gpu)

    _vlog(_logger, _verbose_logging, "cli.main: calling load_config()")
    cfg = load_config()
    logger.info(f"Config loaded: wallet={'configured' if cfg.get('wallet', {}).get('address') else 'NOT configured'}")
    _vlog(_logger, _verbose_logging, f"cli.main: config loaded, keys={list(cfg.keys())}")

    _vlog(_logger, _verbose_logging, f"cli.main: getting wallet, args.wallet={args.wallet is not None}, cfg wallet={cfg.get('wallet', {}).get('address') is not None}")
    wallet_raw = args.wallet or cfg.get("wallet", {}).get("address")
    
    # If wallet is still empty, try loading directly from database
    if not wallet_raw or not wallet_raw.strip():
        _vlog(_logger, _verbose_logging, "cli.main: wallet not found in config, loading directly from database")
        logger.info("Wallet not found in config, loading directly from database...")
        from .db import get_value
        wallet_raw = get_value("settings", "wallet_address")
        _vlog(_logger, _verbose_logging, f"cli.main: wallet from DB={'present' if wallet_raw else 'None'}, length={len(wallet_raw) if wallet_raw else 0}")
        # Update cfg with wallet from DB if found
        if wallet_raw and wallet_raw.strip():
            cfg.setdefault("wallet", {})["address"] = wallet_raw.strip()
            logger.info(f"Wallet loaded from database: {wallet_raw[:10]}...{wallet_raw[-10:]}")
    
    _vlog(_logger, _verbose_logging, f"cli.main: wallet_raw={'present' if wallet_raw else 'None'}, length={len(wallet_raw) if wallet_raw else 0}")
    wallet = wallet_raw.strip() if wallet_raw else None
    _vlog(_logger, _verbose_logging, f"cli.main: wallet={'present' if wallet else 'None'}, length={len(wallet) if wallet else 0}")
    
    _vlog(_logger, _verbose_logging, f"cli.main: checking wallet: {wallet is not None}")
    if wallet:
        _vlog(_logger, _verbose_logging, "cli.main: validating wallet address format")
        # Validate wallet address format
        wallet_len = len(wallet)
        _vlog(_logger, _verbose_logging, f"cli.main: wallet length={wallet_len}, checking 26 <= {wallet_len} <= 62")
        if wallet_len < 26 or wallet_len > 62:
            _vlog(_logger, _verbose_logging, f"cli.main: invalid wallet length, exiting with code 2")
            print(f"Error: Invalid wallet address length. Bitcoin addresses are 26-62 characters long.")
            sys.exit(2)
        _vlog(_logger, _verbose_logging, "cli.main: checking wallet characters")
        valid_chars = all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" for c in wallet)
        _vlog(_logger, _verbose_logging, f"cli.main: valid_chars={valid_chars}")
        if not valid_chars:
            _vlog(_logger, _verbose_logging, "cli.main: invalid wallet characters, exiting with code 2")
            print(f"Error: Invalid wallet address format. Address contains invalid characters.")
            sys.exit(2)
        _vlog(_logger, _verbose_logging, "cli.main: wallet validation passed")
    else:
        _vlog(_logger, _verbose_logging, "cli.main: wallet is None, checking WEB_AVAILABLE and args.no_web")
        if WEB_AVAILABLE and not args.no_web:
            _vlog(_logger, _verbose_logging, "cli.main: setting up logging for web mode")
            # Setup logging to stdout/stderr for Docker logs
            log_level = getattr(logging, cfg.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
            _vlog(_logger, _verbose_logging, f"cli.main: log_level={log_level}")
            logging.basicConfig(
                level = log_level,
                format = '%(asctime)s %(levelname)s %(name)s %(message)s',
                handlers=[logging.StreamHandler()]  # Log to stdout/stderr for Docker logs
            )
            _vlog(_logger, _verbose_logging, "cli.main: logging configured")
            logger = logging.getLogger("SatoshiRig")
            _vlog(_logger, _verbose_logging, "cli.main: logger created")
            _vlog(_logger, _verbose_logging, "cli.main: registering SIGINT handler")
            signal(SIGINT , _handle_sigint)

            web_port = args.web_port or int(os.environ.get("WEB_PORT" , "5000"))
            _vlog(_logger, _verbose_logging, f"cli.main: web_port={web_port}")
            _vlog(_logger, _verbose_logging, "cli.main: importing web server functions")
            from .web.server import update_status , set_miner_state , set_config , set_miner
            _vlog(_logger, _verbose_logging, "cli.main: updating status")
            update_status("wallet_address" , "")
            update_status("running", False)
            _vlog(_logger, _verbose_logging, "cli.main: setting miner state and config")
            set_miner_state(STATE)
            set_miner(None)
            set_config(cfg)
            _vlog(_logger, _verbose_logging, f"cli.main: starting web server thread on 0.0.0.0:{web_port}")
            web_thread = threading.Thread(target = start_web_server , args = ("0.0.0.0" , web_port) , daemon = True)
            _vlog(_logger, _verbose_logging, "cli.main: starting web thread")
            web_thread.start()
            _vlog(_logger, _verbose_logging, "cli.main: web thread started")
            logger.warning("Wallet address not configured. Open the web dashboard, set the wallet address, then restart the miner.")
            _vlog(_logger, _verbose_logging, "cli.main: entering wait loop")
            try:
                while True:
                    _vlog(_logger, _verbose_logging, "cli.main: sleeping 5 seconds")
                    time.sleep(5)
            except KeyboardInterrupt:
                _vlog(_logger, _verbose_logging, "cli.main: KeyboardInterrupt received")
                pass
            _vlog(_logger, _verbose_logging, "cli.main: returning (no wallet)")
            return
        else:
            _vlog(_logger, _verbose_logging, "cli.main: no wallet and no web, exiting with code 2")
            print("Missing wallet address. Provide with --wallet <ADDRESS> or set it in the web dashboard.")
            sys.exit(2)

    # Log wallet status before continuing
    if wallet:
        logger.info(f"Wallet found: {wallet[:10]}...{wallet[-10:]}")
        logger.info("Mining will start automatically")
    else:
        logger.warning("No wallet configured - only Web Dashboard will start")
        logger.warning("Configure wallet in Web Dashboard and start mining manually")
    
    _vlog(_logger, _verbose_logging, "cli.main: setting up logging for mining mode")
    log_level = getattr(logging , cfg.get("logging" , {}).get("level" , "INFO").upper() , logging.INFO)
    _vlog(_logger, _verbose_logging, f"cli.main: log_level={log_level}")
    logging.basicConfig(
        level = log_level ,
        format = '%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[logging.StreamHandler()]  # Log to stdout/stderr for Docker logs
    )
    _vlog(_logger, _verbose_logging, "cli.main: logging configured")
    logger = logging.getLogger("SatoshiRig")
    _vlog(_logger, _verbose_logging, "cli.main: logger created")

    _vlog(_logger, _verbose_logging, "cli.main: registering SIGINT handler")
    signal(SIGINT , _handle_sigint)

    _vlog(_logger, _verbose_logging, f"cli.main: creating PoolClient host={cfg['pool']['host']}, port={cfg['pool']['port']}")
    pool = PoolClient(cfg["pool"]["host"] , int(cfg["pool"]["port"]))
    _vlog(_logger, _verbose_logging, "cli.main: PoolClient created")
    _vlog(_logger, _verbose_logging, f"cli.main: creating Miner wallet={wallet[:10]}..., cfg keys={list(cfg.keys())}")
    miner = Miner(wallet , cfg , pool , STATE , logger)
    _vlog(_logger, _verbose_logging, "cli.main: Miner created")
    
    _vlog(_logger, _verbose_logging, f"cli.main: checking WEB_AVAILABLE and args.no_web: WEB_AVAILABLE={WEB_AVAILABLE}, args.no_web={args.no_web}")
    if WEB_AVAILABLE and not args.no_web :
        _vlog(_logger, _verbose_logging, "cli.main: setting up web dashboard")
        web_port = args.web_port or int(os.environ.get("WEB_PORT" , "5000"))
        _vlog(_logger, _verbose_logging, f"cli.main: web_port={web_port}")
        _vlog(_logger, _verbose_logging, "cli.main: importing web server functions")
        from .web.server import update_status , set_miner_state , set_config , set_miner
        from .web.status import save_statistics_now
        _vlog(_logger, _verbose_logging, "cli.main: updating status")
        update_status("wallet_address" , wallet)
        _vlog(_logger, _verbose_logging, f"cli.main: setting wallet in cfg: {wallet[:10]}...")
        cfg.setdefault("wallet", {})["address"] = wallet
        # Set miner state reference for web API control
        _vlog(_logger, _verbose_logging, "cli.main: setting miner state")
        set_miner_state(STATE)
        # Set miner instance reference for dynamic config updates
        _vlog(_logger, _verbose_logging, "cli.main: setting miner instance")
        set_miner(miner)
        # Set configuration reference for web UI (sanitized - no sensitive data)
        _vlog(_logger, _verbose_logging, "cli.main: setting config")
        set_config(cfg)
        _vlog(_logger, _verbose_logging, "cli.main: attempting to persist config to DB")
        try:
            from .config import persist_config_to_db
            persist_config_to_db(cfg)
            _vlog(_logger, _verbose_logging, "cli.main: config persisted to DB")
        except Exception as exc:
            _vlog(_logger, _verbose_logging, f"cli.main: exception persisting config: {type(exc).__name__}: {exc}")
            logger.warning("Failed to persist wallet to database: %s", exc)
        # Determine blockchain explorer URL from config
        _vlog(_logger, _verbose_logging, "cli.main: determining explorer URL")
        network_config = cfg.get("network" , {})
        _vlog(_logger, _verbose_logging, f"cli.main: network_config source={network_config.get('source')}")
        if network_config.get("source") == "local" :
            # For local RPC, use blockchain.info as default
            explorer_base = "https://blockchain.info"
            _vlog(_logger, _verbose_logging, f"cli.main: using default explorer_base={explorer_base}")
        else :
            # Extract base URL from latest_block_url (e.g., https://blockchain.info/latestblock -> https://blockchain.info)
            latest_block_url = network_config.get("latest_block_url" , "https://blockchain.info/latestblock")
            _vlog(_logger, _verbose_logging, f"cli.main: latest_block_url={latest_block_url}")
            explorer_base = latest_block_url.rsplit("/" , 1)[0]
            _vlog(_logger, _verbose_logging, f"cli.main: explorer_base={explorer_base}")
        explorer_url = f"{explorer_base}/address/{wallet}"
        _vlog(_logger, _verbose_logging, f"cli.main: explorer_url={explorer_url}")
        update_status("explorer_url" , explorer_url)
        
        # Connect to pool automatically (without starting mining) if wallet is configured
        _vlog(_logger, _verbose_logging, f"cli.main: checking wallet for pool connection: {wallet is not None}")
        if wallet:
            _vlog(_logger, _verbose_logging, "cli.main: attempting to connect to pool")
            try:
                import threading
                _vlog(_logger, _verbose_logging, "cli.main: creating connect thread")
                connect_thread = threading.Thread(target=miner.connect_to_pool_only, daemon=True)
                _vlog(_logger, _verbose_logging, "cli.main: starting connect thread")
                connect_thread.start()
                _vlog(_logger, _verbose_logging, "cli.main: connect thread started")
                logger.info("Pool connection initiated (mining not started)")
            except Exception as connect_error:
                _vlog(_logger, _verbose_logging, f"cli.main: exception connecting to pool: {type(connect_error).__name__}: {connect_error}")
                logger.warning(f"Failed to connect to pool on startup: {connect_error}")
        _vlog(_logger, _verbose_logging, f"cli.main: creating web server thread on 0.0.0.0:{web_port}")
        web_thread = threading.Thread(target = start_web_server , args = ("0.0.0.0" , web_port) , daemon = True)
        _vlog(_logger, _verbose_logging, "cli.main: starting web thread")
        web_thread.start()
        _vlog(_logger, _verbose_logging, "cli.main: web thread started")
        logger.info("Web dashboard started on port %s" , web_port)
        
        # Register shutdown handler to save statistics
        _vlog(_logger, _verbose_logging, "cli.main: registering atexit handler for save_statistics_now")
        atexit.register(save_statistics_now)
        _vlog(_logger, _verbose_logging, "cli.main: atexit handler registered")

    _vlog(_logger, _verbose_logging, "cli.main: entering try block to start miner")
    try:
        _vlog(_logger, _verbose_logging, "cli.main: calling miner.start()")
        logger.info("Starting miner...")
        miner.start()
        _vlog(_logger, _verbose_logging, "cli.main: miner.start() completed")
    except Exception as e:
        _vlog(_logger, _verbose_logging, f"cli.main: exception in miner.start(): {type(e).__name__}: {e}")
        logger.error(f"Failed to start miner: {e}", exc_info=True)
        _vlog(_logger, _verbose_logging, f"cli.main: checking WEB_AVAILABLE and args.no_web: WEB_AVAILABLE={WEB_AVAILABLE}, args.no_web={args.no_web}")
        if WEB_AVAILABLE and not args.no_web:
            _vlog(_logger, _verbose_logging, "cli.main: updating status for failed start")
            from .web.server import update_status, update_pool_status
            update_status("running", False)
            update_pool_status(False)
        # If web server is running, keep it alive so user can fix the issue
        if WEB_AVAILABLE and not args.no_web:
            _vlog(_logger, _verbose_logging, "cli.main: web available, entering wait loop")
            logger.warning("Miner failed to start, but web dashboard is still available for configuration.")
            try:
                while True:
                    _vlog(_logger, _verbose_logging, "cli.main: sleeping 5 seconds in wait loop")
                    time.sleep(5)
            except KeyboardInterrupt:
                _vlog(_logger, _verbose_logging, "cli.main: KeyboardInterrupt in wait loop")
                pass
        else:
            _vlog(_logger, _verbose_logging, "cli.main: no web server, re-raising exception")
            raise  # Re-raise if no web server
    finally:
        _vlog(_logger, _verbose_logging, "cli.main: entering finally block")
        # Cleanup: Close pool connection
        _vlog(_logger, _verbose_logging, "cli.main: attempting to close pool connection")
        try:
            pool.close()
            _vlog(_logger, _verbose_logging, "cli.main: pool.close() completed")
            logger.debug("Pool connection closed")
        except Exception as e:
            _vlog(_logger, _verbose_logging, f"cli.main: exception closing pool: {type(e).__name__}: {e}")
            logger.debug(f"Error closing pool connection: {e}")
        # Save statistics on exit
        _vlog(_logger, _verbose_logging, f"cli.main: checking if should save statistics: WEB_AVAILABLE={WEB_AVAILABLE}, args.no_web={args.no_web}")
        if WEB_AVAILABLE and not args.no_web:
            _vlog(_logger, _verbose_logging, "cli.main: attempting to save statistics")
            try:
                from .web.status import save_statistics_now
                save_statistics_now()
                _vlog(_logger, _verbose_logging, "cli.main: save_statistics_now() completed")
            except Exception as e:
                _vlog(_logger, _verbose_logging, f"cli.main: exception saving statistics: {type(e).__name__}: {e}")
                logger.debug(f"Error saving statistics on exit: {e}")
        _vlog(_logger, _verbose_logging, "cli.main: finally block completed, END")


