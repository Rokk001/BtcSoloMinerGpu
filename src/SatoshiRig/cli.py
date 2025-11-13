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

try :
    from .web import start_web_server
    WEB_AVAILABLE = True
except ImportError :
    WEB_AVAILABLE = False


STATE = MinerState()


def _handle_sigint(signal_received , frame) :
    STATE.shutdown_flag = True
    print("Terminating miner, please wait…")
    # Save statistics on shutdown
    try:
        from .web.status import save_statistics_now
        save_statistics_now()
    except Exception:
        pass  # Ignore errors during shutdown


def main() :
    parser = argparse.ArgumentParser(prog = "satoshirig")
    parser.add_argument("--wallet" , "-w" , required = False , help = "BTC wallet address")
    parser.add_argument("--config" , required = False , help = "Path to config.toml")
    parser.add_argument("--backend" , required = False , choices = ["cpu" , "cuda" , "opencl"])
    parser.add_argument("--gpu" , type = int , required = False , help = "GPU device index")
    parser.add_argument("--web-port" , type = int , default = 5000 , help = "Web dashboard port (default: 5000)")
    parser.add_argument("--no-web" , action = "store_true" , help = "Disable web dashboard")
    args = parser.parse_args()

    if args.config :
        os.environ["CONFIG_FILE"] = args.config
    if args.backend :
        os.environ["COMPUTE_BACKEND"] = args.backend
    if args.gpu is not None :
        os.environ["GPU_DEVICE"] = str(args.gpu)

    cfg = load_config()

    wallet_raw = args.wallet or cfg.get("wallet", {}).get("address")
    wallet = wallet_raw.strip() if wallet_raw else None
    
    if wallet:
        # Validate wallet address format
        if len(wallet) < 26 or len(wallet) > 62:
            print(f"Error: Invalid wallet address length. Bitcoin addresses are 26-62 characters long.")
            sys.exit(2)
        if not all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" for c in wallet):
            print(f"Error: Invalid wallet address format. Address contains invalid characters.")
            sys.exit(2)
    else:
        if WEB_AVAILABLE and not args.no_web:
            logging.basicConfig(
                level = getattr(logging , cfg.get("logging" , {}).get("level" , "INFO").upper() , logging.INFO) ,
                filename = cfg.get("logging" , {}).get("file" , None) ,
                format = '%(asctime)s %(levelname)s %(name)s %(message)s'
            )
            logger = logging.getLogger("SatoshiRig")
            signal(SIGINT , _handle_sigint)

            web_port = args.web_port or int(os.environ.get("WEB_PORT" , "5000"))
            from .web.server import update_status , set_miner_state , set_config , set_miner
            update_status("wallet_address" , "")
            update_status("running", False)
            set_miner_state(STATE)
            set_miner(None)
            set_config(cfg)
            web_thread = threading.Thread(target = start_web_server , args = ("0.0.0.0" , web_port) , daemon = True)
            web_thread.start()
            logger.warning("Wallet address not configured. Open the web dashboard, set the wallet address, then restart the miner.")
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                pass
            return
        else:
            print("Missing wallet address. Provide with --wallet <ADDRESS> or set it in config.toml.")
            sys.exit(2)

    logging.basicConfig(
        level = getattr(logging , cfg.get("logging" , {}).get("level" , "INFO").upper() , logging.INFO) ,
        filename = cfg.get("logging" , {}).get("file" , None) ,
        format = '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger("SatoshiRig")

    signal(SIGINT , _handle_sigint)

    pool = PoolClient(cfg["pool"]["host"] , int(cfg["pool"]["port"]))
    miner = Miner(wallet , cfg , pool , STATE , logger)
    
    if WEB_AVAILABLE and not args.no_web :
        web_port = args.web_port or int(os.environ.get("WEB_PORT" , "5000"))
        from .web.server import update_status , set_miner_state , set_config , set_miner
        from .web.status import save_statistics_now
        update_status("wallet_address" , wallet)
        cfg.setdefault("wallet", {})["address"] = wallet
        # Set miner state reference for web API control
        set_miner_state(STATE)
        # Set miner instance reference for dynamic config updates
        set_miner(miner)
        # Set configuration reference for web UI (sanitized - no sensitive data)
        set_config(cfg)
        try:
            save_config(cfg)
        except Exception as exc:
            logger.warning("Failed to persist wallet to config: %s", exc)
        # Determine blockchain explorer URL from config
        network_config = cfg.get("network" , {})
        if network_config.get("source") == "local" :
            # For local RPC, use blockchain.info as default
            explorer_base = "https://blockchain.info"
        else :
            # Extract base URL from latest_block_url (e.g., https://blockchain.info/latestblock -> https://blockchain.info)
            latest_block_url = network_config.get("latest_block_url" , "https://blockchain.info/latestblock")
            explorer_base = latest_block_url.rsplit("/" , 1)[0]
        update_status("explorer_url" , f"{explorer_base}/address/{wallet}")
        web_thread = threading.Thread(target = start_web_server , args = ("0.0.0.0" , web_port) , daemon = True)
        web_thread.start()
        logger.info("Web dashboard started on port %s" , web_port)
        
        # Register shutdown handler to save statistics
        atexit.register(save_statistics_now)

    try:
        miner.start()
    finally:
        # Cleanup: Close pool connection
        try:
            pool.close()
            logger.debug("Pool connection closed")
        except Exception as e:
            logger.debug(f"Error closing pool connection: {e}")
        # Save statistics on exit
        if WEB_AVAILABLE and not args.no_web:
            try:
                from .web.status import save_statistics_now
                save_statistics_now()
            except Exception as e:
                logger.debug(f"Error saving statistics on exit: {e}")


