import os
import sys
import argparse
import logging
import threading
from signal import SIGINT , signal

from .config import load_config
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

    wallet = args.wallet or os.environ.get("WALLET_ADDRESS")
    if not wallet :
        print("Missing wallet address. Provide with --wallet <ADDRESS> or WALLET_ADDRESS env var.")
        sys.exit(2)

    logging.basicConfig(
        level = getattr(logging , cfg.get("logging" , {}).get("level" , "INFO").upper() , logging.INFO) ,
        filename = cfg.get("logging" , {}).get("file" , None) ,
        format = '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger("SatoshiRig")

    signal(SIGINT , _handle_sigint)

    if WEB_AVAILABLE and not args.no_web :
        web_port = args.web_port or int(os.environ.get("WEB_PORT" , "5000"))
        from .web.server import update_status
        update_status("wallet_address" , wallet)
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

    pool = PoolClient(cfg["pool"]["host"] , int(cfg["pool"]["port"]))
    miner = Miner(wallet , cfg , pool , STATE , logger)
    miner.start()


