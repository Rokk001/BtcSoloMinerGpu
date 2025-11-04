import os
import tomllib


DEFAULT_CONFIG_PATHS = [
    os.environ.get("CONFIG_FILE") ,
    os.path.join(os.getcwd() , "config" , "config.toml") ,
    os.path.join(os.path.dirname(os.path.dirname(__file__)) , ".." , "config" , "config.toml") ,
]


def _first_existing_path(paths) :
    for p in paths :
        if p and os.path.isfile(os.path.abspath(p)) :
            return os.path.abspath(p)
    return None


def load_config() :
    cfg_path = _first_existing_path(DEFAULT_CONFIG_PATHS)
    if not cfg_path :
        return {
            "pool": {"host": "solo.ckpool.org" , "port": 3333} ,
            "network": {
                "source": os.environ.get("BLOCK_SOURCE" , "web") ,  # web | local
                "latest_block_url": "https://blockchain.info/latestblock" ,
                "request_timeout_secs": 15 ,
                "rpc_url": os.environ.get("BITCOIN_RPC_URL" , "http://127.0.0.1:8332") ,
                "rpc_user": os.environ.get("BITCOIN_RPC_USER" , "") ,
                "rpc_password": os.environ.get("BITCOIN_RPC_PASSWORD" , "")
            } ,
            "logging": {"file": "miner.log" , "level": "INFO"} ,
            "miner": {
                "restart_delay_secs": 2 ,
                "subscribe_thread_start_delay_secs": 4 ,
                "hash_log_prefix_zeros": 7
            } ,
            "compute": {"backend": os.environ.get("COMPUTE_BACKEND" , "cpu") , "gpu_device": int(os.environ.get("GPU_DEVICE" , "0"))}
        }
    with open(cfg_path , "rb") as f :
        cfg = tomllib.load(f)
    compute = cfg.get("compute" , {})
    if "backend" not in compute :
        compute["backend"] = os.environ.get("COMPUTE_BACKEND" , "cpu")
    if "gpu_device" not in compute :
        compute["gpu_device"] = int(os.environ.get("GPU_DEVICE" , "0"))
    cfg["compute"] = compute
    return cfg


