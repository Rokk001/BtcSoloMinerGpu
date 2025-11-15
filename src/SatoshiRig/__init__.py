
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Log to stdout/stderr for Docker logs
)

__all__ = [
    "cli",
    "miner",
    "config",
]


