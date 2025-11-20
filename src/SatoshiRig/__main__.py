from .cli import main
import logging
import sys

# Configure basic logging immediately on startup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Log to stdout for Docker logs
)

logger = logging.getLogger("SatoshiRig")
logger.info("=" * 80)
logger.info("SatoshiRig starting...")
logger.info("=" * 80)

if __name__ == "__main__":
    logger.info("Entering main() function")
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main(): {e}", exc_info=True)
        sys.exit(1)


