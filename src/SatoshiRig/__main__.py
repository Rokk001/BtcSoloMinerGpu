from .cli import main


import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Log to stdout/stderr for Docker logs
)


if __name__ == "__main__" :
    main()


