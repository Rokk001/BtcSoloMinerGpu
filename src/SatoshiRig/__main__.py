from .cli import main


import logging
logging.basicConfig(level=logging.INFO, filename='miner.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__" :
    main()


