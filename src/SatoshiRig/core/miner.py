import binascii
import hashlib
import logging
import random
import time
from datetime import datetime
from typing import Any , Optional

import requests

from ..clients.pool_client import PoolClient
from .state import MinerState

try :
    from ..web import update_status
except ImportError :
    def update_status(key , value) :
        pass


def now_time() :
    return datetime.now().time()


class Miner :
    def __init__(self , wallet_address: str , config: dict , pool_client: PoolClient , state: MinerState , logger: logging.Logger) :
        self.wallet = wallet_address
        self.cfg = config
        self.pool = pool_client
        self.state = state
        self.log = logger

    def _get_current_block_height(self) -> int :
        net = self.cfg["network"]
        source = (net.get("source") or "web").lower()
        if source == "local" :
            # Bitcoin Core JSON-RPC: getblockcount
            payload = {"jsonrpc": "1.0" , "id": "satoshirig" , "method": "getblockcount" , "params": []}
            auth = None
            if net.get("rpc_user") or net.get("rpc_password") :
                auth = (net.get("rpc_user" , "") , net.get("rpc_password" , ""))
            r = requests.post(net.get("rpc_url") , json = payload , auth = auth , timeout = net.get("request_timeout_secs" , 15))
            r.raise_for_status()
            data = r.json()
            return int(data["result"])  # returns block count (height)
        else :
            r = requests.get(net["latest_block_url"] , timeout = net["request_timeout_secs"]) 
            r.raise_for_status()
            return int(r.json()['height'])

    def _build_block_header(self , prev_hash , merkle_root , ntime , nbits , nonce_hex) :
        return (
            self.state.version + prev_hash + merkle_root + ntime + nbits + nonce_hex +
            '000000800000000000000000000000000000000000000000000000000000000000000000000000000000000080020000'
        )

    def start(self) :
        self.pool.connect()
        sub_details , extranonce1 , extranonce2_size = self.pool.subscribe()
        self.state.subscription_details = sub_details
        self.state.extranonce1 = extranonce1
        self.state.extranonce2_size = extranonce2_size
        self.pool.authorize(self.wallet)
        responses = self.pool.read_notify()
        (
            self.state.job_id ,
            self.state.prev_hash ,
            self.state.coinbase_part1 ,
            self.state.coinbase_part2 ,
            self.state.merkle_branch ,
            self.state.version ,
            self.state.nbits ,
            self.state.ntime ,
            self.state.clean_jobs
        ) = responses[0]['params']
        self.state.updated_prev_hash = self.state.prev_hash
        return self._mine_loop()

    def _mine_loop(self) :
        if self.cfg["miner"]["restart_delay_secs"] and self.state.height_to_best_difficulty.get(-1) == -1 :
            time.sleep(self.cfg["miner"]["restart_delay_secs"])

        target = (self.state.nbits[2 :] + '00' * (int(self.state.nbits[:2] , 16) - 3)).zfill(64)
        extranonce2 = hex(random.getrandbits(32))[2 :].zfill(2 * self.state.extranonce2_size)
        self.state.extranonce2 = extranonce2

        coinbase = self.state.coinbase_part1 + self.state.extranonce1 + extranonce2 + self.state.coinbase_part2
        coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()

        merkle_root = coinbase_hash_bin
        for branch_hash in self.state.merkle_branch :
            merkle_root = hashlib.sha256(
                hashlib.sha256(merkle_root + binascii.unhexlify(branch_hash)).digest()
            ).digest()

        merkle_root = binascii.hexlify(merkle_root).decode()
        merkle_root = ''.join([merkle_root[i] + merkle_root[i + 1] for i in range(0 , len(merkle_root) , 2)][: :-1])

        current_height = self._get_current_block_height()
        self.state.height_to_best_difficulty[current_height + 1] = 0

        reference_diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" , 16)

        self.log.info('Mining block height %s' , current_height + 1)
        update_status("current_height" , current_height + 1)

        prefix_zeros = '0' * self.cfg["miner"]["hash_log_prefix_zeros"]
        hash_count = 0
        start_time = time.time()

        while True :
            if self.state.shutdown_flag :
                update_status("running" , False)
                break

            if self.state.prev_hash != self.state.updated_prev_hash :
                self.log.info('New block detected: %s' , self.state.prev_hash)
                self.log.info('Best difficulty for height %s was %s' , current_height + 1 ,
                              self.state.height_to_best_difficulty[current_height + 1])
                self.state.updated_prev_hash = self.state.prev_hash
                self.state.height_to_best_difficulty[-1] = -1
                return self._mine_loop()

            nonce_hex = hex(random.getrandbits(32))[2 :].zfill(8)
            block_header = self._build_block_header(self.state.prev_hash , merkle_root , self.state.ntime , self.state.nbits , nonce_hex)
            hash_hex = hashlib.sha256(hashlib.sha256(binascii.unhexlify(block_header)).digest()).digest()
            hash_hex = binascii.hexlify(hash_hex).decode()
            hash_count += 1

            if hash_hex.startswith(prefix_zeros) :
                self.log.debug('Candidate hash %s at height %s' , hash_hex , current_height + 1)
                update_status("last_hash" , hash_hex)
            this_hash_int = int(hash_hex , 16)

            difficulty = reference_diff / this_hash_int

            if self.state.height_to_best_difficulty[current_height + 1] < difficulty :
                self.state.height_to_best_difficulty[current_height + 1] = difficulty
                update_status("best_difficulty" , difficulty)

            elapsed = time.time() - start_time
            if elapsed > 0 :
                hash_rate = hash_count / elapsed
                update_status("hash_rate" , hash_rate)

            if hash_hex < target :
                self.log.info('Block solved at height %s' , current_height + 1)
                self.log.info('Block hash %s' , hash_hex)
                self.log.debug('Blockheader %s' , block_header)
                ret = self.pool.submit(self.wallet , self.state.job_id , self.state.extranonce2 , self.state.ntime , nonce_hex)
                self.log.info('Pool response %s' , ret)
                return True


