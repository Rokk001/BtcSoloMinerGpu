"""Compatibility facade for historical imports.

Prefer using: from SatoshiRig.core.miner import Miner
"""

from .core.miner import Miner  # noqa: F401
import binascii
import hashlib
import json
import logging
import os
import random
import socket
import threading
import time
from datetime import datetime
from typing import Optional

import requests
from colorama import Back , Fore , Style

from dataclasses import dataclass
from .config import load_config
@dataclass
class MinerState :
    shutdown_flag: bool = False
    thread_running_flags: list = None
    local_height: int = 0
    height_to_best_difficulty: dict = None
    updated_prev_hash: str = None
    job_id: str = None
    prev_hash: str = None
    coinbase_part1: str = None
    coinbase_part2: str = None
    merkle_branch: list = None
    version: str = None
    nbits: str = None
    ntime: str = None
    clean_jobs: bool = None
    subscription_details: str = None
    extranonce1: str = None
    extranonce2_size: int = None
    extranonce2: str = None


STATE = MinerState(
    shutdown_flag = False ,
    thread_running_flags = [False] * 2 ,
    local_height = 0 ,
    height_to_best_difficulty = {} ,
)


socket_connection: Optional[socket.socket] = None
CFG = load_config()


def now_time() :
    return datetime.now().time()


def log_info(message) :
    level = getattr(logging , str(CFG.get("logging" , {}).get("level" , "INFO")).upper() , logging.INFO)
    logging.basicConfig(level = level , filename = CFG.get("logging" , {}).get("file" , "miner.log") ,
                        format = '%(asctime)s %(message)s')
    logging.info(message)


def get_current_block_height() :
    response = requests.get(CFG["network"]["latest_block_url"] , timeout = CFG["network"]["request_timeout_secs"]) 
    response.raise_for_status()
    return int(response.json()['height'])


def check_for_shutdown(thread_obj) :
    thread_index = thread_obj.thread_index
    if STATE.shutdown_flag :
        if thread_index != -1 :
            STATE.thread_running_flags[thread_index] = False
            thread_obj.exit = True


class TerminableThread(threading.Thread) :
    def __init__(self , arg , thread_index) :
        super(TerminableThread , self).__init__()
        self.exit = False
        self.arg = arg
        self.thread_index = thread_index

    def run(self) :
        self.thread_handler(self.arg , self.thread_index)

    def thread_handler(self , arg , thread_index) :
        while True :
            check_for_shutdown(self)
            if self.exit :
                break
            STATE.thread_running_flags[thread_index] = True
            try :
                self.thread_handler2(arg)
            except Exception as error :
                log_info("ThreadHandler()")
                log_info(str(error))
                print(Fore.RED , error)
            STATE.thread_running_flags[thread_index] = False
            time.sleep(2)

    def thread_handler2(self , arg) :
        raise NotImplementedError("must implement this method")

    def check_self_shutdown(self) :
        check_for_shutdown(self)

    def request_exit(self) :
        self.exit = True
        STATE.thread_running_flags[self.thread_index] = False


def _build_block_header(prev_hash , merkle_root , ntime , nbits , nonce_hex) :
    return (
        STATE.version + prev_hash + merkle_root + ntime + nbits + nonce_hex +
        '000000800000000000000000000000000000000000000000000000000000000000000000000000000000000080020000'
    )


def mining_loop(thread_obj , wallet_address , restarted = False) :
    global socket_connection
    if restarted :
        log_info('[*] Miner restarted')
        print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.BLUE , '[*] Miner Restarted')
        time.sleep(CFG["miner"]["restart_delay_secs"])

    target = (STATE.nbits[2 :] + '00' * (int(STATE.nbits[:2] , 16) - 3)).zfill(64)
    extranonce2 = hex(random.getrandbits(32))[2 :].zfill(2 * STATE.extranonce2_size)
    STATE.extranonce2 = extranonce2

    coinbase = STATE.coinbase_part1 + STATE.extranonce1 + extranonce2 + STATE.coinbase_part2
    coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()

    merkle_root = coinbase_hash_bin
    for branch_hash in STATE.merkle_branch :
        merkle_root = hashlib.sha256(
            hashlib.sha256(merkle_root + binascii.unhexlify(branch_hash)).digest()
        ).digest()

    merkle_root = binascii.hexlify(merkle_root).decode()
    merkle_root = ''.join([merkle_root[i] + merkle_root[i + 1] for i in range(0 , len(merkle_root) , 2)][: :-1])

    current_height = get_current_block_height()
    STATE.height_to_best_difficulty[current_height + 1] = 0

    reference_diff = int("00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" , 16)

    log_info('[*] Working to solve block with height {}'.format(current_height + 1))
    print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.YELLOW , '[*] Working to solve block with ' , Fore.RED ,
          'height {}'.format(current_height + 1))

    prefix_zeros = '0' * CFG["miner"]["hash_log_prefix_zeros"]

    while True :
        thread_obj.check_self_shutdown()
        if thread_obj.exit :
            break

        if STATE.prev_hash != STATE.updated_prev_hash :
            log_info('[*] New block {} detected on network '.format(STATE.prev_hash))
            print(Fore.YELLOW , '[' , now_time() , ']' , Fore.MAGENTA , '[*] New block {} detected on' , Fore.BLUE ,
                  ' network '.format(STATE.prev_hash))
            log_info('[*] Best difficulty while trying to solve block {} was {}'.format(current_height + 1 ,
                                                                                       STATE.height_to_best_difficulty[current_height + 1]))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.GREEN , '[*] Best difficulty while trying to solve block' ,
                  Fore.WHITE , ' {} ' , Fore.BLUE ,
                  'was {}'.format(current_height + 1 ,
                                  STATE.height_to_best_difficulty[current_height + 1]))
            STATE.updated_prev_hash = STATE.prev_hash
            mining_loop(thread_obj , wallet_address , restarted = True)
            print(Back.YELLOW , Fore.MAGENTA , '[' , now_time() , ']' , Fore.BLUE , 'Miner Restart Now...' ,
                  Style.RESET_ALL)
            continue

        nonce_hex = hex(random.getrandbits(32))[2 :].zfill(8)
        block_header = _build_block_header(STATE.prev_hash , merkle_root , STATE.ntime , STATE.nbits , nonce_hex)
        hash_hex = hashlib.sha256(hashlib.sha256(binascii.unhexlify(block_header)).digest()).digest()
        hash_hex = binascii.hexlify(hash_hex).decode()

        if hash_hex.startswith(prefix_zeros) :
            log_info('[*] New hash: {} for block {}'.format(hash_hex , current_height + 1))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.YELLOW , '[*] New hash:' , Fore.WHITE , ' {} for block' ,
                  Fore.WHITE ,
                  ' {}'.format(hash_hex , current_height + 1))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.BLUE , 'Hash:' , str(hash_hex))
        this_hash_int = int(hash_hex , 16)

        difficulty = reference_diff / this_hash_int

        if STATE.height_to_best_difficulty[current_height + 1] < difficulty :
            STATE.height_to_best_difficulty[current_height + 1] = difficulty

        if hash_hex < target :
            log_info('[*] Block {} solved.'.format(current_height + 1))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.YELLOW , '[*] Block {} solved.'.format(current_height + 1))
            log_info('[*] Block hash: {}'.format(hash_hex))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.YELLOW , '[*] Block hash: {}'.format(hash_hex))
            log_info('[*] Blockheader: {}'.format(block_header))
            print(Fore.YELLOW , '[*] Blockheader: {}'.format(block_header))
            payload = json.dumps({
                "params": [wallet_address , STATE.job_id , STATE.extranonce2 , STATE.ntime , nonce_hex] ,
                "id": 1 ,
                "method": "mining.submit"
            }).encode() + b"\n"
            log_info('[*] Payload: {}'.format(payload))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.BLUE , '[*] Payload:' , Fore.GREEN , ' {}'.format(payload))
            socket_connection.sendall(payload)
            response_bytes = socket_connection.recv(1024)
            log_info('[*] Pool response: {}'.format(response_bytes))
            print(Fore.MAGENTA , '[' , now_time() , ']' , Fore.GREEN , '[*] Pool Response:' , Fore.CYAN ,
                  ' {}'.format(response_bytes))
            return True


def block_listener(thread_obj , wallet_address) :
    global socket_connection
    socket_connection = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
    socket_connection.connect((CFG["pool"]["host"] , int(CFG["pool"]["port"])))
    socket_connection.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
    lines = socket_connection.recv(1024).decode().split('\n')
    response = json.loads(lines[0])
    STATE.subscription_details , STATE.extranonce1 , STATE.extranonce2_size = response['result']
    authorize_msg = json.dumps({
        "params": [wallet_address , "password"] ,
        "id": 2 ,
        "method": "mining.authorize"
    }).encode() + b"\n"
    socket_connection.sendall(authorize_msg)
    response_buffer = b''
    while response_buffer.count(b'\n') < 4 and not (b'mining.notify' in response_buffer) :
        response_buffer += socket_connection.recv(1024)

    responses = [json.loads(res) for res in response_buffer.decode().split('\n') if
                 len(res.strip()) > 0 and 'mining.notify' in res]
    (
        STATE.job_id ,
        STATE.prev_hash ,
        STATE.coinbase_part1 ,
        STATE.coinbase_part2 ,
        STATE.merkle_branch ,
        STATE.version ,
        STATE.nbits ,
        STATE.ntime ,
        STATE.clean_jobs
    ) = responses[0]['params']
    STATE.updated_prev_hash = STATE.prev_hash

    while True :
        thread_obj.check_self_shutdown()
        if thread_obj.exit :
            break
        response_buffer = b''
        while response_buffer.count(b'\n') < 4 and not (b'mining.notify' in response_buffer) :
            response_buffer += socket_connection.recv(1024)
        responses = [json.loads(res) for res in response_buffer.decode().split('\n') if
                     len(res.strip()) > 0 and 'mining.notify' in res]
        if responses[0]['params'][1] != STATE.prev_hash :
            (
                STATE.job_id ,
                STATE.prev_hash ,
                STATE.coinbase_part1 ,
                STATE.coinbase_part2 ,
                STATE.merkle_branch ,
                STATE.version ,
                STATE.nbits ,
                STATE.ntime ,
                STATE.clean_jobs
            ) = responses[0]['params']


class MinerThread(TerminableThread) :
    def __init__(self , wallet_address , arg = None) :
        super(MinerThread , self).__init__(arg , thread_index = 0)
        self.wallet_address = wallet_address

    def thread_handler2(self , arg) :
        self.thread_miner(arg)

    def thread_miner(self , arg) :
        STATE.thread_running_flags[self.thread_index] = True
        check_for_shutdown(self)
        try :
            ret = mining_loop(self , self.wallet_address)
            log_info("[*] Miner returned %s\n\n" % ("true" if ret else "false"))
            print(Fore.LIGHTCYAN_EX , "[*] Miner returned %s\n\n" % ("true" if ret else "false"))
        except Exception as error :
            log_info("[*] Miner()")
            print(Back.WHITE , Fore.MAGENTA , "[" , now_time() , "]" , Fore.BLUE , "[*] Miner()")
            log_info(str(error))
        STATE.thread_running_flags[self.thread_index] = False


class SubscribeThread(TerminableThread) :
    def __init__(self , wallet_address , arg = None) :
        super(SubscribeThread , self).__init__(arg , thread_index = 1)
        self.wallet_address = wallet_address

    def thread_handler2(self , arg) :
        self.thread_new_block(arg)

    def thread_new_block(self , arg) :
        STATE.thread_running_flags[self.thread_index] = True
        check_for_shutdown(self)
        try :
            block_listener(self , self.wallet_address)
        except Exception as error :
            log_info("[*] Subscribe thread()")
            print(Fore.MAGENTA , "[" , now_time() , "]" , Fore.YELLOW , "[*] Subscribe thread()")
            log_info(str(error))
        STATE.thread_running_flags[self.thread_index] = False


def start_mining(wallet_address) :
    subscribe_thread = SubscribeThread(wallet_address , None)
    subscribe_thread.start()
    log_info("[*] Subscribe thread started.")
    print(Fore.MAGENTA , "[" , now_time() , "]" , Fore.GREEN , "[*] Subscribe thread started.")
    time.sleep(CFG["miner"]["subscribe_thread_start_delay_secs"])
    miner_thread = MinerThread(wallet_address , None)
    miner_thread.start()
    log_info("[*] Miner Thread Started")
    print(Fore.MAGENTA , "[" , now_time() , "]" , Fore.GREEN , "[*] Miner Thread Started")


