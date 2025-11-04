import json
import socket
from typing import Tuple , List


class PoolClient :
    def __init__(self , host: str , port: int , timeout: int = 30) :
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) :
        self.sock = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host , self.port))

    def subscribe(self) -> Tuple[str , str , int] :
        assert self.sock is not None
        self.sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
        lines = self.sock.recv(1024).decode().split('\n')
        response = json.loads(lines[0])
        subscription_details , extranonce1 , extranonce2_size = response['result']
        return subscription_details , extranonce1 , int(extranonce2_size)

    def authorize(self , wallet_address: str) :
        assert self.sock is not None
        authorize_msg = json.dumps({
            "params": [wallet_address , "password"] ,
            "id": 2 ,
            "method": "mining.authorize"
        }).encode() + b"\n"
        self.sock.sendall(authorize_msg)

    def read_notify(self) -> list:
        assert self.sock is not None
        response_buffer = b''
        while response_buffer.count(b'\n') < 4 and not (b'mining.notify' in response_buffer) :
            response_buffer += self.sock.recv(1024)
        responses = [json.loads(res) for res in response_buffer.decode().split('\n') if
                     len(res.strip()) > 0 and 'mining.notify' in res]
        return responses

    def submit(self , wallet_address: str , job_id: str , extranonce2: str , ntime: str , nonce_hex: str) -> bytes:
        assert self.sock is not None
        payload = json.dumps({
            "params": [wallet_address , job_id , extranonce2 , ntime , nonce_hex] ,
            "id": 1 ,
            "method": "mining.submit"
        }).encode() + b"\n"
        self.sock.sendall(payload)
        return self.sock.recv(1024)


