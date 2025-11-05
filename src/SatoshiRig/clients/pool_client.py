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
        # Robust line-buffered read with simple framing by newlines
        # Keep reading until we see at least one mining.notify message
        # and we have consumed a line ending.
        buffer = bytearray()
        messages: list[str] = []
        self.sock.settimeout(self.timeout)
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                try:
                    newline_index = buffer.index(10)  # '\n'
                except ValueError:
                    break
                line = buffer[:newline_index].decode(errors='ignore').strip()
                del buffer[:newline_index + 1]
                if line:
                    messages.append(line)
            # Stop once we have at least one notify message
            if any('mining.notify' in m for m in messages):
                break
        responses = []
        for m in messages:
            try:
                obj = json.loads(m)
                if 'mining.notify' in m:
                    responses.append(obj)
            except json.JSONDecodeError:
                continue
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


