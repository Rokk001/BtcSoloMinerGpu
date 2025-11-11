import json
import logging
import socket
import time
from typing import Tuple , List


class PoolClient :
    def __init__(self , host: str , port: int , timeout: int = 30) :
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.logger = logging.getLogger("SatoshiRig.pool_client")

    def connect(self) :
        max_retries = 3
        retry_delay = 2  # seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self.sock = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host , self.port))
                return  # Success
            except (socket.error, OSError, ConnectionError) as e:
                last_error = e
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                
                if attempt < max_retries - 1:
                    self.logger.warning(f"Failed to connect to {self.host}:{self.port} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to connect to {self.host}:{self.port} after {max_retries} attempts: {e}")
        
        raise last_error

    def subscribe(self) -> Tuple[str , str , int] :
        if self.sock is None:
            raise RuntimeError("Socket not connected. Call connect() first.")
        try:
            self.sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
            data = self.sock.recv(1024)
            if not data:
                raise ConnectionError("Connection closed by server during subscribe")
            lines = data.decode('utf-8', errors='replace').split('\n')
            response = json.loads(lines[0])
            subscription_details , extranonce1 , extranonce2_size = response['result']
            return subscription_details , extranonce1 , int(extranonce2_size)
        except (socket.error, OSError, ConnectionError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Subscribe failed: {e}")
            raise

    def authorize(self , wallet_address: str) :
        if self.sock is None:
            raise RuntimeError("Socket not connected. Call connect() first.")
        try:
            authorize_msg = json.dumps({
                "params": [wallet_address , "password"] ,
                "id": 2 ,
                "method": "mining.authorize"
            }).encode('utf-8') + b"\n"
            self.sock.sendall(authorize_msg)
        except (socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"Authorize failed: {e}")
            raise

    def read_notify(self) -> list:
        if self.sock is None:
            raise RuntimeError("Socket not connected. Call connect() first.")
        # Robust line-buffered read with simple framing by newlines
        # Keep reading until we see at least one mining.notify message
        # and we have consumed a line ending.
        buffer = bytearray()
        messages: list[str] = []
        self.sock.settimeout(self.timeout)
        max_buffer_size = 1024 * 1024  # 1MB max buffer size to prevent memory leak
        max_iterations = 1000  # Prevent infinite loop
        iteration_count = 0
        try:
            while iteration_count < max_iterations:
                iteration_count += 1
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed by server during read_notify")
                buffer.extend(chunk)
                
                # Prevent buffer from growing unbounded
                if len(buffer) > max_buffer_size:
                    self.logger.warning(f"Buffer size exceeded {max_buffer_size} bytes, truncating")
                    buffer = buffer[-max_buffer_size:]  # Keep only last 1MB
                
                while True:
                    try:
                        newline_index = buffer.index(10)  # '\n'
                    except ValueError:
                        break
                    # Use 'replace' instead of 'ignore' to preserve data integrity
                    line = buffer[:newline_index].decode('utf-8', errors='replace').strip()
                    del buffer[:newline_index + 1]
                    if line:
                        messages.append(line)
                # Stop once we have at least one notify message
                if messages and any('mining.notify' in m for m in messages):
                    break
            else:
                # Loop exhausted without finding mining.notify
                if not messages:
                    self.logger.warning("read_notify: No messages received after max iterations")
                    return []
                self.logger.warning(f"read_notify: Max iterations reached, returning {len(messages)} messages")
        except (socket.timeout, socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"read_notify failed: {e}")
            raise
        responses = []
        for m in messages:
            try:
                obj = json.loads(m)
                if 'mining.notify' in m:
                    responses.append(obj)
            except json.JSONDecodeError as e:
                self.logger.debug(f"Failed to parse message: {m[:100]}... Error: {e}")
                continue
        return responses

    def submit(self , wallet_address: str , job_id: str , extranonce2: str , ntime: str , nonce_hex: str) -> bytes:
        if self.sock is None:
            raise RuntimeError("Socket not connected. Call connect() first.")
        try:
            payload = json.dumps({
                "params": [wallet_address , job_id , extranonce2 , ntime , nonce_hex] ,
                "id": 1 ,
                "method": "mining.submit"
            }).encode('utf-8') + b"\n"
            self.sock.sendall(payload)
            # Set timeout before recv to prevent blocking indefinitely
            self.sock.settimeout(self.timeout)
            response = self.sock.recv(1024)
            if not response:
                raise ConnectionError("Connection closed by server during submit")
            return response
        except (socket.timeout, socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"Submit failed: {e}")
            raise

    def close(self):
        """Close the socket connection"""
        if self.sock:
            try:
                self.sock.close()
            except (socket.error, OSError) as e:
                self.logger.debug(f"Error closing socket: {e}")
            finally:
                self.sock = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures socket is closed"""
        self.close()
        return False


