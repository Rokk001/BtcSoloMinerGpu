import json
import logging
import socket
import threading
import time
from typing import Tuple , List


class PoolClient :
    def __init__(self , host: str , port: int , timeout: int = 30) :
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.logger = logging.getLogger("SatoshiRig.pool_client")
        self._socket_lock = threading.Lock()  # Lock for thread-safe socket operations

    def connect(self) :
        with self._socket_lock:
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
            
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to connect to pool: unknown error")

    def subscribe(self) -> Tuple[str , str , int] :
        with self._socket_lock:
            if self.sock is None:
                raise RuntimeError("Socket not connected. Call connect() first.")
            try:
                self.sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
            
            # Read response - may need multiple recv() calls for large responses
            # Pool responses can be > 1024 bytes, so we need to read until we get a complete line
            # Pool may also send mining.notify messages immediately after subscribe response
            buffer = bytearray()
            max_buffer_size = 64 * 1024  # 64KB max
            self.sock.settimeout(self.timeout)
            
            # Read until we have at least one complete line
            # Pool may send multiple messages (subscribe response + mining.notify), so we need to read all
            lines_read = 0
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed by server during subscribe")
                buffer.extend(chunk)
                
                # Count how many complete lines we have
                lines_read = buffer.count(b'\n')
                
                # If we have at least one complete line, check if we have the subscribe response
                if lines_read > 0:
                    # Decode and check if we already have the subscribe response
                    temp_lines = buffer.decode('utf-8', errors='replace').split('\n')
                    found_subscribe_response = False
                    for line in temp_lines:
                        if not line.strip():
                            continue
                        try:
                            parsed = json.loads(line)
                            if 'result' in parsed and parsed.get('id') == 1:
                                found_subscribe_response = True
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    # If we found the subscribe response, we can stop reading
                    if found_subscribe_response:
                        break
                
                # Prevent buffer overflow
                if len(buffer) > max_buffer_size:
                    raise RuntimeError(f"Subscribe response too large (>{max_buffer_size} bytes)")
            
            # Decode and parse all lines - pool may send multiple messages
            lines = buffer.decode('utf-8', errors='replace').split('\n')
            
            # Find the subscribe response (should have 'result' field and 'id': 1)
            # Pool may also send mining.notify messages, so we need to find the right one
            response = None
            for line in lines:
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                    # Subscribe response should have 'result' field and 'id' matching our request (1)
                    if 'result' in parsed and parsed.get('id') == 1:
                        response = parsed
                        break
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
            
                if response is None:
                    # Log all received lines for debugging
                    self.logger.error(f"No valid subscribe response found. Received lines: {lines[:5]}...")
                    raise RuntimeError(f"Invalid subscribe response: no response with 'result' field found. First line: {lines[0][:200] if lines else 'empty'}")
                
                if 'result' not in response:
                    raise RuntimeError(f"Invalid subscribe response: missing 'result' field: {response}")
                subscription_details , extranonce1 , extranonce2_size = response['result']
                return subscription_details , extranonce1 , int(extranonce2_size)
            except (socket.error, OSError, ConnectionError, json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error(f"Subscribe failed: {e}")
                raise

    def authorize(self , wallet_address: str) :
        with self._socket_lock:
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
        with self._socket_lock:
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
        with self._socket_lock:
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
        with self._socket_lock:
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


