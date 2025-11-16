import json
import logging
import socket
import threading
import time
from typing import Tuple , List

from ..utils.logging_utils import _vlog


class PoolClient :
    def __init__(self , host: str , port: int , timeout: int = 30) :
        _vlog(logging.getLogger("SatoshiRig.pool_client"), True, f"PoolClient.__init__: START host={host}, port={port}, timeout={timeout}")
        self.host = host
        _vlog(logging.getLogger("SatoshiRig.pool_client"), True, f"PoolClient.__init__: self.host={self.host}")
        self.port = port
        _vlog(logging.getLogger("SatoshiRig.pool_client"), True, f"PoolClient.__init__: self.port={self.port}")
        self.timeout = timeout
        _vlog(logging.getLogger("SatoshiRig.pool_client"), True, f"PoolClient.__init__: self.timeout={self.timeout}")
        self.sock: socket.socket | None = None
        _vlog(logging.getLogger("SatoshiRig.pool_client"), True, "PoolClient.__init__: self.sock=None")
        self.logger = logging.getLogger("SatoshiRig.pool_client")
        _vlog(self.logger, True, "PoolClient.__init__: logger created")
        self._socket_lock = threading.Lock()  # Lock for thread-safe socket operations
        _vlog(self.logger, True, "PoolClient.__init__: _socket_lock created")
        self._verbose_logging = True  # Always enable verbose logging for pool client
        _vlog(self.logger, True, "PoolClient.__init__: END")

    def connect(self) :
        _vlog(self.logger, self._verbose_logging, "PoolClient.connect: START")
        _vlog(self.logger, self._verbose_logging, "PoolClient.connect: acquiring _socket_lock")
        with self._socket_lock:
            _vlog(self.logger, self._verbose_logging, "PoolClient.connect: inside _socket_lock")
            max_retries = 3
            _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: max_retries={max_retries}")
            retry_delay = 2  # seconds
            _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: retry_delay={retry_delay}")
            last_error = None
            _vlog(self.logger, self._verbose_logging, "PoolClient.connect: last_error=None")
            
            _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: starting retry loop, max_retries={max_retries}")
            for attempt in range(max_retries):
                _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: attempt {attempt + 1}/{max_retries}")
                try:
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: creating socket AF_INET, SOCK_STREAM")
                    self.sock = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: socket created, sock={self.sock}")
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: setting timeout={self.timeout}")
                    self.sock.settimeout(self.timeout)
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: connecting to ({self.host}, {self.port})")
                    self.sock.connect((self.host , self.port))
                    _vlog(self.logger, self._verbose_logging, "PoolClient.connect: connection successful")
                    _vlog(self.logger, self._verbose_logging, "PoolClient.connect: END (success)")
                    return  # Success
                except (socket.error, OSError, ConnectionError) as e:
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: exception {type(e).__name__}: {e}")
                    last_error = e
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: last_error set to {last_error}")
                    if self.sock:
                        _vlog(self.logger, self._verbose_logging, "PoolClient.connect: closing socket due to error")
                        try:
                            self.sock.close()
                            _vlog(self.logger, self._verbose_logging, "PoolClient.connect: socket closed")
                        except Exception as close_error:
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: error closing socket: {close_error}")
                            pass
                        self.sock = None
                        _vlog(self.logger, self._verbose_logging, "PoolClient.connect: self.sock=None")
                    
                    if attempt < max_retries - 1:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: retrying, attempt {attempt + 1} < {max_retries}")
                        self.logger.warning(f"Failed to connect to {self.host}:{self.port} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: sleeping {retry_delay} seconds")
                        time.sleep(retry_delay)
                        _vlog(self.logger, self._verbose_logging, "PoolClient.connect: sleep completed")
                    else:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: max retries reached, no more retries")
                        self.logger.error(f"Failed to connect to {self.host}:{self.port} after {max_retries} attempts: {e}")
            _vlog(self.logger, self._verbose_logging, "PoolClient.connect: released _socket_lock")
            
            _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: checking last_error: {last_error is not None}")
            if last_error is not None:
                _vlog(self.logger, self._verbose_logging, f"PoolClient.connect: raising last_error: {last_error}")
                raise last_error
            _vlog(self.logger, self._verbose_logging, "PoolClient.connect: raising RuntimeError")
            raise RuntimeError("Failed to connect to pool: unknown error")

    def subscribe(self) -> Tuple[str , str , int] :
        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: START")
        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: acquiring _socket_lock")
        with self._socket_lock:
            _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: inside _socket_lock")
            _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: checking sock: {self.sock is not None}")
            if self.sock is None:
                _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: sock is None, raising RuntimeError")
                raise RuntimeError("Socket not connected. Call connect() first.")
            try:
                _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: sending subscribe message")
                self.sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
                _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: subscribe message sent")
                
                # Read response - may need multiple recv() calls for large responses
                # Pool responses can be > 1024 bytes, so we need to read until we get a complete line
                # Pool may also send mining.notify messages immediately after subscribe response
                _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: initializing buffer")
                buffer = bytearray()
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: buffer created, length={len(buffer)}")
                max_buffer_size = 64 * 1024  # 64KB max
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: max_buffer_size={max_buffer_size}")
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: setting socket timeout={self.timeout}")
                self.sock.settimeout(self.timeout)
                
                # Read until we have at least one complete line
                # Pool may send multiple messages (subscribe response + mining.notify), so we need to read all
                lines_read = 0
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: lines_read={lines_read}")
                read_timeout_count = 0
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: read_timeout_count={read_timeout_count}")
                max_timeout_retries = 3  # Allow a few timeouts if we're getting partial data
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: max_timeout_retries={max_timeout_retries}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: entering read loop")
                while True:
                    try:
                        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: calling recv(4096)")
                        chunk = self.sock.recv(4096)
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: recv returned, chunk length={len(chunk) if chunk else 0}")
                        if not chunk:
                            _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: chunk is empty, raising ConnectionError")
                            raise ConnectionError("Connection closed by server during subscribe")
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: extending buffer with chunk, buffer length before={len(buffer)}")
                        buffer.extend(chunk)
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: buffer length after={len(buffer)}")
                        read_timeout_count = 0  # Reset timeout counter on successful read
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: read_timeout_count reset to 0")
                    except socket.timeout:
                        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: socket.timeout exception")
                        # If we have at least one complete line, try to parse it
                        newline_count = buffer.count(b'\n')
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: newline_count={newline_count}")
                        if newline_count > 0:
                            self.logger.debug(f"Timeout during subscribe read, but have {newline_count} complete lines, attempting to parse")
                            _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: breaking read loop (have complete lines)")
                            break
                        read_timeout_count += 1
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: read_timeout_count incremented to {read_timeout_count}")
                        if read_timeout_count >= max_timeout_retries:
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: max_timeout_retries reached, raising TimeoutError")
                            raise TimeoutError(f"Subscribe read timed out after {max_timeout_retries} attempts")
                        # Wait a bit and retry
                        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: sleeping 0.1 seconds before retry")
                        time.sleep(0.1)
                        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: sleep completed, continuing")
                        continue
                    
                    # Count how many complete lines we have
                    lines_read = buffer.count(b'\n')
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: lines_read={lines_read}")
                    
                    # If we have at least one complete line, check if we have the subscribe response
                    if lines_read > 0:
                        _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: decoding buffer to check for subscribe response")
                        # Decode and check if we already have the subscribe response
                        temp_lines = buffer.decode('utf-8', errors='replace').split('\n')
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: temp_lines count={len(temp_lines)}")
                        found_subscribe_response = False
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: found_subscribe_response={found_subscribe_response}")
                        for line_idx, line in enumerate(temp_lines):
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: checking line {line_idx+1}/{len(temp_lines)}, line length={len(line)}, line.strip()={bool(line.strip())}")
                            if not line.strip():
                                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: line {line_idx+1} is empty, skipping")
                                continue
                            try:
                                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: parsing line {line_idx+1} as JSON")
                                parsed = json.loads(line)
                                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: parsed JSON keys={list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
                                if 'result' in parsed and parsed.get('id') == 1:
                                    _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: found subscribe response in line {line_idx+1}")
                                    found_subscribe_response = True
                                    break
                            except json.JSONDecodeError as json_error:
                                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: JSONDecodeError in line {line_idx+1}: {json_error}")
                                continue
                        
                        # If we found the subscribe response, we can stop reading
                        if found_subscribe_response:
                            _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: found subscribe response, breaking read loop")
                            break
                    
                    # Prevent buffer overflow
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: checking buffer size: {len(buffer)} <= {max_buffer_size}")
                    if len(buffer) > max_buffer_size:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: buffer overflow, raising RuntimeError")
                        raise RuntimeError(f"Subscribe response too large (>{max_buffer_size} bytes)")
                
                _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: read loop completed")
                # Decode and parse all lines - pool may send multiple messages
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: decoding buffer, length={len(buffer)}")
                lines = buffer.decode('utf-8', errors='replace').split('\n')
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: lines count={len(lines)}")
                
                # Find the subscribe response (should have 'result' field and 'id': 1)
                # Pool may also send mining.notify messages, so we need to find the right one
                response = None
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: response=None, searching for subscribe response")
                for line_idx, line in enumerate(lines):
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: checking line {line_idx+1}/{len(lines)}, line.strip()={bool(line.strip())}")
                    if not line.strip():
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: line {line_idx+1} is empty, skipping")
                        continue
                    try:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: parsing line {line_idx+1} as JSON")
                        parsed = json.loads(line)
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: parsed JSON keys={list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
                        # Subscribe response should have 'result' field and 'id' matching our request (1)
                        if 'result' in parsed and parsed.get('id') == 1:
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: found subscribe response in line {line_idx+1}")
                            response = parsed
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: response set, keys={list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
                            break
                    except json.JSONDecodeError as json_error:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: JSONDecodeError in line {line_idx+1}: {json_error}")
                        # Skip invalid JSON lines
                        continue
                
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: checking response: {response is not None}")
                if response is None:
                    # Log all received lines for debugging
                    self.logger.error(f"No valid subscribe response found. Received lines: {lines[:5]}...")
                    _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: raising RuntimeError (no response found)")
                    raise RuntimeError(f"Invalid subscribe response: no response with 'result' field found. First line: {lines[0][:200] if lines else 'empty'}")
                
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: checking 'result' in response: {'result' in response}")
                if 'result' not in response:
                    _vlog(self.logger, self._verbose_logging, "PoolClient.subscribe: raising RuntimeError (missing 'result' field)")
                    raise RuntimeError(f"Invalid subscribe response: missing 'result' field: {response}")
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: unpacking response['result']")
                subscription_details , extranonce1 , extranonce2_size = response['result']
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: subscription_details={subscription_details}, extranonce1={extranonce1}, extranonce2_size={extranonce2_size}")
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: converting extranonce2_size to int: {int(extranonce2_size)}")
                result = (subscription_details , extranonce1 , int(extranonce2_size))
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: returning result, END")
                return result
            except (socket.error, OSError, ConnectionError, json.JSONDecodeError, KeyError, ValueError) as e:
                _vlog(self.logger, self._verbose_logging, f"PoolClient.subscribe: exception {type(e).__name__}: {e}")
                self.logger.error(f"Subscribe failed: {e}")
                raise

    def authorize(self , wallet_address: str) :
        _vlog(self.logger, self._verbose_logging, f"PoolClient.authorize: START wallet_address={wallet_address[:10]}...")
        _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: acquiring _socket_lock")
        with self._socket_lock:
            _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: inside _socket_lock")
            _vlog(self.logger, self._verbose_logging, f"PoolClient.authorize: checking sock: {self.sock is not None}")
            if self.sock is None:
                _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: sock is None, raising RuntimeError")
                raise RuntimeError("Socket not connected. Call connect() first.")
            try:
                _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: creating authorize message")
                authorize_msg_dict = {
                    "params": [wallet_address , "password"] ,
                    "id": 2 ,
                    "method": "mining.authorize"
                }
                _vlog(self.logger, self._verbose_logging, f"PoolClient.authorize: authorize_msg_dict={authorize_msg_dict}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: converting to JSON")
                authorize_msg_json = json.dumps(authorize_msg_dict)
                _vlog(self.logger, self._verbose_logging, f"PoolClient.authorize: authorize_msg_json={authorize_msg_json}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: encoding to UTF-8 and adding newline")
                authorize_msg = authorize_msg_json.encode('utf-8') + b"\n"
                _vlog(self.logger, self._verbose_logging, f"PoolClient.authorize: authorize_msg length={len(authorize_msg)}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: sending authorize message")
                self.sock.sendall(authorize_msg)
                _vlog(self.logger, self._verbose_logging, "PoolClient.authorize: authorize message sent, END")
            except (socket.error, OSError, ConnectionError) as e:
                _vlog(self.logger, self._verbose_logging, f"PoolClient.authorize: exception {type(e).__name__}: {e}")
                self.logger.error(f"Authorize failed: {e}")
                raise

    def read_notify(self) -> list:
        _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: START")
        _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: acquiring _socket_lock")
        with self._socket_lock:
            _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: inside _socket_lock")
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: checking sock: {self.sock is not None}")
            if self.sock is None:
                _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: sock is None, raising RuntimeError")
                raise RuntimeError("Socket not connected. Call connect() first.")
            # Robust line-buffered read with simple framing by newlines
            # Keep reading until we see at least one mining.notify message
            # and we have consumed a line ending.
            _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: initializing buffer")
            buffer = bytearray()
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: buffer created, length={len(buffer)}")
            messages: list[str] = []
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: messages list created, length={len(messages)}")
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: setting socket timeout={self.timeout}")
            self.sock.settimeout(self.timeout)
            max_buffer_size = 1024 * 1024  # 1MB max buffer size to prevent memory leak
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: max_buffer_size={max_buffer_size}")
            max_iterations = 1000  # Prevent infinite loop
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: max_iterations={max_iterations}")
            iteration_count = 0
            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: iteration_count={iteration_count}")
            try:
                _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: entering read loop")
                while iteration_count < max_iterations:
                    iteration_count += 1
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: iteration {iteration_count}/{max_iterations}")
                    _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: calling recv(4096)")
                    chunk = self.sock.recv(4096)
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: recv returned, chunk length={len(chunk) if chunk else 0}")
                    if not chunk:
                        _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: chunk is empty, raising ConnectionError")
                        raise ConnectionError("Connection closed by server during read_notify")
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: extending buffer, buffer length before={len(buffer)}")
                    buffer.extend(chunk)
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: buffer length after={len(buffer)}")
                    
                    # Prevent buffer from growing unbounded
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: checking buffer size: {len(buffer)} <= {max_buffer_size}")
                    if len(buffer) > max_buffer_size:
                        self.logger.warning(f"Buffer size exceeded {max_buffer_size} bytes, truncating")
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: truncating buffer, keeping last {max_buffer_size} bytes")
                        buffer = buffer[-max_buffer_size:]  # Keep only last 1MB
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: buffer length after truncation={len(buffer)}")
                    
                    _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: entering line extraction loop")
                    while True:
                        try:
                            _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: searching for newline (byte 10) in buffer")
                            newline_index = buffer.index(10)  # '\n'
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: newline found at index={newline_index}")
                        except ValueError:
                            _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: no newline found, breaking line extraction loop")
                            break
                        # Use 'replace' instead of 'ignore' to preserve data integrity
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: extracting line from buffer[0:{newline_index}]")
                        line = buffer[:newline_index].decode('utf-8', errors='replace').strip()
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: line extracted, length={len(line)}, line[:50]={line[:50] if line else 'empty'}...")
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: deleting buffer[0:{newline_index + 1}]")
                        del buffer[:newline_index + 1]
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: buffer length after deletion={len(buffer)}")
                        if line:
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: adding line to messages, messages count before={len(messages)}")
                            messages.append(line)
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: messages count after={len(messages)}")
                    _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: line extraction loop completed")
                    # Stop once we have at least one notify message
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: checking for mining.notify in messages: {any('mining.notify' in m for m in messages)}")
                    if messages and any('mining.notify' in m for m in messages):
                        _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: found mining.notify, breaking read loop")
                        break
                else:
                    # Loop exhausted without finding mining.notify
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: read loop exhausted, messages count={len(messages)}")
                    if not messages:
                        self.logger.warning("read_notify: No messages received after max iterations")
                        _vlog(self.logger, self._verbose_logging, "PoolClient.read_notify: returning empty list, END")
                        return []
                    self.logger.warning(f"read_notify: Max iterations reached, returning {len(messages)} messages")
                
                _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: parsing messages, messages count={len(messages)}")
                responses = []
                _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: responses list created, length={len(responses)}")
                for msg_idx, m in enumerate(messages):
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: processing message {msg_idx+1}/{len(messages)}, length={len(m)}, contains 'mining.notify'={'mining.notify' in m}")
                    try:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: parsing message {msg_idx+1} as JSON")
                        obj = json.loads(m)
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: parsed JSON keys={list(obj.keys()) if isinstance(obj, dict) else 'not a dict'}")
                        if 'mining.notify' in m:
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: message {msg_idx+1} contains 'mining.notify', adding to responses")
                            responses.append(obj)
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: responses count={len(responses)}")
                        else:
                            _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: message {msg_idx+1} does not contain 'mining.notify', skipping")
                    except json.JSONDecodeError as e:
                        _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: JSONDecodeError in message {msg_idx+1}: {e}")
                        self.logger.debug(f"Failed to parse message: {m[:100]}... Error: {e}")
                        continue
                _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: returning responses, count={len(responses)}, END")
                return responses
            except (socket.timeout, socket.error, OSError, ConnectionError) as e:
                _vlog(self.logger, self._verbose_logging, f"PoolClient.read_notify: exception {type(e).__name__}: {e}")
                self.logger.error(f"read_notify failed: {e}")
                raise

    def submit(self , wallet_address: str , job_id: str , extranonce2: str , ntime: str , nonce_hex: str) -> bytes:
        _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: START wallet_address={wallet_address[:10]}..., job_id={job_id}, extranonce2={extranonce2}, ntime={ntime}, nonce_hex={nonce_hex}")
        _vlog(self.logger, self._verbose_logging, "PoolClient.submit: acquiring _socket_lock")
        with self._socket_lock:
            _vlog(self.logger, self._verbose_logging, "PoolClient.submit: inside _socket_lock")
            _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: checking sock: {self.sock is not None}")
            if self.sock is None:
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: sock is None, raising RuntimeError")
                raise RuntimeError("Socket not connected. Call connect() first.")
            try:
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: creating submit message")
                submit_msg_dict = {
                    "params": [wallet_address , job_id , extranonce2 , ntime , nonce_hex] ,
                    "id": 1 ,
                    "method": "mining.submit"
                }
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: submit_msg_dict={submit_msg_dict}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: converting to JSON")
                submit_msg_json = json.dumps(submit_msg_dict)
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: submit_msg_json={submit_msg_json}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: encoding to UTF-8 and adding newline")
                payload = submit_msg_json.encode('utf-8') + b"\n"
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: payload length={len(payload)}")
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: sending payload")
                self.sock.sendall(payload)
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: payload sent")
                # Set timeout before recv to prevent blocking indefinitely
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: setting socket timeout={self.timeout}")
                self.sock.settimeout(self.timeout)
                _vlog(self.logger, self._verbose_logging, "PoolClient.submit: calling recv(1024)")
                response = self.sock.recv(1024)
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: recv returned, response length={len(response) if response else 0}")
                if not response:
                    _vlog(self.logger, self._verbose_logging, "PoolClient.submit: response is empty, raising ConnectionError")
                    raise ConnectionError("Connection closed by server during submit")
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: returning response, END")
                return response
            except (socket.timeout, socket.error, OSError, ConnectionError) as e:
                _vlog(self.logger, self._verbose_logging, f"PoolClient.submit: exception {type(e).__name__}: {e}")
                self.logger.error(f"Submit failed: {e}")
                raise

    def close(self):
        """Close the socket connection"""
        _vlog(self.logger, self._verbose_logging, "PoolClient.close: START")
        _vlog(self.logger, self._verbose_logging, "PoolClient.close: acquiring _socket_lock")
        with self._socket_lock:
            _vlog(self.logger, self._verbose_logging, "PoolClient.close: inside _socket_lock")
            _vlog(self.logger, self._verbose_logging, f"PoolClient.close: checking sock: {self.sock is not None}")
            if self.sock:
                try:
                    _vlog(self.logger, self._verbose_logging, "PoolClient.close: closing socket")
                    self.sock.close()
                    _vlog(self.logger, self._verbose_logging, "PoolClient.close: socket closed")
                except (socket.error, OSError) as e:
                    _vlog(self.logger, self._verbose_logging, f"PoolClient.close: exception closing socket: {type(e).__name__}: {e}")
                    self.logger.debug(f"Error closing socket: {e}")
                finally:
                    _vlog(self.logger, self._verbose_logging, "PoolClient.close: setting self.sock=None")
                    self.sock = None
                    _vlog(self.logger, self._verbose_logging, "PoolClient.close: self.sock=None")
            else:
                _vlog(self.logger, self._verbose_logging, "PoolClient.close: sock is None, nothing to close")
        _vlog(self.logger, self._verbose_logging, "PoolClient.close: released _socket_lock, END")

    def __enter__(self):
        """Context manager entry"""
        _vlog(self.logger, self._verbose_logging, "PoolClient.__enter__: START")
        _vlog(self.logger, self._verbose_logging, "PoolClient.__enter__: returning self, END")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures socket is closed"""
        _vlog(self.logger, self._verbose_logging, f"PoolClient.__exit__: START exc_type={exc_type}, exc_val={exc_val}, exc_tb={exc_tb is not None}")
        _vlog(self.logger, self._verbose_logging, "PoolClient.__exit__: calling close()")
        self.close()
        _vlog(self.logger, self._verbose_logging, "PoolClient.__exit__: close() completed, returning False, END")
        return False


