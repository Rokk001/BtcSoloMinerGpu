"""
GPU Compute Module for CUDA and OpenCL support
"""

import binascii
import hashlib
import logging
from typing import Optional, Tuple

logger = logging.getLogger("SatoshiRig.gpu")

# Try to import CUDA
# Note: We import pycuda.driver first, then autoinit only when needed
# This allows the module to be imported even if CUDA is not available at import time
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    CUDA_AVAILABLE = True
    logger.debug(
        "PyCUDA imported successfully (autoinit will be called during initialization)"
    )
except ImportError as e:
    CUDA_AVAILABLE = False
    cuda = None
    SourceModule = None
    logger.debug(f"PyCUDA not available: {e}")
except Exception as e:
    CUDA_AVAILABLE = False
    cuda = None
    SourceModule = None
    logger.warning(f"PyCUDA import failed: {e}")

# Try to import OpenCL
try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
    logger.debug("PyOpenCL imported successfully")
except ImportError as e:
    OPENCL_AVAILABLE = False
    cl = None
    logger.debug(f"PyOpenCL not available: {e}")
except Exception as e:
    OPENCL_AVAILABLE = False
    cl = None
    logger.warning(f"PyOpenCL import failed: {e}")


# CUDA SHA256 Kernel - COMPLETE IMPLEMENTATION
CUDA_SHA256_KERNEL = """
#include <cuda_runtime.h>
#include <stdint.h>

// SHA256 constants
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ void sha256_transform(uint32_t *state, const uint8_t *data) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h, t1, t2;
    
    // Copy state
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)data[i*4] << 24) | ((uint32_t)data[i*4+1] << 16) |
               ((uint32_t)data[i*4+2] << 8) | (uint32_t)data[i*4+3];
    }
    
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add to state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__global__ void mine_sha256(
    uint8_t *block_headers,
    uint32_t *nonces,
    uint8_t *results,
    int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    
    // Get block header for this thread
    uint8_t *header = &block_headers[idx * 80];
    uint32_t nonce = nonces[idx];
    
    // Update nonce in header (bytes 76-79, little-endian for Bitcoin)
    header[76] = (nonce >> 0) & 0xFF;
    header[77] = (nonce >> 8) & 0xFF;
    header[78] = (nonce >> 16) & 0xFF;
    header[79] = (nonce >> 24) & 0xFF;
    
    // Initial hash state (SHA256 initial values)
    uint32_t state1[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // First SHA256 - Bitcoin block header is 80 bytes, needs 2 blocks
    // Block 1: bytes 0-63 (64 bytes)
    sha256_transform(state1, header);
    
    // Block 2: bytes 64-79 (16 bytes) + padding
    uint8_t block2[64];
    // Copy remaining 16 bytes from header
    for (int i = 0; i < 16; i++) {
        block2[i] = header[64 + i];
    }
    // Padding: 0x80 followed by zeros
    block2[16] = 0x80;
    for (int i = 17; i < 56; i++) {
        block2[i] = 0;
    }
    // Length in bits: 80 * 8 = 640 = 0x280 (big-endian)
    block2[56] = 0;
    block2[57] = 0;
    block2[58] = 0;
    block2[59] = 0;
    block2[60] = 0;
    block2[61] = 0;
    block2[62] = 0x02;
    block2[63] = 0x80;
    
    sha256_transform(state1, block2);
    
    // Prepare second hash input (first hash result)
    uint8_t hash1[32];
    for (int i = 0; i < 8; i++) {
        hash1[i*4] = (state1[i] >> 24) & 0xFF;
        hash1[i*4+1] = (state1[i] >> 16) & 0xFF;
        hash1[i*4+2] = (state1[i] >> 8) & 0xFF;
        hash1[i*4+3] = state1[i] & 0xFF;
    }
    
    // Second SHA256
    uint32_t state2[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Pad hash1 for second SHA256 (64 bytes total)
    uint8_t padded[64];
    for (int i = 0; i < 32; i++) padded[i] = hash1[i];
    padded[32] = 0x80;
    for (int i = 33; i < 56; i++) padded[i] = 0;
    // Length in bits: 32 * 8 = 256 = 0x100
    padded[56] = 0; padded[57] = 0; padded[58] = 0; padded[59] = 0;
    padded[60] = 0; padded[61] = 0; padded[62] = 0x01; padded[63] = 0x00;
    
    sha256_transform(state2, padded);
    
    // Store result (little-endian for Bitcoin)
    uint8_t *result = &results[idx * 32];
    for (int i = 0; i < 8; i++) {
        result[i*4] = (state2[i] >> 0) & 0xFF;
        result[i*4+1] = (state2[i] >> 8) & 0xFF;
        result[i*4+2] = (state2[i] >> 16) & 0xFF;
        result[i*4+3] = (state2[i] >> 24) & 0xFF;
    }
}
"""


class CUDAMiner:
    """CUDA-based GPU miner"""

    def __init__(
        self,
        device_id: int = 0,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 256,
        max_workers: int = 8,
    ):
        self.device_id = device_id
        self.log = logger or logging.getLogger("SatoshiRig.gpu.cuda")
        self.context = None
        self.device = None
        self.batch_size = batch_size
        self.max_workers = max_workers

        if not CUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available. Install with: pip install pycuda")

        try:
            # Initialize CUDA (equivalent to pycuda.autoinit, but only when needed)
            # Try to get device count - if it fails, CUDA is not initialized
            # Note: cuda.is_initialized() doesn't exist in all PyCUDA versions, so we use try/except
            try:
                device_count = cuda.Device.count()
            except RuntimeError:
                # CUDA not initialized, initialize it
                cuda.init()
                device_count = cuda.Device.count()

            self.log.debug(f"CUDA initialized, device count: {device_count}")

            if device_count == 0:
                raise RuntimeError(
                    "No CUDA devices found. Make sure NVIDIA GPU is available and container is run with --runtime=nvidia or --gpus"
                )

            if device_id >= cuda.Device.count():
                self.log.warning(
                    f"Device ID {device_id} not available (only {cuda.Device.count()} devices), using device 0"
                )
                device_id = 0

            self.device = cuda.Device(device_id)
            self.context = self.device.make_context()
            device_name = self.device.name()
            self.log.info(f"CUDA device {device_id} initialized: {device_name}")

            # Get device properties for debugging
            props = self.device.get_attributes()
            self.log.debug(f"CUDA device {device_id} properties: {props}")
        except Exception as e:
            # Check if it's a CUDA-specific error
            if cuda and hasattr(cuda, "Error") and isinstance(e, cuda.Error):
                self.log.error(f"CUDA error initializing device {device_id}: {e}")
                raise RuntimeError(
                    f"CUDA initialization failed: {e}. Make sure container is run with --runtime=nvidia or --gpus and NVIDIA drivers are installed."
                )
            # Generic error
            self.log.error(f"Failed to initialize CUDA device {device_id}: {e}")
            raise RuntimeError(f"CUDA initialization failed: {e}")

    def hash_block_header(
        self, block_header_hex: str, num_nonces: int = 1024, start_nonce: int = 0
    ) -> Optional[Tuple[str, int]]:
        """
        Hash block header with multiple nonces on GPU - ECHTE GPU-NUTZUNG
        Args:
            block_header_hex: Block header in hex format (80 bytes)
            num_nonces: Number of nonces to test
            start_nonce: Starting nonce value (default: 0)
        Returns: (best_hash_hex, best_nonce) or None if no valid hash found
        """
        try:
            # Convert block header to bytes
            block_header = binascii.unhexlify(block_header_hex)
            if len(block_header) != 80:
                self.log.error(f"Invalid block header length: {len(block_header)}")
                return None

            # Compile CUDA kernel if not already compiled
            if not hasattr(self, "_kernel") or self._kernel is None:
                try:
                    mod = SourceModule(CUDA_SHA256_KERNEL)
                    self._kernel = mod.get_function("mine_sha256")
                    self.log.info("CUDA SHA256 kernel compiled successfully")
                except Exception as e:
                    self.log.error(f"Failed to compile CUDA kernel: {e}")
                    return None

            import struct

            try:
                import numpy as np
            except ImportError:
                self.log.error(
                    "numpy is required for CUDA GPU mining. Install with: pip install numpy"
                )
                return None

            # Validate input parameters
            if num_nonces <= 0:
                self.log.error(f"Invalid num_nonces: {num_nonces} (must be > 0)")
                return None
            if start_nonce < 0 or start_nonce >= 2**32:
                self.log.error(
                    f"Invalid start_nonce: {start_nonce} (must be 0-{2**32-1})"
                )
                return None

            # Prepare block headers (one per nonce)
            base_header = bytearray(block_header)
            headers = []
            nonces = []
            for i in range(num_nonces):
                nonce = (start_nonce + i) % (2**32)
                header_copy = base_header.copy()
                header_copy[76:80] = struct.pack("<I", nonce)
                headers.append(bytes(header_copy))
                nonces.append(nonce)

            # Allocate GPU memory
            headers_gpu = None
            nonces_gpu = None
            results_gpu = None

            try:
                headers_gpu = cuda.mem_alloc(80 * num_nonces)
                nonces_gpu = cuda.mem_alloc(4 * num_nonces)
                results_gpu = cuda.mem_alloc(32 * num_nonces)

                # Copy data to GPU
                headers_array = np.frombuffer(b"".join(headers), dtype=np.uint8)
                nonces_array = np.array(nonces, dtype=np.uint32)

                cuda.memcpy_htod(headers_gpu, headers_array)
                cuda.memcpy_htod(nonces_gpu, nonces_array)

                # Launch kernel
                # Use 256 threads per block (optimal for most GPUs)
                threads_per_block = 256
                blocks_per_grid = (
                    num_nonces + threads_per_block - 1
                ) // threads_per_block

                self._kernel(
                    headers_gpu,
                    nonces_gpu,
                    results_gpu,
                    np.int32(num_nonces),
                    block=(threads_per_block, 1, 1),
                    grid=(blocks_per_grid, 1),
                )

                # Synchronize to ensure kernel execution is complete before copying results
                cuda.Context.synchronize()

                # Copy results back from GPU
                results_array = np.empty(32 * num_nonces, dtype=np.uint8)
                cuda.memcpy_dtoh(results_array, results_gpu)

                # Find best hash
                best_hash = None
                best_nonce = None

                for i in range(num_nonces):
                    hash_bytes = results_array[i * 32 : (i + 1) * 32]
                    hash_hex = binascii.hexlify(hash_bytes).decode()

                    # Compare numerically (not as strings!)
                    if best_hash is None:
                        best_hash = hash_hex
                        best_nonce = nonces[i]
                    else:
                        try:
                            hash_int = int(hash_hex, 16)
                            best_hash_int = int(best_hash, 16)
                            if hash_int < best_hash_int:
                                best_hash = hash_hex
                                best_nonce = nonces[i]
                        except ValueError:
                            # If conversion fails, use string comparison as fallback
                            if hash_hex < best_hash:
                                best_hash = hash_hex
                                best_nonce = nonces[i]

                return (best_hash, best_nonce) if best_hash else None
            finally:
                # Free GPU memory (always, even on exception)
                if headers_gpu:
                    try:
                        headers_gpu.free()
                    except Exception:
                        pass
                if nonces_gpu:
                    try:
                        nonces_gpu.free()
                    except Exception:
                        pass
                if results_gpu:
                    try:
                        results_gpu.free()
                    except Exception:
                        pass

        except Exception as e:
            self.log.error(f"CUDA hash error: {e}")
            import traceback

            self.log.debug(traceback.format_exc())
            return None

    def cleanup(self):
        """Clean up CUDA context"""
        if self.context:
            try:
                self.context.pop()
            except Exception as e:
                # Log but don't fail on cleanup errors
                if self.log:
                    self.log.debug(f"Error during CUDA context cleanup: {e}")
                pass

    def __del__(self):
        self.cleanup()


# OpenCL SHA256 Kernel - COMPLETE IMPLEMENTATION
OPENCL_SHA256_KERNEL = """
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// SHA256 constants
constant uint k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

uint rotr(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

uint ch(uint x, uint y, uint z) {
    return (x & y) ^ (~x & z);
}

uint maj(uint x, uint y, uint z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

uint sigma0(uint x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

uint sigma1(uint x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

uint gamma0(uint x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

uint gamma1(uint x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

void sha256_transform(uint *state, const uchar *data) {
    uint w[64];
    uint a, b, c, d, e, f, g, h, t1, t2;
    
    // Copy state
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint)data[i*4] << 24) | ((uint)data[i*4+1] << 16) |
               ((uint)data[i*4+2] << 8) | (uint)data[i*4+3];
    }
    
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add to state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__kernel void mine_sha256(
    __global uchar *block_headers,
    __global uint *nonces,
    __global uchar *results,
    int num_blocks
) {
    int idx = get_global_id(0);
    if (idx >= num_blocks) return;
    
    // Get block header for this thread
    __global uchar *header = &block_headers[idx * 80];
    uint nonce = nonces[idx];
    
    // Update nonce in header (bytes 76-79, little-endian for Bitcoin)
    header[76] = (nonce >> 0) & 0xFF;
    header[77] = (nonce >> 8) & 0xFF;
    header[78] = (nonce >> 16) & 0xFF;
    header[79] = (nonce >> 24) & 0xFF;
    
    // Initial hash state (SHA256 initial values)
    uint state1[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // First SHA256 - Bitcoin block header is 80 bytes, needs 2 blocks
    // Block 1: bytes 0-63 (64 bytes)
    sha256_transform(state1, header);
    
    // Block 2: bytes 64-79 (16 bytes) + padding
    uchar block2[64];
    // Copy remaining 16 bytes from header
    for (int i = 0; i < 16; i++) {
        block2[i] = header[64 + i];
    }
    // Padding: 0x80 followed by zeros
    block2[16] = 0x80;
    for (int i = 17; i < 56; i++) {
        block2[i] = 0;
    }
    // Length in bits: 80 * 8 = 640 = 0x280 (big-endian)
    block2[56] = 0;
    block2[57] = 0;
    block2[58] = 0;
    block2[59] = 0;
    block2[60] = 0;
    block2[61] = 0;
    block2[62] = 0x02;
    block2[63] = 0x80;
    
    sha256_transform(state1, block2);
    
    // Prepare second hash input (first hash result)
    uchar hash1[32];
    for (int i = 0; i < 8; i++) {
        hash1[i*4] = (state1[i] >> 24) & 0xFF;
        hash1[i*4+1] = (state1[i] >> 16) & 0xFF;
        hash1[i*4+2] = (state1[i] >> 8) & 0xFF;
        hash1[i*4+3] = state1[i] & 0xFF;
    }
    
    // Second SHA256
    uint state2[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Pad hash1 for second SHA256 (64 bytes total)
    uchar padded[64];
    for (int i = 0; i < 32; i++) padded[i] = hash1[i];
    padded[32] = 0x80;
    for (int i = 33; i < 56; i++) padded[i] = 0;
    // Length in bits: 32 * 8 = 256 = 0x100
    padded[56] = 0; padded[57] = 0; padded[58] = 0; padded[59] = 0;
    padded[60] = 0; padded[61] = 0; padded[62] = 0x01; padded[63] = 0x00;
    
    sha256_transform(state2, padded);
    
    // Store result (little-endian for Bitcoin)
    __global uchar *result = &results[idx * 32];
    for (int i = 0; i < 8; i++) {
        result[i*4] = (state2[i] >> 0) & 0xFF;
        result[i*4+1] = (state2[i] >> 8) & 0xFF;
        result[i*4+2] = (state2[i] >> 16) & 0xFF;
        result[i*4+3] = (state2[i] >> 24) & 0xFF;
    }
}
"""


class OpenCLMiner:
    """OpenCL-based GPU miner"""

    def __init__(
        self,
        device_id: int = 0,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 256,
        max_workers: int = 8,
    ):
        self.device_id = device_id
        self.log = logger or logging.getLogger("SatoshiRig.gpu.opencl")
        self.context = None
        self.device = None
        self.queue = None
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._program = None
        self._kernel = None

        if not OPENCL_AVAILABLE:
            raise RuntimeError(
                "PyOpenCL not available. Install with: pip install pyopencl"
            )

        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")

            # Try to find GPU device
            devices = []
            for platform in platforms:
                devices.extend(platform.get_devices(cl.device_type.GPU))

            if not devices:
                raise RuntimeError("No OpenCL GPU devices found")

            if device_id >= len(devices):
                self.log.warning(f"Device ID {device_id} not available, using device 0")
                device_id = 0

            self.device = devices[device_id]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self.log.info(f"OpenCL device {device_id} initialized: {self.device.name}")

            # Compile OpenCL kernel
            try:
                self._program = cl.Program(self.context, OPENCL_SHA256_KERNEL).build()
                self._kernel = self._program.mine_sha256
                self.log.info("OpenCL SHA256 kernel compiled successfully")
            except Exception as e:
                self.log.error(f"Failed to compile OpenCL kernel: {e}")
                raise RuntimeError(f"OpenCL kernel compilation failed: {e}")

        except Exception as e:
            self.log.error(f"Failed to initialize OpenCL device {device_id}: {e}")
            raise

    def hash_block_header(
        self, block_header_hex: str, num_nonces: int = 1024, start_nonce: int = 0
    ) -> Optional[Tuple[str, int]]:
        """
        Hash block header with multiple nonces on GPU - ECHTE GPU-NUTZUNG
        Args:
            block_header_hex: Block header in hex format (80 bytes)
            num_nonces: Number of nonces to test
            start_nonce: Starting nonce value (default: 0)
        Returns: (best_hash_hex, best_nonce) or None if no valid hash found
        """
        try:
            # Convert block header to bytes
            block_header = binascii.unhexlify(block_header_hex)
            if len(block_header) != 80:
                self.log.error(f"Invalid block header length: {len(block_header)}")
                return None

            if self._kernel is None:
                self.log.error("OpenCL kernel not compiled")
                return None

            import struct

            try:
                import numpy as np
            except ImportError:
                self.log.error(
                    "numpy is required for OpenCL GPU mining. Install with: pip install numpy"
                )
                return None

            # Validate input parameters
            if num_nonces <= 0:
                self.log.error(f"Invalid num_nonces: {num_nonces} (must be > 0)")
                return None
            if start_nonce < 0 or start_nonce >= 2**32:
                self.log.error(
                    f"Invalid start_nonce: {start_nonce} (must be 0-{2**32-1})"
                )
                return None

            # Prepare block headers (one per nonce)
            base_header = bytearray(block_header)
            headers = []
            nonces = []
            for i in range(num_nonces):
                nonce = (start_nonce + i) % (2**32)
                header_copy = base_header.copy()
                header_copy[76:80] = struct.pack("<I", nonce)
                headers.append(bytes(header_copy))
                nonces.append(nonce)

            # Allocate OpenCL buffers
            headers_buf = None
            nonces_buf = None
            results_buf = None

            try:
                headers_buf = cl.Buffer(
                    self.context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=b"".join(headers),
                )
                nonces_buf = cl.Buffer(
                    self.context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=np.array(nonces, dtype=np.uint32),
                )
                results_buf = cl.Buffer(
                    self.context, cl.mem_flags.WRITE_ONLY, 32 * num_nonces
                )

                # Launch kernel
                self._kernel(
                    self.queue,
                    (num_nonces,),
                    None,
                    headers_buf,
                    nonces_buf,
                    results_buf,
                    np.int32(num_nonces),
                )

                # Wait for kernel to finish
                self.queue.finish()

                # Copy results back from GPU
                results_array = np.empty(32 * num_nonces, dtype=np.uint8)
                cl.enqueue_copy(self.queue, results_array, results_buf)

                # Wait for copy to finish (CRITICAL: ensure data is ready)
                self.queue.finish()

                # Find best hash
                best_hash = None
                best_nonce = None

                for i in range(num_nonces):
                    hash_bytes = results_array[i * 32 : (i + 1) * 32]
                    hash_hex = binascii.hexlify(hash_bytes).decode()

                    # Compare numerically (not as strings!)
                    if best_hash is None:
                        best_hash = hash_hex
                        best_nonce = nonces[i]
                    else:
                        try:
                            hash_int = int(hash_hex, 16)
                            best_hash_int = int(best_hash, 16)
                            if hash_int < best_hash_int:
                                best_hash = hash_hex
                                best_nonce = nonces[i]
                        except ValueError:
                            # If conversion fails, use string comparison as fallback
                            if hash_hex < best_hash:
                                best_hash = hash_hex
                                best_nonce = nonces[i]

                return (best_hash, best_nonce) if best_hash else None
            finally:
                # Free OpenCL buffers (always, even on exception)
                if headers_buf:
                    try:
                        headers_buf.release()
                    except Exception:
                        pass
                if nonces_buf:
                    try:
                        nonces_buf.release()
                    except Exception:
                        pass
                if results_buf:
                    try:
                        results_buf.release()
                    except Exception:
                        pass

        except Exception as e:
            self.log.error(f"OpenCL hash error: {e}")
            import traceback

            self.log.debug(traceback.format_exc())
            return None

    def cleanup(self):
        """Clean up OpenCL context and queue"""
        if self.queue:
            try:
                self.queue.finish()
            except Exception as e:
                if self.log:
                    self.log.debug(f"Error finishing OpenCL queue: {e}")
                pass
            finally:
                self.queue = None

        if self.context:
            try:
                # OpenCL contexts are automatically cleaned up when garbage collected
                # but we can explicitly release resources if needed
                pass
            except Exception as e:
                if self.log:
                    self.log.debug(f"Error during OpenCL context cleanup: {e}")
                pass
            finally:
                self.context = None
                self.device = None

    def __del__(self):
        """Destructor - ensures cleanup on object deletion"""
        self.cleanup()


def create_gpu_miner(
    backend: str,
    device_id: int = 0,
    logger: Optional[logging.Logger] = None,
    batch_size: int = 256,
    max_workers: int = 8,
):
    """
    Create a GPU miner instance based on backend type

    Args:
        backend: 'cuda' or 'opencl'
        device_id: GPU device index
        logger: Optional logger instance
        batch_size: Number of nonces per batch (default: 256)
        max_workers: Maximum number of parallel workers (default: 8)

    Returns:
        CUDAMiner or OpenCLMiner instance
    """
    if backend == "cuda":
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend requested but PyCUDA not available")
        return CUDAMiner(
            device_id=device_id,
            logger=logger,
            batch_size=batch_size,
            max_workers=max_workers,
        )
    elif backend == "opencl":
        if not OPENCL_AVAILABLE:
            raise RuntimeError("OpenCL backend requested but PyOpenCL not available")
        return OpenCLMiner(
            device_id=device_id,
            logger=logger,
            batch_size=batch_size,
            max_workers=max_workers,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
