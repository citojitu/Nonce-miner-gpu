// nonce-miner-gpu — CUDA host program
// Same I/O protocol as the CPU miner (compat with ../nonce-miner Node wrapper):
//   args:    <challenge_hex_32> <target_hex_32> <start_nonce> [batch_size]
//   stdout:  "FOUND <decimal_nonce>"
//   stderr:  "RATE <hashes_per_second>"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

extern "C" __global__ void mine_kernel(
    const uint64_t* challenge_lanes,
    const uint64_t* target_be,
    uint64_t start_nonce,
    uint64_t total_threads,
    int* found_flag,
    uint64_t* found_nonce
);

static bool parse_hex32(const char* s, uint8_t out[32]) {
    if (s[0] == '0' && s[1] == 'x') s += 2;
    if (strlen(s) != 64) return false;
    for (int i = 0; i < 32; ++i) {
        int hi = s[i*2], lo = s[i*2+1];
        auto h = [](int c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int H = h(hi), L = h(lo);
        if (H < 0 || L < 0) return false;
        out[i] = (uint8_t)((H << 4) | L);
    }
    return true;
}

static void bytes_to_lanes_le(const uint8_t bytes[32], uint64_t lanes[4]) {
    for (int i = 0; i < 4; ++i) {
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) {
            v |= (uint64_t)bytes[i*8 + j] << (j * 8);  // little-endian within lane
        }
        lanes[i] = v;
    }
}

static void bytes_to_be_u64s(const uint8_t bytes[32], uint64_t out[4]) {
    for (int i = 0; i < 4; ++i) {
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) {
            v = (v << 8) | bytes[i*8 + j];  // big-endian
        }
        out[i] = v;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <challenge_hex> <target_hex> <start_nonce> [batch_size]\n", argv[0]);
        return 1;
    }
    uint8_t challenge_bytes[32], target_bytes[32];
    if (!parse_hex32(argv[1], challenge_bytes)) { fprintf(stderr, "bad challenge hex\n"); return 1; }
    if (!parse_hex32(argv[2], target_bytes))    { fprintf(stderr, "bad target hex\n");    return 1; }

    uint64_t start_nonce = strtoull(argv[3], nullptr, 10);
    uint64_t batch = (argc > 4) ? strtoull(argv[4], nullptr, 10) : 0;
    if (batch == 0) batch = 1ULL << 22;  // 4M per kernel launch — auto-tune in future

    uint64_t challenge_lanes[4], target_be[4];
    bytes_to_lanes_le(challenge_bytes, challenge_lanes);
    bytes_to_be_u64s(target_bytes,   target_be);

    // device buffers
    uint64_t *d_ch, *d_tgt, *d_nonce;
    int *d_found;
    cudaMalloc(&d_ch,    4 * sizeof(uint64_t));
    cudaMalloc(&d_tgt,   4 * sizeof(uint64_t));
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_nonce, sizeof(uint64_t));
    cudaMemcpy(d_ch,  challenge_lanes, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, target_be,       4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "device: %s (sm_%d%d, %d SMs)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fprintf(stderr, "batch_size: %llu (per kernel)\n", (unsigned long long)batch);

    const int block_dim = 256;
    auto start_time = std::chrono::steady_clock::now();
    uint64_t total = 0;
    uint64_t nonce_offset = start_nonce;

    while (true) {
        int zero = 0;
        cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);

        uint64_t threads = batch;
        uint64_t grid = (threads + block_dim - 1) / block_dim;

        mine_kernel<<<(unsigned int)grid, block_dim>>>(
            d_ch, d_tgt, nonce_offset, threads, d_found, d_nonce
        );
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            fprintf(stderr, "cuda err: %s\n", cudaGetErrorString(e));
            return 2;
        }

        int found = 0;
        cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (found) {
            uint64_t nonce;
            cudaMemcpy(&nonce, d_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            printf("FOUND %llu\n", (unsigned long long)nonce);
            fflush(stdout);
            return 0;
        }

        total += threads;
        nonce_offset += threads;

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - start_time).count();
        if (dt >= 2.0) {
            double rate = (double)total / dt;
            fprintf(stderr, "RATE %.0f\n", rate);
            start_time = now;
            total = 0;
        }
    }
    return 0;
}
