// keccak256 GPU kernel — one thread = one nonce attempt.
// Compiled with nvcc as part of nonce-miner-gpu.
//
// Hash format: input = challenge[32] || abi_encode(uint256 nonce)
//   = challenge[32] + 24 zero bytes + nonce_be[8]
// Compare hash (big-endian uint256) against target.
//
// Reference: Keccak-256 (NOT SHA-3) — same as Ethereum/Solidity keccak256.

#include <stdint.h>

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

// Round constants
__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ void keccak_f(uint64_t s[25]) {
    uint64_t t, bc[5];
    for (int r = 0; r < 24; ++r) {
        // Theta
        bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];
        for (int i = 0; i < 5; ++i) {
            t = bc[(i+4)%5] ^ ROTL64(bc[(i+1)%5], 1);
            s[i]    ^= t; s[i+5]  ^= t; s[i+10] ^= t; s[i+15] ^= t; s[i+20] ^= t;
        }
        // Rho + Pi
        t = s[1];
        uint64_t tmp;
        tmp = s[10]; s[10] = ROTL64(t, 1);  t = tmp;
        tmp = s[7];  s[7]  = ROTL64(t, 3);  t = tmp;
        tmp = s[11]; s[11] = ROTL64(t, 6);  t = tmp;
        tmp = s[17]; s[17] = ROTL64(t, 10); t = tmp;
        tmp = s[18]; s[18] = ROTL64(t, 15); t = tmp;
        tmp = s[3];  s[3]  = ROTL64(t, 21); t = tmp;
        tmp = s[5];  s[5]  = ROTL64(t, 28); t = tmp;
        tmp = s[16]; s[16] = ROTL64(t, 36); t = tmp;
        tmp = s[8];  s[8]  = ROTL64(t, 45); t = tmp;
        tmp = s[21]; s[21] = ROTL64(t, 55); t = tmp;
        tmp = s[24]; s[24] = ROTL64(t, 2);  t = tmp;
        tmp = s[4];  s[4]  = ROTL64(t, 14); t = tmp;
        tmp = s[15]; s[15] = ROTL64(t, 27); t = tmp;
        tmp = s[23]; s[23] = ROTL64(t, 41); t = tmp;
        tmp = s[19]; s[19] = ROTL64(t, 56); t = tmp;
        tmp = s[13]; s[13] = ROTL64(t, 8);  t = tmp;
        tmp = s[12]; s[12] = ROTL64(t, 25); t = tmp;
        tmp = s[2];  s[2]  = ROTL64(t, 43); t = tmp;
        tmp = s[20]; s[20] = ROTL64(t, 62); t = tmp;
        tmp = s[14]; s[14] = ROTL64(t, 18); t = tmp;
        tmp = s[22]; s[22] = ROTL64(t, 39); t = tmp;
        tmp = s[9];  s[9]  = ROTL64(t, 61); t = tmp;
        tmp = s[6];  s[6]  = ROTL64(t, 20); t = tmp;
                     s[1]  = ROTL64(t, 44);
        // Chi
        for (int j = 0; j < 25; j += 5) {
            uint64_t a0 = s[j], a1 = s[j+1], a2 = s[j+2], a3 = s[j+3], a4 = s[j+4];
            s[j]   = a0 ^ ((~a1) & a2);
            s[j+1] = a1 ^ ((~a2) & a3);
            s[j+2] = a2 ^ ((~a3) & a4);
            s[j+3] = a3 ^ ((~a4) & a0);
            s[j+4] = a4 ^ ((~a0) & a1);
        }
        // Iota
        s[0] ^= RC[r];
    }
}

// host helpers in main.cu construct the challenge as 4 little-endian u64
// (lane order in keccak state) — kernel just ingests them.

// Compute keccak256 of (challenge[32 bytes] || nonce as abi.encode uint256).
// Returns hash in little-endian-lane format (matching state words).
__device__ void compute_hash(const uint64_t challenge_lanes[4], uint64_t nonce_be, uint64_t out[4]) {
    uint64_t s[25] = {0};
    // Block 1: 136 bytes (rate). Our input is 64 bytes + 0x01 + zeros + 0x80 padding.
    // bytes  0..31  : challenge (lanes 0..3)
    // bytes 32..55  : zero (lanes 4..6)
    // bytes 56..63  : nonce big-endian uint64 = lane 7 in keccak state
    // byte  64      : 0x01 (start of pad10*1)
    // byte  135     : 0x80 (end of pad10*1)
    s[0] = challenge_lanes[0];
    s[1] = challenge_lanes[1];
    s[2] = challenge_lanes[2];
    s[3] = challenge_lanes[3];
    // lanes 4,5,6 are zero
    // lane 7: nonce stored in LAST 8 bytes of a 32-byte uint256.
    //   uint256 big-endian — bytes 56..63 of the *input*. In lane terms (8 bytes each, LE within lane):
    //   byte 56 is most significant of nonce → in keccak lane it becomes LSB of the lane when interpreted LE.
    // We need keccak state words to mirror the input bytes in LITTLE-ENDIAN within each 8-byte lane.
    // Input bytes 56..63 in memory = nonce_be encoded as 8 bytes most-sig-first.
    //   = nonce_be byte representation: B7 B6 B5 B4 B3 B2 B1 B0 (B7 = MSB of nonce_be)
    // As a little-endian u64 lane = B0 | B1<<8 | B2<<16 | ... | B7<<56
    //   = byte_swap(nonce_be) → since nonce_be is the value, in CPU u64 it's already stored LE.
    //   = __byte_perm / __brevll equivalent.
    s[7] = __brevll(nonce_be) >> 0;  // byte-reverse the u64 to convert BE→LE memory layout
    // Wait — __brevll is bit-reverse, not byte-reverse. Use byte-swap intrinsic.
    // CUDA: there's no direct u64 byteswap intrinsic, do manually:
    {
        uint64_t n = nonce_be;
        n = ((n & 0xFF00FF00FF00FF00ULL) >> 8)  | ((n & 0x00FF00FF00FF00FFULL) << 8);
        n = ((n & 0xFFFF0000FFFF0000ULL) >> 16) | ((n & 0x0000FFFF0000FFFFULL) << 16);
        n = (n >> 32) | (n << 32);
        s[7] = n;
    }
    // padding: first byte after input = 0x01 → at byte 64 = lane 8, byte 0
    s[8] = 0x0000000000000001ULL;
    // last byte of rate region: byte 135 = lane 16, byte 7 → 0x80
    s[16] = 0x8000000000000000ULL;

    keccak_f(s);

    // Output 32 bytes = first 4 lanes (little-endian within each)
    out[0] = s[0];
    out[1] = s[1];
    out[2] = s[2];
    out[3] = s[3];
}

// Compare hash (interpreted as big-endian uint256) against target (also BE u256).
// The hash output bytes are: lane0[0..7] lane1[0..7] lane2[0..7] lane3[0..7],
// where lane bytes are little-endian within the lane.
// Big-endian byte order of the hash = byte-reverse each lane individually,
// then read lanes in order 0,1,2,3 to get most-significant-first.
__device__ __forceinline__ uint64_t byteswap64(uint64_t x) {
    x = ((x & 0xFF00FF00FF00FF00ULL) >> 8)  | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = ((x & 0xFFFF0000FFFF0000ULL) >> 16) | ((x & 0x0000FFFF0000FFFFULL) << 16);
    return (x >> 32) | (x << 32);
}

// target_be[0..3] is big-endian uint256 split into 4 u64s (target_be[0] = most-significant word).
// hash_lanes[0..3] is keccak output lanes (little-endian within each lane).
// To compare hash as BE u256: hash_be_word_i = byteswap64(hash_lanes[i]).
// First difference in word order [0..3] decides.
__device__ __forceinline__ bool hash_less_than_target(const uint64_t hash_lanes[4], const uint64_t target_be[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t h = byteswap64(hash_lanes[i]);
        if (h != target_be[i]) return h < target_be[i];
    }
    return false;
}

extern "C" __global__ void mine_kernel(
    const uint64_t* __restrict__ challenge_lanes,   // 4 lanes (32 bytes)
    const uint64_t* __restrict__ target_be,         // 4 u64 BE words
    uint64_t start_nonce,
    uint64_t total_threads,
    int* found_flag,
    uint64_t* found_nonce
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    uint64_t nonce = start_nonce + tid;

    uint64_t ch[4];
    ch[0] = challenge_lanes[0];
    ch[1] = challenge_lanes[1];
    ch[2] = challenge_lanes[2];
    ch[3] = challenge_lanes[3];

    uint64_t hash[4];
    compute_hash(ch, nonce, hash);

    if (hash_less_than_target(hash, target_be)) {
        if (atomicCAS(found_flag, 0, 1) == 0) {
            *found_nonce = nonce;
        }
    }
}
