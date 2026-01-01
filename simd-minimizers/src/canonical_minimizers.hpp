// canonical_minimizers.hpp - Shared declarations for canonical minimizers
// This header is shared between scalar and SIMD implementations

#pragma once

#include <stdint.h>
#include <vector>

// =============================================================================
// Hash constants for ntHash
// =============================================================================

// ntHash lookup table for DNA bases (packed-seq encoding: A=0, C=1, T=2, G=3)
// These are the lower 32 bits of the 64-bit ntHash constants from the Rust implementation
static const uint32_t HASHES_F[4] = {
    0x95c60474,  // A (from 0x3c8b_fbb3_95c6_0474)
    0x62a02b4c,  // C (from 0x3193_c185_62a0_2b4c)
    0x82572324,  // T (from 0x2032_3ed0_8257_2324)
    0x4be24456   // G (from 0x2955_49f5_4be2_4456)
};

// =============================================================================
// Utility functions
// =============================================================================

// Custom rotation functions
inline uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

inline uint32_t rotr32(uint32_t x, int k) {
    return (x >> k) | (x << (32 - k));
}

// Complement a DNA base (packed-seq encoding: A=0, C=1, T=2, G=3)
// A(0) <-> T(2), C(1) <-> G(3)
inline uint8_t complement_base(uint8_t base) {
    static const uint8_t complement_map[4] = {2, 3, 0, 1};
    return complement_map[base & 0x03];
}

// =============================================================================
// Debug level
// =============================================================================

#ifndef DEBUG_LEVEL
extern int DEBUG_LEVEL;
#endif

// =============================================================================
// Scalar function declarations
// =============================================================================

// Scalar ntHash computation
std::vector<uint32_t> nthash_scalar(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k
);

// Scalar sliding window minimum (two-stack algorithm)
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> sliding_lr_min_scalar(
    const std::vector<uint32_t>& hashes,
    uint32_t w
);

// Naive sliding window minimum (for testing)
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> sliding_lr_min_naive(
    const std::vector<uint32_t>& hashes,
    uint32_t w
);

// Scalar canonical minimizers computation
void canonical_minimizers_seq_scalar(
    const uint8_t* seq_data,
    uint32_t start_pos,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_minimizers
);

// =============================================================================
// SIMD function declarations
// =============================================================================

// SIMD ntHash computation
std::vector<uint32_t> nthash_simd(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k
);

// SIMD sliding window minimum
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> sliding_lr_min_simd(
    const std::vector<uint32_t>& hashes,
    uint32_t w
);

// =============================================================================
// C-compatible interface for Rust FFI
// =============================================================================

extern "C" {
    // Main entry point for canonical minimizers
    void canonical_minimizers_seq_simd_avx2(
        const uint8_t* seq_data,
        uint32_t seq_len,
        uint32_t k,
        uint32_t w,
        uint32_t** out_ptr,
        uint32_t* out_len
    );

    // Test functions
    int test_nthash_scalar_unrolled(
        const uint8_t* seq_data,
        uint32_t seq_len,
        uint32_t k
    );

    int test_sliding_min_scalar(
        const uint32_t* hashes,
        uint32_t hash_len,
        uint32_t w
    );

    int test_sliding_min_simd(
        const uint32_t* hashes,
        uint32_t hash_len,
        uint32_t w
    );

    int test_packed_seq(
        const uint8_t* ascii_seq,
        uint32_t seq_len
    );

    int test_noncanonical_minimizers_simd_vs_rust(
        const uint8_t* ascii_seq,
        uint32_t seq_len,
        uint32_t k,
        uint32_t w,
        const uint32_t* rust_positions,
        uint32_t rust_len
    );

    // Benchmark functions
    uint64_t benchmark_nthash_scalar_unrolled(
        const uint8_t* seq_data,
        uint32_t seq_len,
        uint32_t k,
        uint32_t iterations
    );

    uint64_t benchmark_sliding_min_scalar(
        const uint32_t* hashes,
        uint32_t hash_len,
        uint32_t w,
        uint32_t iterations
    );

    uint64_t benchmark_sliding_min_simd(
        const uint32_t* hashes,
        uint32_t hash_len,
        uint32_t w,
        uint32_t iterations
    );

    uint64_t benchmark_packed_seq_simd(
        const uint8_t* ascii_seq,
        uint32_t seq_len,
        uint32_t k,
        uint32_t iterations
    );

    uint64_t benchmark_nthash_packed_seq(
        const uint8_t* ascii_seq,
        uint32_t seq_len,
        uint32_t k,
        uint32_t iterations
    );

    uint64_t benchmark_hash_and_slidmin_only(
        const uint8_t* ascii_seq,
        uint32_t seq_len,
        uint32_t k,
        uint32_t w,
        uint32_t iterations
    );

    // Utility functions
    uint32_t get_cpp_forward_hashes(
        const uint8_t* seq_data,
        uint32_t seq_len,
        uint32_t k,
        uint32_t* out_hashes,
        uint32_t max_hashes
    );

    void free_minimizers(uint32_t* ptr);
}
