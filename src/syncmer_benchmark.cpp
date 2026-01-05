// syncmer_benchmark.cpp - Standalone syncmer benchmark using SIMD
//
// Syncmers are k-mers where the minimizer (of size m) appears at the
// prefix (position 0) or suffix (position k-m) of the k-mer.
//
// Build with: make syncmer_benchmark
// Run with:   ./syncmer_benchmark

#include "canonical_minimizers.hpp"
#include "packed_seq.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace packed_seq;

// Generate random DNA sequence
static std::vector<uint8_t> generate_random_dna(size_t len) {
    const char bases[] = "ACGT";
    std::vector<uint8_t> seq(len);
    for (size_t i = 0; i < len; i++) {
        uint32_t x = static_cast<uint32_t>(i);
        x = x * 1103515245 + 12345;
        seq[i] = bases[(x >> 16) % 4];
    }
    return seq;
}

// Forward declaration for SIMD syncmers
extern void syncmers_simd_fused_packed(
    const packed_seq::PackedSeq& seq,
    uint32_t seq_len,
    uint32_t k,      // syncmer k-mer size
    uint32_t m,      // minimizer size (s-mer)
    std::vector<uint32_t>& out_positions
);

// Forward declaration for non-canonical minimizers
extern void minimizers_simd_fused_packed(
    const packed_seq::PackedSeq& seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions
);

int main(int argc, char** argv) {
    // Default parameters
    size_t seq_len = 1000000;  // 1 MB
    uint32_t k = 21;           // syncmer k-mer size
    uint32_t m = 11;           // minimizer size (s-mer)
    uint32_t iterations = 10;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            seq_len = atol(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-n seq_len] [-k syncmer_size] [-m minimizer_size] [-i iterations]\n", argv[0]);
            printf("  -n  Sequence length (default: 1000000)\n");
            printf("  -k  Syncmer k-mer size (default: 21)\n");
            printf("  -m  Minimizer size within syncmer (default: 11)\n");
            printf("  -i  Number of iterations (default: 10)\n");
            return 0;
        }
    }

    uint32_t w = k - m + 1;

    printf("Syncmer Benchmark (SIMD)\n");
    printf("========================\n");
    printf("Sequence length: %zu\n", seq_len);
    printf("Syncmer k-mer size (k): %u\n", k);
    printf("Minimizer size (m): %u\n", m);
    printf("Window size (w = k - m + 1): %u\n", w);
    printf("Iterations: %u\n", iterations);
    printf("\n");

    // Generate random DNA
    auto seq_data = generate_random_dna(seq_len);
    PackedSeq seq_packed = PackedSeq::from_ascii(seq_data.data(), seq_len);

    // Benchmark minimizers only (for comparison)
    {
        std::vector<uint32_t> positions;
        positions.reserve((seq_len - m + 1) / w);

        auto start = std::chrono::high_resolution_clock::now();
        volatile uint32_t result_sum = 0;

        for (uint32_t iter = 0; iter < iterations; iter++) {
            positions.clear();
            minimizers_simd_fused_packed(seq_packed, seq_len, m, w, positions);
            if (!positions.empty()) result_sum += positions[0];
        }

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double ms_per_iter = (double)us / iterations / 1000.0;
        double mb = (double)seq_len / 1000000.0;
        double throughput = mb / (ms_per_iter / 1000.0);

        printf("Minimizers SIMD (m=%u, w=%u):\n", m, w);
        printf("  Time: %.2f ms/iter\n", ms_per_iter);
        printf("  Throughput: %.1f MB/s\n", throughput);
        printf("  Unique positions: %zu\n", positions.size());
        printf("\n");
    }

    // Benchmark syncmers SIMD
    {
        std::vector<uint32_t> syncmer_positions;
        syncmer_positions.reserve((seq_len - m + 1) / w);

        auto start = std::chrono::high_resolution_clock::now();
        volatile uint32_t result_sum = 0;

        for (uint32_t iter = 0; iter < iterations; iter++) {
            syncmer_positions.clear();
            syncmers_simd_fused_packed(seq_packed, seq_len, k, m, syncmer_positions);
            if (!syncmer_positions.empty()) result_sum += syncmer_positions[0];
        }

        auto end = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double ms_per_iter = (double)us / iterations / 1000.0;
        double mb = (double)seq_len / 1000000.0;
        double throughput = mb / (ms_per_iter / 1000.0);
        uint32_t num_windows = seq_len - k + 1;

        printf("Syncmers SIMD (k=%u, m=%u):\n", k, m);
        printf("  Time: %.2f ms/iter\n", ms_per_iter);
        printf("  Throughput: %.1f MB/s\n", throughput);
        printf("  Unique syncmer positions: %zu\n", syncmer_positions.size());
        printf("  Expected fraction: ~%.1f%% (2/w)\n", 200.0 / w);
        printf("\n");
    }

    // Parameter sweep
    printf("Parameter sweep (SIMD):\n");
    printf("  k     m     w   | Minimizers | Syncmers | Ratio  | MB/s\n");
    printf("------------------|------------|----------|--------|------\n");

    uint32_t test_params[][2] = {
        {15, 7},
        {21, 11},
        {31, 15},
        {31, 21},
    };

    for (auto& params : test_params) {
        uint32_t test_k = params[0];
        uint32_t test_m = params[1];
        uint32_t test_w = test_k - test_m + 1;

        // Get minimizer count
        std::vector<uint32_t> min_pos;
        min_pos.reserve((seq_len - test_m + 1) / test_w);
        minimizers_simd_fused_packed(seq_packed, seq_len, test_m, test_w, min_pos);

        // Get syncmer count and time
        std::vector<uint32_t> syncmer_pos;
        syncmer_pos.reserve((seq_len - test_m + 1) / test_w);

        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t iter = 0; iter < iterations; iter++) {
            syncmer_pos.clear();
            syncmers_simd_fused_packed(seq_packed, seq_len, test_k, test_m, syncmer_pos);
        }
        auto end = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double ms_per_iter = (double)us / iterations / 1000.0;
        double mb = (double)seq_len / 1000000.0;
        double throughput = mb / (ms_per_iter / 1000.0);
        double ratio = (double)syncmer_pos.size() / min_pos.size();

        printf("  %2u   %2u   %2u   | %10zu | %8zu | %5.1f%% | %5.1f\n",
               test_k, test_m, test_w, min_pos.size(), syncmer_pos.size(), ratio * 100.0, throughput);
    }

    return 0;
}
