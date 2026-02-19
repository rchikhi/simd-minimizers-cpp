// example.cpp - Standalone example demonstrating the C++ minimizers API
//
// Build: make example
// Or:    g++ -std=c++17 -O3 -mavx2 -march=native -o example src/example.cpp
//        src/canonical_minimizers_simd.cpp src/canonical_minimizers_scalar.cpp

#include "canonical_minimizers.hpp"
#include "packed_seq.hpp"
#include <vector>
#include <cstdio>
#include <cstring>

int main() {
    // DNA sequence (ASCII)
    const char* dna = "ACGTGCTCAGAGACTCAGAGGAACGTACGT";
    size_t len = strlen(dna);

    // Parameters
    uint32_t k = 15;  // k-mer size
    uint32_t w = 11;  // window size (k+w-1 must be odd for canonical)

    // Pre-allocate output buffer
    uint32_t l = k + w - 1;
    std::vector<uint32_t> positions(len - l + 1);

    // Compute canonical minimizers (zero-copy API)
    uint32_t num_positions = canonical_minimizers_to_buffer(
        reinterpret_cast<const uint8_t*>(dna), len, k, w,
        positions.data(), positions.size());

    // Print results
    printf("Found %u minimizers:\n", num_positions);
    for (uint32_t i = 0; i < num_positions; i++) {
        printf("  position %u\n", positions[i]);
    }

    return 0;
}
