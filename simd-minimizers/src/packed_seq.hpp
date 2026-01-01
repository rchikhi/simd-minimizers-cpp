// packed_seq.hpp - C++ port of Rust's packed-seq crate
// Provides 2-bit DNA packing and SIMD parallel iteration over 8 chunks
//
// Key features:
// - Pack DNA (ACTG) into 2 bits per base
// - Split sequence into 8 chunks for SIMD parallelism
// - Iterate all 8 chunks simultaneously using AVX2 gather
// - Support delayed iteration for rolling hash algorithms

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cassert>

namespace packed_seq {

// Type aliases matching Rust's wide crate
using u32x8 = __m256i;
using u64x4 = __m256i;

// =============================================================================
// 2-bit DNA packing: A=0, C=1, T=2, G=3
// Storage: 4 bases per byte, LSB = first base (little-endian)
// =============================================================================

// Pack an ASCII character to 2-bit representation
inline uint8_t pack_char(uint8_t c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'T': case 't': return 2;
        case 'G': case 'g': return 3;
        default: return 0;  // Treat unknown as A
    }
}

// Unpack 2-bit representation to ASCII
inline char unpack_char(uint8_t b) {
    static const char bases[4] = {'A', 'C', 'T', 'G'};
    return bases[b & 0x03];
}

// Get complement of a 2-bit base: A<->T, C<->G
inline uint8_t complement(uint8_t b) {
    // A(0) <-> T(2), C(1) <-> G(3)
    // XOR with 2 swaps A<->T and C<->G
    return b ^ 2;
}

// =============================================================================
// PackedSeq - 2-bit packed DNA sequence
// =============================================================================

class PackedSeq {
private:
    std::vector<uint8_t> data_;  // Packed data (4 bases per byte)
    size_t len_;                  // Number of bases
    size_t offset_;               // Bit offset within first byte (0, 2, 4, or 6)

public:
    PackedSeq() : len_(0), offset_(0) {}

    // Create from ASCII sequence
    // Note: We add padding to handle SIMD gather operations that may read beyond the sequence
    static PackedSeq from_ascii(const uint8_t* ascii, size_t len) {
        PackedSeq seq;
        seq.len_ = len;
        seq.offset_ = 0;
        // Add extra padding (8 * 4 bytes = 32 bytes) to handle SIMD reads beyond sequence
        // Initialize to zeros so invalid reads produce consistent values
        size_t data_size = (len + 3) / 4 + 32;
        seq.data_.resize(data_size, 0);

        for (size_t i = 0; i < len; i++) {
            size_t byte_idx = i / 4;
            size_t bit_offset = (i % 4) * 2;
            uint8_t base = pack_char(ascii[i]);
            seq.data_[byte_idx] |= (base << bit_offset);
        }
        return seq;
    }

    // Create from already-packed data (0-3 per byte, NOT 4 bases per byte)
    static PackedSeq from_packed_bytes(const uint8_t* packed, size_t len) {
        PackedSeq seq;
        seq.len_ = len;
        seq.offset_ = 0;
        seq.data_.resize((len + 3) / 4);

        for (size_t i = 0; i < len; i++) {
            size_t byte_idx = i / 4;
            size_t bit_offset = (i % 4) * 2;
            uint8_t base = packed[i] & 0x03;
            seq.data_[byte_idx] |= (base << bit_offset);
        }
        return seq;
    }

    size_t len() const { return len_; }
    const uint8_t* data() const { return data_.data(); }

    // Get base at position i
    uint8_t get(size_t i) const {
        assert(i < len_);
        size_t byte_idx = (offset_ / 2 + i) / 4;
        size_t bit_offset = ((offset_ / 2 + i) % 4) * 2;
        return (data_[byte_idx] >> bit_offset) & 0x03;
    }

    // Convert back to ASCII
    std::vector<uint8_t> to_ascii() const {
        std::vector<uint8_t> result(len_);
        for (size_t i = 0; i < len_; i++) {
            result[i] = unpack_char(get(i));
        }
        return result;
    }
};

// =============================================================================
// SIMD Utilities
// =============================================================================

// Gather 4 u64 values from memory at given byte offsets
inline u64x4 gather_u64(const uint8_t* base, u64x4 offsets) {
    return _mm256_i64gather_epi64(
        reinterpret_cast<const long long*>(base),
        offsets,
        1  // scale = 1 byte
    );
}

// Deinterleave two u32x8 vectors
// Input:  a = [a0, a1, a2, a3, a4, a5, a6, a7]
//         b = [b0, b1, b2, b3, b4, b5, b6, b7]
// Output: even = [a0, a2, a4, a6, b0, b2, b4, b6]
//         odd  = [a1, a3, a5, a7, b1, b3, b5, b7]
inline std::pair<u32x8, u32x8> deinterleave(u32x8 a, u32x8 b) {
    // Cast to float for shuffle operations
    __m256 af = _mm256_castsi256_ps(a);
    __m256 bf = _mm256_castsi256_ps(b);

    // Shuffle to extract even/odd elements within 128-bit lanes
    // 0b10_00_10_00 = select elements 0,2 from each pair
    // 0b11_01_11_01 = select elements 1,3 from each pair
    __m256 shuffle_even = _mm256_shuffle_ps(af, bf, 0b10001000);  // 0x88
    __m256 shuffle_odd  = _mm256_shuffle_ps(af, bf, 0b11011101);  // 0xDD

    // Permute to get final order across 128-bit lanes
    __m256d even_d = _mm256_permute4x64_pd(_mm256_castps_pd(shuffle_even), 0b11011000);  // 0xD8
    __m256d odd_d  = _mm256_permute4x64_pd(_mm256_castps_pd(shuffle_odd), 0b11011000);

    return {
        _mm256_castpd_si256(even_d),
        _mm256_castpd_si256(odd_d)
    };
}

// =============================================================================
// SimdSeqIterator - Parallel iteration over 8 chunks
// =============================================================================

// State for SIMD parallel iteration
struct SimdIterState {
    const uint8_t* data;      // Pointer to packed data
    size_t len;               // Total length in bases
    size_t chunk_size;        // Bases per chunk (n)
    size_t par_len;           // Total items per lane (n + context - 1)
    size_t bytes_per_chunk;   // Bytes between chunk starts
    size_t pos;               // Current position within chunks
    u32x8 cur;                // Current buffered data (8 lanes)
    size_t buf_pos;           // Position within buffered data (0-15)
};

// Initialize SIMD iterator state
// context: overlap between chunks (typically k for k-mer processing)
inline SimdIterState simd_iter_init(const PackedSeq& seq, size_t context) {
    SimdIterState state;
    state.data = seq.data();
    state.len = seq.len();

    // Calculate chunk parameters (matching Rust's algorithm)
    // num_kmers_stride = len.saturating_sub(context - 1)
    // n = num_kmers_stride.div_ceil(L).next_multiple_of(C8)
    // where L=8 (SIMD lanes) and C8=4 (chars per byte for 2-bit encoding)
    size_t num_kmers = (state.len > context - 1) ? (state.len - (context - 1)) : 0;
    size_t div_ceil_8 = (num_kmers + 7) / 8;  // div_ceil by L=8
    size_t n = ((div_ceil_8 + 3) / 4) * 4;    // next_multiple_of(C8=4)
    state.chunk_size = n;
    // par_len = n + context - 1 (matching Rust's par_len = n + context + o - 1 with o=0)
    state.par_len = (num_kmers == 0) ? 0 : (n + context - 1);
    state.bytes_per_chunk = n / 4;  // 4 bases per byte

    state.pos = 0;
    state.cur = _mm256_setzero_si256();
    state.buf_pos = 16;  // Force initial load

    return state;
}

// Get chunk offsets as u64x4 (for two gather operations)
inline std::pair<u64x4, u64x4> get_chunk_offsets(const SimdIterState& state, size_t byte_offset) {
    size_t bpc = state.bytes_per_chunk;

    // First 4 chunks: offsets 0, 1*bpc, 2*bpc, 3*bpc
    alignas(32) int64_t off_lo[4] = {
        (int64_t)(byte_offset + 0 * bpc),
        (int64_t)(byte_offset + 1 * bpc),
        (int64_t)(byte_offset + 2 * bpc),
        (int64_t)(byte_offset + 3 * bpc)
    };

    // Last 4 chunks: offsets 4*bpc, 5*bpc, 6*bpc, 7*bpc
    alignas(32) int64_t off_hi[4] = {
        (int64_t)(byte_offset + 4 * bpc),
        (int64_t)(byte_offset + 5 * bpc),
        (int64_t)(byte_offset + 6 * bpc),
        (int64_t)(byte_offset + 7 * bpc)
    };

    return {
        _mm256_load_si256(reinterpret_cast<const __m256i*>(off_lo)),
        _mm256_load_si256(reinterpret_cast<const __m256i*>(off_hi))
    };
}

// Load next batch of 16 characters into iterator state
inline void simd_iter_load(SimdIterState& state) {
    // Calculate byte offset within each chunk
    size_t byte_offset = state.pos / 4;

    // Build gather indices for all 8 lanes
    // Each lane reads from: lane * bytes_per_chunk + byte_offset
    alignas(32) int32_t indices[8];
    for (int i = 0; i < 8; i++) {
        indices[i] = (int32_t)(i * state.bytes_per_chunk + byte_offset);
    }
    __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));

    // Gather 4 bytes (16 bases) for each of 8 lanes
    state.cur = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(state.data),
        idx,
        1  // scale = 1 byte
    );
    state.buf_pos = 0;
}

// Get next 8 characters (one from each chunk) as u32x8
// Each lane contains a 2-bit character value (0-3)
inline u32x8 simd_iter_next(SimdIterState& state) {
    // Load new data every 16 characters
    if (state.buf_pos >= 16) {
        simd_iter_load(state);
    }

    // Extract lowest 2 bits from each lane
    u32x8 mask = _mm256_set1_epi32(0x03);
    u32x8 chars = _mm256_and_si256(state.cur, mask);

    // Shift right by 2 bits for next extraction
    state.cur = _mm256_srli_epi32(state.cur, 2);
    state.buf_pos++;
    state.pos++;

    return chars;
}

// Check if iterator has more elements
inline bool simd_iter_has_next(const SimdIterState& state) {
    return state.pos < state.par_len;
}

// =============================================================================
// DelayedSimdIterator - Returns (add, remove) pairs for rolling hash
// =============================================================================

struct DelayedSimdIterState {
    SimdIterState inner;
    size_t delay;
    std::vector<u32x8> ring_buf;
    size_t ring_mask;
    size_t write_idx;
    size_t read_idx;
    u32x8 upcoming;      // Current characters
    u32x8 upcoming_d;    // Delayed characters
    size_t inner_pos;    // Position within inner u32x8 (0-15)
};

// Initialize delayed iterator
// delay: how many positions back to look for the "remove" character
inline DelayedSimdIterState delayed_iter_init(const PackedSeq& seq, size_t context, size_t delay) {
    DelayedSimdIterState state;
    state.inner = simd_iter_init(seq, context);
    state.delay = delay;

    // Ring buffer size: power of 2 >= delay/16 + 2
    size_t buf_len = 1;
    while (buf_len < delay / 16 + 2) buf_len *= 2;
    state.ring_buf.resize(buf_len, _mm256_setzero_si256());
    state.ring_mask = buf_len - 1;

    state.write_idx = 0;
    state.read_idx = (buf_len - delay / 16) & state.ring_mask;

    state.upcoming = _mm256_setzero_si256();
    state.upcoming_d = _mm256_setzero_si256();
    state.inner_pos = 16;  // Force initial load

    return state;
}

// Load next batch of data for delayed iterator
inline void delayed_iter_load(DelayedSimdIterState& state) {
    if (!simd_iter_has_next(state.inner)) return;

    // Load new u32x8 into ring buffer
    u32x8 next = simd_iter_next(state.inner);
    state.ring_buf[state.write_idx] = next;
    state.upcoming = next;

    // Get delayed value from ring buffer
    state.upcoming_d = state.ring_buf[state.read_idx];

    // Advance indices
    state.write_idx = (state.write_idx + 1) & state.ring_mask;
    state.read_idx = (state.read_idx + 1) & state.ring_mask;
}

// Get next (add, remove) pair as (u32x8, u32x8)
inline std::pair<u32x8, u32x8> delayed_iter_next(DelayedSimdIterState& state) {
    // Load new data every 16 characters
    if (state.inner_pos >= 16) {
        delayed_iter_load(state);
        state.inner_pos = 0;
    }

    // Extract lowest 2 bits from each lane
    u32x8 mask = _mm256_set1_epi32(0x03);
    u32x8 add = _mm256_and_si256(state.upcoming, mask);
    u32x8 remove = _mm256_and_si256(state.upcoming_d, mask);

    // Shift right by 2 bits for next extraction
    state.upcoming = _mm256_srli_epi32(state.upcoming, 2);
    state.upcoming_d = _mm256_srli_epi32(state.upcoming_d, 2);
    state.inner_pos++;

    return {add, remove};
}

// =============================================================================
// Delayed2SimdIterator - Returns (add, remove_k, remove_l) triples
// Used for canonical minimizers: delay_k for ntHash, delay_l for canonical count
// =============================================================================

struct Delayed2SimdIterState {
    SimdIterState inner;
    size_t delay_k;          // Delay for ntHash (k-1)
    size_t delay_l;          // Delay for canonical (l-1 = k+w-2)
    std::vector<u32x8> ring_buf_k;
    std::vector<u32x8> ring_buf_l;
    size_t ring_mask_k;
    size_t ring_mask_l;
    size_t write_idx_k, read_idx_k;
    size_t write_idx_l, read_idx_l;
    u32x8 upcoming;          // Current characters
    u32x8 upcoming_dk;       // Delayed k characters
    u32x8 upcoming_dl;       // Delayed l characters
    size_t inner_pos;        // Position within inner u32x8 (0-15)
};

// Initialize delayed2 iterator for canonical minimizers
inline Delayed2SimdIterState delayed2_iter_init(const PackedSeq& seq, size_t context, size_t delay_k, size_t delay_l) {
    Delayed2SimdIterState state;
    state.inner = simd_iter_init(seq, context);
    state.delay_k = delay_k;
    state.delay_l = delay_l;

    // Ring buffer for delay_k
    size_t buf_len_k = 1;
    while (buf_len_k < delay_k / 16 + 2) buf_len_k *= 2;
    state.ring_buf_k.resize(buf_len_k, _mm256_setzero_si256());
    state.ring_mask_k = buf_len_k - 1;
    state.write_idx_k = 0;
    state.read_idx_k = (buf_len_k - delay_k / 16) & state.ring_mask_k;

    // Ring buffer for delay_l
    size_t buf_len_l = 1;
    while (buf_len_l < delay_l / 16 + 2) buf_len_l *= 2;
    state.ring_buf_l.resize(buf_len_l, _mm256_setzero_si256());
    state.ring_mask_l = buf_len_l - 1;
    state.write_idx_l = 0;
    state.read_idx_l = (buf_len_l - delay_l / 16) & state.ring_mask_l;

    state.upcoming = _mm256_setzero_si256();
    state.upcoming_dk = _mm256_setzero_si256();
    state.upcoming_dl = _mm256_setzero_si256();
    state.inner_pos = 16;  // Force initial load

    return state;
}

// Load next batch of data for delayed2 iterator
inline void delayed2_iter_load(Delayed2SimdIterState& state) {
    if (!simd_iter_has_next(state.inner)) return;

    u32x8 next = simd_iter_next(state.inner);

    // Store in both ring buffers
    state.ring_buf_k[state.write_idx_k] = next;
    state.ring_buf_l[state.write_idx_l] = next;
    state.upcoming = next;

    // Get delayed values
    state.upcoming_dk = state.ring_buf_k[state.read_idx_k];
    state.upcoming_dl = state.ring_buf_l[state.read_idx_l];

    // Advance indices
    state.write_idx_k = (state.write_idx_k + 1) & state.ring_mask_k;
    state.read_idx_k = (state.read_idx_k + 1) & state.ring_mask_k;
    state.write_idx_l = (state.write_idx_l + 1) & state.ring_mask_l;
    state.read_idx_l = (state.read_idx_l + 1) & state.ring_mask_l;
}

// Get next (add, remove_k, remove_l) triple
inline std::tuple<u32x8, u32x8, u32x8> delayed2_iter_next(Delayed2SimdIterState& state) {
    if (state.inner_pos >= 16) {
        delayed2_iter_load(state);
        state.inner_pos = 0;
    }

    u32x8 mask = _mm256_set1_epi32(0x03);
    u32x8 add = _mm256_and_si256(state.upcoming, mask);
    u32x8 remove_k = _mm256_and_si256(state.upcoming_dk, mask);
    u32x8 remove_l = _mm256_and_si256(state.upcoming_dl, mask);

    state.upcoming = _mm256_srli_epi32(state.upcoming, 2);
    state.upcoming_dk = _mm256_srli_epi32(state.upcoming_dk, 2);
    state.upcoming_dl = _mm256_srli_epi32(state.upcoming_dl, 2);
    state.inner_pos++;

    return {add, remove_k, remove_l};
}

// =============================================================================
// Convenience functions for ntHash-style iteration
// =============================================================================

// Process sequence with SIMD parallelism, calling callback for each 8 characters
template<typename F>
void par_iter_bp(const PackedSeq& seq, size_t context, F&& callback) {
    SimdIterState state = simd_iter_init(seq, context);
    while (simd_iter_has_next(state)) {
        u32x8 chars = simd_iter_next(state);
        callback(chars);
    }
}

// Process sequence with delayed iteration for rolling hash
template<typename F>
void par_iter_bp_delayed(const PackedSeq& seq, size_t context, size_t delay, F&& callback) {
    DelayedSimdIterState state = delayed_iter_init(seq, context, delay);
    size_t total = state.inner.chunk_size;
    for (size_t i = 0; i < total && simd_iter_has_next(state.inner); i++) {
        auto [add, remove] = delayed_iter_next(state);
        callback(add, remove);
    }
}

// Process sequence with two delayed iterations for canonical minimizers
template<typename F>
void par_iter_bp_delayed_2(const PackedSeq& seq, size_t context, size_t delay_k, size_t delay_l, F&& callback) {
    Delayed2SimdIterState state = delayed2_iter_init(seq, context, delay_k, delay_l);
    size_t total = state.inner.chunk_size;
    for (size_t i = 0; i < total && simd_iter_has_next(state.inner); i++) {
        auto [add, remove_k, remove_l] = delayed2_iter_next(state);
        callback(add, remove_k, remove_l);
    }
}

} // namespace packed_seq
