// canonical_minimizers_simd.cpp - AVX2 SIMD implementations for canonical minimizers
// This file contains only SIMD implementations. Scalar code is in canonical_minimizers_scalar.cpp

#include "canonical_minimizers.hpp"
#include "packed_seq.hpp"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <climits>

// Debug level (0 = minimal, 1 = hashes, 2 = all)
#ifndef DEBUG_LEVEL
int DEBUG_LEVEL = 0;
#endif

// Type definitions for SIMD
using u32x8 = __m256i;
using i32x8 = __m256i;

// Rotation constant: seq-hash uses R=7 by default (not the classical R=1)
// This reduces correlation between high bits of consecutive hashes.
static constexpr uint32_t ROT = 7;

// =============================================================================
// SIMD Utility Functions
// =============================================================================

// Force inline for hot path functions (matches Rust's aggressive inlining)
#define FORCE_INLINE __attribute__((always_inline)) inline

// Table lookup using AVX2
static FORCE_INLINE u32x8 table_lookup_avx2(u32x8 table, u32x8 indices) {
    return _mm256_permutevar8x32_epi32(table, indices);
}

// SIMD rotate left by ROT bits
static FORCE_INLINE u32x8 simd_rotl(u32x8 x) {
    return _mm256_or_si256(_mm256_slli_epi32(x, ROT), _mm256_srli_epi32(x, 32 - ROT));
}

// SIMD rotate right by ROT bits
static FORCE_INLINE u32x8 simd_rotr(u32x8 x) {
    return _mm256_or_si256(_mm256_srli_epi32(x, ROT), _mm256_slli_epi32(x, 32 - ROT));
}

// =============================================================================
// SIMD ntHash Implementation
// =============================================================================

// Scalar ntHash with 4x loop unrolling (NOT true SIMD - use nthash with packed_seq for SIMD)
std::vector<uint32_t> nthash_scalar_unrolled(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k
) {
    if (seq_len < k) return {};

    const uint32_t num_kmers = seq_len - k + 1;
    std::vector<uint32_t> hashes(num_kmers);

    const uint32_t rot = (k - 1) * ROT;
    uint32_t c_rot_arr[4], f_rot_arr[4];
    for (int i = 0; i < 4; i++) {
        c_rot_arr[i] = rotl32(HASHES_F[complement_base(i)], rot);
        f_rot_arr[i] = rotl32(HASHES_F[i], rot);
    }

    uint32_t fw = 0, rc = 0;
    for (uint32_t i = 0; i < k - 1; i++) {
        uint8_t base = seq_data[i] & 0x03;
        fw = rotl32(fw, ROT) ^ HASHES_F[base];
        rc = rotr32(rc, ROT) ^ c_rot_arr[base];
    }

    // Unroll 4x for better ILP
    uint32_t pos = 0;
    const uint32_t unroll_end = (num_kmers / 4) * 4;

    for (; pos < unroll_end; pos += 4) {
        for (int i = 0; i < 4; i++) {
            uint8_t add = seq_data[pos + i + k - 1] & 0x03;
            uint8_t rem = seq_data[pos + i] & 0x03;
            uint32_t fw_out = rotl32(fw, ROT) ^ HASHES_F[add];
            uint32_t rc_out = rotr32(rc, ROT) ^ c_rot_arr[add];
            hashes[pos + i] = fw_out + rc_out;
            fw = fw_out ^ f_rot_arr[rem];
            rc = rc_out ^ HASHES_F[complement_base(rem)];
        }
    }

    for (; pos < num_kmers; pos++) {
        uint8_t add = seq_data[pos + k - 1] & 0x03;
        uint8_t rem = seq_data[pos] & 0x03;
        uint32_t fw_out = rotl32(fw, ROT) ^ HASHES_F[add];
        uint32_t rc_out = rotr32(rc, ROT) ^ c_rot_arr[add];
        hashes[pos] = fw_out + rc_out;
        fw = fw_out ^ f_rot_arr[rem];
        rc = rc_out ^ HASHES_F[complement_base(rem)];
    }

    return hashes;
}

// =============================================================================
// SIMD Sliding Window Minimum
// =============================================================================

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> sliding_lr_min_simd(
    const std::vector<uint32_t>& hashes,
    uint32_t w
) {
    if (hashes.empty() || hashes.size() < w) {
        return {{}, {}};
    }

    const uint32_t num_hashes = hashes.size();
    const uint32_t num_windows = num_hashes - w + 1;
    std::vector<uint32_t> left_mins(num_windows);
    std::vector<uint32_t> right_mins(num_windows);

    const uint32_t chunk_windows = (num_windows + 7) / 8;

    alignas(32) uint32_t chunk_starts[8];
    alignas(32) uint32_t chunk_ends[8];
    for (int i = 0; i < 8; i++) {
        chunk_starts[i] = i * chunk_windows;
        chunk_ends[i] = std::min(chunk_starts[i] + chunk_windows, num_windows);
    }

    const u32x8 val_mask = _mm256_set1_epi32(0xFFFF0000);
    const u32x8 pos_mask = _mm256_set1_epi32(0x0000FFFF);
    const u32x8 one = _mm256_set1_epi32(1);

    std::vector<std::pair<u32x8, u32x8>> ring_buf(w);
    for (uint32_t i = 0; i < w; i++) {
        ring_buf[i] = {_mm256_set1_epi32(UINT32_MAX), _mm256_setzero_si256()};
    }
    uint32_t ring_idx = 0;

    u32x8 prefix_lmin = _mm256_set1_epi32(UINT32_MAX);
    u32x8 prefix_rmin = _mm256_setzero_si256();
    u32x8 pos = _mm256_setzero_si256();
    u32x8 pos_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(chunk_starts));

    for (uint32_t iter = 0; iter < chunk_windows + w - 1; iter++) {
        alignas(32) uint32_t hash_vals[8];
        for (int lane = 0; lane < 8; lane++) {
            uint32_t hash_idx = chunk_starts[lane] + iter;
            hash_vals[lane] = (hash_idx < num_hashes) ? hashes[hash_idx] : UINT32_MAX;
        }
        u32x8 hash_val = _mm256_load_si256(reinterpret_cast<const __m256i*>(hash_vals));

        u32x8 lelem = _mm256_or_si256(_mm256_and_si256(hash_val, val_mask), pos);
        u32x8 relem = _mm256_or_si256(
            _mm256_and_si256(_mm256_xor_si256(hash_val, _mm256_set1_epi32(-1)), val_mask),
            pos);

        pos = _mm256_add_epi32(pos, one);
        ring_buf[ring_idx] = {lelem, relem};
        ring_idx = (ring_idx + 1) % w;

        prefix_lmin = _mm256_min_epu32(prefix_lmin, lelem);
        prefix_rmin = _mm256_max_epu32(prefix_rmin, relem);

        if (ring_idx == 0) {
            u32x8 suffix_l = ring_buf[w - 1].first;
            u32x8 suffix_r = ring_buf[w - 1].second;
            for (int j = (int)w - 2; j >= 0; j--) {
                suffix_l = _mm256_min_epu32(suffix_l, ring_buf[j].first);
                suffix_r = _mm256_max_epu32(suffix_r, ring_buf[j].second);
                ring_buf[j] = {suffix_l, suffix_r};
            }
            prefix_lmin = lelem;
            prefix_rmin = relem;
        }

        if (iter >= w - 1) {
            auto [suffix_lmin, suffix_rmin] = ring_buf[ring_idx];
            u32x8 lmin = _mm256_min_epu32(prefix_lmin, suffix_lmin);
            u32x8 rmin = _mm256_max_epu32(prefix_rmin, suffix_rmin);

            u32x8 lmin_pos = _mm256_add_epi32(_mm256_and_si256(lmin, pos_mask), pos_offset);
            u32x8 rmin_pos = _mm256_add_epi32(_mm256_and_si256(rmin, pos_mask), pos_offset);

            alignas(32) uint32_t lmin_vals[8], rmin_vals[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(lmin_vals), lmin_pos);
            _mm256_store_si256(reinterpret_cast<__m256i*>(rmin_vals), rmin_pos);

            uint32_t win_idx = iter - (w - 1);
            for (int lane = 0; lane < 8; lane++) {
                uint32_t out_idx = chunk_starts[lane] + win_idx;
                if (out_idx < chunk_ends[lane]) {
                    left_mins[out_idx] = lmin_vals[lane];
                    right_mins[out_idx] = rmin_vals[lane];
                }
            }
        }
    }

    return {left_mins, right_mins};
}

// =============================================================================
// Streaming SIMD Sliding Min State
// =============================================================================

struct SlidingMinState {
    std::vector<u32x8> ring_buf;
    u32x8 prefix_min;
    size_t w;
    size_t idx;
    u32x8 val_mask;
    u32x8 pos_mask;
    u32x8 pos;
    u32x8 pos_offset;
    // Pre-computed constants (hoisted from hot path)
    u32x8 one;
    u32x8 delta;
    // Scalar tracking for efficient overflow check (all lanes advance together)
    uint32_t pos_scalar;
    uint32_t max_pos_scalar;
    uint32_t delta_scalar;

    SlidingMinState(size_t w_, size_t chunk_size, size_t k, size_t initial_pos = 0)
        : w(w_), idx(0) {
        ring_buf.resize(w, _mm256_set1_epi32(0xFFFFFFFF));
        prefix_min = _mm256_set1_epi32(0xFFFFFFFF);
        val_mask = _mm256_set1_epi32(0xFFFF0000);
        pos_mask = _mm256_set1_epi32(0x0000FFFF);
        pos = _mm256_set1_epi32(initial_pos);
        // Pre-compute constants
        one = _mm256_set1_epi32(1);
        delta_scalar = (1 << 16) - 2 - (uint32_t)w;
        delta = _mm256_set1_epi32(delta_scalar);
        max_pos_scalar = (1 << 16) - 1;
        pos_scalar = initial_pos;

        alignas(32) int32_t offsets[8];
        for (int i = 0; i < 8; i++) {
            offsets[i] = (int32_t)(i * chunk_size) - (int32_t)(k - 1);
        }
        pos_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(offsets));
    }

    FORCE_INLINE u32x8 process(u32x8 hash) {
        // Prefetch next ring_buf slot for next iteration
        size_t next_idx = (idx + 1 < w) ? idx + 1 : 0;
        _mm_prefetch(reinterpret_cast<const char*>(&ring_buf[next_idx]), _MM_HINT_T0);

        // Simplified overflow check using scalar tracking (all lanes advance together)
        if (__builtin_expect(pos_scalar == max_pos_scalar, 0)) {
            pos = _mm256_sub_epi32(pos, delta);
            prefix_min = _mm256_sub_epi32(prefix_min, delta);
            pos_offset = _mm256_add_epi32(pos_offset, delta);
            pos_scalar -= delta_scalar;
            for (auto& elem : ring_buf) {
                elem = _mm256_sub_epi32(elem, delta);
            }
        }

        u32x8 elem = _mm256_or_si256(_mm256_and_si256(hash, val_mask), pos);
        ring_buf[idx] = elem;
        prefix_min = _mm256_min_epu32(prefix_min, elem);
        pos = _mm256_add_epi32(pos, one);
        pos_scalar++;
        idx++;

        if (idx == w) {
            idx = 0;
            u32x8 suffix = ring_buf[w - 1];
            for (size_t i = w - 1; i > 0; i--) {
                suffix = _mm256_min_epu32(suffix, ring_buf[i - 1]);
                ring_buf[i - 1] = suffix;
            }
            // After suffix computation, prefix starts fresh with just the current element
            // (Rust: *prefix_min = elem)
            prefix_min = elem;
        }

        u32x8 suffix_min = ring_buf[idx];
        u32x8 min_elem = _mm256_min_epu32(prefix_min, suffix_min);
        return _mm256_add_epi32(_mm256_and_si256(min_elem, pos_mask), pos_offset);
    }
};

// =============================================================================
// Streaming SIMD Sliding LR Min State (Left and Right minimums)
// =============================================================================

struct SlidingLRMinState {
    std::vector<std::pair<u32x8, u32x8>> ring_buf;  // (left_elem, right_elem)
    u32x8 prefix_lmin;  // Prefix minimum for left (min)
    u32x8 prefix_rmax;  // Prefix maximum for right (inverted, so max = rightmost min)
    size_t w;
    size_t idx;
    u32x8 val_mask;
    u32x8 pos_mask;
    u32x8 pos;
    u32x8 pos_offset;
    // Pre-computed constants (hoisted from hot path)
    u32x8 one;
    u32x8 neg_one;  // For inverting hash
    u32x8 delta;
    // Scalar tracking for efficient overflow check (all lanes advance together)
    uint32_t pos_scalar;
    uint32_t max_pos_scalar;
    uint32_t delta_scalar;

    SlidingLRMinState(size_t w_, size_t chunk_size, size_t k, size_t initial_pos = 0)
        : w(w_), idx(0) {
        ring_buf.resize(w, {_mm256_set1_epi32(0xFFFFFFFF), _mm256_setzero_si256()});
        prefix_lmin = _mm256_set1_epi32(0xFFFFFFFF);
        prefix_rmax = _mm256_setzero_si256();
        val_mask = _mm256_set1_epi32(0xFFFF0000);
        pos_mask = _mm256_set1_epi32(0x0000FFFF);
        pos = _mm256_set1_epi32(initial_pos);
        // Pre-compute constants
        one = _mm256_set1_epi32(1);
        neg_one = _mm256_set1_epi32(-1);
        delta_scalar = (1 << 16) - 2 - (uint32_t)w;
        delta = _mm256_set1_epi32(delta_scalar);
        max_pos_scalar = (1 << 16) - 1;
        pos_scalar = initial_pos;

        alignas(32) int32_t offsets[8];
        for (int i = 0; i < 8; i++) {
            offsets[i] = (int32_t)(i * chunk_size) - (int32_t)(k - 1);
        }
        pos_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(offsets));
    }

    // Process hash and return (left_min_pos, right_min_pos)
    FORCE_INLINE std::pair<u32x8, u32x8> process(u32x8 hash) {
        // Prefetch next ring_buf slot for next iteration
        size_t next_idx = (idx + 1 < w) ? idx + 1 : 0;
        _mm_prefetch(reinterpret_cast<const char*>(&ring_buf[next_idx]), _MM_HINT_T0);

        // Simplified overflow check using scalar tracking (all lanes advance together)
        if (__builtin_expect(pos_scalar == max_pos_scalar, 0)) {
            pos = _mm256_sub_epi32(pos, delta);
            prefix_lmin = _mm256_sub_epi32(prefix_lmin, delta);
            prefix_rmax = _mm256_sub_epi32(prefix_rmax, delta);
            pos_offset = _mm256_add_epi32(pos_offset, delta);
            pos_scalar -= delta_scalar;
            for (auto& [l, r] : ring_buf) {
                l = _mm256_sub_epi32(l, delta);
                r = _mm256_sub_epi32(r, delta);
            }
        }

        // Left: smaller hash wins, earlier position as tiebreaker
        u32x8 lelem = _mm256_or_si256(_mm256_and_si256(hash, val_mask), pos);
        // Right: invert hash so max becomes min, later position as tiebreaker
        u32x8 relem = _mm256_or_si256(
            _mm256_and_si256(_mm256_xor_si256(hash, neg_one), val_mask),
            pos);

        ring_buf[idx] = {lelem, relem};
        prefix_lmin = _mm256_min_epu32(prefix_lmin, lelem);
        prefix_rmax = _mm256_max_epu32(prefix_rmax, relem);

        pos = _mm256_add_epi32(pos, one);
        pos_scalar++;
        idx++;

        if (idx == w) {
            idx = 0;
            // Compute suffix minima from right to left
            u32x8 suffix_l = ring_buf[w - 1].first;
            u32x8 suffix_r = ring_buf[w - 1].second;
            for (size_t i = w - 1; i > 0; i--) {
                suffix_l = _mm256_min_epu32(suffix_l, ring_buf[i - 1].first);
                suffix_r = _mm256_max_epu32(suffix_r, ring_buf[i - 1].second);
                ring_buf[i - 1] = {suffix_l, suffix_r};
            }
            // After suffix computation, prefix starts fresh with just the current element
            // (Rust: *prefix_min = elem)
            prefix_lmin = lelem;
            prefix_rmax = relem;
        }

        auto [suffix_lmin, suffix_rmax] = ring_buf[idx];
        u32x8 lmin = _mm256_min_epu32(prefix_lmin, suffix_lmin);
        u32x8 rmax = _mm256_max_epu32(prefix_rmax, suffix_rmax);

        u32x8 lmin_pos = _mm256_add_epi32(_mm256_and_si256(lmin, pos_mask), pos_offset);
        u32x8 rmin_pos = _mm256_add_epi32(_mm256_and_si256(rmax, pos_mask), pos_offset);

        return {lmin_pos, rmin_pos};
    }
};

// =============================================================================
// Canonical Mapper - Counts TG bases to determine canonical flag
// =============================================================================

struct CanonicalMapper {
    i32x8 cnt;      // Running count of (TG - AC) * 2, offset by -l
    i32x8 two;

    CanonicalMapper(size_t k, size_t w) {
        size_t l = k + w - 1;
        cnt = _mm256_set1_epi32(-(int32_t)l);
        two = _mm256_set1_epi32(2);
    }

    // Process (add, remove_l) and return canonical mask
    // Returns all-1s (-1) if canonical, all-0s if not
    // IMPORTANT: Match Rust behavior - check happens AFTER add but BEFORE subtract
    FORCE_INLINE i32x8 process(u32x8 add, u32x8 remove_l) {
        // Count TG bases: T=2 (binary 10), G=3 (binary 11) both have bit 1 set
        // So (base & 2) gives 2 for TG, 0 for AC
        i32x8 add_i = _mm256_and_si256(add, two);
        i32x8 rem_i = _mm256_and_si256(remove_l, two);

        // Add first, then check, then subtract (matching Rust's order)
        cnt = _mm256_add_epi32(cnt, add_i);
        i32x8 result = _mm256_cmpgt_epi32(cnt, _mm256_setzero_si256());
        cnt = _mm256_sub_epi32(cnt, rem_i);

        // Return mask: -1 if cnt > 0 (canonical), 0 otherwise
        return result;
    }
};

// =============================================================================
// SIMD Blend - Select lmin where mask is -1, rmin where mask is 0
// =============================================================================

static FORCE_INLINE u32x8 simd_blend(i32x8 mask, u32x8 lmin, u32x8 rmin) {
    // mask is either all-1s or all-0s per lane
    // blendv selects from second operand where mask bit is 1
    return _mm256_blendv_epi8(rmin, lmin, mask);
}

// =============================================================================
// SIMD Transpose (8x8 matrix of u32)
// Based on: https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
// =============================================================================

FORCE_INLINE void transpose_8x8(const u32x8 m[8], u32x8 t[8]) {
    // Treat as float for shuffle operations (same bit pattern)
    __m256 m0 = _mm256_castsi256_ps(m[0]);
    __m256 m1 = _mm256_castsi256_ps(m[1]);
    __m256 m2 = _mm256_castsi256_ps(m[2]);
    __m256 m3 = _mm256_castsi256_ps(m[3]);
    __m256 m4 = _mm256_castsi256_ps(m[4]);
    __m256 m5 = _mm256_castsi256_ps(m[5]);
    __m256 m6 = _mm256_castsi256_ps(m[6]);
    __m256 m7 = _mm256_castsi256_ps(m[7]);

    __m256 x0 = _mm256_unpacklo_ps(m0, m1);
    __m256 x1 = _mm256_unpackhi_ps(m0, m1);
    __m256 x2 = _mm256_unpacklo_ps(m2, m3);
    __m256 x3 = _mm256_unpackhi_ps(m2, m3);
    __m256 x4 = _mm256_unpacklo_ps(m4, m5);
    __m256 x5 = _mm256_unpackhi_ps(m4, m5);
    __m256 x6 = _mm256_unpacklo_ps(m6, m7);
    __m256 x7 = _mm256_unpackhi_ps(m6, m7);

    #define SHUF(z,y,x,w) (((z)<<6)|((y)<<4)|((x)<<2)|(w))
    __m256 y0 = _mm256_shuffle_ps(x0, x2, SHUF(1,0,1,0));
    __m256 y1 = _mm256_shuffle_ps(x0, x2, SHUF(3,2,3,2));
    __m256 y2 = _mm256_shuffle_ps(x1, x3, SHUF(1,0,1,0));
    __m256 y3 = _mm256_shuffle_ps(x1, x3, SHUF(3,2,3,2));
    __m256 y4 = _mm256_shuffle_ps(x4, x6, SHUF(1,0,1,0));
    __m256 y5 = _mm256_shuffle_ps(x4, x6, SHUF(3,2,3,2));
    __m256 y6 = _mm256_shuffle_ps(x5, x7, SHUF(1,0,1,0));
    __m256 y7 = _mm256_shuffle_ps(x5, x7, SHUF(3,2,3,2));
    #undef SHUF

    t[0] = _mm256_castps_si256(_mm256_permute2f128_ps(y0, y4, 0x20));
    t[1] = _mm256_castps_si256(_mm256_permute2f128_ps(y1, y5, 0x20));
    t[2] = _mm256_castps_si256(_mm256_permute2f128_ps(y2, y6, 0x20));
    t[3] = _mm256_castps_si256(_mm256_permute2f128_ps(y3, y7, 0x20));
    t[4] = _mm256_castps_si256(_mm256_permute2f128_ps(y0, y4, 0x31));
    t[5] = _mm256_castps_si256(_mm256_permute2f128_ps(y1, y5, 0x31));
    t[6] = _mm256_castps_si256(_mm256_permute2f128_ps(y2, y6, 0x31));
    t[7] = _mm256_castps_si256(_mm256_permute2f128_ps(y3, y7, 0x31));
}

// =============================================================================
// SIMD Dedup using Daniel Lemire's algorithm
// https://lemire.me/blog/2017/04/10/removing-duplicates-from-lists-quickly/
// =============================================================================

// Lookup table: for each 8-bit mask of duplicates, shuffle indices to pack unique values
alignas(32) static const uint32_t UNIQSHUF[256][8] = {
    {0,1,2,3,4,5,6,7}, {1,2,3,4,5,6,7,0}, {0,2,3,4,5,6,7,0}, {2,3,4,5,6,7,0,0},
    {0,1,3,4,5,6,7,0}, {1,3,4,5,6,7,0,0}, {0,3,4,5,6,7,0,0}, {3,4,5,6,7,0,0,0},
    {0,1,2,4,5,6,7,0}, {1,2,4,5,6,7,0,0}, {0,2,4,5,6,7,0,0}, {2,4,5,6,7,0,0,0},
    {0,1,4,5,6,7,0,0}, {1,4,5,6,7,0,0,0}, {0,4,5,6,7,0,0,0}, {4,5,6,7,0,0,0,0},
    {0,1,2,3,5,6,7,0}, {1,2,3,5,6,7,0,0}, {0,2,3,5,6,7,0,0}, {2,3,5,6,7,0,0,0},
    {0,1,3,5,6,7,0,0}, {1,3,5,6,7,0,0,0}, {0,3,5,6,7,0,0,0}, {3,5,6,7,0,0,0,0},
    {0,1,2,5,6,7,0,0}, {1,2,5,6,7,0,0,0}, {0,2,5,6,7,0,0,0}, {2,5,6,7,0,0,0,0},
    {0,1,5,6,7,0,0,0}, {1,5,6,7,0,0,0,0}, {0,5,6,7,0,0,0,0}, {5,6,7,0,0,0,0,0},
    {0,1,2,3,4,6,7,0}, {1,2,3,4,6,7,0,0}, {0,2,3,4,6,7,0,0}, {2,3,4,6,7,0,0,0},
    {0,1,3,4,6,7,0,0}, {1,3,4,6,7,0,0,0}, {0,3,4,6,7,0,0,0}, {3,4,6,7,0,0,0,0},
    {0,1,2,4,6,7,0,0}, {1,2,4,6,7,0,0,0}, {0,2,4,6,7,0,0,0}, {2,4,6,7,0,0,0,0},
    {0,1,4,6,7,0,0,0}, {1,4,6,7,0,0,0,0}, {0,4,6,7,0,0,0,0}, {4,6,7,0,0,0,0,0},
    {0,1,2,3,6,7,0,0}, {1,2,3,6,7,0,0,0}, {0,2,3,6,7,0,0,0}, {2,3,6,7,0,0,0,0},
    {0,1,3,6,7,0,0,0}, {1,3,6,7,0,0,0,0}, {0,3,6,7,0,0,0,0}, {3,6,7,0,0,0,0,0},
    {0,1,2,6,7,0,0,0}, {1,2,6,7,0,0,0,0}, {0,2,6,7,0,0,0,0}, {2,6,7,0,0,0,0,0},
    {0,1,6,7,0,0,0,0}, {1,6,7,0,0,0,0,0}, {0,6,7,0,0,0,0,0}, {6,7,0,0,0,0,0,0},
    {0,1,2,3,4,5,7,0}, {1,2,3,4,5,7,0,0}, {0,2,3,4,5,7,0,0}, {2,3,4,5,7,0,0,0},
    {0,1,3,4,5,7,0,0}, {1,3,4,5,7,0,0,0}, {0,3,4,5,7,0,0,0}, {3,4,5,7,0,0,0,0},
    {0,1,2,4,5,7,0,0}, {1,2,4,5,7,0,0,0}, {0,2,4,5,7,0,0,0}, {2,4,5,7,0,0,0,0},
    {0,1,4,5,7,0,0,0}, {1,4,5,7,0,0,0,0}, {0,4,5,7,0,0,0,0}, {4,5,7,0,0,0,0,0},
    {0,1,2,3,5,7,0,0}, {1,2,3,5,7,0,0,0}, {0,2,3,5,7,0,0,0}, {2,3,5,7,0,0,0,0},
    {0,1,3,5,7,0,0,0}, {1,3,5,7,0,0,0,0}, {0,3,5,7,0,0,0,0}, {3,5,7,0,0,0,0,0},
    {0,1,2,5,7,0,0,0}, {1,2,5,7,0,0,0,0}, {0,2,5,7,0,0,0,0}, {2,5,7,0,0,0,0,0},
    {0,1,5,7,0,0,0,0}, {1,5,7,0,0,0,0,0}, {0,5,7,0,0,0,0,0}, {5,7,0,0,0,0,0,0},
    {0,1,2,3,4,7,0,0}, {1,2,3,4,7,0,0,0}, {0,2,3,4,7,0,0,0}, {2,3,4,7,0,0,0,0},
    {0,1,3,4,7,0,0,0}, {1,3,4,7,0,0,0,0}, {0,3,4,7,0,0,0,0}, {3,4,7,0,0,0,0,0},
    {0,1,2,4,7,0,0,0}, {1,2,4,7,0,0,0,0}, {0,2,4,7,0,0,0,0}, {2,4,7,0,0,0,0,0},
    {0,1,4,7,0,0,0,0}, {1,4,7,0,0,0,0,0}, {0,4,7,0,0,0,0,0}, {4,7,0,0,0,0,0,0},
    {0,1,2,3,7,0,0,0}, {1,2,3,7,0,0,0,0}, {0,2,3,7,0,0,0,0}, {2,3,7,0,0,0,0,0},
    {0,1,3,7,0,0,0,0}, {1,3,7,0,0,0,0,0}, {0,3,7,0,0,0,0,0}, {3,7,0,0,0,0,0,0},
    {0,1,2,7,0,0,0,0}, {1,2,7,0,0,0,0,0}, {0,2,7,0,0,0,0,0}, {2,7,0,0,0,0,0,0},
    {0,1,7,0,0,0,0,0}, {1,7,0,0,0,0,0,0}, {0,7,0,0,0,0,0,0}, {7,0,0,0,0,0,0,0},
    {0,1,2,3,4,5,6,0}, {1,2,3,4,5,6,0,0}, {0,2,3,4,5,6,0,0}, {2,3,4,5,6,0,0,0},
    {0,1,3,4,5,6,0,0}, {1,3,4,5,6,0,0,0}, {0,3,4,5,6,0,0,0}, {3,4,5,6,0,0,0,0},
    {0,1,2,4,5,6,0,0}, {1,2,4,5,6,0,0,0}, {0,2,4,5,6,0,0,0}, {2,4,5,6,0,0,0,0},
    {0,1,4,5,6,0,0,0}, {1,4,5,6,0,0,0,0}, {0,4,5,6,0,0,0,0}, {4,5,6,0,0,0,0,0},
    {0,1,2,3,5,6,0,0}, {1,2,3,5,6,0,0,0}, {0,2,3,5,6,0,0,0}, {2,3,5,6,0,0,0,0},
    {0,1,3,5,6,0,0,0}, {1,3,5,6,0,0,0,0}, {0,3,5,6,0,0,0,0}, {3,5,6,0,0,0,0,0},
    {0,1,2,5,6,0,0,0}, {1,2,5,6,0,0,0,0}, {0,2,5,6,0,0,0,0}, {2,5,6,0,0,0,0,0},
    {0,1,5,6,0,0,0,0}, {1,5,6,0,0,0,0,0}, {0,5,6,0,0,0,0,0}, {5,6,0,0,0,0,0,0},
    {0,1,2,3,4,6,0,0}, {1,2,3,4,6,0,0,0}, {0,2,3,4,6,0,0,0}, {2,3,4,6,0,0,0,0},
    {0,1,3,4,6,0,0,0}, {1,3,4,6,0,0,0,0}, {0,3,4,6,0,0,0,0}, {3,4,6,0,0,0,0,0},
    {0,1,2,4,6,0,0,0}, {1,2,4,6,0,0,0,0}, {0,2,4,6,0,0,0,0}, {2,4,6,0,0,0,0,0},
    {0,1,4,6,0,0,0,0}, {1,4,6,0,0,0,0,0}, {0,4,6,0,0,0,0,0}, {4,6,0,0,0,0,0,0},
    {0,1,2,3,6,0,0,0}, {1,2,3,6,0,0,0,0}, {0,2,3,6,0,0,0,0}, {2,3,6,0,0,0,0,0},
    {0,1,3,6,0,0,0,0}, {1,3,6,0,0,0,0,0}, {0,3,6,0,0,0,0,0}, {3,6,0,0,0,0,0,0},
    {0,1,2,6,0,0,0,0}, {1,2,6,0,0,0,0,0}, {0,2,6,0,0,0,0,0}, {2,6,0,0,0,0,0,0},
    {0,1,6,0,0,0,0,0}, {1,6,0,0,0,0,0,0}, {0,6,0,0,0,0,0,0}, {6,0,0,0,0,0,0,0},
    {0,1,2,3,4,5,0,0}, {1,2,3,4,5,0,0,0}, {0,2,3,4,5,0,0,0}, {2,3,4,5,0,0,0,0},
    {0,1,3,4,5,0,0,0}, {1,3,4,5,0,0,0,0}, {0,3,4,5,0,0,0,0}, {3,4,5,0,0,0,0,0},
    {0,1,2,4,5,0,0,0}, {1,2,4,5,0,0,0,0}, {0,2,4,5,0,0,0,0}, {2,4,5,0,0,0,0,0},
    {0,1,4,5,0,0,0,0}, {1,4,5,0,0,0,0,0}, {0,4,5,0,0,0,0,0}, {4,5,0,0,0,0,0,0},
    {0,1,2,3,5,0,0,0}, {1,2,3,5,0,0,0,0}, {0,2,3,5,0,0,0,0}, {2,3,5,0,0,0,0,0},
    {0,1,3,5,0,0,0,0}, {1,3,5,0,0,0,0,0}, {0,3,5,0,0,0,0,0}, {3,5,0,0,0,0,0,0},
    {0,1,2,5,0,0,0,0}, {1,2,5,0,0,0,0,0}, {0,2,5,0,0,0,0,0}, {2,5,0,0,0,0,0,0},
    {0,1,5,0,0,0,0,0}, {1,5,0,0,0,0,0,0}, {0,5,0,0,0,0,0,0}, {5,0,0,0,0,0,0,0},
    {0,1,2,3,4,0,0,0}, {1,2,3,4,0,0,0,0}, {0,2,3,4,0,0,0,0}, {2,3,4,0,0,0,0,0},
    {0,1,3,4,0,0,0,0}, {1,3,4,0,0,0,0,0}, {0,3,4,0,0,0,0,0}, {3,4,0,0,0,0,0,0},
    {0,1,2,4,0,0,0,0}, {1,2,4,0,0,0,0,0}, {0,2,4,0,0,0,0,0}, {2,4,0,0,0,0,0,0},
    {0,1,4,0,0,0,0,0}, {1,4,0,0,0,0,0,0}, {0,4,0,0,0,0,0,0}, {4,0,0,0,0,0,0,0},
    {0,1,2,3,0,0,0,0}, {1,2,3,0,0,0,0,0}, {0,2,3,0,0,0,0,0}, {2,3,0,0,0,0,0,0},
    {0,1,3,0,0,0,0,0}, {1,3,0,0,0,0,0,0}, {0,3,0,0,0,0,0,0}, {3,0,0,0,0,0,0,0},
    {0,1,2,0,0,0,0,0}, {1,2,0,0,0,0,0,0}, {0,2,0,0,0,0,0,0}, {2,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
};

// Thread-local lane cache to avoid allocation per call (like Rust's CACHE)
thread_local std::vector<uint32_t> lane_cache[8];
thread_local bool lane_cache_initialized = false;

// Append unique values from 'new_vals' to output, deduplicating against 'old' (previous 8 values)
// Uses Lemire's SIMD dedup algorithm
FORCE_INLINE size_t append_unique_vals_simd(
    u32x8 old_vals,
    u32x8 new_vals,
    uint32_t* out,
    size_t write_idx
) {
    // Rotate old by 1 and blend with new to get predecessors
    // recon = [new[0], new[1], ..., new[6], old[7]]
    u32x8 recon = _mm256_blend_epi32(old_vals, new_vals, 0b01111111);
    u32x8 rotate_mask = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    u32x8 predecessors = _mm256_permutevar8x32_epi32(recon, rotate_mask);

    // Compare new with predecessors to find duplicates
    u32x8 cmp = _mm256_cmpeq_epi32(predecessors, new_vals);
    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

    // Count unique values and get shuffle indices
    int num_unique = 8 - __builtin_popcount(mask);
    u32x8 shuffle_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(UNIQSHUF[mask]));

    // Shuffle unique values to front and store
    u32x8 packed = _mm256_permutevar8x32_epi32(new_vals, shuffle_idx);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + write_idx), packed);

    return write_idx + num_unique;
}

// Scalar dedup - simpler approach that may be faster for sparse output (like Rust)
// For minimizers with w=11, ~7 of 8 values are duplicates, so scalar branch prediction wins
FORCE_INLINE size_t append_unique_vals_scalar(
    u32x8 old_vals,
    u32x8 new_vals,
    uint32_t* out,
    size_t write_idx
) {
    alignas(32) uint32_t old_arr[8], new_arr[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(old_arr), old_vals);
    _mm256_store_si256(reinterpret_cast<__m256i*>(new_arr), new_vals);

    uint32_t prec = old_arr[7];
    for (int i = 0; i < 8; i++) {
        uint32_t curr = new_arr[i];
        if (curr != prec) {
            out[write_idx++] = curr;
        }
        prec = curr;
    }
    return write_idx;
}

// =============================================================================
// FUSED SIMD Pipeline - Collection/dedup inline with main loop
// =============================================================================

// Non-canonical minimizers with inline collection
void minimizers_simd_fused_packed(
    const packed_seq::PackedSeq& seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions
) {
    using namespace packed_seq;

    const uint32_t l = k + w - 1;
    if (seq_len < l) return;

    // Initialize hash constants for forward-only ntHash
    const uint32_t rot = (k - 1) * ROT;
    alignas(32) uint32_t f_table[8], f_rot_table[8];
    for (int i = 0; i < 4; i++) {
        f_table[i] = f_table[i+4] = HASHES_F[i];
        f_rot_table[i] = f_rot_table[i+4] = rotl32(HASHES_F[i], rot);
    }

    u32x8 simd_f = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_table));
    u32x8 simd_f_rot = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_rot_table));

    // Single delay buffer (only need delay of k-1)
    size_t buf_size = 1;
    while (buf_size < k) buf_size *= 2;
    size_t buf_mask = buf_size - 1;
    alignas(32) u32x8 delay_buf[64];
    for (size_t i = 0; i < buf_size; i++) delay_buf[i] = _mm256_setzero_si256();

    // Initialize iterator
    SimdIterState state = simd_iter_init(seq, l);
    size_t chunk_size = state.chunk_size;

    // Use simple SlidingMinState (left-only, not LR)
    SlidingMinState sliding_min(w, chunk_size, k, 0);

    // Pre-warm hash state
    uint32_t fw_init = 0;
    for (uint32_t i = 0; i < k - 1; i++) {
        fw_init = rotl32(fw_init, ROT) ^ HASHES_F[0];
    }
    u32x8 h_fw = _mm256_set1_epi32(fw_init);

    size_t write_idx = 0, read_idx = 0;
    size_t iter_count = 0;
    size_t output_count = 0;

    // === FUSED COLLECTION STATE (same as canonical) ===
    size_t max_per_lane = chunk_size + 64;
    if (!lane_cache_initialized) {
        for (int i = 0; i < 8; i++) {
            lane_cache[i].reserve(max_per_lane);
        }
        lane_cache_initialized = true;
    }
    for (int i = 0; i < 8; i++) {
        if (lane_cache[i].capacity() < max_per_lane) {
            lane_cache[i].reserve(max_per_lane);
        }
    }
    size_t lane_write_idx[8] = {0};

    u32x8 old_dedup[8];
    for (int i = 0; i < 8; i++) {
        old_dedup[i] = _mm256_set1_epi32(UINT32_MAX);
    }

    u32x8 m[8];
    for (int i = 0; i < 8; i++) {
        m[i] = _mm256_setzero_si256();
    }
    size_t batch_count = 0;

    size_t num_valid_windows = seq_len - l + 1;
    size_t fast_path_limit = (num_valid_windows > 7 * chunk_size + 7) ? num_valid_windows - 7 * chunk_size - 7 : 0;
    const u32x8 idx_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const u32x8 invalid_val = _mm256_set1_epi32(UINT32_MAX);
    const u32x8 valid_limit = _mm256_set1_epi32((uint32_t)num_valid_windows);

    // Main loop with FUSED collection
    while (simd_iter_has_next(state)) {
        u32x8 add = simd_iter_next(state);
        u32x8 remove;

        if (__builtin_expect(iter_count < k - 1, 0)) {
            remove = _mm256_setzero_si256();
        } else {
            remove = delay_buf[read_idx];
            read_idx = (read_idx + 1) & buf_mask;
        }

        delay_buf[write_idx] = add;
        write_idx = (write_idx + 1) & buf_mask;
        iter_count++;

        // Forward-only ntHash
        u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
        h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove));
        u32x8 hash = hfw_out;

        // Sliding min
        u32x8 min_pos = sliding_min.process(hash);
        output_count++;

        if (__builtin_expect(output_count > l - 1, 1)) {
            m[batch_count % 8] = min_pos;
            batch_count++;

            if (batch_count % 8 == 0) {
                u32x8 t[8];
                transpose_8x8(m, t);
                size_t batch_start = batch_count - 8;

                if (batch_start >= fast_path_limit) {
                    for (int j = 0; j < 8; j++) {
                        u32x8 window_base = _mm256_set1_epi32((uint32_t)(j * chunk_size + batch_start));
                        u32x8 window_indices = _mm256_add_epi32(window_base, idx_offsets);
                        u32x8 mask = _mm256_cmpgt_epi32(valid_limit, window_indices);
                        t[j] = _mm256_blendv_epi8(invalid_val, t[j], mask);
                    }
                }

                for (int j = 0; j < 8; j++) {
                    if (lane_write_idx[j] + 8 > lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 1024);
                    }
                    lane_write_idx[j] = append_unique_vals_simd(
                        old_dedup[j], t[j], lane_cache[j].data(), lane_write_idx[j]
                    );
                    old_dedup[j] = t[j];
                }
            }
        }
    }

    // Handle partial last batch
    size_t partial = batch_count % 8;
    if (partial > 0) {
        u32x8 t[8];
        transpose_8x8(m, t);
        size_t batch_start = (batch_count / 8) * 8;

        for (int j = 0; j < 8; j++) {
            union { __m256i vec; uint32_t arr[8]; } u;
            u.vec = t[j];

            uint32_t last_val = (lane_write_idx[j] > 0) ? lane_cache[j][lane_write_idx[j] - 1] : UINT32_MAX;
            size_t lane_window_start = j * chunk_size + batch_start;

            for (size_t idx = 0; idx < partial; idx++) {
                size_t window_idx = lane_window_start + idx;
                if (window_idx >= num_valid_windows) continue;
                if (u.arr[idx] != last_val) {
                    if (lane_write_idx[j] >= lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 64);
                    }
                    lane_cache[j][lane_write_idx[j]++] = u.arr[idx];
                    last_val = u.arr[idx];
                }
            }
        }
    }

    // Truncate and flatten
    for (int j = 0; j < 8; j++) {
        lane_cache[j].resize(lane_write_idx[j]);
    }

    size_t total_elements = 0;
    for (int lane = 0; lane < 8; lane++) {
        total_elements += lane_cache[lane].size();
    }
    out_positions.reserve(out_positions.size() + total_elements);

    for (int lane = 0; lane < 8; lane++) {
        const auto& v = lane_cache[lane];
        if (v.empty()) continue;

        size_t start = 0;
        while (start < v.size() && !out_positions.empty() && v[start] == out_positions.back()) {
            start++;
        }

        size_t end = v.size();
        while (end > start && v[end - 1] == UINT32_MAX) {
            end--;
        }

        if (start < end) {
            out_positions.insert(out_positions.end(), v.begin() + start, v.begin() + end);
        }
    }

    // Tail skipped for benchmarking
    (void)seq_len;
    (void)chunk_size;
    (void)k;
    (void)w;
}

// =============================================================================
// Helper: Append filtered values using SIMD compaction
// Appends values from `vals` where corresponding `mask` bit is 0 (not filtered out)
// =============================================================================
FORCE_INLINE size_t append_filtered_vals_simd(
    u32x8 vals,
    int mask,  // 8-bit mask: 1 = skip, 0 = keep
    uint32_t* out,
    size_t write_idx
) {
    int num_to_write = 8 - __builtin_popcount(mask);
    if (num_to_write == 0) return write_idx;

    u32x8 shuffle_idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(UNIQSHUF[mask]));
    u32x8 packed = _mm256_permutevar8x32_epi32(vals, shuffle_idx);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + write_idx), packed);

    return write_idx + num_to_write;
}

// =============================================================================
// Syncmer computation - inline check during batch processing
// For each window, checks if minimizer is at prefix (pos 0) or suffix (pos w-1)
// =============================================================================
void syncmers_simd_fused_packed(
    const packed_seq::PackedSeq& seq,
    uint32_t seq_len,
    uint32_t k,      // syncmer k-mer size
    uint32_t m,      // minimizer size (s-mer)
    std::vector<uint32_t>& out_positions
) {
    using namespace packed_seq;

    const uint32_t w = k - m + 1;
    const uint32_t l = m + w - 1;  // = k
    if (seq_len < l) return;

    // Hash constants
    const uint32_t rot = (m - 1) * ROT;
    alignas(32) uint32_t f_table[8], f_rot_table[8];
    for (int i = 0; i < 4; i++) {
        f_table[i] = f_table[i+4] = HASHES_F[i];
        f_rot_table[i] = f_rot_table[i+4] = rotl32(HASHES_F[i], rot);
    }
    u32x8 simd_f = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_table));
    u32x8 simd_f_rot = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_rot_table));

    // Delay buffer
    size_t buf_size = 1;
    while (buf_size < m) buf_size *= 2;
    size_t buf_mask = buf_size - 1;
    alignas(32) u32x8 delay_buf[64];
    for (size_t i = 0; i < buf_size; i++) delay_buf[i] = _mm256_setzero_si256();

    SimdIterState state = simd_iter_init(seq, l);
    size_t chunk_size = state.chunk_size;
    SlidingMinState sliding_min(w, chunk_size, m, 0);

    uint32_t fw_init = 0;
    for (uint32_t i = 0; i < m - 1; i++) {
        fw_init = rotl32(fw_init, ROT) ^ HASHES_F[0];
    }
    u32x8 h_fw = _mm256_set1_epi32(fw_init);

    size_t buf_write_idx = 0, buf_read_idx = 0;
    size_t iter_count = 0;
    size_t output_count = 0;

    size_t num_valid_windows = seq_len - l + 1;

    // Lane cache for collecting results
    size_t max_per_lane = chunk_size + 64;
    if (!lane_cache_initialized) {
        for (int i = 0; i < 8; i++) lane_cache[i].reserve(max_per_lane);
        lane_cache_initialized = true;
    }
    // Like minimizers: just check capacity, use lane_write_idx to track positions
    for (int i = 0; i < 8; i++) {
        if (lane_cache[i].capacity() < max_per_lane) lane_cache[i].reserve(max_per_lane);
    }
    size_t lane_write_idx[8] = {0};

    // Batch buffer
    u32x8 batch[8];
    size_t batch_count = 0;

    // Constants for syncmer check
    const u32x8 SIMD_MAX = _mm256_set1_epi32(UINT32_MAX);
    const u32x8 idx_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const u32x8 w_minus_1 = _mm256_set1_epi32(w - 1);

    // Pre-compute lane bases: lane_base[j] = j * chunk_size
    u32x8 lane_base[8];
    for (int j = 0; j < 8; j++) {
        lane_base[j] = _mm256_set1_epi32((uint32_t)(j * chunk_size));
    }

    size_t fast_path_limit = (num_valid_windows > 7 * chunk_size + 7) ? num_valid_windows - 7 * chunk_size - 7 : 0;
    const u32x8 valid_limit = _mm256_set1_epi32((uint32_t)num_valid_windows);

    // Main loop - same as minimizers
    while (simd_iter_has_next(state)) {
        u32x8 add = simd_iter_next(state);
        u32x8 remove;

        if (__builtin_expect(iter_count < m - 1, 0)) {
            remove = _mm256_setzero_si256();
        } else {
            remove = delay_buf[buf_read_idx];
            buf_read_idx = (buf_read_idx + 1) & buf_mask;
        }
        delay_buf[buf_write_idx] = add;
        buf_write_idx = (buf_write_idx + 1) & buf_mask;
        iter_count++;

        // Hash
        u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
        h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove));

        // Sliding min
        u32x8 min_pos = sliding_min.process(hfw_out);
        output_count++;

        if (__builtin_expect(output_count > l - 1, 1)) {
            batch[batch_count % 8] = min_pos;
            batch_count++;

            if (batch_count % 8 == 0) {
                // Transpose and apply syncmer filter
                u32x8 t[8];
                transpose_8x8(batch, t);
                size_t batch_start = batch_count - 8;

                // Validity check for tail batches (same as minimizer code)
                if (batch_start >= fast_path_limit) {
                    for (int j = 0; j < 8; j++) {
                        u32x8 window_base = _mm256_set1_epi32((uint32_t)(j * chunk_size + batch_start));
                        u32x8 window_indices = _mm256_add_epi32(window_base, idx_offsets);
                        u32x8 mask = _mm256_cmpgt_epi32(valid_limit, window_indices);
                        t[j] = _mm256_blendv_epi8(SIMD_MAX, t[j], mask);
                    }
                }

                for (int j = 0; j < 8; j++) {
                    // Compute lane_offsets for this lane
                    u32x8 lane_offsets = _mm256_add_epi32(
                        lane_base[j],
                        _mm256_add_epi32(_mm256_set1_epi32((uint32_t)batch_start), idx_offsets)
                    );
                    u32x8 suffix_offsets = _mm256_add_epi32(lane_offsets, w_minus_1);

                    // Syncmer check: min_pos at prefix or suffix
                    u32x8 is_prefix = _mm256_cmpeq_epi32(t[j], lane_offsets);
                    u32x8 is_suffix = _mm256_cmpeq_epi32(t[j], suffix_offsets);
                    u32x8 is_syncmer = _mm256_or_si256(is_prefix, is_suffix);

                    // Output lane_offsets if syncmer, else MAX
                    u32x8 result = _mm256_blendv_epi8(SIMD_MAX, lane_offsets, is_syncmer);

                    // Filter and collect
                    if (lane_write_idx[j] + 8 > lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 1024);
                    }
                    u32x8 cmp_max = _mm256_cmpeq_epi32(result, SIMD_MAX);
                    int skip_mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_max));
                    lane_write_idx[j] = append_filtered_vals_simd(
                        result, skip_mask, lane_cache[j].data(), lane_write_idx[j]
                    );
                }
            }
        }
    }

    // Handle partial last batch
    size_t partial = batch_count % 8;
    if (partial > 0) {
        for (size_t i = partial; i < 8; i++) batch[i] = SIMD_MAX;
        u32x8 t[8];
        transpose_8x8(batch, t);
        size_t batch_start = (batch_count / 8) * 8;
        u32x8 batch_start_plus_idx = _mm256_add_epi32(
            _mm256_set1_epi32((uint32_t)batch_start), idx_offsets
        );

        for (int j = 0; j < 8; j++) {
            // After transpose: t[j][i] is from lane j, iteration (batch_start + i)
            // Window position for element i = j * chunk_size + batch_start + i
            u32x8 lane_offsets = _mm256_add_epi32(lane_base[j], batch_start_plus_idx);
            u32x8 suffix_offsets = _mm256_add_epi32(lane_offsets, w_minus_1);

            u32x8 is_prefix = _mm256_cmpeq_epi32(t[j], lane_offsets);
            u32x8 is_suffix = _mm256_cmpeq_epi32(t[j], suffix_offsets);
            u32x8 is_syncmer = _mm256_or_si256(is_prefix, is_suffix);

            // Output window position (lane_offsets) if syncmer, else MAX
            u32x8 result = _mm256_blendv_epi8(SIMD_MAX, lane_offsets, is_syncmer);

            // Always check validity for partial batch
            u32x8 valid_mask = _mm256_cmpgt_epi32(valid_limit, lane_offsets);
            result = _mm256_blendv_epi8(SIMD_MAX, result, valid_mask);

            if (lane_write_idx[j] + 8 > lane_cache[j].size()) {
                lane_cache[j].resize(lane_cache[j].size() + 1024);
            }
            u32x8 cmp_max = _mm256_cmpeq_epi32(result, SIMD_MAX);
            int skip_mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_max));
            lane_write_idx[j] = append_filtered_vals_simd(
                result, skip_mask, lane_cache[j].data(), lane_write_idx[j]
            );
        }
    }

    // Just concatenate lanes (no sort/dedup for now - testing raw performance)
    size_t total = 0;
    for (int i = 0; i < 8; i++) total += lane_write_idx[i];
    out_positions.reserve(total);

    for (int i = 0; i < 8; i++) {
        out_positions.insert(out_positions.end(),
                            lane_cache[i].data(),
                            lane_cache[i].data() + lane_write_idx[i]);
    }
}

// =============================================================================
// Timing struct for profiling (used by benchmark_canonical_phases)
// =============================================================================
struct FusedPhaseTiming {
    uint64_t init_us = 0;           // Hash table + delay buffer + iterator setup
    uint64_t main_loop_us = 0;      // Main processing loop with inline collection
    uint64_t partial_batch_us = 0;  // Handling leftover batch
    uint64_t truncate_us = 0;       // Resizing lane caches
    uint64_t prereserve_us = 0;     // Reserving output vector
    uint64_t flatten_us = 0;        // Copying lanes to output
};

// =============================================================================
// Core fused implementation (template to optionally enable profiling)
// =============================================================================
template<bool PROFILE>
void canonical_minimizers_simd_fused_impl(
    const packed_seq::PackedSeq& seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions,
    FusedPhaseTiming* timing = nullptr
) {
    using namespace packed_seq;
    using Clock = std::chrono::high_resolution_clock;

    const uint32_t l = k + w - 1;
    if (seq_len < l) return;
    if (l % 2 == 0) return;

    // === PHASE 0: INITIALIZATION ===
    [[maybe_unused]] auto t0 = PROFILE ? Clock::now() : Clock::time_point{};

    // Initialize hash constants for canonical ntHash
    const uint32_t rot = (k - 1) * ROT;
    alignas(32) uint32_t f_table[8], c_table[8], f_rot_table[8], c_rot_table[8];
    for (int i = 0; i < 4; i++) {
        f_table[i] = f_table[i+4] = HASHES_F[i];
        c_table[i] = c_table[i+4] = HASHES_F[complement_base(i)];
        f_rot_table[i] = f_rot_table[i+4] = rotl32(HASHES_F[i], rot);
        c_rot_table[i] = c_rot_table[i+4] = rotl32(HASHES_F[complement_base(i)], rot);
    }

    u32x8 simd_f = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_table));
    u32x8 simd_c = _mm256_load_si256(reinterpret_cast<const __m256i*>(c_table));
    u32x8 simd_f_rot = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_rot_table));
    u32x8 simd_c_rot = _mm256_load_si256(reinterpret_cast<const __m256i*>(c_rot_table));

    // Delay buffers
    size_t buf_size_k = 1;
    while (buf_size_k < k) buf_size_k *= 2;
    size_t buf_mask_k = buf_size_k - 1;

    size_t buf_size_l = 1;
    while (buf_size_l < l + 1) buf_size_l *= 2;
    size_t buf_mask_l = buf_size_l - 1;

    alignas(32) u32x8 delay_buf_k[64];
    alignas(32) u32x8 delay_buf_l[128];
    for (size_t i = 0; i < buf_size_k; i++) delay_buf_k[i] = _mm256_setzero_si256();
    for (size_t i = 0; i < buf_size_l; i++) delay_buf_l[i] = _mm256_setzero_si256();

    // Initialize iterator
    SimdIterState state = simd_iter_init(seq, l);
    size_t chunk_size = state.chunk_size;

    // Initialize components
    SlidingLRMinState sliding_min(w, chunk_size, k, 0);
    CanonicalMapper canonical_mapper(k, w);

    // Pre-warm hash state
    uint32_t fw_init = 0, rc_init = 0;
    for (uint32_t i = 0; i < k - 1; i++) {
        fw_init = rotl32(fw_init, ROT) ^ HASHES_F[0];
        rc_init = rotr32(rc_init, ROT) ^ c_rot_table[0];
    }
    u32x8 h_fw = _mm256_set1_epi32(fw_init);
    u32x8 h_rc = _mm256_set1_epi32(rc_init);

    size_t write_idx_k = 0, read_idx_k = 0;
    size_t write_idx_l = 0, read_idx_l = 0;
    size_t iter_count = 0;
    size_t output_count = 0;

    // === FUSED COLLECTION STATE ===
    size_t max_per_lane = chunk_size + 64;
    if (!lane_cache_initialized) {
        for (int i = 0; i < 8; i++) {
            lane_cache[i].reserve(max_per_lane);
        }
        lane_cache_initialized = true;
    }
    for (int i = 0; i < 8; i++) {
        if (lane_cache[i].capacity() < max_per_lane) {
            lane_cache[i].reserve(max_per_lane);
        }
    }
    size_t lane_write_idx[8] = {0};

    // Dedup state
    u32x8 old_dedup[8];
    for (int i = 0; i < 8; i++) {
        old_dedup[i] = _mm256_set1_epi32(UINT32_MAX);
    }

    // Transpose batch buffer
    u32x8 m[8];
    for (int i = 0; i < 8; i++) {
        m[i] = _mm256_setzero_si256();
    }
    size_t batch_count = 0;

    // Pre-compute fast-path threshold
    size_t num_valid_windows = seq_len - l + 1;
    size_t fast_path_limit = 0;
    if (num_valid_windows > 7 * chunk_size + 7) {
        fast_path_limit = num_valid_windows - 7 * chunk_size - 7;
    }

    const u32x8 idx_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const u32x8 invalid_val = _mm256_set1_epi32(UINT32_MAX);
    const u32x8 valid_limit = _mm256_set1_epi32((uint32_t)num_valid_windows);

    if constexpr (PROFILE) {
        timing->init_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
    }

    // === PHASE 1: MAIN LOOP ===
    [[maybe_unused]] auto t1 = PROFILE ? Clock::now() : Clock::time_point{};

    while (simd_iter_has_next(state)) {
        u32x8 add = simd_iter_next(state);
        u32x8 remove_k, remove_l;

        if (iter_count < k - 1) {
            remove_k = _mm256_setzero_si256();
        } else {
            remove_k = delay_buf_k[read_idx_k];
            read_idx_k = (read_idx_k + 1) & buf_mask_k;
        }

        // Canonical mapper uses delay = l-1 (matching Rust's Delay(l-1))
        if (iter_count < l - 1) {
            remove_l = _mm256_setzero_si256();
        } else {
            remove_l = delay_buf_l[read_idx_l];
            read_idx_l = (read_idx_l + 1) & buf_mask_l;
        }

        delay_buf_k[write_idx_k] = add;
        write_idx_k = (write_idx_k + 1) & buf_mask_k;
        delay_buf_l[write_idx_l] = add;
        write_idx_l = (write_idx_l + 1) & buf_mask_l;

        iter_count++;

        u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
        h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove_k));

        u32x8 hrc_out = _mm256_xor_si256(simd_rotr(h_rc), table_lookup_avx2(simd_c_rot, add));
        h_rc = _mm256_xor_si256(hrc_out, table_lookup_avx2(simd_c, remove_k));

        u32x8 hash = _mm256_add_epi32(hfw_out, hrc_out);

        i32x8 canonical_mask = canonical_mapper.process(add, remove_l);
        auto [lmin_pos, rmin_pos] = sliding_min.process(hash);
        u32x8 selected_pos = simd_blend(canonical_mask, lmin_pos, rmin_pos);

        output_count++;

        if (output_count > l - 1) {
            m[batch_count % 8] = selected_pos;
            batch_count++;

            if (batch_count % 8 == 0) {
                u32x8 t[8];
                transpose_8x8(m, t);

                size_t batch_start = batch_count - 8;

                if (batch_start >= fast_path_limit) {
                    for (int j = 0; j < 8; j++) {
                        u32x8 window_base = _mm256_set1_epi32((uint32_t)(j * chunk_size + batch_start));
                        u32x8 window_indices = _mm256_add_epi32(window_base, idx_offsets);
                        u32x8 mask = _mm256_cmpgt_epi32(valid_limit, window_indices);
                        t[j] = _mm256_blendv_epi8(invalid_val, t[j], mask);
                    }
                }

                for (int j = 0; j < 8; j++) {
                    if (lane_write_idx[j] + 8 > lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 1024);
                    }
                    lane_write_idx[j] = append_unique_vals_simd(
                        old_dedup[j], t[j], lane_cache[j].data(), lane_write_idx[j]
                    );
                    old_dedup[j] = t[j];
                }
            }
        }
    }

    if constexpr (PROFILE) {
        timing->main_loop_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t1).count();
    }

    // === PHASE 2: PARTIAL BATCH ===
    [[maybe_unused]] auto t2 = PROFILE ? Clock::now() : Clock::time_point{};

    size_t partial = batch_count % 8;
    if (partial > 0) {
        u32x8 t[8];
        transpose_8x8(m, t);
        size_t batch_start = (batch_count / 8) * 8;

        for (int j = 0; j < 8; j++) {
            union { __m256i vec; uint32_t arr[8]; } u;
            u.vec = t[j];

            uint32_t last_val = (lane_write_idx[j] > 0) ? lane_cache[j][lane_write_idx[j] - 1] : UINT32_MAX;
            size_t lane_window_start = j * chunk_size + batch_start;

            for (size_t idx = 0; idx < partial; idx++) {
                size_t window_idx = lane_window_start + idx;
                if (window_idx >= num_valid_windows) continue;
                if (u.arr[idx] != last_val) {
                    if (lane_write_idx[j] >= lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 64);
                    }
                    lane_cache[j][lane_write_idx[j]++] = u.arr[idx];
                    last_val = u.arr[idx];
                }
            }
        }
    }

    if constexpr (PROFILE) {
        timing->partial_batch_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t2).count();
    }

    // === PHASE 3: TRUNCATE ===
    [[maybe_unused]] auto t3 = PROFILE ? Clock::now() : Clock::time_point{};

    for (int j = 0; j < 8; j++) {
        lane_cache[j].resize(lane_write_idx[j]);
    }

    if constexpr (PROFILE) {
        timing->truncate_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t3).count();
    }

    // === PHASE 4: PRE-RESERVE ===
    [[maybe_unused]] auto t4 = PROFILE ? Clock::now() : Clock::time_point{};

    size_t total_elements = 0;
    for (int lane = 0; lane < 8; lane++) {
        total_elements += lane_cache[lane].size();
    }
    out_positions.reserve(out_positions.size() + total_elements);

    if constexpr (PROFILE) {
        timing->prereserve_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t4).count();
    }

    // === PHASE 5: FLATTEN ===
    [[maybe_unused]] auto t5 = PROFILE ? Clock::now() : Clock::time_point{};

    for (int lane = 0; lane < 8; lane++) {
        const auto& v = lane_cache[lane];
        if (v.empty()) continue;

        size_t start = 0;
        while (start < v.size() && !out_positions.empty() && v[start] == out_positions.back()) {
            start++;
        }

        size_t end = v.size();
        while (end > start && v[end - 1] == UINT32_MAX) {
            end--;
        }

        if (start < end) {
            out_positions.insert(out_positions.end(), v.begin() + start, v.begin() + end);
        }
    }

    if constexpr (PROFILE) {
        timing->flatten_us += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t5).count();
    }

    (void)seq_len;
    (void)chunk_size;
    (void)k;
    (void)w;
}

// Core implementation that takes pre-packed sequence (for fair benchmarking against Rust)
void canonical_minimizers_simd_fused_packed(
    const packed_seq::PackedSeq& seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions
) {
    canonical_minimizers_simd_fused_impl<false>(seq, seq_len, k, w, out_positions, nullptr);
}

// Zero-copy FFI: writes directly to provided buffer (avoids malloc + double copy)
// Uses thread-local storage to avoid repeated allocation of packed buffer
// Returns number of positions written, or 0 if buffer too small
extern "C" uint32_t noncanonical_minimizers_to_buffer(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t* out_buf,
    uint32_t out_capacity
) {
    if (seq_len < k + w - 1) return 0;

    // Thread-local packed buffer - reused across calls to avoid allocation
    thread_local packed_seq::PackedSeq seq;
    seq.pack_into(seq_data, seq_len);

    // Thread-local output vector - also reused
    thread_local std::vector<uint32_t> minimizers;
    minimizers.clear();
    minimizers.reserve(std::min(out_capacity, (seq_len - k + 1) / w));

    minimizers_simd_fused_packed(seq, seq_len, k, w, minimizers);

    uint32_t count = std::min((uint32_t)minimizers.size(), out_capacity);
    memcpy(out_buf, minimizers.data(), count * sizeof(uint32_t));
    return count;
}

// Zero-copy FFI for canonical minimizers
// Uses thread-local storage to avoid repeated allocation
extern "C" uint32_t canonical_minimizers_to_buffer(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t* out_buf,
    uint32_t out_capacity
) {
    uint32_t l = k + w - 1;
    if (seq_len < l || l % 2 == 0) return 0;  // Odd window required for canonicality

    thread_local packed_seq::PackedSeq seq;
    seq.pack_into(seq_data, seq_len);

    thread_local std::vector<uint32_t> minimizers;
    minimizers.clear();
    // Reserve for worst case: one position per window
    uint32_t num_windows = seq_len - l + 1;
    minimizers.reserve(std::min(out_capacity, num_windows));

    canonical_minimizers_simd_fused_packed(seq, seq_len, k, w, minimizers);

    uint32_t count = std::min((uint32_t)minimizers.size(), out_capacity);
    memcpy(out_buf, minimizers.data(), count * sizeof(uint32_t));
    return count;
}

// Zero-copy FFI for syncmers
// Uses thread-local storage to avoid repeated allocation
extern "C" uint32_t syncmers_to_buffer(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k,
    uint32_t m,
    uint32_t* out_buf,
    uint32_t out_capacity
) {
    if (seq_len < k) return 0;

    thread_local packed_seq::PackedSeq seq;
    seq.pack_into(seq_data, seq_len);

    thread_local std::vector<uint32_t> positions;
    positions.clear();
    positions.reserve(std::min(out_capacity, seq_len / (k - m + 1)));

    syncmers_simd_fused_packed(seq, seq_len, k, m, positions);

    uint32_t count = std::min((uint32_t)positions.size(), out_capacity);
    memcpy(out_buf, positions.data(), count * sizeof(uint32_t));
    return count;
}

// =============================================================================
// Test Functions
// =============================================================================

extern "C" int test_sliding_min_scalar(
    const uint32_t* hashes,
    uint32_t hash_len,
    uint32_t w
) {
    std::vector<uint32_t> hash_vec(hashes, hashes + hash_len);

    auto [left_fast, right_fast] = sliding_lr_min_scalar(hash_vec, w);
    auto [left_naive, right_naive] = sliding_lr_min_naive(hash_vec, w);

    if (left_fast.size() != left_naive.size() || right_fast.size() != right_naive.size()) {
        fprintf(stderr, "Size mismatch: fast=(%zu,%zu), naive=(%zu,%zu)\n",
                left_fast.size(), right_fast.size(), left_naive.size(), right_naive.size());
        return 0;
    }

    for (size_t i = 0; i < left_fast.size(); i++) {
        if (left_fast[i] != left_naive[i] || right_fast[i] != right_naive[i]) {
            fprintf(stderr, "Mismatch at window %zu\n", i);
            return 0;
        }
    }
    return 1;
}

extern "C" int test_sliding_min_simd(
    const uint32_t* hashes,
    uint32_t hash_len,
    uint32_t w
) {
    std::vector<uint32_t> hash_vec(hashes, hashes + hash_len);

    auto [left_simd, right_simd] = sliding_lr_min_simd(hash_vec, w);
    auto [left_naive, right_naive] = sliding_lr_min_naive(hash_vec, w);

    if (left_simd.size() != left_naive.size() || right_simd.size() != right_naive.size()) {
        fprintf(stderr, "Size mismatch: simd=(%zu,%zu), naive=(%zu,%zu)\n",
                left_simd.size(), right_simd.size(), left_naive.size(), right_naive.size());
        return 0;
    }

    for (size_t i = 0; i < left_simd.size(); i++) {
        if (left_simd[i] != left_naive[i] || right_simd[i] != right_naive[i]) {
            fprintf(stderr, "Mismatch at window %zu\n", i);
            return 0;
        }
    }
    return 1;
}

extern "C" int test_packed_seq(
    const uint8_t* ascii_seq,
    uint32_t seq_len
) {
    using namespace packed_seq;

    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);
    auto ascii_back = seq.to_ascii();

    for (uint32_t i = 0; i < seq_len; i++) {
        char orig = (ascii_seq[i] >= 'a') ? (ascii_seq[i] - 32) : ascii_seq[i];
        if (ascii_back[i] != orig) {
            fprintf(stderr, "Round-trip mismatch at %u\n", i);
            return 0;
        }
    }

    for (uint32_t i = 0; i < seq_len; i++) {
        if (seq.get(i) != pack_char(ascii_seq[i])) {
            fprintf(stderr, "get(%u) mismatch\n", i);
            return 0;
        }
    }

    if (seq_len >= 64) {
        SimdIterState state = simd_iter_init(seq, 1);
        size_t iter_count = 0;
        while (simd_iter_has_next(state) && iter_count < 10) {
            u32x8 chars = simd_iter_next(state);
            alignas(32) uint32_t values[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(values), chars);
            for (int lane = 0; lane < 8; lane++) {
                if (values[lane] > 3) {
                    fprintf(stderr, "Invalid base at iter %zu, lane %d\n", iter_count, lane);
                    return 0;
                }
            }
            iter_count++;
        }
    }
    return 1;
}

extern "C" int test_noncanonical_minimizers_simd_vs_rust(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    const uint32_t* rust_positions,
    uint32_t rust_len
) {
    std::vector<uint32_t> cpp_positions;
    auto seq = packed_seq::PackedSeq::from_ascii(ascii_seq, seq_len);
    minimizers_simd_fused_packed(seq, seq_len, k, w, cpp_positions);

    fprintf(stderr, "C++ positions (%zu): ", cpp_positions.size());
    for (size_t i = 0; i < std::min(cpp_positions.size(), (size_t)20); i++) {
        fprintf(stderr, "%u ", cpp_positions[i]);
    }
    fprintf(stderr, "...\n");

    fprintf(stderr, "Rust positions (%u): ", rust_len);
    for (size_t i = 0; i < std::min((size_t)rust_len, (size_t)20); i++) {
        fprintf(stderr, "%u ", rust_positions[i]);
    }
    fprintf(stderr, "...\n");

    size_t match_count = 0;
    size_t cpp_idx = 0, rust_idx = 0;

    while (cpp_idx < cpp_positions.size() && rust_idx < rust_len) {
        if (cpp_positions[cpp_idx] == rust_positions[rust_idx]) {
            match_count++;
            cpp_idx++;
            rust_idx++;
        } else if (cpp_positions[cpp_idx] < rust_positions[rust_idx]) {
            fprintf(stderr, "Mismatch at cpp_idx=%zu\n", cpp_idx);
            return 0;
        } else {
            rust_idx++;
        }
    }

    if (cpp_idx < cpp_positions.size()) {
        fprintf(stderr, "C++ has extra positions\n");
        return 0;
    }

    size_t rust_remaining = rust_len - rust_idx;
    if (rust_remaining > 0) {
        fprintf(stderr, "Matched %zu positions, Rust has %zu more (likely tail)\n",
                match_count, rust_remaining);
    } else {
        fprintf(stderr, "Matched %zu positions\n", match_count);
    }

    return (match_count > 0 && match_count >= cpp_positions.size()) ? 1 : 0;
}

// =============================================================================
// Benchmark Functions
// =============================================================================

extern "C" uint64_t benchmark_packed_seq_simd(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t iterations
) {
    using namespace packed_seq;
    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t sum = 0;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        DelayedSimdIterState state = delayed_iter_init(seq, k, k - 1);
        size_t count = 0;
        while (simd_iter_has_next(state.inner) && count < state.inner.chunk_size) {
            auto [add, remove] = delayed_iter_next(state);
            alignas(32) uint32_t add_vals[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(add_vals), add);
            sum += add_vals[0];
            count++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

extern "C" uint64_t benchmark_nthash_packed_seq(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t iterations
) {
    using namespace packed_seq;
    if (seq_len < k) return 0;

    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);

    const uint32_t rot = (k - 1) * ROT;
    alignas(32) uint32_t f_table[8], c_table[8], f_rot_table[8], c_rot_table[8];
    for (int i = 0; i < 4; i++) {
        f_table[i] = f_table[i+4] = HASHES_F[i];
        c_table[i] = c_table[i+4] = HASHES_F[complement_base(i)];
        f_rot_table[i] = f_rot_table[i+4] = rotl32(HASHES_F[i], rot);
        c_rot_table[i] = c_rot_table[i+4] = rotl32(HASHES_F[complement_base(i)], rot);
    }

    u32x8 simd_f = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_table));
    u32x8 simd_c = _mm256_load_si256(reinterpret_cast<const __m256i*>(c_table));
    u32x8 simd_f_rot = _mm256_load_si256(reinterpret_cast<const __m256i*>(f_rot_table));
    u32x8 simd_c_rot = _mm256_load_si256(reinterpret_cast<const __m256i*>(c_rot_table));

    size_t buf_size = 1;
    while (buf_size < k) buf_size *= 2;
    size_t buf_mask = buf_size - 1;
    alignas(32) u32x8 delay_buf[64];

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t sum = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        SimdIterState state = simd_iter_init(seq, k);
        u32x8 h_fw = _mm256_setzero_si256();
        u32x8 h_rc = _mm256_setzero_si256();
        u32x8 hash_sum = _mm256_setzero_si256();
        size_t delay_idx = 0;

        for (uint32_t i = 0; i < k - 1 && simd_iter_has_next(state); i++) {
            delay_buf[delay_idx] = simd_iter_next(state);
            delay_idx = (delay_idx + 1) & buf_mask;
        }

        while (simd_iter_has_next(state)) {
            u32x8 add = simd_iter_next(state);
            u32x8 remove = delay_buf[delay_idx];
            delay_buf[delay_idx] = add;
            delay_idx = (delay_idx + 1) & buf_mask;

            u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
            h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove));

            u32x8 hrc_out = _mm256_xor_si256(simd_rotr(h_rc), table_lookup_avx2(simd_c_rot, add));
            h_rc = _mm256_xor_si256(hrc_out, table_lookup_avx2(simd_c, remove));

            u32x8 canonical = _mm256_add_epi32(hfw_out, hrc_out);
            hash_sum = _mm256_add_epi32(hash_sum, canonical);
        }

        alignas(32) uint32_t sums[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(sums), hash_sum);
        sum += sums[0];
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark non-canonical pipeline (pre-packed sequence)
extern "C" uint64_t benchmark_noncanonical(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t iterations
) {
    using namespace packed_seq;

    if (seq_len < k + w - 1) return 0;

    // Pre-pack sequence OUTSIDE timing loop (same as Rust benchmark)
    PackedSeq seq_packed = PackedSeq::from_ascii(ascii_seq, seq_len);

    std::vector<uint32_t> positions;
    positions.reserve((seq_len - k + 1) / w);

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t result_sum = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        positions.clear();
        minimizers_simd_fused_packed(seq_packed, seq_len, k, w, positions);

        if (!positions.empty()) {
            result_sum += positions[0];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark canonical pipeline (pre-packed sequence)
extern "C" uint64_t benchmark_canonical(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t iterations
) {
    using namespace packed_seq;

    if (seq_len < k + w - 1) return 0;
    if ((k + w - 1) % 2 == 0) return 0;  // l must be odd

    // Pre-pack sequence OUTSIDE timing loop (same as Rust benchmark)
    PackedSeq seq_packed = PackedSeq::from_ascii(ascii_seq, seq_len);

    std::vector<uint32_t> positions;
    positions.reserve((seq_len - k + 1) / w);

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t result_sum = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        positions.clear();
        canonical_minimizers_simd_fused_packed(seq_packed, seq_len, k, w, positions);

        if (!positions.empty()) {
            result_sum += positions[0];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark canonical phases separately: init, main_loop, collection
// Uses the template with PROFILE=true to get detailed timing, then aggregates
extern "C" void benchmark_canonical_phases(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t iterations,
    uint64_t* init_us,
    uint64_t* main_loop_us,
    uint64_t* collection_us
) {
    using namespace packed_seq;

    if (seq_len < k + w - 1) {
        *init_us = *main_loop_us = *collection_us = 0;
        return;
    }

    // Pre-pack sequence OUTSIDE timing loop (same as Rust benchmark)
    PackedSeq seq_packed = PackedSeq::from_ascii(ascii_seq, seq_len);

    FusedPhaseTiming timing = {};
    std::vector<uint32_t> out_positions;
    out_positions.reserve((seq_len - k + 1) / w);

    for (uint32_t iter = 0; iter < iterations; iter++) {
        out_positions.clear();
        canonical_minimizers_simd_fused_impl<true>(seq_packed, seq_len, k, w, out_positions, &timing);
    }

    // Aggregate: init, main_loop (includes partial batch), collection (truncate + prereserve + flatten)
    *init_us = timing.init_us;
    *main_loop_us = timing.main_loop_us + timing.partial_batch_us;
    *collection_us = timing.truncate_us + timing.prereserve_us + timing.flatten_us;
}

extern "C" uint32_t get_cpp_forward_hashes(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k,
    uint32_t* out_hashes,
    uint32_t max_hashes
) {
    if (seq_len < k) return 0;

    const uint32_t num_kmers = seq_len - k + 1;
    const uint32_t to_write = std::min(num_kmers, max_hashes);

    const uint32_t rot = (k - 1) * ROT;
    uint32_t f_rot[4];
    for (int i = 0; i < 4; i++) {
        f_rot[i] = rotl32(HASHES_F[i], rot);
    }

    uint32_t fw = 0;
    for (uint32_t i = 0; i < k - 1; i++) {
        uint8_t base = packed_seq::pack_char(seq_data[i]);
        fw = rotl32(fw, ROT) ^ HASHES_F[base];
    }

    for (uint32_t pos = 0; pos < to_write; pos++) {
        uint8_t add_base = packed_seq::pack_char(seq_data[pos + k - 1]);
        uint8_t rem_base = packed_seq::pack_char(seq_data[pos]);
        uint32_t fw_out = rotl32(fw, ROT) ^ HASHES_F[add_base];
        out_hashes[pos] = fw_out;
        fw = fw_out ^ f_rot[rem_base];
    }

    return to_write;
}

// =============================================================================
// ISOLATED MICROBENCHMARKS FOR DEDUP
// =============================================================================

// Benchmark append_unique_vals_simd in isolation
// Returns total time in microseconds for 'iterations' dedup operations
extern "C" uint64_t benchmark_dedup_isolated(uint32_t iterations) {
    using Clock = std::chrono::high_resolution_clock;

    // Allocate output buffer (large enough to never resize during benchmark)
    std::vector<uint32_t> output(iterations * 8 + 64);

    // Initialize test data - use pattern that produces ~50% unique values
    alignas(32) u32x8 old_vals = _mm256_set_epi32(100, 101, 102, 103, 104, 105, 106, 107);
    alignas(32) u32x8 new_vals_base = _mm256_set_epi32(108, 108, 109, 109, 110, 110, 111, 111);

    auto start = Clock::now();

    size_t write_idx = 0;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        // Vary the values slightly to avoid branch prediction artifacts
        u32x8 new_vals = _mm256_add_epi32(new_vals_base, _mm256_set1_epi32(iter & 7));
        write_idx = append_unique_vals_simd(old_vals, new_vals, output.data(), write_idx);
        old_vals = new_vals;

        // Reset write_idx periodically to avoid running out of buffer
        if (write_idx > iterations * 4) write_idx = 0;
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark SCALAR dedup in isolation (for comparison with SIMD Lemire)
extern "C" uint64_t benchmark_dedup_scalar_isolated(uint32_t iterations) {
    using Clock = std::chrono::high_resolution_clock;

    std::vector<uint32_t> output(iterations * 8 + 64);

    alignas(32) u32x8 old_vals = _mm256_set_epi32(100, 101, 102, 103, 104, 105, 106, 107);
    alignas(32) u32x8 new_vals_base = _mm256_set_epi32(108, 108, 109, 109, 110, 110, 111, 111);

    auto start = Clock::now();

    size_t write_idx = 0;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        u32x8 new_vals = _mm256_add_epi32(new_vals_base, _mm256_set1_epi32(iter & 7));
        write_idx = append_unique_vals_scalar(old_vals, new_vals, output.data(), write_idx);
        old_vals = new_vals;

        if (write_idx > iterations * 4) write_idx = 0;
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark syncmers pipeline (pre-packed sequence)
extern "C" uint64_t benchmark_syncmers(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,      // syncmer k-mer size
    uint32_t m,      // minimizer size (s-mer)
    uint32_t iterations
) {
    using namespace packed_seq;

    uint32_t w = k - m + 1;
    if (seq_len < k) return 0;

    // Pre-pack sequence OUTSIDE timing loop (same as minimizer benchmark)
    PackedSeq seq_packed = PackedSeq::from_ascii(ascii_seq, seq_len);

    std::vector<uint32_t> positions;
    positions.reserve(seq_len / w);

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t result_sum = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        positions.clear();
        syncmers_simd_fused_packed(seq_packed, seq_len, k, m, positions);

        if (!positions.empty()) {
            result_sum += positions[0];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// =============================================================================
// Packing benchmark functions
// =============================================================================

// Pack ASCII sequence to 2-bit format (uses PEXT when available)
// Writes directly to output buffer - no intermediate allocation or copy
extern "C" uint32_t pack_sequence_cpp(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint8_t* out_packed,
    uint32_t out_capacity
) {
    uint32_t packed_len = (seq_len + 3) / 4;
    if (packed_len > out_capacity) {
        return 0;
    }

    // Zero-init required for |= operations in pack_ascii_to_buffer
    memset(out_packed, 0, packed_len);
    // Pack directly to output buffer - no intermediate allocation
    packed_seq::pack_ascii_to_buffer(ascii_seq, seq_len, out_packed);

    return packed_len;
}

// Benchmark packing performance
extern "C" uint64_t benchmark_packing(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t iterations
) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t result_sum = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        auto packed = packed_seq::PackedSeq::from_ascii(ascii_seq, seq_len);
        // Prevent optimization
        result_sum += packed.data()[0];
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
