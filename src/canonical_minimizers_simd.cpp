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

struct NtHashSimdState {
    u32x8 h_fw;
    u32x8 h_rc;
    u32x8 simd_f;
    u32x8 simd_c;
    u32x8 simd_f_rot;
    u32x8 simd_c_rot;
};

// nthash_init is no longer needed - hash state is initialized inline in the pipeline functions
// Keeping the struct definition for reference but removing the unused initializer

static FORCE_INLINE u32x8 nthash_step(NtHashSimdState& state, u32x8 a, u32x8 r) {
    u32x8 hfw_out = _mm256_xor_si256(simd_rotl(state.h_fw), table_lookup_avx2(state.simd_f, a));
    state.h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(state.simd_f_rot, r));

    u32x8 hrc_out = _mm256_xor_si256(simd_rotr(state.h_rc), table_lookup_avx2(state.simd_c_rot, a));
    state.h_rc = _mm256_xor_si256(hrc_out, table_lookup_avx2(state.simd_c, r));

    return _mm256_add_epi32(hfw_out, hrc_out);
}

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
    u32x8 max_pos;
    u32x8 pos;
    u32x8 pos_offset;

    SlidingMinState(size_t w_, size_t chunk_size, size_t k, size_t initial_pos = 0)
        : w(w_), idx(0) {
        ring_buf.resize(w, _mm256_set1_epi32(0xFFFFFFFF));
        prefix_min = _mm256_set1_epi32(0xFFFFFFFF);
        val_mask = _mm256_set1_epi32(0xFFFF0000);
        pos_mask = _mm256_set1_epi32(0x0000FFFF);
        max_pos = _mm256_set1_epi32((1 << 16) - 1);
        pos = _mm256_set1_epi32(initial_pos);

        alignas(32) int32_t offsets[8];
        for (int i = 0; i < 8; i++) {
            offsets[i] = (int32_t)(i * chunk_size) - (int32_t)(k - 1);
        }
        pos_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(offsets));
    }

    // Handle position overflow - trigger when ALL lanes reach max_pos
    // (all lanes process synchronously, so they should all overflow together)
    FORCE_INLINE void check_and_reset_overflow() {
        u32x8 cmp = _mm256_cmpeq_epi32(pos, max_pos);
        // Check if ALL lanes matched (all bytes of cmp are 0xFF)
        if (_mm256_movemask_epi8(cmp) == -1) {
            u32x8 delta = _mm256_set1_epi32((1 << 16) - 2 - (uint32_t)w);
            pos = _mm256_sub_epi32(pos, delta);
            prefix_min = _mm256_sub_epi32(prefix_min, delta);
            pos_offset = _mm256_add_epi32(pos_offset, delta);
            for (auto& elem : ring_buf) {
                elem = _mm256_sub_epi32(elem, delta);
            }
        }
    }

    FORCE_INLINE u32x8 process(u32x8 hash) {
        check_and_reset_overflow();

        u32x8 elem = _mm256_or_si256(_mm256_and_si256(hash, val_mask), pos);
        ring_buf[idx] = elem;
        prefix_min = _mm256_min_epu32(prefix_min, elem);
        pos = _mm256_add_epi32(pos, _mm256_set1_epi32(1));
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
    u32x8 max_pos;
    u32x8 pos;
    u32x8 pos_offset;

    SlidingLRMinState(size_t w_, size_t chunk_size, size_t k, size_t initial_pos = 0)
        : w(w_), idx(0) {
        ring_buf.resize(w, {_mm256_set1_epi32(0xFFFFFFFF), _mm256_setzero_si256()});
        prefix_lmin = _mm256_set1_epi32(0xFFFFFFFF);
        prefix_rmax = _mm256_setzero_si256();
        val_mask = _mm256_set1_epi32(0xFFFF0000);
        pos_mask = _mm256_set1_epi32(0x0000FFFF);
        max_pos = _mm256_set1_epi32((1 << 16) - 1);
        pos = _mm256_set1_epi32(initial_pos);

        alignas(32) int32_t offsets[8];
        for (int i = 0; i < 8; i++) {
            offsets[i] = (int32_t)(i * chunk_size) - (int32_t)(k - 1);
        }
        pos_offset = _mm256_load_si256(reinterpret_cast<const __m256i*>(offsets));
    }

    // Handle position overflow - trigger when ALL lanes reach max_pos
    // (all lanes process synchronously, so they should all overflow together)
    FORCE_INLINE void check_and_reset_overflow() {
        u32x8 cmp = _mm256_cmpeq_epi32(pos, max_pos);
        // Check if ALL lanes matched (all bytes of cmp are 0xFF)
        if (_mm256_movemask_epi8(cmp) == -1) {
            u32x8 delta = _mm256_set1_epi32((1 << 16) - 2 - (uint32_t)w);
            pos = _mm256_sub_epi32(pos, delta);
            prefix_lmin = _mm256_sub_epi32(prefix_lmin, delta);
            prefix_rmax = _mm256_sub_epi32(prefix_rmax, delta);
            pos_offset = _mm256_add_epi32(pos_offset, delta);
            for (auto& [l, r] : ring_buf) {
                l = _mm256_sub_epi32(l, delta);
                r = _mm256_sub_epi32(r, delta);
            }
        }
    }

    // Process hash and return (left_min_pos, right_min_pos)
    FORCE_INLINE std::pair<u32x8, u32x8> process(u32x8 hash) {
        // Check for position overflow
        check_and_reset_overflow();

        // Left: smaller hash wins, earlier position as tiebreaker
        u32x8 lelem = _mm256_or_si256(_mm256_and_si256(hash, val_mask), pos);
        // Right: invert hash so max becomes min, later position as tiebreaker
        u32x8 relem = _mm256_or_si256(
            _mm256_and_si256(_mm256_xor_si256(hash, _mm256_set1_epi32(-1)), val_mask),
            pos);

        ring_buf[idx] = {lelem, relem};
        prefix_lmin = _mm256_min_epu32(prefix_lmin, lelem);
        prefix_rmax = _mm256_max_epu32(prefix_rmax, relem);

        pos = _mm256_add_epi32(pos, _mm256_set1_epi32(1));
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
// SIMD Minimizers with packed_seq
// =============================================================================

void minimizers_simd_packed_seq(
    const packed_seq::PackedSeq& seq,
    uint32_t k,
    uint32_t w,
    std::vector<u32x8>& out_positions,
    bool canonical = false
) {
    using namespace packed_seq;

    if (seq.len() < k + w - 1) return;

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
    for (size_t i = 0; i < buf_size; i++) {
        delay_buf[i] = _mm256_setzero_si256();
    }

    SimdIterState state = simd_iter_init(seq, k + w - 1);
    size_t chunk_size = state.chunk_size;

    SlidingMinState sliding_min(w, chunk_size, k, 0);

    uint32_t fw_init = 0, rc_init = 0;
    for (uint32_t i = 0; i < k - 1; i++) {
        fw_init = rotl32(fw_init, ROT) ^ HASHES_F[0];
        rc_init = rotr32(rc_init, ROT) ^ c_rot_table[0];
    }
    u32x8 h_fw = _mm256_set1_epi32(fw_init);
    u32x8 h_rc = _mm256_set1_epi32(rc_init);

    size_t write_idx = 0, read_idx = 0;
    size_t iter_count = 0, output_count = 0;

    while (simd_iter_has_next(state)) {
        u32x8 add = simd_iter_next(state);
        u32x8 remove;

        if (iter_count < k - 1) {
            remove = _mm256_setzero_si256();
        } else {
            remove = delay_buf[read_idx];
            read_idx = (read_idx + 1) & buf_mask;
        }

        delay_buf[write_idx] = add;
        write_idx = (write_idx + 1) & buf_mask;
        iter_count++;

        u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
        h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove));

        u32x8 hash;
        if (canonical) {
            u32x8 hrc_out = _mm256_xor_si256(simd_rotr(h_rc), table_lookup_avx2(simd_c_rot, add));
            h_rc = _mm256_xor_si256(hrc_out, table_lookup_avx2(simd_c, remove));
            hash = _mm256_add_epi32(hfw_out, hrc_out);
        } else {
            hash = hfw_out;
        }

        u32x8 min_pos = sliding_min.process(hash);
        output_count++;

        if (output_count > (k + w - 2)) {
            out_positions.push_back(min_pos);
        }
    }
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

void collect_and_dedup_positions(
    const std::vector<u32x8>& simd_positions,
    size_t num_valid_windows,
    std::vector<uint32_t>& out_positions,
    uint32_t max_position = UINT32_MAX
) {
    if (simd_positions.empty()) return;

    size_t num_outputs = simd_positions.size();

    // Initialize thread-local lane cache (like Rust's CACHE)
    // Just ensure capacity, use on-demand growth to avoid zeroing all upfront
    if (!lane_cache_initialized) {
        for (int i = 0; i < 8; i++) {
            lane_cache[i].reserve(8192);
        }
        lane_cache_initialized = true;
    }

    // Clear but keep capacity
    for (int i = 0; i < 8; i++) {
        lane_cache[i].clear();
        if (lane_cache[i].capacity() < num_outputs + 8) {
            lane_cache[i].reserve(num_outputs + 8);
        }
    }
    size_t write_idx[8] = {0};

    // Previous values for dedup (start with MAX so first element is always kept)
    u32x8 old[8];
    for (int i = 0; i < 8; i++) {
        old[i] = _mm256_set1_epi32(UINT32_MAX);
    }

    // Process in groups of 8 using transpose + SIMD dedup (matching Rust algorithm)
    u32x8 m[8];
    for (int i = 0; i < 8; i++) {
        m[i] = _mm256_setzero_si256();
    }

    // Pre-compute fast path threshold (like Rust's fast_path_limit)
    size_t fast_path_limit = 0;
    if (num_valid_windows > 7 * num_outputs + 7) {
        fast_path_limit = num_valid_windows - 7 * num_outputs - 7;
    }

    for (size_t i = 0; i < num_outputs; i++) {
        m[i % 8] = simd_positions[i];

        if (i % 8 == 7) {
            u32x8 t[8];
            transpose_8x8(m, t);

            size_t batch_start = i - 7;

            if (batch_start < fast_path_limit) {
                // FAST PATH: All windows valid
                for (int j = 0; j < 8; j++) {
                    // On-demand growth (like Rust)
                    if (write_idx[j] + 8 > lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 1024);
                    }
                    write_idx[j] = append_unique_vals_simd(
                        old[j], t[j], lane_cache[j].data(), write_idx[j]
                    );
                    old[j] = t[j];
                }
            } else {
                // SLOW PATH: Mark invalid positions with UINT32_MAX
                for (int j = 0; j < 8; j++) {
                    size_t lane_window_start = j * num_outputs + batch_start;
                    alignas(32) uint32_t vals[8];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(vals), t[j]);
                    for (int idx = 0; idx < 8; idx++) {
                        if (lane_window_start + idx >= num_valid_windows) {
                            vals[idx] = UINT32_MAX;
                        }
                    }
                    t[j] = _mm256_load_si256(reinterpret_cast<const __m256i*>(vals));
                }

                for (int j = 0; j < 8; j++) {
                    // On-demand growth (like Rust)
                    if (write_idx[j] + 8 > lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 1024);
                    }
                    write_idx[j] = append_unique_vals_simd(
                        old[j], t[j], lane_cache[j].data(), write_idx[j]
                    );
                    old[j] = t[j];
                }
            }
        }
    }

    // Handle partial last group with scalar dedup
    size_t k_partial = num_outputs % 8;
    if (k_partial > 0) {
        u32x8 t[8];
        transpose_8x8(m, t);
        size_t batch_start = (num_outputs / 8) * 8;

        for (int j = 0; j < 8; j++) {
            // Use union to avoid store-load stall (like Rust's transmute)
            union { __m256i vec; uint32_t arr[8]; } u;
            u.vec = t[j];

            uint32_t last_val = (write_idx[j] > 0) ? lane_cache[j][write_idx[j] - 1] : UINT32_MAX;
            size_t lane_window_start = j * num_outputs + batch_start;

            for (size_t idx = 0; idx < k_partial; idx++) {
                size_t window_idx = lane_window_start + idx;
                if (window_idx >= num_valid_windows) continue;
                if (u.arr[idx] != last_val) {
                    // Ensure space for write
                    if (write_idx[j] >= lane_cache[j].size()) {
                        lane_cache[j].resize(lane_cache[j].size() + 64);
                    }
                    lane_cache[j][write_idx[j]++] = u.arr[idx];
                    last_val = u.arr[idx];
                }
            }
        }
    }

    // Truncate lanes to actual written length (like Rust's v[j].truncate(write_idx[j]))
    for (int j = 0; j < 8; j++) {
        lane_cache[j].resize(write_idx[j]);
    }

    // Pre-reserve output capacity (key optimization!)
    size_t total_elements = 0;
    for (int j = 0; j < 8; j++) {
        total_elements += lane_cache[j].size();
    }
    out_positions.reserve(out_positions.size() + total_elements);

    // Flatten with bulk insert (like Rust's extend_from_slice)
    // Note: Invalid positions are marked with UINT32_MAX, need to filter them
    for (int lane = 0; lane < 8; lane++) {
        const auto& v = lane_cache[lane];
        if (v.empty()) continue;

        size_t start = 0;
        // Skip boundary duplicates
        while (start < v.size() && !out_positions.empty() && v[start] == out_positions.back()) {
            start++;
        }

        // Find end of valid positions (skip trailing UINT32_MAX)
        size_t end = v.size();
        while (end > start && v[end - 1] == UINT32_MAX) {
            end--;
        }

        // BULK INSERT (uses memcpy internally)
        if (start < end) {
            out_positions.insert(out_positions.end(), v.begin() + start, v.begin() + end);
        }
    }
}

void minimizers_simd_packed_seq_collected(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions,
    bool canonical = false
) {
    using namespace packed_seq;

    uint32_t l = k + w - 1;
    if (seq_len < l) return;

    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);
    std::vector<u32x8> simd_positions;
    simd_positions.reserve((seq_len - l + 1) / 8 + 1);
    minimizers_simd_packed_seq(seq, k, w, simd_positions, canonical);

    SimdIterState state = simd_iter_init(seq, l);
    size_t chunk_size = state.chunk_size;
    size_t num_valid_windows = seq_len - l + 1;
    uint32_t max_position = seq_len - k + 1;

    collect_and_dedup_positions(simd_positions, num_valid_windows, out_positions, max_position);
}

// =============================================================================
// SIMD Canonical Minimizers - Full pipeline matching Rust implementation
// =============================================================================

void canonical_minimizers_simd_packed_seq(
    const packed_seq::PackedSeq& seq,
    uint32_t k,
    uint32_t w,
    std::vector<u32x8>& out_positions
) {
    using namespace packed_seq;

    const uint32_t l = k + w - 1;
    if (seq.len() < l) return;
    if (l % 2 == 0) return;  // l must be odd for canonical

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

    // Use simple delay buffers (power of 2 size) like the working minimizers_simd_packed_seq
    size_t buf_size_k = 1;
    while (buf_size_k < k) buf_size_k *= 2;
    size_t buf_mask_k = buf_size_k - 1;

    size_t buf_size_l = 1;
    while (buf_size_l < l + 1) buf_size_l *= 2;  // Note: delay is l, so need l+1 slots
    size_t buf_mask_l = buf_size_l - 1;

    alignas(32) u32x8 delay_buf_k[64];
    alignas(32) u32x8 delay_buf_l[128];  // May need more for larger l
    for (size_t i = 0; i < buf_size_k; i++) delay_buf_k[i] = _mm256_setzero_si256();
    for (size_t i = 0; i < buf_size_l; i++) delay_buf_l[i] = _mm256_setzero_si256();

    // Initialize iterator
    SimdIterState state = simd_iter_init(seq, l);
    size_t chunk_size = state.chunk_size;

    // Initialize components
    SlidingLRMinState sliding_min(w, chunk_size, k, 0);
    CanonicalMapper canonical_mapper(k, w);

    // Pre-warm hash state with k-1 zeros (matching Rust's nthash_mapper)
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

    // Main loop - process all iterations
    while (simd_iter_has_next(state)) {
        u32x8 add = simd_iter_next(state);
        u32x8 remove_k, remove_l;

        // Get delayed character for ntHash (delay = k-1)
        if (iter_count < k - 1) {
            remove_k = _mm256_setzero_si256();
        } else {
            remove_k = delay_buf_k[read_idx_k];
            read_idx_k = (read_idx_k + 1) & buf_mask_k;
        }

        // Get delayed character for canonical (delay = l-1, matching Rust's Delay(l-1))
        if (iter_count < l - 1) {
            remove_l = _mm256_setzero_si256();
        } else {
            remove_l = delay_buf_l[read_idx_l];
            read_idx_l = (read_idx_l + 1) & buf_mask_l;
        }

        // Store current character in delay buffers
        delay_buf_k[write_idx_k] = add;
        write_idx_k = (write_idx_k + 1) & buf_mask_k;
        delay_buf_l[write_idx_l] = add;
        write_idx_l = (write_idx_l + 1) & buf_mask_l;

        iter_count++;

        // Compute canonical ntHash
        u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
        h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove_k));

        u32x8 hrc_out = _mm256_xor_si256(simd_rotr(h_rc), table_lookup_avx2(simd_c_rot, add));
        h_rc = _mm256_xor_si256(hrc_out, table_lookup_avx2(simd_c, remove_k));

        u32x8 hash = _mm256_add_epi32(hfw_out, hrc_out);

        // Compute canonical mask
        i32x8 canonical_mask = canonical_mapper.process(add, remove_l);

        // Compute both left and right minimums
        auto [lmin_pos, rmin_pos] = sliding_min.process(hash);

        // Blend: select lmin where canonical, rmin otherwise
        u32x8 selected_pos = simd_blend(canonical_mask, lmin_pos, rmin_pos);

        output_count++;

        // Skip first l-1 outputs (warmup)
        if (output_count > l - 1) {
            out_positions.push_back(selected_pos);
        }
    }
}

void canonical_minimizers_simd_collected(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions
) {
    using namespace packed_seq;

    const uint32_t l = k + w - 1;
    if (seq_len < l) return;
    if (l % 2 == 0) return;

    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);
    std::vector<u32x8> simd_positions;
    simd_positions.reserve((seq_len - l + 1) / 8 + 1);

    canonical_minimizers_simd_packed_seq(seq, k, w, simd_positions);

    SimdIterState state = simd_iter_init(seq, l);
    size_t chunk_size = state.chunk_size;
    size_t num_valid_windows = seq_len - l + 1;
    // max_position should be the number of valid k-mer positions (not windows)
    // k-mers are valid at positions 0 to seq_len - k, so max_position = seq_len - k + 1
    uint32_t max_position = seq_len - k + 1;

    collect_and_dedup_positions(simd_positions, num_valid_windows, out_positions, max_position);

    // Process tail: windows that don't fit into the 8 SIMD chunks
    // The SIMD part covers windows at positions [0, 8 * chunk_size)
    // The tail covers windows at positions [8 * chunk_size, num_valid_windows)
    // Matching Rust: tail positions are offset by 8 * head_len where head_len = simd_positions.size()
    size_t simd_coverage = 8 * chunk_size;
    size_t head_len = simd_positions.size();
    size_t tail_offset = 8 * head_len;  // Position offset for tail

    if (simd_coverage < seq_len && simd_coverage >= l - 1) {
        // The tail sequence starts at position 8 * chunk_size
        // Process using scalar canonical minimizers
        const uint8_t* tail_seq = ascii_seq + simd_coverage;
        size_t tail_len = seq_len - simd_coverage;

        if (tail_len >= l) {
            // Convert tail to packed seq for scalar processing
            std::vector<uint8_t> tail_packed;
            for (size_t i = 0; i < tail_len; i++) {
                tail_packed.push_back(pack_char(tail_seq[i]));
            }

            // Call scalar implementation
            std::vector<uint32_t> tail_positions;
            canonical_minimizers_seq_scalar(tail_packed.data(), 0, tail_len, k, w, tail_positions);

            // Append tail positions with offset and dedup against existing
            for (uint32_t pos : tail_positions) {
                uint32_t adjusted_pos = pos + (uint32_t)tail_offset;
                if (adjusted_pos < max_position) {
                    if (out_positions.empty() || out_positions.back() != adjusted_pos) {
                        out_positions.push_back(adjusted_pos);
                    }
                }
            }
        }
    }
}

// =============================================================================
// FUSED SIMD Pipeline - Collection/dedup inline with main loop
// =============================================================================

// Non-canonical version with NEW inline collection (for fair benchmarking)
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

        if (iter_count < k - 1) {
            remove = _mm256_setzero_si256();
        } else {
            remove = delay_buf[read_idx];
            read_idx = (read_idx + 1) & buf_mask;
        }

        delay_buf[write_idx] = add;
        write_idx = (write_idx + 1) & buf_mask;
        iter_count++;

        // Forward-only ntHash (non-canonical)
        u32x8 hfw_out = _mm256_xor_si256(simd_rotl(h_fw), table_lookup_avx2(simd_f, add));
        h_fw = _mm256_xor_si256(hfw_out, table_lookup_avx2(simd_f_rot, remove));
        u32x8 hash = hfw_out;  // Non-canonical: just use forward hash

        // Left-only sliding min
        u32x8 min_pos = sliding_min.process(hash);

        output_count++;

        if (output_count > l - 1) {
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
// Timing struct for profiling (used by profile_collection_phases and benchmark_canonical_phases)
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

// ASCII wrapper that calls the packed version (original interface for FFI)
void canonical_minimizers_simd_fused(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_positions
) {
    using namespace packed_seq;
    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);
    canonical_minimizers_simd_fused_packed(seq, seq_len, k, w, out_positions);
}

// =============================================================================
// PROFILING: Time each phase of the collection separately
// =============================================================================
extern "C" void profile_collection_phases(
    const uint8_t* ascii_seq, uint32_t seq_len, uint32_t k, uint32_t w, uint32_t iterations,
    uint64_t* main_loop_us, uint64_t* partial_batch_us, uint64_t* truncate_us,
    uint64_t* prereserve_us, uint64_t* flatten_us, uint64_t* tail_us
) {
    using namespace packed_seq;
    using Clock = std::chrono::high_resolution_clock;

    const uint32_t l = k + w - 1;
    if (seq_len < l) return;

    // Pack sequence once (outside timing loop)
    PackedSeq seq_packed = PackedSeq::from_ascii(ascii_seq, seq_len);

    // Accumulate timing across iterations
    FusedPhaseTiming total_timing;
    uint64_t total_tail = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        std::vector<uint32_t> out_positions;
        FusedPhaseTiming iter_timing;

        // Call the template with profiling enabled
        canonical_minimizers_simd_fused_impl<true>(seq_packed, seq_len, k, w, out_positions, &iter_timing);

        // Accumulate phase timings
        total_timing.main_loop_us += iter_timing.main_loop_us;
        total_timing.partial_batch_us += iter_timing.partial_batch_us;
        total_timing.truncate_us += iter_timing.truncate_us;
        total_timing.prereserve_us += iter_timing.prereserve_us;
        total_timing.flatten_us += iter_timing.flatten_us;

        // === PHASE 6: TAIL (timed separately) ===
        auto t_tail_start = Clock::now();

        // Compute chunk_size for tail offset calculation
        size_t num_kmers = seq_len - l + 1;
        size_t div_ceil_8 = (num_kmers + 7) / 8;
        size_t chunk_size = ((div_ceil_8 + 3) / 4) * 4;
        size_t simd_coverage = 8 * chunk_size;

        if (simd_coverage < seq_len) {
            const uint8_t* tail_seq = ascii_seq + simd_coverage;
            size_t tail_len = seq_len - simd_coverage;

            if (tail_len >= l) {
                std::vector<uint8_t> tail_packed_data;
                for (size_t i = 0; i < tail_len; i++) {
                    tail_packed_data.push_back(pack_char(tail_seq[i]));
                }

                std::vector<uint32_t> tail_positions;
                canonical_minimizers_seq_scalar(tail_packed_data.data(), 0, tail_len, k, w, tail_positions);

                uint32_t max_position = seq_len - k + 1;
                for (uint32_t pos : tail_positions) {
                    uint32_t adjusted_pos = pos + (uint32_t)simd_coverage;
                    if (adjusted_pos < max_position) {
                        if (out_positions.empty() || out_positions.back() != adjusted_pos) {
                            out_positions.push_back(adjusted_pos);
                        }
                    }
                }
            }
        }

        auto t_tail_end = Clock::now();
        total_tail += std::chrono::duration_cast<std::chrono::microseconds>(t_tail_end - t_tail_start).count();
    }

    *main_loop_us = total_timing.main_loop_us;
    *partial_batch_us = total_timing.partial_batch_us;
    *truncate_us = total_timing.truncate_us;
    *prereserve_us = total_timing.prereserve_us;
    *flatten_us = total_timing.flatten_us;
    *tail_us = total_tail;
}

// Debug: compare fused vs collected
extern "C" void debug_compare_fused_vs_collected(
    const uint8_t* seq_data, uint32_t seq_len, uint32_t k, uint32_t w
) {
    size_t l = k + w - 1;
    size_t num_valid_windows = seq_len - l + 1;
    size_t num_kmers = num_valid_windows;  // Same thing for minimizers
    size_t div_ceil_8 = (num_kmers + 7) / 8;
    size_t chunk_size = ((div_ceil_8 + 3) / 4) * 4;
    size_t par_len = chunk_size + l - 1;

    std::vector<uint32_t> fused_out, collected_out;
    canonical_minimizers_simd_fused(seq_data, seq_len, k, w, fused_out);
    canonical_minimizers_simd_collected(seq_data, seq_len, k, w, collected_out);

    fprintf(stderr, "seq_len=%u, k=%u, w=%u, l=%zu\n", seq_len, k, w, l);
    fprintf(stderr, "num_valid_windows=%zu, chunk_size=%zu, par_len=%zu\n",
            num_valid_windows, chunk_size, par_len);
    fprintf(stderr, "Lane boundaries:\n");
    for (int j = 0; j < 8; j++) {
        size_t lane_start = j * chunk_size;
        size_t lane_end = std::min(lane_start + chunk_size, num_valid_windows);
        fprintf(stderr, "  Lane %d: [%zu, %zu) (%zu windows)\n",
                j, lane_start, lane_end, (lane_start < num_valid_windows) ? lane_end - lane_start : 0);
    }
    fprintf(stderr, "fused: %zu positions, collected: %zu positions\n",
            fused_out.size(), collected_out.size());

    // Find first difference
    size_t max_i = std::max(fused_out.size(), collected_out.size());
    for (size_t i = 0; i < max_i; i++) {
        uint32_t f = (i < fused_out.size()) ? fused_out[i] : UINT32_MAX;
        uint32_t c = (i < collected_out.size()) ? collected_out[i] : UINT32_MAX;
        if (f != c) {
            fprintf(stderr, "DIFF at index %zu: fused=%u, collected=%u\n", i, f, c);
            // Print context
            for (size_t j = (i > 3 ? i - 3 : 0); j < std::min(i + 5, max_i); j++) {
                uint32_t fv = (j < fused_out.size()) ? fused_out[j] : UINT32_MAX;
                uint32_t cv = (j < collected_out.size()) ? collected_out[j] : UINT32_MAX;
                fprintf(stderr, "  [%zu] fused=%u collected=%u %s\n", j, fv, cv,
                        (fv != cv) ? "<-- DIFF" : "");
            }
            break;
        }
    }
}

// =============================================================================
// Main Entry Point - Now uses SIMD canonical minimizers
// =============================================================================

extern "C" void canonical_minimizers_seq_simd_avx2(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t** out_ptr,
    uint32_t* out_len
) {
    try {
        if (seq_len == 0) {
            *out_ptr = nullptr;
            *out_len = 0;
            return;
        }

        if ((k + w - 1) % 2 == 0) {
            *out_ptr = nullptr;
            *out_len = 0;
            return;
        }

        std::vector<uint32_t> minimizers;
        minimizers.reserve((seq_len - k + 1) / w);

        // For short sequences, use scalar implementation for correctness
        // SIMD requires longer sequences to work correctly with 8-way parallelism
        const uint32_t l = k + w - 1;
        if (seq_len < l + 64) {  // Use scalar for sequences < 64 bases beyond minimum
            canonical_minimizers_seq_scalar(seq_data, 0, seq_len, k, w, minimizers);
        } else {
            // Use fused canonical pipeline (hash + sliding min streamed together)
            canonical_minimizers_simd_fused(seq_data, seq_len, k, w, minimizers);
        }

        *out_len = minimizers.size();
        if (*out_len > 0) {
            *out_ptr = (uint32_t*)malloc(*out_len * sizeof(uint32_t));
            if (*out_ptr == nullptr) {
                *out_len = 0;
                return;
            }
            memcpy(*out_ptr, minimizers.data(), *out_len * sizeof(uint32_t));
        } else {
            *out_ptr = nullptr;
        }
    } catch (...) {
        *out_ptr = nullptr;
        *out_len = 0;
    }
}

extern "C" void free_minimizers(uint32_t* ptr) {
    if (ptr) free(ptr);
}

// =============================================================================
// Test Functions
// =============================================================================

extern "C" int test_nthash_scalar_unrolled(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k
) {
    auto unrolled_hashes = nthash_scalar_unrolled(seq_data, seq_len, k);
    auto scalar_hashes = nthash_scalar(seq_data, seq_len, k);

    if (unrolled_hashes.size() != scalar_hashes.size()) {
        fprintf(stderr, "Size mismatch: unrolled=%zu, scalar=%zu\n",
                unrolled_hashes.size(), scalar_hashes.size());
        return 0;
    }

    for (size_t i = 0; i < unrolled_hashes.size(); i++) {
        if (unrolled_hashes[i] != scalar_hashes[i]) {
            fprintf(stderr, "Hash mismatch at pos %zu: unrolled=0x%08x, scalar=0x%08x\n",
                    i, unrolled_hashes[i], scalar_hashes[i]);
            return 0;
        }
    }
    return 1;
}

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
    minimizers_simd_packed_seq_collected(ascii_seq, seq_len, k, w, cpp_positions, false);

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

extern "C" uint64_t benchmark_nthash_scalar_unrolled(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k,
    uint32_t iterations
) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t sum = 0;
    for (uint32_t i = 0; i < iterations; i++) {
        auto hashes = nthash_scalar_unrolled(seq_data, seq_len, k);
        sum += hashes.size();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

extern "C" uint64_t benchmark_sliding_min_scalar(
    const uint32_t* hashes,
    uint32_t hash_len,
    uint32_t w,
    uint32_t iterations
) {
    std::vector<uint32_t> hash_vec(hashes, hashes + hash_len);
    auto start = std::chrono::high_resolution_clock::now();
    volatile size_t sum = 0;
    for (uint32_t i = 0; i < iterations; i++) {
        auto [left, right] = sliding_lr_min_scalar(hash_vec, w);
        sum += left.size() + right.size();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

extern "C" uint64_t benchmark_sliding_min_simd(
    const uint32_t* hashes,
    uint32_t hash_len,
    uint32_t w,
    uint32_t iterations
) {
    std::vector<uint32_t> hash_vec(hashes, hashes + hash_len);
    auto start = std::chrono::high_resolution_clock::now();
    volatile size_t sum = 0;
    for (uint32_t i = 0; i < iterations; i++) {
        auto [left, right] = sliding_lr_min_simd(hash_vec, w);
        sum += left.size() + right.size();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

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

// NOTE: This benchmark only measures hash + sliding min, NO collection!
// For fair comparison with full pipelines, use benchmark_noncanonical_full instead.
extern "C" uint64_t benchmark_hash_and_slidmin_only(
    const uint8_t* ascii_seq,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    uint32_t iterations
) {
    using namespace packed_seq;
    if (seq_len < k + w - 1) return 0;

    PackedSeq seq = PackedSeq::from_ascii(ascii_seq, seq_len);
    std::vector<u32x8> positions;
    positions.reserve((seq_len - k - w + 2) / 8 + 1);

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint32_t result_sum = 0;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        positions.clear();
        minimizers_simd_packed_seq(seq, k, w, positions);

        if (!positions.empty()) {
            alignas(32) uint32_t vals[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(vals), positions[0]);
            result_sum += vals[0];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark non-canonical FULL pipeline (for comparison with canonical)
// Uses NEW fused collection (same as canonical) for fair comparison
extern "C" uint64_t benchmark_noncanonical_full(
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
        // Use NEW fused collection (same as canonical_minimizers_simd_fused_packed)
        minimizers_simd_fused_packed(seq_packed, seq_len, k, w, positions);

        if (!positions.empty()) {
            result_sum += positions[0];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark canonical FULL pipeline DIRECTLY (no FFI result handling)
// This times only the core algorithm without malloc/memcpy/free overhead
// Pre-packs sequence OUTSIDE timing loop for fair comparison with Rust
extern "C" uint64_t benchmark_canonical_full_direct(
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
// ISOLATED MICROBENCHMARKS FOR COLLECTION COMPONENTS
// These measure individual operations in isolation to identify bottlenecks
// =============================================================================

// Benchmark transpose_8x8 in isolation
// Returns total time in microseconds for 'iterations' transpose operations
extern "C" uint64_t benchmark_transpose_isolated(uint32_t iterations) {
    using Clock = std::chrono::high_resolution_clock;

    // Initialize test data with realistic values
    alignas(32) u32x8 m[8];
    alignas(32) u32x8 t[8];
    for (int i = 0; i < 8; i++) {
        m[i] = _mm256_set_epi32(i*8+7, i*8+6, i*8+5, i*8+4, i*8+3, i*8+2, i*8+1, i*8+0);
    }

    auto start = Clock::now();

    volatile uint32_t sink = 0;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        transpose_8x8(m, t);
        // Prevent optimization by using result
        sink += _mm256_extract_epi32(t[0], 0);
        // Feed back to prevent loop unrolling optimizations
        m[0] = _mm256_add_epi32(m[0], _mm256_set1_epi32(1));
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

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

// Benchmark 8x dedup calls (one batch) to simulate actual collection pattern
// Returns total time in microseconds for 'iterations' batches (each batch = 8 dedup calls)
extern "C" uint64_t benchmark_dedup_batch_isolated(uint32_t iterations) {
    using Clock = std::chrono::high_resolution_clock;

    // 8 lane caches like in real collection
    std::vector<uint32_t> lane_caches[8];
    for (int i = 0; i < 8; i++) {
        lane_caches[i].resize(iterations * 8 + 64);
    }
    size_t write_idx[8] = {0};

    // Previous values per lane
    alignas(32) u32x8 old_vals[8];
    for (int i = 0; i < 8; i++) {
        old_vals[i] = _mm256_set1_epi32(UINT32_MAX);
    }

    // Transposed result (simulated)
    alignas(32) u32x8 t[8];
    for (int i = 0; i < 8; i++) {
        t[i] = _mm256_set_epi32(i*8+7, i*8+6, i*8+5, i*8+4, i*8+3, i*8+2, i*8+1, i*8+0);
    }

    auto start = Clock::now();

    for (uint32_t iter = 0; iter < iterations; iter++) {
        // Simulate 8 dedup calls per batch
        for (int j = 0; j < 8; j++) {
            write_idx[j] = append_unique_vals_simd(old_vals[j], t[j], lane_caches[j].data(), write_idx[j]);
            old_vals[j] = t[j];
        }

        // Vary values and reset periodically
        t[0] = _mm256_add_epi32(t[0], _mm256_set1_epi32(1));
        if (write_idx[0] > iterations * 4) {
            for (int j = 0; j < 8; j++) write_idx[j] = 0;
        }
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Benchmark lane cache resize check pattern
// Returns total time in microseconds for 'iterations' resize checks
extern "C" uint64_t benchmark_lane_resize_isolated(uint32_t iterations) {
    using Clock = std::chrono::high_resolution_clock;

    std::vector<uint32_t> lane_cache;
    lane_cache.reserve(8192);  // Start with realistic initial capacity

    auto start = Clock::now();

    volatile size_t sink = 0;
    size_t write_idx = 0;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        // This is the pattern used in collection
        if (write_idx + 8 > lane_cache.size()) {
            lane_cache.resize(lane_cache.size() + 1024);
        }
        // Simulate write
        write_idx += 4;  // Typical unique count
        sink = write_idx;

        // Occasionally reset to vary the pattern
        if (iter % 1000 == 999) {
            lane_cache.resize(0);
            lane_cache.reserve(8192);
            write_idx = 0;
        }
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Combined benchmark: transpose + 8x dedup (full collection batch)
// Returns total time in microseconds for 'iterations' complete batches
extern "C" uint64_t benchmark_collection_batch_isolated(uint32_t iterations) {
    using Clock = std::chrono::high_resolution_clock;

    // 8 lane caches
    std::vector<uint32_t> lane_caches[8];
    for (int i = 0; i < 8; i++) {
        lane_caches[i].resize(iterations * 8 + 64);
    }
    size_t write_idx[8] = {0};

    alignas(32) u32x8 old_vals[8];
    for (int i = 0; i < 8; i++) {
        old_vals[i] = _mm256_set1_epi32(UINT32_MAX);
    }

    // Input matrix (8 u32x8 values to transpose)
    alignas(32) u32x8 m[8];
    alignas(32) u32x8 t[8];
    for (int i = 0; i < 8; i++) {
        m[i] = _mm256_set_epi32(i*8+7, i*8+6, i*8+5, i*8+4, i*8+3, i*8+2, i*8+1, i*8+0);
    }

    auto start = Clock::now();

    for (uint32_t iter = 0; iter < iterations; iter++) {
        // Step 1: Transpose
        transpose_8x8(m, t);

        // Step 2: 8x Dedup
        for (int j = 0; j < 8; j++) {
            write_idx[j] = append_unique_vals_simd(old_vals[j], t[j], lane_caches[j].data(), write_idx[j]);
            old_vals[j] = t[j];
        }

        // Vary values
        m[0] = _mm256_add_epi32(m[0], _mm256_set1_epi32(1));
        if (write_idx[0] > iterations * 4) {
            for (int j = 0; j < 8; j++) write_idx[j] = 0;
        }
    }

    auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
