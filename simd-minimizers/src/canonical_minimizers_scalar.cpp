// canonical_minimizers_scalar.cpp - Scalar implementations for canonical minimizers
// This file contains all non-SIMD (scalar) implementations

#include "canonical_minimizers.hpp"
#include <algorithm>
#include <cstdio>
#include <climits>

// =============================================================================
// Scalar ntHash implementation
// =============================================================================

std::vector<uint32_t> nthash_scalar(
    const uint8_t* seq_data,
    uint32_t seq_len,
    uint32_t k
) {
    if (seq_len < k) return {};

    const uint32_t num_kmers = seq_len - k + 1;
    std::vector<uint32_t> hashes;
    hashes.reserve(num_kmers);

    const uint32_t rot = k - 1;
    uint32_t c_rot[4], f_rot[4];
    for (int i = 0; i < 4; i++) {
        c_rot[i] = rotl32(HASHES_F[complement_base(i)], rot);
        f_rot[i] = rotl32(HASHES_F[i], rot);
    }

    uint32_t fw = 0, rc = 0;
    for (uint32_t i = 0; i < k - 1; i++) {
        uint8_t base = seq_data[i] & 0x03;
        fw = rotl32(fw, 1) ^ HASHES_F[base];
        rc = rotr32(rc, 1) ^ c_rot[base];
    }

    for (uint32_t pos = 0; pos < num_kmers; pos++) {
        uint8_t add_base = seq_data[pos + k - 1] & 0x03;
        uint8_t rem_base = seq_data[pos] & 0x03;

        uint32_t fw_out = rotl32(fw, 1) ^ HASHES_F[add_base];
        uint32_t rc_out = rotr32(rc, 1) ^ c_rot[add_base];
        hashes.push_back(fw_out + rc_out);

        fw = fw_out ^ f_rot[rem_base];
        rc = rc_out ^ HASHES_F[complement_base(rem_base)];
    }
    return hashes;
}

// =============================================================================
// Scalar sliding window minimum (two-stack algorithm)
// =============================================================================

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> sliding_lr_min_scalar(
    const std::vector<uint32_t>& hashes,
    uint32_t w
) {
    if (hashes.empty() || hashes.size() < w) {
        return {{}, {}};
    }

    const uint32_t num_windows = hashes.size() - w + 1;
    std::vector<uint32_t> left_mins(num_windows);
    std::vector<uint32_t> right_mins(num_windows);

    // Constants for encoding: upper 16 bits = hash value, lower 16 bits = position
    const uint32_t val_mask = 0xFFFF0000;
    const uint32_t pos_mask = 0x0000FFFF;
    const uint32_t max_pos = (1 << 16) - 1;

    // Ring buffer for storing (left_elem, right_elem) pairs
    std::vector<std::pair<uint32_t, uint32_t>> ring_buf(w, {UINT32_MAX, 0});
    uint32_t ring_idx = 0;

    // Prefix minimums (left uses min, right uses max with inverted hash)
    uint32_t prefix_lmin = UINT32_MAX;
    uint32_t prefix_rmin = 0;

    // Position tracking for overflow handling
    uint32_t pos = 0;
    uint32_t pos_offset = 0;

    for (uint32_t i = 0; i < hashes.size(); i++) {
        // Handle position overflow (16-bit limit)
        if (pos == max_pos) {
            uint32_t delta = (1 << 16) - 2 - w;
            pos -= delta;
            prefix_lmin = (prefix_lmin != UINT32_MAX) ? prefix_lmin - delta : UINT32_MAX;
            prefix_rmin = (prefix_rmin != 0) ? prefix_rmin - delta : 0;
            pos_offset += delta;
            for (auto& [l, r] : ring_buf) {
                l = (l != UINT32_MAX) ? l - delta : UINT32_MAX;
                r = (r != 0) ? r - delta : 0;
            }
        }

        // Encode current element
        // Left: smaller hash wins, earlier position as tiebreaker
        uint32_t lelem = (hashes[i] & val_mask) | pos;
        // Right: invert hash so max becomes min, later position as tiebreaker
        uint32_t relem = (~hashes[i] & val_mask) | pos;

        pos++;

        // Push to ring buffer
        ring_buf[ring_idx] = {lelem, relem};
        ring_idx = (ring_idx + 1) % w;

        // Update prefix minimums
        prefix_lmin = std::min(prefix_lmin, lelem);
        prefix_rmin = std::max(prefix_rmin, relem);

        // When ring buffer wraps, compute suffix minima
        if (ring_idx == 0) {
            uint32_t suffix_l = ring_buf[w - 1].first;
            uint32_t suffix_r = ring_buf[w - 1].second;
            for (int j = (int)w - 2; j >= 0; j--) {
                suffix_l = std::min(suffix_l, ring_buf[j].first);
                suffix_r = std::max(suffix_r, ring_buf[j].second);
                ring_buf[j] = {suffix_l, suffix_r};
            }
            prefix_lmin = lelem;
            prefix_rmin = relem;
        }

        // After warming up (first w-1 elements), output minimizer positions
        if (i >= w - 1) {
            auto [suffix_lmin, suffix_rmin] = ring_buf[ring_idx];
            uint32_t lmin = std::min(prefix_lmin, suffix_lmin);
            uint32_t rmin = std::max(prefix_rmin, suffix_rmin);

            uint32_t win_idx = i - (w - 1);
            left_mins[win_idx] = (lmin & pos_mask) + pos_offset;
            right_mins[win_idx] = (rmin & pos_mask) + pos_offset;
        }
    }

    return {left_mins, right_mins};
}

// =============================================================================
// Naive sliding window minimum (O(n*w) for testing)
// =============================================================================

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> sliding_lr_min_naive(
    const std::vector<uint32_t>& hashes,
    uint32_t w
) {
    if (hashes.empty() || hashes.size() < w) {
        return {{}, {}};
    }

    const uint32_t num_windows = hashes.size() - w + 1;
    std::vector<uint32_t> left_mins(num_windows);
    std::vector<uint32_t> right_mins(num_windows);

    const uint32_t val_mask = 0xFFFF0000;
    const uint32_t pos_mask = 0x0000FFFF;

    for (uint32_t win = 0; win < num_windows; win++) {
        // Left minimum: smaller key wins, earlier position on tie
        uint32_t left_idx = win;
        uint32_t left_key = (hashes[win] & val_mask) | (win & pos_mask);
        for (uint32_t i = 1; i < w; i++) {
            uint32_t idx = win + i;
            uint32_t key = (hashes[idx] & val_mask) | (idx & pos_mask);
            if (key < left_key) {
                left_key = key;
                left_idx = idx;
            }
        }

        // Right minimum: larger inverted key wins, later position on tie
        uint32_t right_idx = win;
        uint32_t right_key = (~hashes[win] & val_mask) | (win & pos_mask);
        for (uint32_t i = 1; i < w; i++) {
            uint32_t idx = win + i;
            uint32_t key = (~hashes[idx] & val_mask) | (idx & pos_mask);
            if (key > right_key) {
                right_key = key;
                right_idx = idx;
            }
        }

        left_mins[win] = left_idx;
        right_mins[win] = right_idx;
    }

    return {left_mins, right_mins};
}

// =============================================================================
// Scalar canonical minimizers
// =============================================================================

void canonical_minimizers_seq_scalar(
    const uint8_t* seq_data,
    uint32_t start_pos,
    uint32_t seq_len,
    uint32_t k,
    uint32_t w,
    std::vector<uint32_t>& out_minimizers) {

    const uint32_t l = k + w - 1;
    if (seq_len < l || start_pos >= seq_len - l + 1 || l % 2 == 0) return;

    // Build lookup tables for ntHash
    const uint32_t rot = k - 1;
    uint32_t c_rot[4], f_rot[4];
    for (int i = 0; i < 4; i++) {
        c_rot[i] = rotl32(HASHES_F[complement_base(i)], rot);
        f_rot[i] = rotl32(HASHES_F[i], rot);
    }

    // 1. Compute all k-mer hashes using canonical ntHash
    std::vector<uint32_t> kmer_hashes;
    kmer_hashes.reserve(seq_len - k + 1 - start_pos);

    uint32_t fw_hash = 0, rc_hash = 0;

    // Initialize with first k-1 bases
    for (uint32_t i = 0; i < k - 1; i++) {
        uint8_t base = seq_data[start_pos + i] & 0x03;
        fw_hash = rotl32(fw_hash, 1) ^ HASHES_F[base];
        rc_hash = rotr32(rc_hash, 1) ^ c_rot[base];
    }

    // Compute rolling hashes for all k-mers
    for (uint32_t pos = start_pos; pos <= seq_len - k; pos++) {
        uint8_t add_base = seq_data[pos + k - 1] & 0x03;
        uint8_t remove_base = seq_data[pos] & 0x03;

        uint32_t fw_hash_out = rotl32(fw_hash, 1) ^ HASHES_F[add_base];
        uint32_t rc_hash_out = rotr32(rc_hash, 1) ^ c_rot[add_base];
        kmer_hashes.push_back(fw_hash_out + rc_hash_out);

        fw_hash = fw_hash_out ^ f_rot[remove_base];
        rc_hash = rc_hash_out ^ HASHES_F[complement_base(remove_base)];
    }

    // 2. Compute sliding minimums (left and right) and canonical flags
    const uint32_t val_mask = 0xFFFF0000;
    const uint32_t pos_mask = 0x0000FFFF;
    const uint32_t num_windows = kmer_hashes.size() - w + 1;

    uint32_t last_pos = UINT32_MAX;

    for (uint32_t win = 0; win < num_windows; win++) {
        // Find left minimum (smaller key wins, earlier position on tie)
        uint32_t left_idx = win;
        uint32_t left_key = (kmer_hashes[win] & val_mask) | (win & pos_mask);
        for (uint32_t i = 1; i < w; i++) {
            uint32_t idx = win + i;
            uint32_t key = (kmer_hashes[idx] & val_mask) | (idx & pos_mask);
            if (key < left_key) { left_key = key; left_idx = idx; }
        }

        // Find right minimum (larger inverted key wins, later position on tie)
        uint32_t right_idx = win;
        uint32_t right_key = (~kmer_hashes[win] & val_mask) | (win & pos_mask);
        for (uint32_t i = 1; i < w; i++) {
            uint32_t idx = win + i;
            uint32_t key = (~kmer_hashes[idx] & val_mask) | (idx & pos_mask);
            if (key > right_key) { right_key = key; right_idx = idx; }
        }

        // Compute canonical: count bases with bit 1 set (T=2, G=3)
        int32_t count = -(int32_t)l;
        for (uint32_t i = 0; i < l; i++) {
            count += (seq_data[start_pos + win + i] & 0x02);
        }

        // Select minimizer and deduplicate
        uint32_t min_pos = (count > 0) ? left_idx : right_idx;
        if (min_pos != last_pos) {
            out_minimizers.push_back(min_pos);
            last_pos = min_pos;
        }
    }
}
