//! Bindings for the C++ implementation of canonical_minimizers_seq_simd_avx2
//!
//! This module provides a safe Rust interface to the C++ implementation.
#![allow(dead_code)]

use std::os::raw::{c_uchar, c_uint};
use std::slice;
use std::ptr;

extern "C" {
    /// C++ function to compute canonical minimizers
    fn canonical_minimizers_seq_simd_avx2(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        out_ptr: *mut *mut c_uint,
        out_len: *mut c_uint,
    );

    /// Debug: compare fused vs collected
    fn debug_compare_fused_vs_collected(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
    );

    /// Test scalar unrolled ntHash against scalar - returns 1 if match, 0 if mismatch
    fn test_nthash_scalar_unrolled(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
    ) -> i32;

    /// Get forward-only (non-canonical) hash values for comparison with Rust
    fn get_cpp_forward_hashes(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        out_hashes: *mut c_uint,
        max_hashes: c_uint,
    ) -> c_uint;

    /// Benchmark scalar unrolled ntHash - returns time in microseconds
    fn benchmark_nthash_scalar_unrolled(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Test scalar sliding window minimum - returns 1 if two-stack matches naive, 0 if mismatch
    fn test_sliding_min_scalar(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
    ) -> i32;

    /// Benchmark scalar sliding window minimum - returns time in microseconds
    fn benchmark_sliding_min_scalar(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Test SIMD sliding min vs scalar - returns 1 if match, 0 if mismatch
    fn test_sliding_min_simd(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
    ) -> i32;

    /// Benchmark SIMD sliding min - returns time in microseconds
    fn benchmark_sliding_min_simd(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Test packed_seq implementation - returns 1 if correct, 0 if error
    fn test_packed_seq(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
    ) -> i32;

    /// Benchmark packed_seq SIMD iteration - returns time in microseconds
    fn benchmark_packed_seq_simd(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark ntHash using packed_seq SIMD - returns time in microseconds
    fn benchmark_nthash_packed_seq(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark fused pipeline: ntHash + streaming sliding min
    fn benchmark_hash_and_slidmin_only(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark non-canonical FULL pipeline (for comparison)
    fn benchmark_noncanonical_full(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark canonical FULL pipeline DIRECTLY (no FFI result handling)
    fn benchmark_canonical_full_direct(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark canonical phases separately: init, main_loop, collection
    fn benchmark_canonical_phases(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
        init_us: *mut u64,
        main_loop_us: *mut u64,
        collection_us: *mut u64,
    );

    /// Test C++ SIMD non-canonical minimizers against Rust positions
    fn test_noncanonical_minimizers_simd_vs_rust(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        rust_positions: *const c_uint,
        rust_len: c_uint,
    ) -> i32;

    /// Profile collection phases - returns time in microseconds for each phase
    fn profile_collection_phases(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
        main_loop_us: *mut u64,
        partial_batch_us: *mut u64,
        truncate_us: *mut u64,
        prereserve_us: *mut u64,
        flatten_us: *mut u64,
        tail_us: *mut u64,
    );

    // Isolated microbenchmarks for collection components
    fn benchmark_transpose_isolated(iterations: c_uint) -> u64;
    fn benchmark_dedup_isolated(iterations: c_uint) -> u64;
    fn benchmark_dedup_batch_isolated(iterations: c_uint) -> u64;
    fn benchmark_lane_resize_isolated(iterations: c_uint) -> u64;
    fn benchmark_collection_batch_isolated(iterations: c_uint) -> u64;
}

/// Test C++ scalar unrolled ntHash against regular scalar implementation
pub fn cpp_test_nthash_scalar_unrolled(seq_data: &[u8], k: usize) -> bool {
    unsafe { test_nthash_scalar_unrolled(seq_data.as_ptr(), seq_data.len() as c_uint, k as c_uint) == 1 }
}

/// Benchmark C++ SIMD ntHash, returns time in microseconds
pub fn cpp_benchmark_nthash_simd(seq_data: &[u8], k: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_nthash_scalar_unrolled(
            seq_data.as_ptr(),
            seq_data.len() as c_uint,
            k as c_uint,
            iterations as c_uint,
        )
    }
}

/// Get forward-only hash values from C++ for comparison with Rust
pub fn cpp_get_forward_hashes(seq_data: &[u8], k: usize) -> Vec<u32> {
    if seq_data.len() < k {
        return Vec::new();
    }
    let num_kmers = seq_data.len() - k + 1;
    let mut hashes = vec![0u32; num_kmers];
    unsafe {
        let written = get_cpp_forward_hashes(
            seq_data.as_ptr(),
            seq_data.len() as c_uint,
            k as c_uint,
            hashes.as_mut_ptr(),
            num_kmers as c_uint,
        );
        hashes.truncate(written as usize);
    }
    hashes
}

/// Test C++ scalar sliding window minimum (two-stack vs naive)
pub fn cpp_test_sliding_min_scalar(hashes: &[u32], w: usize) -> bool {
    unsafe {
        test_sliding_min_scalar(
            hashes.as_ptr(),
            hashes.len() as c_uint,
            w as c_uint,
        ) == 1
    }
}

/// Benchmark C++ scalar sliding window minimum, returns time in microseconds
pub fn cpp_benchmark_sliding_min_scalar(hashes: &[u32], w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_sliding_min_scalar(
            hashes.as_ptr(),
            hashes.len() as c_uint,
            w as c_uint,
            iterations as c_uint,
        )
    }
}

/// Test C++ SIMD sliding min vs scalar
pub fn cpp_test_sliding_min_simd(hashes: &[u32], w: usize) -> bool {
    unsafe {
        test_sliding_min_simd(
            hashes.as_ptr(),
            hashes.len() as c_uint,
            w as c_uint,
        ) == 1
    }
}

/// Benchmark C++ SIMD sliding window minimum, returns time in microseconds
pub fn cpp_benchmark_sliding_min_simd(hashes: &[u32], w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_sliding_min_simd(
            hashes.as_ptr(),
            hashes.len() as c_uint,
            w as c_uint,
            iterations as c_uint,
        )
    }
}

/// Test C++ packed_seq implementation
pub fn cpp_test_packed_seq(ascii_seq: &[u8]) -> bool {
    unsafe {
        test_packed_seq(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
        ) == 1
    }
}

/// Benchmark C++ packed_seq SIMD iteration, returns time in microseconds
pub fn cpp_benchmark_packed_seq_simd(ascii_seq: &[u8], k: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_packed_seq_simd(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            iterations as c_uint,
        )
    }
}

/// Benchmark C++ ntHash using packed_seq SIMD, returns time in microseconds
pub fn cpp_benchmark_nthash_packed_seq(ascii_seq: &[u8], k: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_nthash_packed_seq(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            iterations as c_uint,
        )
    }
}

/// Benchmark C++ fused pipeline (ntHash + streaming sliding min), returns time in microseconds
pub fn cpp_benchmark_fused_pipeline(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_hash_and_slidmin_only(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint,
        )
    }
}

/// Benchmark C++ non-canonical FULL pipeline, returns time in microseconds
pub fn cpp_benchmark_noncanonical_full(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_noncanonical_full(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint,
        )
    }
}

/// Benchmark C++ canonical FULL pipeline DIRECTLY (no FFI result handling), returns time in microseconds
pub fn cpp_benchmark_canonical_full_direct(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_canonical_full_direct(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint,
        )
    }
}

/// Phase timing for canonical pipeline
pub struct CanonicalPhaseTiming {
    pub init_us: u64,
    /// Main loop includes: hash, sliding min, canonical mapper, AND inline collection (transpose + dedup)
    pub main_loop_us: u64,
    /// Final flatten only: truncate lanes + copy to output vector
    pub final_flatten_us: u64,
}

/// Benchmark C++ canonical phases separately
pub fn cpp_benchmark_canonical_phases(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> CanonicalPhaseTiming {
    let mut timing = CanonicalPhaseTiming {
        init_us: 0,
        main_loop_us: 0,
        final_flatten_us: 0,
    };
    unsafe {
        benchmark_canonical_phases(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint,
            &mut timing.init_us,
            &mut timing.main_loop_us,
            &mut timing.final_flatten_us,
        );
    }
    timing
}

/// Test C++ SIMD non-canonical minimizers against Rust positions
pub fn cpp_test_noncanonical_minimizers_simd(ascii_seq: &[u8], k: usize, w: usize, rust_positions: &[u32]) -> bool {
    unsafe {
        test_noncanonical_minimizers_simd_vs_rust(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            rust_positions.as_ptr(),
            rust_positions.len() as c_uint,
        ) == 1
    }
}

/// Wrapper for the C++ implementation of canonical_minimizers_seq_simd_avx2.
/// 
/// This function computes the canonical minimizers of a sequence using 
/// AVX2 SIMD instructions.
///
/// The function handles the FFI details and memory management, providing a safe interface.
pub fn cpp_canonical_minimizer_positions(
    seq_data: &[u8],
    k: usize,
    w: usize,
    out_minimizers: &mut Vec<u32>
) {
    let l = k + w - 1;
    
    // Check if the window length is odd, which is required for canonicality
    if l % 2 == 0 {
        panic!("Window length (k+w-1) must be odd to guarantee canonicality");
    }
    
    // Check if there's enough sequence data
    if seq_data.len() < l {
        return;
    }
    
    // Prepare output parameters
    let mut out_ptr: *mut c_uint = ptr::null_mut();
    let mut out_len: c_uint = 0;
    
    unsafe {
        // Call the C++ implementation
        canonical_minimizers_seq_simd_avx2(
            seq_data.as_ptr(),
            seq_data.len() as c_uint,
            k as c_uint,
            w as c_uint,
            &mut out_ptr,
            &mut out_len,
        );
        
        
        // If we got results, copy them to the Rust vector
        if !out_ptr.is_null() && out_len > 0 {
            // Create a slice from the returned buffer
            let result_slice = slice::from_raw_parts(out_ptr, out_len as usize);
            
            // Extend the output vector with the results
            out_minimizers.extend_from_slice(result_slice);
            
            // Free the memory allocated by C++
            libc::free(out_ptr as *mut libc::c_void);
        }
    }
}

/// Profile C++ collection phases - returns timing in microseconds for each phase
pub struct CollectionPhaseTiming {
    pub main_loop_us: u64,
    pub partial_batch_us: u64,
    pub truncate_us: u64,
    pub prereserve_us: u64,
    pub flatten_us: u64,
    pub tail_us: u64,
}

pub fn cpp_profile_collection_phases(
    ascii_seq: &[u8],
    k: usize,
    w: usize,
    iterations: usize,
) -> CollectionPhaseTiming {
    let mut timing = CollectionPhaseTiming {
        main_loop_us: 0,
        partial_batch_us: 0,
        truncate_us: 0,
        prereserve_us: 0,
        flatten_us: 0,
        tail_us: 0,
    };
    unsafe {
        profile_collection_phases(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint,
            &mut timing.main_loop_us,
            &mut timing.partial_batch_us,
            &mut timing.truncate_us,
            &mut timing.prereserve_us,
            &mut timing.flatten_us,
            &mut timing.tail_us,
        );
    }
    timing
}

/// Debug: compare fused vs collected C++ implementations
pub fn cpp_debug_compare_fused_vs_collected(seq_data: &[u8], k: usize, w: usize) {
    unsafe {
        debug_compare_fused_vs_collected(
            seq_data.as_ptr(),
            seq_data.len() as c_uint,
            k as c_uint,
            w as c_uint,
        );
    }
}

/// Isolated timing for collection components
pub struct IsolatedCollectionTiming {
    pub transpose_us: u64,
    pub dedup_us: u64,
    pub dedup_batch_us: u64,
    pub lane_resize_us: u64,
    pub collection_batch_us: u64,
}

/// Benchmark isolated collection components
/// Returns timing for 'iterations' of each operation
pub fn cpp_benchmark_collection_components(iterations: usize) -> IsolatedCollectionTiming {
    unsafe {
        IsolatedCollectionTiming {
            transpose_us: benchmark_transpose_isolated(iterations as c_uint),
            dedup_us: benchmark_dedup_isolated(iterations as c_uint),
            dedup_batch_us: benchmark_dedup_batch_isolated(iterations as c_uint),
            lane_resize_us: benchmark_lane_resize_isolated(iterations as c_uint),
            collection_batch_us: benchmark_collection_batch_isolated(iterations as c_uint),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_minimizer_positions;
    use packed_seq::{PackedSeqVec, SeqVec};

    #[test]
    fn test_cpp_vs_rust_implementation() {
        // Test sequences
        let sequences = vec![
            b"ACGTACGTACGT".to_vec(),
            b"AAAAAAACCCCCCCGGGGGGGTTTTTTT".to_vec(),
            b"ACGTACGTACGTACGTACGTACGTACGTACGT".to_vec(),
        ];
        
        for seq in sequences {
            // Test with different k and w values
            for k in [3, 5, 7, 9] {
                for w in [3, 5, 7, 9] {
                    let l = k + w - 1;
                    // Skip cases where l is even (not canonical)
                    if l % 2 == 0 || seq.len() < l {
                        continue;
                    }
                    
                    // Get results from Rust implementation
                    let packed_seq = PackedSeqVec::from_ascii(&seq);
                    let mut rust_results = Vec::new();
                    canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut rust_results);
                    
                    // Special case debugging for k=9, w=3
                    if k == 9 && w == 3 && seq.len() == 12 {
                        println!("DEBUG: Rust implementation for k=9, w=3 produced: {:?}", rust_results);
                    }
                    
                    // Get results from C++ implementation
                    let mut cpp_results = Vec::new();
                    cpp_canonical_minimizer_positions(&seq, k, w, &mut cpp_results);
                    
                    if k == 9 && w == 3 && seq.len() == 12 {
                        println!("DEBUG: C++ implementation for k=9, w=3 produced: {:?}", cpp_results);
                    }
                    
                    // Compare results
                    assert_eq!(
                        rust_results,
                        cpp_results,
                        "Results differ for k={}, w={}, seq={:?}",
                        k, w, String::from_utf8_lossy(&seq)
                    );
                }
            }
        }
    }

    #[test]
    fn test_cpp_nthash_scalar_unrolled() {
        // Test with various sequence lengths and k values
        for seq_len in [100, 512, 1000, 10000] {
            // Generate random sequence
            let seq: Vec<u8> = (0..seq_len).map(|i| (i % 4) as u8).collect();

            for k in [5, 11, 21, 31] {
                if seq_len < k {
                    continue;
                }

                assert!(
                    cpp_test_nthash_scalar_unrolled(&seq, k),
                    "Scalar unrolled ntHash mismatch for seq_len={}, k={}",
                    seq_len, k
                );
            }
        }
    }

    #[test]
    fn test_cpp_sliding_min_scalar() {
        // Test with various hash lengths and window sizes
        for hash_len in [100, 500, 1000, 5000] {
            // Generate pseudo-random hashes
            let hashes: Vec<u32> = (0..hash_len)
                .map(|i| {
                    // Simple LCG for reproducible pseudo-random values
                    let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
                    x ^ (x >> 16)
                })
                .collect();

            for w in [3, 5, 7, 11, 21] {
                if hash_len < w {
                    continue;
                }

                assert!(
                    cpp_test_sliding_min_scalar(&hashes, w),
                    "Scalar sliding min mismatch for hash_len={}, w={}",
                    hash_len, w
                );
            }
        }
    }

    #[test]
    fn test_cpp_sliding_min_simd() {
        // Test with various hash lengths and window sizes
        for hash_len in [100, 500, 1000, 5000] {
            // Generate pseudo-random hashes
            let hashes: Vec<u32> = (0..hash_len)
                .map(|i| {
                    // Simple LCG for reproducible pseudo-random values
                    let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
                    x ^ (x >> 16)
                })
                .collect();

            for w in [3, 5, 7, 11, 21] {
                if hash_len < w {
                    continue;
                }

                assert!(
                    cpp_test_sliding_min_simd(&hashes, w),
                    "SIMD sliding min mismatch for hash_len={}, w={}",
                    hash_len, w
                );
            }
        }
    }

    #[test]
    fn test_cpp_packed_seq() {
        // Test with various sequence lengths
        for seq_len in [100, 500, 1000, 5000] {
            // Generate pseudo-random DNA sequence
            let bases = b"ACGT";
            let seq: Vec<u8> = (0..seq_len)
                .map(|i| bases[i % 4])
                .collect();

            assert!(
                cpp_test_packed_seq(&seq),
                "packed_seq test failed for seq_len={}",
                seq_len
            );
        }

        // Test with mixed case
        let mixed_seq = b"AcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGtAcGt";
        assert!(
            cpp_test_packed_seq(mixed_seq),
            "packed_seq test failed for mixed case sequence"
        );
    }

    #[test]
    fn test_cpp_hash_vs_rust() {
        use crate::private::nthash::{nthash_seq_scalar, NtHasher};
        use packed_seq::{PackedSeqVec, SeqVec};

        // Simple test sequence
        let seq = b"ACGTACGTACGTACGTACGT";
        let k = 5;

        // Get Rust hashes (forward-only, non-canonical)
        let packed_seq = PackedSeqVec::from_ascii(seq);
        let rust_hashes: Vec<u32> = nthash_seq_scalar::<false, NtHasher>(packed_seq.as_slice(), k).collect();

        // Get C++ hashes
        let cpp_hashes = cpp_get_forward_hashes(seq, k);

        eprintln!("Rust hashes: {:?}", rust_hashes.iter().map(|h| format!("0x{:08x}", h)).collect::<Vec<_>>());
        eprintln!("C++  hashes: {:?}", cpp_hashes.iter().map(|h| format!("0x{:08x}", h)).collect::<Vec<_>>());

        assert_eq!(rust_hashes.len(), cpp_hashes.len(),
            "Hash count mismatch: Rust={}, C++={}", rust_hashes.len(), cpp_hashes.len());

        for (i, (rust, cpp)) in rust_hashes.iter().zip(cpp_hashes.iter()).enumerate() {
            if rust != cpp {
                eprintln!("Hash mismatch at position {}: Rust=0x{:08x}, C++=0x{:08x}", i, rust, cpp);
            }
            assert_eq!(rust, cpp,
                "Hash mismatch at position {}: Rust=0x{:08x}, C++=0x{:08x}", i, rust, cpp);
        }
    }

    #[test]
    fn test_cpp_nthash_vs_rust() {
        use crate::private::nthash::{nthash_seq_scalar, NtHasher};
        use packed_seq::{PackedSeqVec, SeqVec};

        // Generate test sequence
        let bases = b"ACGT";
        let seq: Vec<u8> = (0..100)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
                bases[(x >> 16) as usize % 4]
            })
            .collect();

        let k = 11;

        // Get Rust hashes
        let packed_seq = PackedSeqVec::from_ascii(&seq);
        let rust_hashes: Vec<u32> = nthash_seq_scalar::<false, NtHasher>(packed_seq.as_slice(), k).collect();

        // Get C++ hashes for comparison
        let cpp_hashes = cpp_get_forward_hashes(&seq, k);

        // Verify hashes match
        assert_eq!(rust_hashes.len(), cpp_hashes.len(),
            "Hash count mismatch: Rust={}, C++={}", rust_hashes.len(), cpp_hashes.len());
        for (i, (rust, cpp)) in rust_hashes.iter().zip(cpp_hashes.iter()).enumerate() {
            assert_eq!(rust, cpp, "Hash mismatch at position {}", i);
        }
    }

    #[test]
    fn test_cpp_noncanonical_minimizers_simd_vs_rust() {
        use crate::minimizer_positions;
        use packed_seq::{PackedSeqVec, SeqVec};

        // Test with various sequence lengths - start small for debugging
        for seq_len in [500, 1000, 5000, 10000] {
            // Generate pseudo-random DNA sequence
            let bases = b"ACGT";
            let seq: Vec<u8> = (0..seq_len)
                .map(|i| {
                    // Use LCG for reproducible pseudo-random sequence
                    let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
                    bases[(x >> 16) as usize % 4]
                })
                .collect();

            // Test with different k and w values
            for k in [11, 21, 31] {
                for w in [5, 11] {
                    let l = k + w - 1;
                    if seq_len < l {
                        continue;
                    }

                    // Get Rust results (non-canonical minimizers)
                    let packed_seq = PackedSeqVec::from_ascii(&seq);
                    let mut rust_positions = Vec::new();
                    minimizer_positions(packed_seq.as_slice(), k, w, &mut rust_positions);

                    eprintln!("Testing seq_len={}, k={}, w={}", seq_len, k, w);
                    eprintln!("  Rust positions ({} total): {:?}...", rust_positions.len(),
                        &rust_positions[..std::cmp::min(20, rust_positions.len())]);

                    // Compare with C++
                    let result = cpp_test_noncanonical_minimizers_simd(&seq, k, w, &rust_positions);

                    if !result {
                        eprintln!("  FAILED!");
                    }

                    assert!(
                        result,
                        "C++ SIMD non-canonical minimizers mismatch for seq_len={}, k={}, w={}",
                        seq_len, k, w
                    );
                }
            }
        }
    }

    #[test]
    fn test_debug_fused_vs_collected() {
        // Generate a sequence that fails - 1000 bases
        let bases = b"ACGT";
        let seq: Vec<u8> = (0..1000)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
                bases[(x >> 16) as usize % 4]
            })
            .collect();

        let k = 21;
        let w = 11;

        // Get Rust results for comparison
        let packed_seq = PackedSeqVec::from_ascii(&seq);
        let mut rust_positions = Vec::new();
        canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut rust_positions);

        eprintln!("Rust produces {} positions, last 5: {:?}",
                  rust_positions.len(),
                  &rust_positions[rust_positions.len().saturating_sub(5)..]);

        cpp_debug_compare_fused_vs_collected(&seq, k, w);
    }
}