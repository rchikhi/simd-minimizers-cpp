//! Bindings for the C++ implementation of canonical_minimizers_seq_simd_avx2
//!
//! This module provides a safe Rust interface to the C++ implementation.
#![allow(dead_code)]

use std::os::raw::{c_uchar, c_uint};
use std::slice;
use std::ptr;

unsafe extern "C" {
    /// C++ function to compute canonical minimizers
    fn canonical_minimizers_seq_simd_avx2(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        out_ptr: *mut *mut c_uint,
        out_len: *mut c_uint,
    );

    /// Test scalar unrolled ntHash against scalar - returns 1 if match, 0 if mismatch
    fn test_nthash_scalar_unrolled(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
    ) -> i32;

    /// Test scalar sliding window minimum - returns 1 if two-stack matches naive, 0 if mismatch
    fn test_sliding_min_scalar(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
    ) -> i32;

    /// Test SIMD sliding min vs scalar - returns 1 if match, 0 if mismatch
    fn test_sliding_min_simd(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
    ) -> i32;

    /// Test packed_seq implementation - returns 1 if correct, 0 if error
    fn test_packed_seq(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
    ) -> i32;

    /// Test C++ non-canonical minimizers vs Rust positions
    fn test_noncanonical_minimizers_simd_vs_rust(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        rust_positions: *const c_uint,
        rust_len: c_uint,
    ) -> i32;

    /// Get forward (non-canonical) hashes from C++ for comparison
    fn get_cpp_forward_hashes(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        out_hashes: *mut c_uint,
        max_hashes: c_uint,
    ) -> c_uint;

    /// Get canonical hashes from C++ scalar implementation
    fn get_cpp_canonical_hashes_scalar(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        out_hashes: *mut c_uint,
        max_hashes: c_uint,
    ) -> c_uint;

    /// Debug trace for canonical hash computation
    fn debug_canonical_hash_trace(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
    );

    /// Benchmark scalar unrolled ntHash - returns time in microseconds
    fn benchmark_nthash_scalar_unrolled(
        seq_data: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark scalar sliding window minimum - returns time in microseconds
    fn benchmark_sliding_min_scalar(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark SIMD sliding min - returns time in microseconds
    fn benchmark_sliding_min_simd(
        hashes: *const c_uint,
        hash_len: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

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

    /// Benchmark fused hash + sliding min (no collection) - returns time in microseconds
    fn benchmark_hash_and_slidmin_only(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark non-canonical full pipeline - returns time in microseconds
    fn benchmark_noncanonical_full(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark canonical full pipeline (direct, no FFI overhead) - returns time in microseconds
    fn benchmark_canonical_full_direct(
        ascii_seq: *const c_uchar,
        seq_len: c_uint,
        k: c_uint,
        w: c_uint,
        iterations: c_uint,
    ) -> u64;

    /// Benchmark canonical phases - returns timing in 3 separate phases
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

    /// Free minimizers buffer allocated by C++
    fn free_minimizers(ptr: *mut c_uint);
}

// ============================================================================
// Safe Rust wrappers
// ============================================================================

/// Compute canonical minimizer positions using C++ AVX2 SIMD implementation.
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

// ============================================================================
// Test wrappers
// ============================================================================

pub fn cpp_test_nthash_scalar_unrolled(seq_data: &[u8], k: usize) -> bool {
    unsafe { test_nthash_scalar_unrolled(seq_data.as_ptr(), seq_data.len() as c_uint, k as c_uint) == 1 }
}

pub fn cpp_test_sliding_min_scalar(hashes: &[u32], w: usize) -> bool {
    unsafe { test_sliding_min_scalar(hashes.as_ptr(), hashes.len() as c_uint, w as c_uint) == 1 }
}

pub fn cpp_test_sliding_min_simd(hashes: &[u32], w: usize) -> bool {
    unsafe { test_sliding_min_simd(hashes.as_ptr(), hashes.len() as c_uint, w as c_uint) == 1 }
}

pub fn cpp_test_packed_seq(ascii_seq: &[u8]) -> bool {
    unsafe { test_packed_seq(ascii_seq.as_ptr(), ascii_seq.len() as c_uint) == 1 }
}

pub fn cpp_test_noncanonical_minimizers_simd(ascii_seq: &[u8], k: usize, w: usize, rust_positions: &[u32]) -> bool {
    unsafe {
        test_noncanonical_minimizers_simd_vs_rust(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            rust_positions.as_ptr(),
            rust_positions.len() as c_uint
        ) == 1
    }
}

/// Get forward (non-canonical) hashes from C++ for comparison with Rust
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

/// Get canonical hashes from C++ scalar implementation for comparison with Rust
pub fn cpp_get_canonical_hashes_scalar(seq_data: &[u8], k: usize) -> Vec<u32> {
    if seq_data.len() < k {
        return Vec::new();
    }
    let num_kmers = seq_data.len() - k + 1;
    let mut hashes = vec![0u32; num_kmers];
    unsafe {
        let written = get_cpp_canonical_hashes_scalar(
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

// ============================================================================
// Benchmark wrappers
// ============================================================================

pub fn cpp_benchmark_nthash_simd(seq_data: &[u8], k: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_nthash_scalar_unrolled(
            seq_data.as_ptr(),
            seq_data.len() as c_uint,
            k as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_sliding_min_scalar(hashes: &[u32], w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_sliding_min_scalar(
            hashes.as_ptr(),
            hashes.len() as c_uint,
            w as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_sliding_min_simd(hashes: &[u32], w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_sliding_min_simd(
            hashes.as_ptr(),
            hashes.len() as c_uint,
            w as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_packed_seq_simd(ascii_seq: &[u8], k: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_packed_seq_simd(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_nthash_packed_seq(ascii_seq: &[u8], k: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_nthash_packed_seq(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_fused_pipeline(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_hash_and_slidmin_only(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_noncanonical_full(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_noncanonical_full(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint
        )
    }
}

pub fn cpp_benchmark_canonical_full_direct(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> u64 {
    unsafe {
        benchmark_canonical_full_direct(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint
        )
    }
}

/// Timing for the three phases of canonical minimizer computation
pub struct CanonicalPhaseTiming {
    pub init_us: u64,
    pub main_loop_us: u64,
    pub final_flatten_us: u64,
}

pub fn cpp_benchmark_canonical_phases(ascii_seq: &[u8], k: usize, w: usize, iterations: usize) -> CanonicalPhaseTiming {
    let mut init_us = 0u64;
    let mut main_loop_us = 0u64;
    let mut final_flatten_us = 0u64;
    unsafe {
        benchmark_canonical_phases(
            ascii_seq.as_ptr(),
            ascii_seq.len() as c_uint,
            k as c_uint,
            w as c_uint,
            iterations as c_uint,
            &mut init_us,
            &mut main_loop_us,
            &mut final_flatten_us,
        );
    }
    CanonicalPhaseTiming {
        init_us,
        main_loop_us,
        final_flatten_us,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use packed_seq::{PackedSeqVec, SeqVec};

    #[test]
    fn test_cpp_vs_rust_implementation() {
        use seq_hash::{KmerHasher, NtHasher};

        let test_seqs = [
            b"ACGTACGTACGT".as_slice(),
            b"AAAAAAACCCCCCCGGGGGGGTTTTTTT",
            b"ACGTACGTACGTACGTACGTACGTACGTACGT",
        ];

        for seq in test_seqs {
            for k in [3, 5, 7, 9] {
                for w in [3, 5, 7, 9] {
                    let l = k + w - 1;
                    if l % 2 == 0 || seq.len() < l {
                        continue;
                    }

                    let packed_seq = PackedSeqVec::from_ascii(seq);

                    // Get Rust results using scalar API (to avoid SIMD differences)
                    let rust_pos = crate::canonical_minimizers(k, w).run_scalar_once(packed_seq.as_slice());

                    // Get C++ results
                    let mut cpp_pos = Vec::new();
                    cpp_canonical_minimizer_positions(seq, k, w, &mut cpp_pos);

                    // Debug: print raw results for first failing case
                    if rust_pos != cpp_pos {
                        eprintln!("DEBUG: seq={:?}, k={}, w={}, l={}",
                            std::str::from_utf8(seq).unwrap(), k, w, l);
                        eprintln!("  Rust scalar: {:?}", rust_pos);
                        eprintln!("  C++  raw:    {:?}", cpp_pos);

                        // Also print canonical hashes for debugging
                        let hasher = NtHasher::<true>::new(k);
                        let hashes: Vec<u32> = hasher.hash_kmers_scalar(packed_seq.as_slice()).collect();
                        eprintln!("  Canonical hashes: {:?}", hashes);
                    }

                    // Sort and deduplicate both for comparison
                    let mut sorted_rust: Vec<u32> = rust_pos.clone();
                    let mut sorted_cpp: Vec<u32> = cpp_pos.clone();
                    sorted_rust.sort();
                    sorted_rust.dedup();
                    sorted_cpp.sort();
                    sorted_cpp.dedup();

                    assert_eq!(
                        sorted_rust, sorted_cpp,
                        "C++ and Rust results differ for k={}, w={}, seq len={}",
                        k, w, seq.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_cpp_sliding_min() {
        // Generate some test hashes
        let hashes: Vec<u32> = (0..1000).map(|i| (i * 12345 + 67890) as u32).collect();

        for w in [3, 5, 7, 11] {
            assert!(
                cpp_test_sliding_min_scalar(&hashes, w),
                "C++ sliding min scalar failed for w={}",
                w
            );
            assert!(
                cpp_test_sliding_min_simd(&hashes, w),
                "C++ sliding min SIMD failed for w={}",
                w
            );
        }
    }

    #[test]
    fn test_cpp_packed_seq() {
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        assert!(cpp_test_packed_seq(seq), "C++ packed_seq test failed");
    }

    #[test]
    fn test_cpp_canonical_hashes_trace() {
        use packed_seq::complement_base;

        // Simple test case: "AAA" with k=3
        let seq = b"AAA";
        let k: usize = 3;
        let r: u32 = 7; // Rotation constant

        // ntHash constants
        let hashes_f: [u32; 4] = [
            0x95c60474,  // A
            0x62a02b4c,  // C
            0x82572324,  // T
            0x4be24456,  // G
        ];

        let rot = ((k - 1) * r as usize) as u32;

        // Compute lookup tables like Rust does
        let c: [u32; 4] = std::array::from_fn(|i| hashes_f[complement_base(i as u8) as usize]);
        let f_rot: [u32; 4] = hashes_f.map(|h| h.rotate_left(rot));
        let c_rot: [u32; 4] = c.map(|h| h.rotate_left(rot));

        eprintln!("=== Rust Debug Trace ===");
        eprintln!("k={}, R={}, rot={}", k, r, rot);
        eprintln!("HASHES_F: [0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                  hashes_f[0], hashes_f[1], hashes_f[2], hashes_f[3]);
        eprintln!("c:        [0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                  c[0], c[1], c[2], c[3]);
        eprintln!("c_rot:    [0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                  c_rot[0], c_rot[1], c_rot[2], c_rot[3]);
        eprintln!("f_rot:    [0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]",
                  f_rot[0], f_rot[1], f_rot[2], f_rot[3]);

        // Compute fw_init and rc_init
        let mut fw: u32 = 0;
        let mut rc: u32 = 0;
        for i in 0..k-1 {
            fw = fw.rotate_left(r) ^ hashes_f[0];
            rc = rc.rotate_right(r) ^ c_rot[0];
            eprintln!("  Init iter {}: fw=0x{:08x}, rc=0x{:08x}", i, fw, rc);
        }
        eprintln!("After init: fw=0x{:08x}, rc=0x{:08x}", fw, rc);

        // Warmup
        for i in 0..k-1 {
            let add_base = packed_seq::pack_char(seq[i]);
            eprintln!("  Warmup {}: add_base={} (char '{}')", i, add_base, seq[i] as char);
            let fw_out = fw.rotate_left(r) ^ hashes_f[add_base as usize];
            let rc_out = rc.rotate_right(r) ^ c_rot[add_base as usize];
            eprintln!("    fw_out=0x{:08x}, rc_out=0x{:08x}", fw_out, rc_out);
            fw = fw_out ^ f_rot[0];
            rc = rc_out ^ c[0];
            eprintln!("    after remove: fw=0x{:08x}, rc=0x{:08x}", fw, rc);
        }
        eprintln!("After warmup: fw=0x{:08x}, rc=0x{:08x}", fw, rc);

        // First hash
        let add_base = packed_seq::pack_char(seq[k - 1]);
        let remove_base = packed_seq::pack_char(seq[0]);
        eprintln!("Main pos=0: add={} (char '{}'), remove={} (char '{}')",
                  add_base, seq[k-1] as char, remove_base, seq[0] as char);
        let fw_out = fw.rotate_left(r) ^ hashes_f[add_base as usize];
        let rc_out = rc.rotate_right(r) ^ c_rot[add_base as usize];
        eprintln!("  fw_out=0x{:08x}, rc_out=0x{:08x}", fw_out, rc_out);
        let hash = fw_out.wrapping_add(rc_out);
        eprintln!("  hash = fw_out + rc_out = 0x{:08x} ({})", hash, hash);

        eprintln!("\n=== C++ Debug Trace ===");
        unsafe {
            debug_canonical_hash_trace(seq.as_ptr(), seq.len() as c_uint, k as c_uint);
        }
    }

    #[test]
    fn test_cpp_canonical_hashes_vs_rust() {
        use seq_hash::{KmerHasher, NtHasher};

        // Test case: very simple sequence "AAA" with k=3
        let seq = b"AAA";
        let k = 3;

        let packed_seq = PackedSeqVec::from_ascii(seq);

        // Get Rust canonical hash (just one k-mer: AAA)
        let hasher = NtHasher::<true>::new(k);
        let rust_hashes: Vec<u32> = hasher.hash_kmers_scalar(packed_seq.as_slice()).collect();
        eprintln!("seq='AAA', k=3");
        eprintln!("Rust canonical hashes: {:?}", rust_hashes);

        // Get C++ canonical hashes
        let cpp_hashes = cpp_get_canonical_hashes_scalar(seq, k);
        eprintln!("C++ canonical hashes:  {:?}", cpp_hashes);

        // Now test the failing case
        let seq = b"ACGTACGTACGT";
        let k = 7;

        let packed_seq = PackedSeqVec::from_ascii(seq);

        // Get Rust canonical hashes
        let hasher = NtHasher::<true>::new(k);
        let rust_hashes: Vec<u32> = hasher.hash_kmers_scalar(packed_seq.as_slice()).collect();
        eprintln!("\nseq='ACGTACGTACGT', k=7");
        eprintln!("Rust canonical hashes: {:?}", rust_hashes);

        // Get C++ canonical hashes from scalar implementation
        let cpp_hashes = cpp_get_canonical_hashes_scalar(seq, k);
        eprintln!("C++ canonical hashes:  {:?}", cpp_hashes);

        // Compare
        assert_eq!(rust_hashes.len(), cpp_hashes.len(), "Hash count mismatch");
        for (i, (rust, cpp)) in rust_hashes.iter().zip(cpp_hashes.iter()).enumerate() {
            if rust != cpp {
                eprintln!("Hash mismatch at position {}:", i);
                eprintln!("  Rust: 0x{:08x}", rust);
                eprintln!("  C++:  0x{:08x}", cpp);
            }
            assert_eq!(rust, cpp, "Canonical hash mismatch at position {}", i);
        }
    }
    #[test]
    fn test_cpp_nthash_vs_rust() {
        use seq_hash::{KmerHasher, NtHasher};

        // Generate test sequence
        let bases = b"ACGT";
        let seq: Vec<u8> = (0..100)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
                bases[(x >> 16) as usize % 4]
            })
            .collect();

        let k = 11;

        // Get Rust forward (non-canonical) hashes with R=7 (default, matching updated C++)
        let packed_seq = PackedSeqVec::from_ascii(&seq);
        let fwd_hasher = NtHasher::<false>::new(k);
        let rust_hashes: Vec<u32> = fwd_hasher.hash_kmers_scalar(packed_seq.as_slice()).collect();

        // Get C++ forward hashes
        let cpp_hashes = cpp_get_forward_hashes(&seq, k);

        // Compare counts
        assert_eq!(
            rust_hashes.len(), cpp_hashes.len(),
            "Hash count mismatch: Rust={}, C++={}",
            rust_hashes.len(), cpp_hashes.len()
        );

        // Compare hash values
        for (i, (rust, cpp)) in rust_hashes.iter().zip(cpp_hashes.iter()).enumerate() {
            if rust != cpp {
                eprintln!("Hash mismatch at position {}:", i);
                eprintln!("  Rust hash: 0x{:08x}", rust);
                eprintln!("  C++  hash: 0x{:08x}", cpp);
                eprintln!("  Kmer: {:?}", std::str::from_utf8(&seq[i..i+k]).unwrap());
            }
            assert_eq!(rust, cpp, "Hash mismatch at position {}", i);
        }
    }
}
