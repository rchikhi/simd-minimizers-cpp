use super::*;
use crate::{minimizers::*, nthash::*};
use collect::collect;
use itertools::Itertools;
use packed_seq::{AsciiSeq, AsciiSeqVec, PackedSeq, PackedSeqVec, SeqVec};
use rand::Rng;
use std::{iter::once, sync::LazyLock};

/// Swap G and T, so that the lex order is the same as for the packed version.
fn swap_gt(c: u8) -> u8 {
    match c {
        b'G' => b'T',
        b'T' => b'G',
        c => c,
    }
}

static ASCII_SEQ: LazyLock<AsciiSeqVec> = LazyLock::new(|| AsciiSeqVec::random(1024));
static SLICE: LazyLock<Vec<u8>> =
    LazyLock::new(|| ASCII_SEQ.seq.iter().copied().map(swap_gt).collect_vec());
static PACKED_SEQ: LazyLock<PackedSeqVec> =
    LazyLock::new(|| PackedSeqVec::from_ascii(&ASCII_SEQ.seq));

fn test_on_inputs(f: impl Fn(usize, usize, &[u8], AsciiSeq, PackedSeq)) {
    let slice = &*SLICE;
    let ascii_seq = &*ASCII_SEQ;
    let packed_seq = &*PACKED_SEQ;
    let mut rng = rand::rng();
    let mut ks = vec![1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65];
    let mut ws = vec![1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65];
    let mut lens = (0..100).collect_vec();
    ks.extend((0..10).map(|_| rng.random_range(6..100)).collect_vec());
    ws.extend((0..10).map(|_| rng.random_range(6..100)).collect_vec());
    lens.extend((0..10).map(|_| rng.random_range(100..1024)).collect_vec());
    for &k in &ks {
        for &w in &ws {
            for &len in &lens {
                let slice = slice.slice(0..len);
                let ascii_seq = ascii_seq.slice(0..len);
                let packed_seq = packed_seq.slice(0..len);

                f(k, w, slice, ascii_seq, packed_seq);
            }
        }
    }
}

fn test_nthash<const RC: bool, H: CharHasher>() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if w > 1 {
            return;
        }

        let naive = ascii_seq
            .0
            .windows(k)
            .map(|seq| nthash_kmer::<RC, H>(AsciiSeq(seq)))
            .collect::<Vec<_>>();
        let scalar_ascii = nthash_seq_scalar::<RC, H>(ascii_seq, k).collect::<Vec<_>>();
        let scalar_packed = nthash_seq_scalar::<RC, H>(packed_seq, k).collect::<Vec<_>>();
        let simd_ascii = collect(nthash_seq_simd::<RC, AsciiSeq, H>(ascii_seq, k, 1));
        let simd_packed = collect(nthash_seq_simd::<RC, PackedSeq, H>(packed_seq, k, 1));

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
        assert_eq!(simd_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(simd_packed, naive, "k={}, len={}", k, len);
    });
}

#[test]
fn nthash_forward() {
    test_nthash::<false, NtHasher>();
}

#[test]
fn nthash_canonical() {
    test_nthash::<true, NtHasher>();
}

#[test]
fn nthash_forward_mul() {
    test_nthash::<false, MulHasher>();
}

#[test]
fn nthash_canonical_mul() {
    test_nthash::<true, MulHasher>();
}

#[test]
fn nthash_canonical_is_revcomp() {
    fn f<H: CharHasher>() {
        let seq = &*ASCII_SEQ;
        let seq_rc = AsciiSeqVec::from_vec(
            seq.seq
                .iter()
                .rev()
                .map(|c| packed_seq::complement_char(*c))
                .collect_vec(),
        );
        for k in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
        ] {
            for len in (0..100).chain(once(1024)) {
                let seq = seq.slice(0..len);
                let seq_rc = seq_rc.slice(seq_rc.len() - len..seq_rc.len());
                let scalar = nthash_seq_scalar::<true, H>(seq, k).collect::<Vec<_>>();
                let scalar_rc = nthash_seq_scalar::<true, H>(seq_rc, k).collect::<Vec<_>>();
                let scalar_rc_rc = scalar_rc.iter().rev().copied().collect_vec();
                assert_eq!(
                    scalar_rc_rc,
                    scalar,
                    "k={}, len={} {:032b} {:032b}",
                    k,
                    len,
                    scalar.first().unwrap_or(&0),
                    scalar_rc_rc.first().unwrap_or(&0)
                );
            }
        }
    }
    f::<NtHasher>();
    f::<MulHasher>();
}

#[test]
fn test_anti_lex_hash() {
    use anti_lex::*;
    test_on_inputs(|k, w, slice, ascii_seq, packed_seq| {
        if w > 1 {
            return;
        }
        // naive
        let naive = ascii_seq
            .0
            .windows(k)
            .map(|seq| anti_lex_hash_kmer(AsciiSeq(seq)))
            .collect::<Vec<_>>();
        let scalar_ascii = anti_lex_hash_seq_scalar(ascii_seq, k).collect::<Vec<_>>();
        let scalar_packed = anti_lex_hash_seq_scalar(packed_seq, k).collect::<Vec<_>>();
        let simd_ascii = collect(anti_lex_hash_seq_simd(ascii_seq, k, 1));
        let simd_packed = collect(anti_lex_hash_seq_simd(packed_seq, k, 1));
        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(scalar_packed, naive, "k={}, len={}", k, len);
        assert_eq!(simd_ascii, naive, "k={}, len={}", k, len);
        assert_eq!(simd_packed, naive, "k={}, len={}", k, len);

        let scalar_slice = anti_lex_hash_seq_scalar(slice, k).collect::<Vec<_>>();
        let simd_slice = collect(anti_lex_hash_seq_simd(slice, k, 1));
        assert_eq!(simd_slice, scalar_slice, "k={}, len={}", k, len);
    });
}

#[test]
fn minimizers_fwd() {
    fn f<H: CharHasher>() {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            let naive = ascii_seq
                .0
                .windows(w + k - 1)
                .enumerate()
                .map(|(pos, seq)| (pos + minimizer::<H>(AsciiSeq(seq), k)) as u32)
                .collect::<Vec<_>>();

            let scalar_ascii = minimizers_seq_scalar::<H>(ascii_seq, k, w).collect::<Vec<_>>();
            let scalar_packed = minimizers_seq_scalar::<H>(packed_seq, k, w).collect::<Vec<_>>();
            let simd_ascii = collect(minimizers_seq_simd::<_, H>(ascii_seq, k, w));
            let simd_packed = collect(minimizers_seq_simd::<_, H>(packed_seq, k, w));

            let len = ascii_seq.len();
            assert_eq!(naive, scalar_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(naive, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(naive, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(naive, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f::<NtHasher>();
    f::<MulHasher>();
}

#[test]
fn minimizers_canonical() {
    fn f<H: CharHasher>() {
        test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
            if (k + w - 1) % 2 == 0 {
                return;
            }
            let scalar_ascii =
                canonical_minimizers_seq_scalar::<H>(ascii_seq, k, w).collect::<Vec<_>>();
            let scalar_packed =
                canonical_minimizers_seq_scalar::<H>(packed_seq, k, w).collect::<Vec<_>>();
            let simd_ascii = collect(canonical_minimizers_seq_simd::<_, H>(ascii_seq, k, w));
            let simd_packed = collect(canonical_minimizers_seq_simd::<_, H>(packed_seq, k, w));

            let len = ascii_seq.len();
            assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
            assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
        });
    }
    f::<NtHasher>();
    f::<MulHasher>();
}

#[test]
fn minimizer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        let mut scalar_ascii = vec![];
        minimizer_positions_scalar(ascii_seq, k, w, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        minimizer_positions_scalar(packed_seq, k, w, &mut scalar_packed);
        let mut simd_ascii = vec![];
        super::minimizer_positions(ascii_seq, k, w, &mut simd_ascii);
        let mut simd_packed = vec![];
        super::minimizer_positions(packed_seq, k, w, &mut simd_packed);

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}

#[test]
fn test_canonical_minimizer_positions() {
    test_on_inputs(|k, w, _slice, ascii_seq, packed_seq| {
        if (k + w - 1) % 2 == 0 {
            return;
        }
        let mut scalar_ascii = vec![];
        canonical_minimizer_positions_scalar(ascii_seq, k, w, &mut scalar_ascii);
        let mut scalar_packed = vec![];
        canonical_minimizer_positions_scalar(packed_seq, k, w, &mut scalar_packed);
        let mut simd_ascii = vec![];
        super::canonical_minimizer_positions(ascii_seq, k, w, &mut simd_ascii);
        let mut simd_packed = vec![];
        super::canonical_minimizer_positions(packed_seq, k, w, &mut simd_packed);

        let len = ascii_seq.len();
        assert_eq!(scalar_ascii, scalar_packed, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_ascii, "k={k}, w={w}, len={len}");
        assert_eq!(scalar_ascii, simd_packed, "k={k}, w={w}, len={len}");
    });
}

// Test that C++ and Rust implementations produce identical results
#[test]
fn test_cpp_vs_rust_canonical_minimizers() {
    use crate::cpp_bindings::cpp_canonical_minimizer_positions;

    /// Debug helper: prints detailed hash/minimizer info for troubleshooting
    fn debug_canonical_pipeline(k: usize, w: usize, data: &[u8]) {
        let test_packed_seq = PackedSeqVec::from_ascii(data);

        // Canonicality per window
        let canonical_windows: Vec<_> = crate::canonical::canonical_windows_seq_scalar(
            test_packed_seq.as_slice(), k, w).collect();
        eprintln!("Windows ({}): {:?}", canonical_windows.len(),
            canonical_windows.iter().map(|c| if *c {'L'} else {'R'}).collect::<String>());

        // Hash values
        let hashes: Vec<_> = crate::nthash::nthash_seq_scalar::<true, crate::NtHasher>(
            test_packed_seq.as_slice(), k).collect();
        eprintln!("Hashes: {:?}", hashes.iter().map(|h| format!("{:08x}", h)).collect::<Vec<_>>());

        // Left/right minimizers
        let left: Vec<_> = crate::sliding_min::sliding_min_scalar::<true>(hashes.iter().cloned(), w).collect();
        let right: Vec<_> = crate::sliding_min::sliding_min_scalar::<false>(hashes.iter().cloned(), w).collect();
        eprintln!("Left mins:  {:?}", left);
        eprintln!("Right mins: {:?}", right);

        // Final selection
        let selected: Vec<_> = canonical_windows.iter().zip(left.iter().zip(right.iter()))
            .map(|(c, (l, r))| if *c { *l } else { *r }).collect();
        eprintln!("Selected: {:?}", selected);
    }

    fn compare_implementations(k: usize, w: usize, data: &[u8]) {
        if (k + w - 1) % 2 == 0 || data.len() < k + w - 1 {
            return; // Skip invalid cases
        }

        let packed_seq = PackedSeqVec::from_ascii(data);
        let mut rust_results = Vec::new();
        canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut rust_results);

        let mut cpp_results = Vec::new();
        cpp_canonical_minimizer_positions(data, k, w, &mut cpp_results);

        // Debug output for specific test cases (run with --nocapture to see)
        if (k == 3 || k == 9) && w == 3 && data.len() == 12 {
            eprintln!("\n=== DEBUG k={}, w={} ===", k, w);
            eprintln!("Rust: {:?}", rust_results);
            eprintln!("C++:  {:?}", cpp_results);
            debug_canonical_pipeline(k, w, data);
        }

        let mut sorted_rust = rust_results.clone();
        let mut sorted_cpp = cpp_results.clone();
        sorted_rust.sort_unstable();
        sorted_cpp.sort_unstable();

        assert_eq!(sorted_rust, sorted_cpp,
            "Results differ for k={}, w={}, seq={:?}",
            k, w, String::from_utf8_lossy(&data[..std::cmp::min(data.len(), 100)]));
    }

    // Test with fixed sequences
    for seq in [b"ACGTACGTACGT".as_slice(), b"AAAAAAACCCCCCCGGGGGGGTTTTTTT",
                b"ACGTACGTACGTACGTACGTACGTACGTACGT"] {
        for k in [3, 5, 7, 9] {
            for w in [3, 5, 7, 9] {
                compare_implementations(k, w, seq);
            }
        }
    }

    // Test with random sequences
    let random_seq = &*ASCII_SEQ;
    for len in [64, 128, 256, 512] {
        let seq = random_seq.slice(0..len);
        for k in [3, 5, 7, 9] {
            for w in [3, 5, 7, 9] {
                compare_implementations(k, w, &seq.0);
            }
        }
    }
}

// Test with larger sequences to catch deduplication issues
#[test]
fn test_cpp_vs_rust_canonical_minimizers_large() {
    use crate::cpp_bindings::cpp_canonical_minimizer_positions;
    use rand::Rng;

    // Generate a random 1M sequence
    let bases = b"ACGT";
    let mut rng = rand::rng();
    let seq_data: Vec<u8> = (0..1_000_000).map(|_| bases[rng.random_range(0..4)]).collect();
    let packed_seq = PackedSeqVec::from_ascii(&seq_data);

    let k = 5;
    let w = 5;

    // Get results from Rust implementation
    let mut rust_results = Vec::new();
    canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut rust_results);

    // Get results from C++ implementation
    let mut cpp_results = Vec::new();
    cpp_canonical_minimizer_positions(&seq_data, k, w, &mut cpp_results);

    // Check for consecutive duplicates (should not happen after dedup)
    let mut cpp_dups = Vec::new();
    for i in 1..cpp_results.len() {
        if cpp_results[i] == cpp_results[i-1] {
            cpp_dups.push((i, cpp_results[i]));
        }
    }

    if !cpp_dups.is_empty() {
        println!("C++ has {} consecutive duplicates:", cpp_dups.len());
        for (idx, pos) in cpp_dups.iter().take(10) {
            println!("  At index {}: position {}", idx, pos);
        }
    }

    // Compare unsorted first (to catch ordering issues)
    if rust_results != cpp_results {
        println!("Outputs differ (unsorted):");
        println!("  Rust len: {}", rust_results.len());
        println!("  C++ len: {}", cpp_results.len());

        // Find first difference
        let min_len = rust_results.len().min(cpp_results.len());
        for i in 0..min_len {
            if rust_results[i] != cpp_results[i] {
                println!("  First diff at index {}: Rust={}, C++={}", i, rust_results[i], cpp_results[i]);
                let start = i.saturating_sub(3);
                let end = (i + 4).min(min_len);
                println!("  Context Rust[{}..{}]: {:?}", start, end, &rust_results[start..end]);
                println!("  Context C++[{}..{}]: {:?}", start, end, &cpp_results[start..end]);
                break;
            }
        }
    }

    // Assert no consecutive duplicates in C++ output
    assert!(cpp_dups.is_empty(),
        "C++ implementation has {} consecutive duplicates after dedup", cpp_dups.len());

    // Compare sorted results
    let mut sorted_rust = rust_results.clone();
    let mut sorted_cpp = cpp_results.clone();
    sorted_rust.sort_unstable();
    sorted_cpp.sort_unstable();

    assert_eq!(sorted_rust, sorted_cpp,
        "Sorted results differ: Rust has {} elements, C++ has {}",
        sorted_rust.len(), sorted_cpp.len());
}