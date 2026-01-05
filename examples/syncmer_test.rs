/// Test syncmers: compare Rust SIMD vs C++ SIMD
///
/// Run with:
/// cargo run --release --example syncmer_test

use packed_seq::{PackedSeqVec, Seq, SeqVec};

fn generate_random_dna(len: usize) -> Vec<u8> {
    let bases = b"ACGT";
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
            bases[(x >> 16) as usize % 4]
        })
        .collect()
}

fn main() {
    let seq_len = 1_000_000;  // 1M for real test
    let k = 21; // syncmer k-mer size
    let m = 11; // minimizer size (s-mer)
    let w = k - m + 1; // window size = 11

    println!("Syncmer test");
    println!("============");
    println!("Sequence length: {}", seq_len);
    println!("k={}, m={}, w={}", k, m, w);
    println!();

    // Generate test data
    let seq_data = generate_random_dna(seq_len);
    let packed_seq = PackedSeqVec::from_ascii(&seq_data);

    // Print first 25 bases
    print!("First 25 bases: ");
    for i in 0..25 {
        print!("{}", packed_seq.as_slice().get(i));
    }
    println!();

    // Step 1: Get non-dedup minimizer positions using the public run_all API
    let all_positions = simd_minimizers::minimizers(m, w).run_all_once(packed_seq.as_slice());

    println!("\nFirst 15 minimizer positions (non-dedup): {:?}", &all_positions[..all_positions.len().min(15)]);

    // Step 2: Filter for syncmers - k-mers where minimizer is at prefix or suffix
    let num_windows = seq_len - k + 1;
    let mut syncmers: Vec<u32> = Vec::new();
    let mut prev = u32::MAX;

    for (window_idx, &min_pos) in all_positions.iter().enumerate() {
        if window_idx >= num_windows {
            break;
        }

        // Check syncmer condition: prefix or suffix
        let is_prefix = min_pos as usize == window_idx;
        let is_suffix = min_pos as usize == window_idx + w - 1;

        if (is_prefix || is_suffix) && min_pos != prev {
            syncmers.push(min_pos);
            prev = min_pos;
        }
    }

    // Also get dedup'd minimizers for comparison
    let mut min_positions_dedup: Vec<u32> = Vec::new();
    simd_minimizers::minimizers(m, w).run(packed_seq.as_slice(), &mut min_positions_dedup);

    println!("\nMinimizer positions (Rust SIMD, dedup'd): {:?}", &min_positions_dedup[..min_positions_dedup.len().min(15)]);

    println!("\nRust syncmers (SIMD non-dedup) ({}):", syncmers.len());
    if syncmers.len() <= 20 {
        print!("  ");
        for p in &syncmers {
            print!("{} ", p);
        }
        println!();
    } else {
        println!("  First 10: {:?}", &syncmers[..10]);
    }

    // Step 3: Call C++ to get its results
    println!("\n--- C++ comparison ---");
    let mut cpp_syncmers: Vec<u32> = Vec::new();
    simd_minimizers::cpp::cpp_syncmers_simd(&seq_data, k as u32, m as u32, &mut cpp_syncmers);

    println!("C++ syncmers ({}):", cpp_syncmers.len());
    cpp_syncmers.sort();
    if cpp_syncmers.len() <= 20 {
        print!("  ");
        for p in &cpp_syncmers {
            print!("{} ", p);
        }
        println!();
    } else {
        println!("  First 10: {:?}", &cpp_syncmers[..10]);
    }

    // Compare
    syncmers.sort();
    let match_result = syncmers == cpp_syncmers;
    println!("\nMatch: {}", if match_result { "YES" } else { "NO" });

    if !match_result {
        println!("\nDifferences:");
        let mut si = 0usize;
        let mut ci = 0usize;
        let mut shown = 0;
        while shown < 10 && (si < syncmers.len() || ci < cpp_syncmers.len()) {
            if si >= syncmers.len() {
                println!("  Extra in C++: {}", cpp_syncmers[ci]);
                ci += 1;
                shown += 1;
            } else if ci >= cpp_syncmers.len() {
                println!("  Missing in C++: {}", syncmers[si]);
                si += 1;
                shown += 1;
            } else if syncmers[si] < cpp_syncmers[ci] {
                println!("  Missing in C++: {}", syncmers[si]);
                si += 1;
                shown += 1;
            } else if cpp_syncmers[ci] < syncmers[si] {
                println!("  Extra in C++: {}", cpp_syncmers[ci]);
                ci += 1;
                shown += 1;
            } else {
                si += 1;
                ci += 1;
            }
        }
    }
}
