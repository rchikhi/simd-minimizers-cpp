/// Benchmark C++ SIMD syncmers
///
/// Note: Rust syncmers not benchmarked here - upstream PR20 has proper SIMD implementation.
/// This fork's naive Rust implementation (run_all + scalar filter) is ~2x slower.
///
/// Run with:
/// cargo run --release --example syncmer_benchmark

use packed_seq::{PackedSeqVec, SeqVec};
use std::time::Instant;

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
    let seq_len = 1_000_000;
    let iterations = 10;

    println!("Syncmer Benchmark (C++ SIMD)");
    println!("============================");
    println!("Sequence length: {}", seq_len);
    println!("Iterations: {}", iterations);
    println!();

    // Generate test data
    let seq_data = generate_random_dna(seq_len);
    let packed_seq = PackedSeqVec::from_ascii(&seq_data);

    // Benchmark Rust SIMD minimizers (for comparison baseline)
    {
        let m = 11;
        let w = 11;
        let mut positions: Vec<u32> = Vec::new();

        let start = Instant::now();
        for _ in 0..iterations {
            positions.clear();
            simd_minimizers::minimizers(m, w).run(packed_seq.as_slice(), &mut positions);
        }
        let elapsed = start.elapsed();

        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let mb = seq_len as f64 / 1_000_000.0;
        let throughput = mb / (ms_per_iter / 1000.0);

        println!("Rust Minimizers SIMD (m={}, w={}):", m, w);
        println!("  Time: {:.2} ms/iter", ms_per_iter);
        println!("  Throughput: {:.1} MB/s", throughput);
        println!("  Unique positions: {}", positions.len());
        println!();
    }

    // Benchmark C++ SIMD syncmers
    {
        let k = 21usize;
        let m = 11usize;

        // Use direct benchmark function (pre-packs sequence once, times only algorithm)
        let cpp_us = simd_minimizers::cpp::cpp_benchmark_syncmers(&seq_data, k, m, iterations);
        let cpp_time = std::time::Duration::from_micros(cpp_us / iterations as u64);

        let ms_per_iter = cpp_time.as_secs_f64() * 1000.0;
        let mb = seq_len as f64 / 1_000_000.0;
        let throughput = mb / cpp_time.as_secs_f64();

        // Also get count (one call to get output size)
        let mut cpp_syncmers: Vec<u32> = Vec::new();
        simd_minimizers::cpp::cpp_syncmers_simd(&seq_data, k as u32, m as u32, &mut cpp_syncmers);

        println!("C++ Syncmers SIMD (k={}, m={}):", k, m);
        println!("  Time: {:.2} ms/iter", ms_per_iter);
        println!("  Throughput: {:.1} MB/s", throughput);
        println!("  Unique syncmer positions: {}", cpp_syncmers.len());
        println!();
    }

    // Parameter sweep (C++ only)
    println!("Parameter sweep (C++ SIMD):");
    println!("  k     m     w   | C++ MB/s | C++ syncmers");
    println!("------------------|----------|-------------");

    let test_params: [(usize, usize); 4] = [
        (15, 7),
        (21, 11),
        (31, 15),
        (31, 21),
    ];

    for (test_k, test_m) in test_params {
        let test_w = test_k - test_m + 1;
        let mut cpp_syncmers: Vec<u32> = Vec::new();

        // C++ benchmark (use direct function that pre-packs sequence)
        let cpp_us = simd_minimizers::cpp::cpp_benchmark_syncmers(&seq_data, test_k, test_m, iterations);
        let cpp_time = std::time::Duration::from_micros(cpp_us / iterations as u64);
        let cpp_throughput = (seq_len as f64 / 1_000_000.0) / cpp_time.as_secs_f64();

        // Get C++ count (one call)
        simd_minimizers::cpp::cpp_syncmers_simd(&seq_data, test_k as u32, test_m as u32, &mut cpp_syncmers);

        println!(
            "  {:2}   {:2}   {:2}   |  {:5.1}   |   {:10}",
            test_k, test_m, test_w, cpp_throughput, cpp_syncmers.len()
        );
    }
}
