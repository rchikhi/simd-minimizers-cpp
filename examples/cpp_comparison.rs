/// Benchmark comparing Rust vs C++ implementations for canonical minimizers.
///
/// Run with:
/// RUSTFLAGS="-C target-cpu=native" cargo run --release --example cpp_comparison

use std::time::{Duration, Instant};
use packed_seq::{PackedSeqVec, SeqVec};
use seq_hash::{KmerHasher, NtHasher};
use simd_minimizers::cpp::{
    cpp_benchmark_nthash_simd, cpp_benchmark_sliding_min_scalar,
    cpp_benchmark_sliding_min_simd, cpp_benchmark_packed_seq_simd,
    cpp_benchmark_nthash_packed_seq, cpp_benchmark_fused_pipeline,
    cpp_benchmark_noncanonical_full, cpp_benchmark_canonical_full_direct,
    cpp_benchmark_canonical_phases, cpp_canonical_minimizer_positions,
};

fn generate_random_dna(len: usize) -> Vec<u8> {
    let bases = b"ACGT";
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
            bases[(x >> 16) as usize % 4]
        })
        .collect()
}

fn measure_time<F: FnMut()>(mut f: F, iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations as u32
}

fn main() {
    let seq_len = 1_000_000;
    let k = 21;
    let w = 11;
    let iterations = 10;

    println!("Step-by-step benchmark: seq_len={}, k={}, w={}", seq_len, k, w);
    println!("============================================================");

    // Generate test data
    let seq_data = generate_random_dna(seq_len);
    let packed_seq = PackedSeqVec::from_ascii(&seq_data);

    // Full Rust SIMD pipeline (canonical)
    let rust_full_simd_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::canonical_minimizers(k, w).run(packed_seq.as_slice(), &mut out);
    }, iterations);

    // Full Rust SIMD pipeline (non-canonical)
    let rust_noncanonical_full_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::minimizers(k, w).run(packed_seq.as_slice(), &mut out);
    }, iterations);

    // Full Rust scalar pipeline (canonical)
    let rust_full_scalar_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::canonical_minimizers(k, w).run_scalar(packed_seq.as_slice(), &mut out);
    }, iterations);

    // Full C++ SIMD pipeline (via FFI)
    let cpp_full_simd_time = measure_time(|| {
        let mut out = Vec::new();
        cpp_canonical_minimizer_positions(&seq_data, k, w, &mut out);
    }, iterations);

    // Convert ASCII to packed for C++ SIMD ntHash benchmark
    let packed_for_cpp: Vec<u8> = seq_data.iter().map(|&c| match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'T' | b't' => 2,
        b'G' | b'g' => 3,
        _ => 0,
    }).collect();

    // C++ SIMD ntHash
    let cpp_simd_nthash_us = cpp_benchmark_nthash_simd(&packed_for_cpp, k, iterations);
    let cpp_simd_nthash_time = Duration::from_micros(cpp_simd_nthash_us / iterations as u64);

    // Generate hashes for sliding min benchmark (using Rust hasher)
    let hasher = NtHasher::<false>::new(k);
    let hashes_for_sliding: Vec<u32> = hasher.hash_kmers_scalar(packed_seq.as_slice()).collect();

    // C++ sliding window minimum (scalar)
    let cpp_sliding_min_us = cpp_benchmark_sliding_min_scalar(&hashes_for_sliding, w, iterations);
    let cpp_sliding_min_time = Duration::from_micros(cpp_sliding_min_us / iterations as u64);

    // C++ sliding window minimum (SIMD)
    let cpp_sliding_min_simd_us = cpp_benchmark_sliding_min_simd(&hashes_for_sliding, w, iterations);
    let cpp_sliding_min_simd_time = Duration::from_micros(cpp_sliding_min_simd_us / iterations as u64);

    // C++ packed_seq SIMD iteration
    let cpp_packed_seq_us = cpp_benchmark_packed_seq_simd(&seq_data, k, iterations);
    let cpp_packed_seq_time = Duration::from_micros(cpp_packed_seq_us / iterations as u64);

    // C++ ntHash using packed_seq SIMD
    let cpp_nthash_packed_us = cpp_benchmark_nthash_packed_seq(&seq_data, k, iterations);
    let cpp_nthash_packed_time = Duration::from_micros(cpp_nthash_packed_us / iterations as u64);

    // C++ fused pipeline: ntHash + streaming sliding min
    let cpp_fused_us = cpp_benchmark_fused_pipeline(&seq_data, k, w, iterations);
    let cpp_fused_time = Duration::from_micros(cpp_fused_us / iterations as u64);

    // C++ non-canonical FULL pipeline (for comparison with canonical)
    let cpp_noncanonical_full_us = cpp_benchmark_noncanonical_full(&seq_data, k, w, iterations);
    let cpp_noncanonical_full_time = Duration::from_micros(cpp_noncanonical_full_us / iterations as u64);

    // C++ canonical FULL pipeline DIRECT (no FFI result handling - just timing the core algorithm)
    let cpp_canonical_direct_us = cpp_benchmark_canonical_full_direct(&seq_data, k, w, iterations);
    let cpp_canonical_direct_time = Duration::from_micros(cpp_canonical_direct_us / iterations as u64);

    let mb = seq_len as f64 / 1_000_000.0;

    println!();
    println!("Step                      | Time (ms) | MB/s");
    println!("--------------------------|-----------|--------");
    println!("C++ SIMD ntHash           | {:9.2} | {:7.1}",
        cpp_simd_nthash_time.as_secs_f64() * 1000.0,
        mb / cpp_simd_nthash_time.as_secs_f64());
    println!("C++ packed_seq iteration  | {:9.2} | {:7.1}",
        cpp_packed_seq_time.as_secs_f64() * 1000.0,
        mb / cpp_packed_seq_time.as_secs_f64());
    println!("C++ ntHash (packed_seq)   | {:9.2} | {:7.1}",
        cpp_nthash_packed_time.as_secs_f64() * 1000.0,
        mb / cpp_nthash_packed_time.as_secs_f64());
    println!("C++ sliding min (scalar)  | {:9.2} | {:7.1}",
        cpp_sliding_min_time.as_secs_f64() * 1000.0,
        mb / cpp_sliding_min_time.as_secs_f64());
    println!("C++ sliding min (SIMD)    | {:9.2} | {:7.1}",
        cpp_sliding_min_simd_time.as_secs_f64() * 1000.0,
        mb / cpp_sliding_min_simd_time.as_secs_f64());
    println!("C++ fused (hash+slidmin)  | {:9.2} | {:7.1}",
        cpp_fused_time.as_secs_f64() * 1000.0,
        mb / cpp_fused_time.as_secs_f64());
    println!("--------------------------|-----------|--------");
    println!("Rust FULL canonical       | {:9.2} | {:7.1}",
        rust_full_simd_time.as_secs_f64() * 1000.0,
        mb / rust_full_simd_time.as_secs_f64());
    println!("Rust FULL non-canonical   | {:9.2} | {:7.1}",
        rust_noncanonical_full_time.as_secs_f64() * 1000.0,
        mb / rust_noncanonical_full_time.as_secs_f64());
    println!("Rust FULL (scalar)        | {:9.2} | {:7.1}",
        rust_full_scalar_time.as_secs_f64() * 1000.0,
        mb / rust_full_scalar_time.as_secs_f64());
    println!("C++ FULL canonical (FFI)  | {:9.2} | {:7.1}",
        cpp_full_simd_time.as_secs_f64() * 1000.0,
        mb / cpp_full_simd_time.as_secs_f64());
    println!("C++ FULL canonical (dir)  | {:9.2} | {:7.1}",
        cpp_canonical_direct_time.as_secs_f64() * 1000.0,
        mb / cpp_canonical_direct_time.as_secs_f64());
    println!("C++ FULL non-canonical    | {:9.2} | {:7.1}",
        cpp_noncanonical_full_time.as_secs_f64() * 1000.0,
        mb / cpp_noncanonical_full_time.as_secs_f64());

    println!();
    println!("FFI overhead analysis:");
    println!("  C++ canonical direct vs FFI:        {:.2}x",
        cpp_full_simd_time.as_secs_f64() / cpp_canonical_direct_time.as_secs_f64());

    println!();
    println!("Speedups:");
    println!("  Rust canonical vs Rust scalar:      {:.2}x",
        rust_full_scalar_time.as_secs_f64() / rust_full_simd_time.as_secs_f64());
    println!("  Rust canonical vs non-canonical:    {:.2}x slower",
        rust_full_simd_time.as_secs_f64() / rust_noncanonical_full_time.as_secs_f64());
    println!("  C++ canonical vs non-canonical:     {:.2}x slower",
        cpp_full_simd_time.as_secs_f64() / cpp_noncanonical_full_time.as_secs_f64());
    println!("  C++ canonical vs Rust canonical:    {:.2}x slower",
        cpp_canonical_direct_time.as_secs_f64() / rust_full_simd_time.as_secs_f64());
    println!("  C++ non-canon vs Rust non-canon:    {:.2}x slower",
        cpp_noncanonical_full_time.as_secs_f64() / rust_noncanonical_full_time.as_secs_f64());

    // Phase breakdown for canonical pipeline
    println!();
    println!("============================================================");
    println!("C++ Canonical Pipeline Phase Breakdown:");
    println!("============================================================");

    let phases = cpp_benchmark_canonical_phases(&seq_data, k, w, iterations);
    let total_phase_us = phases.init_us + phases.main_loop_us + phases.final_flatten_us;

    println!();
    println!("Phase breakdown (canonical pipeline, per iteration):");
    println!("  Init:                                       {:>6} us ({:.1}%)",
             phases.init_us / iterations as u64,
             100.0 * phases.init_us as f64 / total_phase_us as f64);
    println!("  Main loop (hash+slidmin+inline collection): {:>6} us ({:.1}%)",
             phases.main_loop_us / iterations as u64,
             100.0 * phases.main_loop_us as f64 / total_phase_us as f64);
    println!("  Final flatten (copy lanes to output):       {:>6} us ({:.1}%)",
             phases.final_flatten_us / iterations as u64,
             100.0 * phases.final_flatten_us as f64 / total_phase_us as f64);
}
