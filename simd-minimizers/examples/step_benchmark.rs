/// Benchmark to compare the individual steps of the minimizer pipeline
/// between Rust SIMD and C++ scalar implementations.
///
/// Run with:
/// RUSTFLAGS="-C target-cpu=native" cargo run --release --example step_benchmark

use std::time::{Duration, Instant};
use packed_seq::{PackedSeq, PackedSeqVec, SeqVec};
use rand::{self, Rng};
use simd_minimizers::private::nthash::{nthash_seq_simd, nthash_seq_scalar, NtHasher};
use simd_minimizers::private::sliding_min::sliding_min_scalar;
use simd_minimizers::private::canonical::canonical_windows_seq_scalar;
use simd_minimizers::private::cpp::{cpp_benchmark_nthash_simd, cpp_benchmark_sliding_min_scalar, cpp_benchmark_sliding_min_simd, cpp_benchmark_packed_seq_simd, cpp_benchmark_nthash_packed_seq, cpp_benchmark_fused_pipeline, cpp_benchmark_noncanonical_full, cpp_benchmark_canonical_full_direct, cpp_benchmark_canonical_phases, cpp_benchmark_collection_components};

fn generate_random_dna(len: usize) -> Vec<u8> {
    let bases = b"ACGT";
    let mut rng = rand::rng();
    (0..len)
        .map(|_| bases[rng.random_range(0..4) as usize])
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
    let seq = packed_seq.as_slice();

    // Benchmark Rust SIMD ntHash
    let rust_simd_nthash_time = measure_time(|| {
        let (head, tail) = nthash_seq_simd::<true, PackedSeq, NtHasher>(seq, k, w);
        // Force evaluation by consuming iterators
        let _: Vec<_> = head.collect();
        let _: Vec<_> = tail.collect();
    }, iterations);

    // Benchmark Rust scalar ntHash
    let rust_scalar_nthash_time = measure_time(|| {
        let it = nthash_seq_scalar::<true, NtHasher>(seq, k);
        let _: Vec<_> = it.collect();
    }, iterations);

    // Benchmark Rust scalar sliding min (left)
    let rust_sliding_min_time = measure_time(|| {
        let hashes = nthash_seq_scalar::<true, NtHasher>(seq, k);
        let it = sliding_min_scalar::<true>(hashes, w);
        let _: Vec<_> = it.collect();
    }, iterations);

    // Benchmark Rust scalar canonical windows
    let rust_canonical_time = measure_time(|| {
        let it = canonical_windows_seq_scalar(seq, k, w);
        let _: Vec<_> = it.collect();
    }, iterations);

    // Full Rust SIMD pipeline
    let rust_full_simd_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::canonical_minimizer_positions(seq, k, w, &mut out);
    }, iterations);

    // Full Rust scalar pipeline
    let rust_full_scalar_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::canonical_minimizer_positions_scalar(seq, k, w, &mut out);
    }, iterations);

    // Full C++ SIMD pipeline (via FFI)
    let cpp_full_simd_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::cpp::canonical_minimizer_positions(&seq_data, k, w, &mut out);
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

    // Generate hashes for sliding min benchmark
    let hashes_for_sliding: Vec<u32> = nthash_seq_scalar::<true, NtHasher>(seq, k).collect();

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

    // Rust non-canonical FULL pipeline
    let rust_noncanonical_full_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::minimizer_positions(seq, k, w, &mut out);
    }, iterations);

    let mb = seq_len as f64 / 1_000_000.0;

    println!();
    println!("Step                      | Time (ms) | MB/s");
    println!("--------------------------|-----------|--------");
    println!("Rust SIMD ntHash          | {:9.2} | {:7.1}",
        rust_simd_nthash_time.as_secs_f64() * 1000.0,
        mb / rust_simd_nthash_time.as_secs_f64());
    println!("Rust scalar ntHash        | {:9.2} | {:7.1}",
        rust_scalar_nthash_time.as_secs_f64() * 1000.0,
        mb / rust_scalar_nthash_time.as_secs_f64());
    println!("C++ SIMD ntHash           | {:9.2} | {:7.1}",
        cpp_simd_nthash_time.as_secs_f64() * 1000.0,
        mb / cpp_simd_nthash_time.as_secs_f64());
    println!("C++ packed_seq iteration  | {:9.2} | {:7.1}",
        cpp_packed_seq_time.as_secs_f64() * 1000.0,
        mb / cpp_packed_seq_time.as_secs_f64());
    println!("C++ ntHash (packed_seq)   | {:9.2} | {:7.1}",
        cpp_nthash_packed_time.as_secs_f64() * 1000.0,
        mb / cpp_nthash_packed_time.as_secs_f64());
    println!("Rust sliding min (scalar) | {:9.2} | {:7.1}",
        rust_sliding_min_time.as_secs_f64() * 1000.0,
        mb / rust_sliding_min_time.as_secs_f64());
    println!("Rust canonical (scalar)   | {:9.2} | {:7.1}",
        rust_canonical_time.as_secs_f64() * 1000.0,
        mb / rust_canonical_time.as_secs_f64());
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
    println!("  Rust canonical vs non-canonical:    {:.2}x",
        rust_noncanonical_full_time.as_secs_f64() / rust_full_simd_time.as_secs_f64());
    println!("  C++ canonical vs non-canonical:     {:.2}x",
        cpp_noncanonical_full_time.as_secs_f64() / cpp_full_simd_time.as_secs_f64());
    println!("  C++ canonical vs Rust canonical:    {:.2}x",
        rust_full_simd_time.as_secs_f64() / cpp_full_simd_time.as_secs_f64());
    println!("  C++ non-canon vs Rust non-canon:    {:.2}x",
        rust_noncanonical_full_time.as_secs_f64() / cpp_noncanonical_full_time.as_secs_f64());

    // Detailed timing breakdown
    println!();
    println!("============================================================");
    println!("Detailed Timing Breakdown:");
    println!("============================================================");

    // Phase breakdown for canonical pipeline
    let phases = cpp_benchmark_canonical_phases(&seq_data, k, w, iterations);
    let total_phase_us = phases.init_us + phases.main_loop_us + phases.final_flatten_us;

    // Isolated component benchmarks
    let component_iterations = 1_000_000u32;
    let timing = cpp_benchmark_collection_components(component_iterations as usize);
    let transpose_ns = timing.transpose_us as f64 * 1000.0 / component_iterations as f64;
    let dedup_batch_ns = timing.dedup_batch_us as f64 * 1000.0 / component_iterations as f64;
    let collection_batch_ns = timing.collection_batch_us as f64 * 1000.0 / component_iterations as f64;

    // Measured overhead from fused vs full comparison
    let fused_us = cpp_fused_time.as_secs_f64() * 1_000_000.0;
    let full_us = cpp_noncanonical_full_time.as_secs_f64() * 1_000_000.0;
    let measured_inline_us = full_us - fused_us;

    // Estimated from isolated benchmarks
    let num_windows = seq_len - k - w + 2;
    let num_batches = num_windows / 8;
    let isolated_estimate_us = collection_batch_ns * num_batches as f64 / 1000.0;

    println!();
    println!("Phase breakdown (canonical pipeline, per iteration):");
    println!("  Main loop (hash+slidmin+inline collection): {:>6} us ({:.1}%)",
             phases.main_loop_us / iterations as u64,
             100.0 * phases.main_loop_us as f64 / total_phase_us as f64);
    println!("  Final flatten (copy lanes to output):       {:>6} us ({:.1}%)",
             phases.final_flatten_us / iterations as u64,
             100.0 * phases.final_flatten_us as f64 / total_phase_us as f64);

    println!();
    println!("Inline collection breakdown (isolated, per batch of 8 windows):");
    println!("  transpose_8x8:       {:>6.1} ns ({:.0}%)", transpose_ns, 100.0 * transpose_ns / collection_batch_ns);
    println!("  8× dedup:            {:>6.1} ns ({:.0}%)", dedup_batch_ns, 100.0 * dedup_batch_ns / collection_batch_ns);
    println!("  Full batch:          {:>6.1} ns", collection_batch_ns);

    println!();
    println!("Collection overhead for {} bases ({} batches):", seq_len, num_batches);
    println!("  Measured (FULL - fused): {:>7.0} us", measured_inline_us);
    println!("  Isolated estimate:       {:>7.0} us", isolated_estimate_us);
    println!("  Ratio: {:.2}x (cache locality benefit)", measured_inline_us / isolated_estimate_us);

    println!();
    println!("Bottleneck: 8× dedup (append_unique_vals_simd) is {:.0}% of inline collection",
             100.0 * dedup_batch_ns / collection_batch_ns);
}
