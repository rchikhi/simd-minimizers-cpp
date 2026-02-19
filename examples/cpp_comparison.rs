/// Benchmark comparing Rust vs C++ implementations for canonical minimizers.
///
/// Run with:
/// RUSTFLAGS="-C target-cpu=native" cargo run --release --example cpp_comparison

use std::time::{Duration, Instant};
use packed_seq::{PackedSeqVec, SeqVec};
use simd_minimizers::cpp::{
    cpp_benchmark_packed_seq_simd, cpp_benchmark_nthash_packed_seq,
    cpp_benchmark_noncanonical,
    cpp_benchmark_canonical, cpp_benchmark_canonical_phases,
    cpp_canonical_minimizer_positions, cpp_noncanonical_minimizer_positions,
    cpp_benchmark_syncmers,
    cpp_benchmark_dedup_simd, cpp_benchmark_dedup_scalar,
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

fn get_cpu_info() -> (String, String) {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = cpuinfo.lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let mhz = cpuinfo.lines()
        .find(|l| l.starts_with("cpu MHz"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| format!("{:.2} GHz", s.trim().parse::<f64>().unwrap_or(0.0) / 1000.0))
        .unwrap_or_else(|| "unknown".to_string());
    (model, mhz)
}

fn main() {
    let seq_len = 1_000_000;
    let k = 21;
    let w = 11;
    let iterations = 20;

    let (cpu_model, cpu_ghz) = get_cpu_info();
    println!("C++ compiler: {}", env!("CPP_COMPILER"));
    println!("CPU: {} @ {}", cpu_model, cpu_ghz);
    println!("Benchmark: seq_len={}, k={}, w={}", seq_len, k, w);

    // Generate test data
    let seq_data = generate_random_dna(seq_len);
    let packed_seq = PackedSeqVec::from_ascii(&seq_data);

    // Full Rust SIMD pipeline (canonical)
    let rust_full_simd_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::canonical_minimizers(k, w).run(packed_seq.as_slice(), &mut out);
    }, iterations);

    // Rust SIMD pipeline (non-canonical, packing excluded)
    let rust_noncanonical_prepacked_time = measure_time(|| {
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

    // C++ packed_seq SIMD iteration
    let cpp_packed_seq_us = cpp_benchmark_packed_seq_simd(&seq_data, k, iterations);
    let cpp_packed_seq_time = Duration::from_micros(cpp_packed_seq_us / iterations as u64);

    // C++ ntHash using packed_seq SIMD
    let cpp_nthash_packed_us = cpp_benchmark_nthash_packed_seq(&seq_data, k, iterations);
    let cpp_nthash_packed_time = Duration::from_micros(cpp_nthash_packed_us / iterations as u64);

    // C++ non-canonical pipeline (packing excluded from timing)
    let cpp_noncanonical_prepacked_us = cpp_benchmark_noncanonical(&seq_data, k, w, iterations);
    let cpp_noncanonical_prepacked_time = Duration::from_micros(cpp_noncanonical_prepacked_us / iterations as u64);

    // C++ canonical pipeline (no FFI result handling - just timing the core algorithm)
    let cpp_canonical_direct_us = cpp_benchmark_canonical(&seq_data, k, w, iterations);
    let cpp_canonical_direct_time = Duration::from_micros(cpp_canonical_direct_us / iterations as u64);

    let mb = seq_len as f64 / 1_000_000.0;

    // =========================================================================
    // Component Benchmarks (for understanding bottlenecks)
    // =========================================================================
    println!();
    println!("Component Benchmarks:");
    println!("                          | Time (ms) | MB/s");
    println!("--------------------------|-----------|--------");
    println!("C++ packed_seq iteration  | {:9.2} | {:7.1}",
        cpp_packed_seq_time.as_secs_f64() * 1000.0,
        mb / cpp_packed_seq_time.as_secs_f64());
    println!("C++ ntHash (packed_seq)   | {:9.2} | {:7.1}",
        cpp_nthash_packed_time.as_secs_f64() * 1000.0,
        mb / cpp_nthash_packed_time.as_secs_f64());
    println!("Rust scalar baseline      | {:9.2} | {:7.1}",
        rust_full_scalar_time.as_secs_f64() * 1000.0,
        mb / rust_full_scalar_time.as_secs_f64());

    // =========================================================================
    // End-to-End Performance (ASCII → Minimizers)
    // =========================================================================
    println!();
    println!("============================================================");
    println!("Rust vs C++ Performance Comparison");
    println!("============================================================");

    // Rust end-to-end: non-canonical (pack + algorithm)
    let rust_e2e_noncanonical = measure_time(|| {
        let packed = PackedSeqVec::from_ascii(&seq_data);
        let mut out = Vec::new();
        simd_minimizers::minimizers(k, w).run(packed.as_slice(), &mut out);
    }, iterations);

    // Rust end-to-end: canonical (pack + algorithm)
    let rust_e2e_canonical = measure_time(|| {
        let packed = PackedSeqVec::from_ascii(&seq_data);
        let mut out = Vec::new();
        simd_minimizers::canonical_minimizers(k, w).run(packed.as_slice(), &mut out);
    }, iterations);

    // C++ e2e: non-canonical (uses zero-copy FFI internally)
    let cpp_e2e_noncanonical = measure_time(|| {
        let mut out = Vec::new();
        cpp_noncanonical_minimizer_positions(&seq_data, k, w, &mut out);
    }, iterations);

    // C++ e2e: canonical already measured above as cpp_full_simd_time

    // Closed syncmer parameters: syncmer length 21, minimizer size 11
    // C++ API: k=21 (syncmer length), m=11 (minimizer size)
    // Rust API: closed_syncmers(k=11, w=11), syncmer length = k+w-1 = 21
    // Both check: minimizer at prefix or suffix of window
    let syncmer_k = 21;
    let syncmer_m = 11;
    let cpp_e2e_syncmers = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::cpp::cpp_syncmers_simd(&seq_data, syncmer_k as u32, syncmer_m as u32, &mut out);
    }, iterations);

    // Rust e2e: closed syncmers (pack + algorithm)
    // C++ syncmers check prefix or suffix, which is "closed" in Rust terminology
    let rust_e2e_syncmers = measure_time(|| {
        let packed = PackedSeqVec::from_ascii(&seq_data);
        let mut out = Vec::new();
        simd_minimizers::closed_syncmers(syncmer_m, syncmer_m).run(packed.as_slice(), &mut out);
    }, iterations);

    // Rust pre-packed: closed syncmers
    let rust_syncmers_prepacked_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::closed_syncmers(syncmer_m, syncmer_m).run(packed_seq.as_slice(), &mut out);
    }, iterations);

    // Rust e2e: canonical closed syncmers (pack + algorithm)
    let rust_e2e_canonical_syncmers = measure_time(|| {
        let packed = PackedSeqVec::from_ascii(&seq_data);
        let mut out = Vec::new();
        simd_minimizers::canonical_closed_syncmers(syncmer_m, syncmer_m).run(packed.as_slice(), &mut out);
    }, iterations);

    // Rust pre-packed: canonical closed syncmers
    let rust_canonical_syncmers_prepacked_time = measure_time(|| {
        let mut out = Vec::new();
        simd_minimizers::canonical_closed_syncmers(syncmer_m, syncmer_m).run(packed_seq.as_slice(), &mut out);
    }, iterations);

    // C++ pre-packed: syncmers (for completeness)
    let cpp_syncmers_prepacked_us = cpp_benchmark_syncmers(&seq_data, syncmer_k, syncmer_m, iterations);
    let cpp_syncmers_prepacked_time = Duration::from_micros(cpp_syncmers_prepacked_us / iterations as u64);

    println!("End-to-End (ASCII → minimizers, packing INCLUDED):");
    println!("                          | Time (ms) | MB/s   | ns/nt");
    println!("--------------------------|-----------|--------|------");
    println!("Rust non-canonical        | {:9.2} | {:6.1} | {:5.2}",
        rust_e2e_noncanonical.as_secs_f64() * 1000.0,
        mb / rust_e2e_noncanonical.as_secs_f64(),
        rust_e2e_noncanonical.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  non-canonical        | {:9.2} | {:6.1} | {:5.2}",
        cpp_e2e_noncanonical.as_secs_f64() * 1000.0,
        mb / cpp_e2e_noncanonical.as_secs_f64(),
        cpp_e2e_noncanonical.as_secs_f64() * 1e9 / seq_len as f64);
    println!("Rust canonical            | {:9.2} | {:6.1} | {:5.2}",
        rust_e2e_canonical.as_secs_f64() * 1000.0,
        mb / rust_e2e_canonical.as_secs_f64(),
        rust_e2e_canonical.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  canonical            | {:9.2} | {:6.1} | {:5.2}",
        cpp_full_simd_time.as_secs_f64() * 1000.0,
        mb / cpp_full_simd_time.as_secs_f64(),
        cpp_full_simd_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("Rust syncmers             | {:9.2} | {:6.1} | {:5.2}",
        rust_e2e_syncmers.as_secs_f64() * 1000.0,
        mb / rust_e2e_syncmers.as_secs_f64(),
        rust_e2e_syncmers.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  syncmers             | {:9.2} | {:6.1} | {:5.2}",
        cpp_e2e_syncmers.as_secs_f64() * 1000.0,
        mb / cpp_e2e_syncmers.as_secs_f64(),
        cpp_e2e_syncmers.as_secs_f64() * 1e9 / seq_len as f64);
    println!("Rust canonical syncmers   | {:9.2} | {:6.1} | {:5.2}",
        rust_e2e_canonical_syncmers.as_secs_f64() * 1000.0,
        mb / rust_e2e_canonical_syncmers.as_secs_f64(),
        rust_e2e_canonical_syncmers.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  canonical syncmers   |         - |      - |     -");

    println!();
    println!("Pre-packed (algorithm only, packing EXCLUDED):");
    println!("                          | Time (ms) | MB/s   | ns/nt");
    println!("--------------------------|-----------|--------|------");
    println!("Rust non-canonical        | {:9.2} | {:6.1} | {:5.2}",
        rust_noncanonical_prepacked_time.as_secs_f64() * 1000.0,
        mb / rust_noncanonical_prepacked_time.as_secs_f64(),
        rust_noncanonical_prepacked_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  non-canonical        | {:9.2} | {:6.1} | {:5.2}",
        cpp_noncanonical_prepacked_time.as_secs_f64() * 1000.0,
        mb / cpp_noncanonical_prepacked_time.as_secs_f64(),
        cpp_noncanonical_prepacked_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("Rust canonical            | {:9.2} | {:6.1} | {:5.2}",
        rust_full_simd_time.as_secs_f64() * 1000.0,
        mb / rust_full_simd_time.as_secs_f64(),
        rust_full_simd_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  canonical            | {:9.2} | {:6.1} | {:5.2}",
        cpp_canonical_direct_time.as_secs_f64() * 1000.0,
        mb / cpp_canonical_direct_time.as_secs_f64(),
        cpp_canonical_direct_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("Rust syncmers             | {:9.2} | {:6.1} | {:5.2}",
        rust_syncmers_prepacked_time.as_secs_f64() * 1000.0,
        mb / rust_syncmers_prepacked_time.as_secs_f64(),
        rust_syncmers_prepacked_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  syncmers             | {:9.2} | {:6.1} | {:5.2}",
        cpp_syncmers_prepacked_time.as_secs_f64() * 1000.0,
        mb / cpp_syncmers_prepacked_time.as_secs_f64(),
        cpp_syncmers_prepacked_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("Rust canonical syncmers   | {:9.2} | {:6.1} | {:5.2}",
        rust_canonical_syncmers_prepacked_time.as_secs_f64() * 1000.0,
        mb / rust_canonical_syncmers_prepacked_time.as_secs_f64(),
        rust_canonical_syncmers_prepacked_time.as_secs_f64() * 1e9 / seq_len as f64);
    println!("C++  canonical syncmers   |         - |      - |     -");

    // 3-phase breakdown (for non-regression: main loop should dominate)
    println!();
    println!("Phase Breakdown (C++ canonical):");
    let phases = cpp_benchmark_canonical_phases(&seq_data, k, w, iterations);
    let total_us = phases.init_us + phases.main_loop_us + phases.final_flatten_us;
    let pct = |x: u64| 100.0 * x as f64 / total_us as f64;
    println!("  Init:       {:5} us ({:4.1}%)", phases.init_us / iterations as u64, pct(phases.init_us));
    println!("  Main loop:  {:5} us ({:4.1}%)", phases.main_loop_us / iterations as u64, pct(phases.main_loop_us));
    println!("  Collect:    {:5} us ({:4.1}%)", phases.final_flatten_us / iterations as u64, pct(phases.final_flatten_us));

    // Dedup comparison (SIMD Lemire vs scalar)
    println!();
    println!("Dedup Strategy Comparison (C++):");
    let dedup_iterations = 1000;
    let dedup_simd_us = cpp_benchmark_dedup_simd(dedup_iterations);
    let dedup_scalar_us = cpp_benchmark_dedup_scalar(dedup_iterations);
    println!("  SIMD Lemire: {:6} us / {} iters = {:5.2} us/iter",
        dedup_simd_us, dedup_iterations, dedup_simd_us as f64 / dedup_iterations as f64);
    println!("  Scalar:      {:6} us / {} iters = {:5.2} us/iter",
        dedup_scalar_us, dedup_iterations, dedup_scalar_us as f64 / dedup_iterations as f64);
    let speedup = dedup_simd_us as f64 / dedup_scalar_us as f64;
    if speedup > 1.0 {
        println!("  => Scalar is {:.1}x faster", speedup);
    } else {
        println!("  => SIMD is {:.1}x faster", 1.0 / speedup);
    }
}
