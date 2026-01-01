/// Benchmark to compare the performance of the Rust and C++ implementations 
/// for computing canonical minimizers.
/// 
/// Run with:
/// RUSTFLAGS="-C target-cpu=native" cargo run --release --example cpp_comparison

use std::time::{Duration, Instant};
use packed_seq::{PackedSeqVec, SeqVec};
use rand::{self, Rng};

// Generate a random DNA sequence of the given length
fn generate_random_dna(len: usize) -> Vec<u8> {
    let bases = b"ACGT";
    let mut rng = rand::rng();
    (0..len)
        .map(|_| bases[rng.random_range(0..4) as usize])
        .collect()
}

// Measure the execution time of a function
fn measure_time<F: FnMut()>(mut f: F) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

fn main() {
    // Parameters to test
    let sequence_lengths = [10_000, 100_000, 1_000_000];
    let k_values = [5, 11, 21, 31];
    let w_values = [5, 11, 21, 31];
    
    println!("Benchmarking canonical minimizers computation - Rust vs C++ implementation");
    println!("====================================================================================");
    println!("Sequence Length | k | w | Rust Time (ms) | Rust MB/s | C++ Time (ms) | C++ MB/s | Speedup");
    println!("---------------|---|---|----------------|-----------|---------------|----------|--------");
    
    for &seq_len in &sequence_lengths {
        // Generate a random DNA sequence
        let seq_data = generate_random_dna(seq_len);
        let packed_seq = PackedSeqVec::from_ascii(&seq_data);
        
        for &k in &k_values {
            for &w in &w_values {
                // Skip cases where k+w-1 is even (not canonical)
                if (k + w - 1) % 2 == 0 {
                    continue;
                }
                
                // Allocate output vectors
                let mut rust_out = Vec::new();
                let mut cpp_out = Vec::new();
                
                // Measure Rust implementation
                let rust_time = measure_time(|| {
                    rust_out.clear();
                    simd_minimizers::canonical_minimizer_positions(
                        packed_seq.as_slice(), k, w, &mut rust_out);
                });
                
                // Calculate Rust throughput in MB/s
                let rust_throughput = (seq_len as f64 / 1_000_000.0) / rust_time.as_secs_f64();
                
                // Measure C++ implementation
                let cpp_time = measure_time(|| {
                    cpp_out.clear();
                    simd_minimizers::cpp::canonical_minimizer_positions(
                        &seq_data, k, w, &mut cpp_out);
                });
                
                // Calculate C++ throughput in MB/s
                let cpp_throughput = (seq_len as f64 / 1_000_000.0) / cpp_time.as_secs_f64();
                
                // Calculate speedup
                let speedup = rust_time.as_secs_f64() / cpp_time.as_secs_f64();
                
                // Format and print results
                println!("{:15} | {:2} | {:2} | {:14.2} | {:9.2} | {:13.2} | {:8.2} | {:7.2}x", 
                    seq_len, k, w, 
                    rust_time.as_millis(),
                    rust_throughput,
                    cpp_time.as_millis(),
                    cpp_throughput,
                    speedup);
                
                // Verify results match
                rust_out.sort();
                cpp_out.sort();
                assert_eq!(rust_out, cpp_out, "Results don't match for seq_len={}, k={}, w={}", 
                    seq_len, k, w);
            }
        }
    }
}