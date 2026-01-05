/// Benchmark sequence packing: Rust PEXT vs C++ scalar
///
/// Run with: cargo run --release --example packing_benchmark

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
    let seq_len = 10_000_000; // 10 MB
    let iterations = 20;

    println!("Packing Benchmark");
    println!("=================");
    println!("Sequence length: {} ({} MB)", seq_len, seq_len / 1_000_000);
    println!("Iterations: {}", iterations);
    println!();

    // Generate test data
    let seq_data = generate_random_dna(seq_len);

    // Benchmark Rust packing (uses PEXT on x86)
    {
        let start = Instant::now();
        for _ in 0..iterations {
            let packed = PackedSeqVec::from_ascii(&seq_data);
            std::hint::black_box(&packed);
        }
        let elapsed = start.elapsed();

        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let mb = seq_len as f64 / 1_000_000.0;
        let throughput = mb / (ms_per_iter / 1000.0);

        println!("Rust Packing (PEXT):");
        println!("  Time: {:.2} ms/iter", ms_per_iter);
        println!("  Throughput: {:.0} MB/s", throughput);
        println!();
    }

    // Benchmark C++ packing (scalar loop)
    {
        let start = Instant::now();
        for _ in 0..iterations {
            // cpp_pack_sequence returns packed bytes
            let packed = simd_minimizers::cpp::cpp_pack_sequence(&seq_data);
            std::hint::black_box(&packed);
        }
        let elapsed = start.elapsed();

        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let mb = seq_len as f64 / 1_000_000.0;
        let throughput = mb / (ms_per_iter / 1000.0);

        println!("C++ Packing (PEXT):");
        println!("  Time: {:.2} ms/iter", ms_per_iter);
        println!("  Throughput: {:.0} MB/s", throughput);
        println!();
    }

    // Size comparison
    println!("Parameter sweep:");
    println!("  Size (MB) | Rust (GB/s) | C++ (GB/s) | Ratio");
    println!("------------|-------------|------------|-------");

    for size_mb in [1, 5, 10, 50, 100] {
        let size = size_mb * 1_000_000;
        if size > seq_len {
            break;
        }
        let seq_slice = &seq_data[..size];

        // Rust
        let start = Instant::now();
        for _ in 0..iterations {
            let packed = PackedSeqVec::from_ascii(seq_slice);
            std::hint::black_box(&packed);
        }
        let rust_elapsed = start.elapsed();
        let rust_throughput = (size as f64 / 1e9) / (rust_elapsed.as_secs_f64() / iterations as f64);

        // C++
        let start = Instant::now();
        for _ in 0..iterations {
            let packed = simd_minimizers::cpp::cpp_pack_sequence(seq_slice);
            std::hint::black_box(&packed);
        }
        let cpp_elapsed = start.elapsed();
        let cpp_throughput = (size as f64 / 1e9) / (cpp_elapsed.as_secs_f64() / iterations as f64);

        let ratio = rust_throughput / cpp_throughput;
        println!(
            "  {:9} |    {:.2}      |    {:.2}     | {:.1}x",
            size_mb, rust_throughput, cpp_throughput, ratio
        );
    }
}
