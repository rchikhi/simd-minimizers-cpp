/// Profile C++ collection phases to identify bottlenecks.
///
/// Run with:
/// RUSTFLAGS="-C target-cpu=native" cargo run --release --example profile_collection

use rand::{self, Rng};
use simd_minimizers::private::cpp::{cpp_profile_collection_phases, CollectionPhaseTiming};

fn generate_random_dna(len: usize) -> Vec<u8> {
    let bases = b"ACGT";
    let mut rng = rand::rng();
    (0..len)
        .map(|_| bases[rng.random_range(0..4) as usize])
        .collect()
}

fn main() {
    let seq_len = 1_000_000;
    let k = 21;
    let w = 11;
    let iterations = 10;

    println!("C++ Collection Phase Profiling: seq_len={}, k={}, w={}, iterations={}",
             seq_len, k, w, iterations);
    println!("================================================================");

    // Generate test data
    let seq_data = generate_random_dna(seq_len);

    // Profile collection phases
    let timing = cpp_profile_collection_phases(&seq_data, k, w, iterations);

    let total = timing.main_loop_us + timing.partial_batch_us + timing.truncate_us
              + timing.prereserve_us + timing.flatten_us + timing.tail_us;

    let per_iter_total = total / iterations as u64;
    let mb = seq_len as f64 / 1_000_000.0;
    let overall_mbs = mb / (per_iter_total as f64 / 1_000_000.0);

    println!();
    println!("Phase                  | Total (us) | Per-iter (us) |  % of total | MB/s");
    println!("-----------------------|------------|---------------|-------------|----------");

    let print_phase = |name: &str, us: u64| {
        let per_iter = us / iterations as u64;
        let pct = 100.0 * us as f64 / total as f64;
        let mbs = mb / (per_iter as f64 / 1_000_000.0);
        println!("{:22} | {:>10} | {:>13} | {:>10.1}% | {:>8.1}",
                 name, us, per_iter, pct, mbs);
    };

    print_phase("Main loop (hash+dedup)", timing.main_loop_us);
    print_phase("Partial batch", timing.partial_batch_us);
    print_phase("Truncate", timing.truncate_us);
    print_phase("Pre-reserve", timing.prereserve_us);
    print_phase("Flatten", timing.flatten_us);
    print_phase("Tail processing", timing.tail_us);
    println!("-----------------------|------------|---------------|-------------|----------");
    println!("{:22} | {:>10} | {:>13} | {:>10}% | {:>8.1}",
             "TOTAL", total, per_iter_total, "100.0", overall_mbs);

    println!();
    println!("Observations:");

    // Identify bottlenecks
    let phases = [
        ("Main loop", timing.main_loop_us),
        ("Partial batch", timing.partial_batch_us),
        ("Truncate", timing.truncate_us),
        ("Pre-reserve", timing.prereserve_us),
        ("Flatten", timing.flatten_us),
        ("Tail processing", timing.tail_us),
    ];

    let max_phase = phases.iter().max_by_key(|x| x.1).unwrap();
    let max_pct = 100.0 * max_phase.1 as f64 / total as f64;

    println!("  - Biggest bottleneck: {} ({:.1}% of total)", max_phase.0, max_pct);

    // Calculate expected throughput if we only had main loop
    let main_loop_mbs = mb / (timing.main_loop_us as f64 / iterations as f64 / 1_000_000.0);
    let collection_overhead = timing.partial_batch_us + timing.truncate_us
                            + timing.prereserve_us + timing.flatten_us + timing.tail_us;
    let collection_mbs = mb / (collection_overhead as f64 / iterations as f64 / 1_000_000.0);

    println!("  - Main loop alone: {:.1} MB/s", main_loop_mbs);
    println!("  - Collection overhead: {:.1} MB/s", collection_mbs);
    println!("  - Collection accounts for {:.1}% of time",
             100.0 * collection_overhead as f64 / total as f64);
}
