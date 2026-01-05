# CLAUDE.md

## Overview

SIMD-accelerated library for computing random minimizers of DNA sequences and general text. Computes all minimizers of a human genome in 4 seconds (canonical: 6s) single-threaded.

Includes both Rust (primary) and C++ implementations with similar performance when C++ uses Clang.

## Build

RUSTFLAGS automatically set via `.cargo/config.toml` - no manual flags needed.

```bash
cargo build --release
cargo test --package simd-minimizers
cargo run --release --example cpp_comparison
cargo run --release --example syncmer_benchmark
cargo run --release --example packing_benchmark   # Compare Rust vs C++ packing
cargo run --release --example variance            # Measure sampling variance
cargo run --release --example bench -- -a -b -c -d  # -a/-b fwd simd/scalar, -c/-d canonical simd/scalar
./scripts/bench_stats.sh [N]                      # Run cpp_comparison N times (default 50), compute avg/min/max
```

For C++ standalone binaries: `make`

## Workspace Structure

- Root (`/`) - Main library crate (simd-minimizers)
- `bench/` - Criterion benchmarking suite (`cd bench && cargo bench`)

## Architecture

Splits sequences into **8 chunks** processed in parallel via SIMD (AVX2/NEON):

1. **Split input** into 8 chunks (lib.rs)
2. **Rolling hash** using ntHash for DNA, mulHash for ASCII (auto-selected)
3. **Sliding window min** on top 16 bits, ties broken by position (sliding_min.rs)
4. **Dedup** consecutive equal positions (collect.rs)
5. **Collect** from all 8 lanes into single output vector

### Key Modules

- `lib.rs` - Public API and Builder pattern
- `minimizers.rs` / `canonical.rs` - Core minimizer logic
- `sliding_min.rs` - Two-stack sliding window minimum
- `collect.rs` - Gathering results from 8 SIMD lanes
- `cpp_bindings.rs` - FFI bindings to C++ implementation

### Input Types (via packed-seq crate)

- `PackedSeq` / `PackedSeqVec` - 2-bit packed DNA (most efficient)
- `AsciiSeq` / `AsciiSeqVec` - Wrapper for `ACTGactg` bytes
- `&[u8]` - General ASCII text

## API

```rust
// Deduplicated minimizer positions
minimizers(k, w).run(seq, &mut positions)

// All positions (one per window) - useful for syncmers
minimizers(k, w).run_all(seq, &mut positions)

// Canonical minimizers (same positions for seq and reverse-complement)
canonical_minimizers(k, w).run(seq, &mut positions)
```

Reuse output vectors between calls for best performance.

## C++ Integration

C++ implementation in `src/canonical_minimizers_simd.cpp` compiled via `build.rs` using Clang (auto-detected, use `CC_GCC=1` to force GCC).

Performance: C++ ~461 MB/s vs Rust ~439 MB/s (non-canonical), roughly equal for canonical.

## Testing

```bash
cargo test --package simd-minimizers  # Main library tests
cargo test                            # Full workspace
```

Tests compare SIMD vs scalar implementations and verify canonical minimizer round-trips.

## Notes

See `.claude/` for detailed notes:
- `architecture_notes.md` - Hash functions, canonical minimizers, full module list, performance tips
- `syncmer_implementation_notes.md` - Syncmer API, usage examples, benchmarks
- `cpp_optimization_notes.md` - C++ optimization history
- `pgo_notes.md` - Profile-guided optimization

## Publication

SimdMinimizers: Computing random minimizers, fast.
Ragnar Groot Koerkamp, Igor Martayan
[doi.org/10.1101/2025.01.27.634998](https://doi.org/10.1101/2025.01.27.634998)
