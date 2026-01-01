# simd-minimizers

[![crates.io](https://img.shields.io/crates/v/simd-minimizers)](https://crates.io/crates/simd-minimizers)
[![docs](https://img.shields.io/docsrs/simd-minimizers)](https://docs.rs/simd-minimizers)

A SIMD-accelerated library to compute random minimizers.

It can compute all the minimizers of a human genome in 4 seconds using a single thread.
It also provides a *canonical* version that ensures that a sequence and its reverse-complement always select the same positions, which takes 6 seconds on a human genome.

The underlying algorithm is described in the following [preprint](https://doi.org/10.1101/2025.01.27.634998):

-   SimdMinimizers: Computing random minimizers, fast.
    Ragnar Groot Koerkamp, Igor Martayan
    bioRxiv 2025.01.27 [doi.org/10.1101/2025.01.27.634998](https://doi.org/10.1101/2025.01.27.634998)


## Requirements

This library supports AVX2 and NEON instruction sets.
Make sure to set `RUSTFLAGS="-C target-cpu=native"` when compiling to use the instruction sets available on your architecture.

    RUSTFLAGS="-C target-cpu=native" cargo run --release


## Implementations

This library provides two implementations of the algorithm:

1. **Rust Implementation**: The primary implementation in Rust using SIMD intrinsics through the `wide` crate.
2. **C++ Implementation**: An alternative implementation in C++ using AVX2 intrinsics directly, which may provide better performance in some cases.

Both implementations should produce identical results, but performance may vary depending on your specific use case and compiler optimizations.


## Usage example

Full documentation can be found on [docs.rs](https://docs.rs/simd-minimizers).

### Rust Implementation

```rust
// Packed SIMD version with Rust implementation
use packed_seq::{complement_char, PackedSeqVec, SeqVec};
let seq = b"ACGTGCTCAGAGACTCAG";
let k = 5;
let w = 7;

let packed_seq = PackedSeqVec::from_ascii(seq);
let mut minimizer_positions = Vec::new();
simd_minimizers::canonical_minimizer_positions(packed_seq.as_slice(), k, w, &mut minimizer_positions);
assert_eq!(minimizer_positions, vec![3, 5, 12]);
```

### C++ Implementation

```rust
// Using the C++ implementation directly
let seq = b"ACGTGCTCAGAGACTCAG";
let k = 5;
let w = 7;

let mut minimizer_positions = Vec::new();
simd_minimizers::cpp::canonical_minimizer_positions(seq, k, w, &mut minimizer_positions);
assert_eq!(minimizer_positions, vec![3, 5, 12]);
```

## Benchmarking

You can compare the performance of both implementations using the provided benchmark:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example cpp_comparison
```

The C++ implementation may be faster in some cases due to better SIMD compiler optimizations.
