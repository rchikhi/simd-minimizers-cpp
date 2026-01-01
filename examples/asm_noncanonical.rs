/// Minimal example for extracting assembly of non-canonical minimizers.
///
/// Compile with:
///   RUSTFLAGS="-C target-cpu=native --emit asm" cargo build --release --example asm_noncanonical
///
/// Then find the assembly at:
///   target/release/examples/asm_noncanonical-*.s
///
/// Search for `rust_noncanonical:` in the assembly file to find the function.

use packed_seq::{PackedSeqVec, SeqVec};

#[inline(never)]
#[unsafe(no_mangle)]
pub fn rust_noncanonical(seq: packed_seq::PackedSeq, k: usize, w: usize, out: &mut Vec<u32>) {
    simd_minimizers::minimizers(k, w).run(seq, out);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn rust_canonical(seq: packed_seq::PackedSeq, k: usize, w: usize, out: &mut Vec<u32>) {
    simd_minimizers::canonical_minimizers(k, w).run(seq, out);
}

fn main() {
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let packed = PackedSeqVec::from_ascii(seq);

    let mut out = Vec::new();
    rust_noncanonical(packed.as_slice(), 21, 11, &mut out);
    println!("Non-canonical: {:?}", out);

    out.clear();
    rust_canonical(packed.as_slice(), 21, 11, &mut out);
    println!("Canonical: {:?}", out);
}
