/// Minimal example for extracting assembly of non-canonical minimizers.
/// Compile with: RUSTFLAGS="-C target-cpu=native --emit asm" cargo build --release --example asm_noncanonical
use packed_seq::{PackedSeqVec, SeqVec};
use simd_minimizers::minimizer_positions;

#[inline(never)]
#[no_mangle]
pub fn rust_noncanonical(seq: packed_seq::PackedSeq, k: usize, w: usize, out: &mut Vec<u32>) {
    minimizer_positions(seq, k, w, out);
}

fn main() {
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let packed = PackedSeqVec::from_ascii(seq);
    let mut out = Vec::new();
    rust_noncanonical(packed.as_slice(), 21, 11, &mut out);
    println!("{:?}", out);
}
