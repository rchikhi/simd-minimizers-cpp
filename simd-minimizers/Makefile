CXX = g++
RUSTC = rustc
CXXFLAGS = -std=c++17 -O3 -mavx2 -Wall -Wno-array-bounds -Wno-unused-variable -Wno-ignored-attributes -Wno-narrowing
RUSTFLAGS = --edition=2021 -O

# Rust dependencies
RUST_DEPS = -C prefer-dynamic -L dependency=./target/debug/deps --extern packed_seq=./target/debug/deps/libpacked_seq*.rlib --extern simd_minimizers=./target/debug/deps/libsimd_minimizers*.rlib -C target-cpu=native

.PHONY: all clean

all: test_hash_cpp test_hash_rust

# C++ executable
test_hash_cpp: test_hash.cpp src/canonical_minimizers_simd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< src/canonical_minimizers_simd.cpp

# Rust executable
test_hash_rust: test_hash.rs
	cargo run --bin test_hash_rust
	cp ../target/debug/test_hash_rust .

clean:
	rm -f test_hash_cpp test_hash_rust
