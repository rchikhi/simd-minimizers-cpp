# Makefile for standalone C++ builds (for debugging/testing outside Cargo)
#
# Usage:
#   make          - Build example executable
#   make check    - Compile check only
#   make clean    - Remove built files

CXX = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -Wall -Wno-array-bounds -Wno-unused-variable -Wno-ignored-attributes -Wno-narrowing

# Source files
SRCS = src/canonical_minimizers_simd.cpp src/canonical_minimizers_scalar.cpp
HDRS = src/canonical_minimizers.hpp src/packed_seq.hpp

.PHONY: all clean check syncmer_benchmark

all: example

example: src/example.cpp $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ src/example.cpp $(SRCS)

syncmer_benchmark: src/syncmer_benchmark.cpp $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ src/syncmer_benchmark.cpp $(SRCS)

# Just compile to check for errors
check: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -c src/canonical_minimizers_simd.cpp -o /tmp/simd.o
	$(CXX) $(CXXFLAGS) -c src/canonical_minimizers_scalar.cpp -o /tmp/scalar.o
	@echo "Compilation successful"
	@rm -f /tmp/simd.o /tmp/scalar.o

clean:
	rm -f example syncmer_benchmark /tmp/simd.o /tmp/scalar.o
