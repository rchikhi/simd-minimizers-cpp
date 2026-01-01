# Makefile for standalone C++ builds (for debugging/testing outside Cargo)
#
# Usage:
#   make          - Build test executable
#   make clean    - Remove built files
#   make test     - Build and run tests

CXX = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -Wall -Wno-array-bounds -Wno-unused-variable -Wno-ignored-attributes -Wno-narrowing

# Source files
SRCS = src/canonical_minimizers_simd.cpp src/canonical_minimizers_scalar.cpp
HDRS = src/canonical_minimizers.hpp src/packed_seq.hpp

.PHONY: all clean test

all: test_cpp

# Build test executable (requires adding a main() to one of the cpp files or creating test_main.cpp)
test_cpp: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -DTEST_MAIN -o $@ $(SRCS)

# Just compile to check for errors
check: $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -c src/canonical_minimizers_simd.cpp -o /tmp/simd.o
	$(CXX) $(CXXFLAGS) -c src/canonical_minimizers_scalar.cpp -o /tmp/scalar.o
	@echo "Compilation successful"
	@rm -f /tmp/simd.o /tmp/scalar.o

clean:
	rm -f test_cpp /tmp/simd.o /tmp/scalar.o
