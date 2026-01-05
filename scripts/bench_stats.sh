#!/bin/bash
# Run cpp_comparison benchmark N times and compute statistics
# Usage: ./scripts/bench_stats.sh [iterations]

N=${1:-50}
echo "Running $N iterations..."

# E2E benchmarks (with packing)
echo ""
echo "=== E2E (with packing) ==="
bash -c "for i in \$(seq 1 $N); do RUSTFLAGS=\"-C target-cpu=native\" cargo run --release --example cpp_comparison 2>&1 | grep -E \"^(Rust|C\\+\\+).*\\|.*\\|.*\\|\" | head -4; done" 2>/dev/null | python3 -c "
import sys
from collections import defaultdict

data = defaultdict(list)
for line in sys.stdin:
    parts = line.split('|')
    if len(parts) >= 3:
        name = parts[0].strip()
        try:
            mbps = float(parts[2].strip())
            data[name].append(mbps)
        except:
            pass

print(f'{'Name':22s} {'N':>3s} {'Avg':>7s} {'Min':>7s} {'Max':>7s} {'Range':>6s}')
print('-' * 52)
for name in sorted(data.keys()):
    vals = data[name]
    rng = max(vals)/min(vals) if min(vals) > 0 else 0
    print(f'{name:22s} {len(vals):3d} {sum(vals)/len(vals):7.1f} {min(vals):7.1f} {max(vals):7.1f} {rng:5.2f}x')
"

# Pre-packed benchmarks (algorithm only)
echo ""
echo "=== Pre-packed (algorithm only) ==="
bash -c "for i in \$(seq 1 $N); do RUSTFLAGS=\"-C target-cpu=native\" cargo run --release --example cpp_comparison 2>&1 | grep -E \"^(Rust|C\\+\\+).*\\|.*\\|.*\\|\" | tail -5 | head -4; done" 2>/dev/null | python3 -c "
import sys
from collections import defaultdict

data = defaultdict(list)
for line in sys.stdin:
    parts = line.split('|')
    if len(parts) >= 3:
        name = parts[0].strip()
        try:
            mbps = float(parts[2].strip())
            data[name].append(mbps)
        except:
            pass

print(f'{'Name':22s} {'N':>3s} {'Avg':>7s} {'Min':>7s} {'Max':>7s} {'Range':>6s}')
print('-' * 52)
for name in sorted(data.keys()):
    vals = data[name]
    rng = max(vals)/min(vals) if min(vals) > 0 else 0
    print(f'{name:22s} {len(vals):3d} {sum(vals)/len(vals):7.1f} {min(vals):7.1f} {max(vals):7.1f} {rng:5.2f}x')
"
