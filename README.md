# tensor_alloc

`tensor_alloc` is a high-performance, userspace memory allocator designed specifically for tensor and machine learning computation workloads. Rather than treating memory requests as opaque byte counts, `tensor_alloc` accepts structured tensor metadata (dimensions, rank, strides, data type, device, lifetime hint, and authoritative byte count), enabling fine-grained size-classing, 64-byte vector alignment, userspace pool reuse, and comprehensive memory metrics tracking.

## Current Milestone Status: Phase 1 (CPU MVP)

- **Platform:** macOS (Apple Silicon arm64) and POSIX-compliant systems.
- **Backend:** CPU-only (`TENSOR_DEVICE_CPU`). Guaranteed 64-byte alignment (`TENSOR_ALLOC_ALIGNMENT`).
- **Policy:** Deterministic power-of-two size class baseline (21 pools from 64 B to 64 MB) plus aligned large-allocation path.
- **Safety:** Explicit metadata validation, integer overflow protection, allocation ownership registry, and double-free detection.
- **Metrics:** Thread-safe tracking of requested vs. reserved memory, internal waste ratios, peak watermarks, and pool reuse rates.
- **Testing:** 100% passing test suite across Debug, Release, and AddressSanitizer/UndefinedBehaviorSanitizer builds.

---

## Prerequisites

- **CMake:** Version 3.16 or higher.
- **C/C++ Toolchain:** C11 / C++17 capable compiler (e.g., Apple Clang 17+, Clang 10+, or GCC 9+).
- **Python:** Python 3.8+ (for synthetic workload trace generation and analysis tools).

---

## Build and Test Instructions

### 1. Release Build (Recommended for Benchmarks)
```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel
ctest --test-dir build_release --output-on-failure
```

### 2. Debug Build with AddressSanitizer and UndefinedBehaviorSanitizer
```bash
cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build_asan --parallel
ctest --test-dir build_asan --output-on-failure
```

---

## Minimal C / C++ Usage Example

```cpp
#include "tensor_alloc.h"
#include <stdio.h>

int main() {
    // 1. Describe 3D float32 tensor of shape [16, 32, 64]
    int64_t shape[3] = {16, 32, 64};
    tensor_alloc_request_t req{};
    req.shape = shape;
    req.ndim = 3;
    req.strides = NULL; // NULL indicates standard C-contiguous
    req.dtype = TENSOR_DTYPE_FLOAT32;
    req.device = TENSOR_DEVICE_CPU;
    req.lifetime_hint = TENSOR_LIFETIME_TEMPORARY;
    req.requested_bytes = 16 * 32 * 64 * sizeof(float); // 131,072 bytes (128 KB)

    // 2. Allocate 64-byte aligned memory
    void* tensor_ptr = NULL;
    tensor_status_t status = tensor_malloc(&req, &tensor_ptr);
    if (status != TENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "Allocation failed: %s\n", tensor_status_str(status));
        return 1;
    }

    // 3. Inspect metrics
    tensor_print_metrics(stdout);

    // 4. Free memory back to userspace pool
    tensor_free(tensor_ptr);

    // 5. Clean up all pools upon shutdown
    tensor_shutdown();
    return 0;
}
```

Run the compiled example:
```bash
./build_release/basic_usage
```

---

## Microbenchmark and Trace Replay

### Run Microbenchmark
Compare `tensor_alloc` against system `malloc/free` across synthetic machine learning workload patterns:
```bash
./build_release/allocator_microbench
```

#### Locally Measured Results (Apple M-series arm64, macOS Darwin 24.6.0, AppleClang 17.0.0):
| Workload Pattern | Allocator | Total Operations | Throughput | Latency (p50) | Latency (p99) | Pool Reuse % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Repeated Same Size (4KB)** | `sys_malloc` | 100,000 | 35.69M ops/s | 41.0 ns | 42.0 ns | 0.0% |
| **Repeated Same Size (4KB)** | `tensor_alloc` | 100,000 | 9.16M ops/s | 125.0 ns | 125.0 ns | **100.0%** |
| **Mixed Sizes (64B – 1MB)** | `sys_malloc` | 60,000 | 29.57M ops/s | 41.0 ns | 42.0 ns | 0.0% |
| **Mixed Sizes (64B – 1MB)** | `tensor_alloc` | 60,000 | 9.28M ops/s | 125.0 ns | 167.0 ns | **99.97%** |
| **Bursty (Batch 100)** | `sys_malloc` | 50,000 | 18.62M ops/s | 42.0 ns | 84.0 ns | 0.0% |
| **Bursty (Batch 100)** | `tensor_alloc` | 50,000 | 10.92M ops/s | 83.0 ns | 125.0 ns | **99.60%** |

### Run Trace Replay
Replay a sequence of tensor allocations and deallocations:
```bash
./build_release/replay_trace 50000
```
*Result: Replayed 60,392 events in 13.04 ms (4.63M ops/sec) with 65.57% pool reuse rate.*

### Generate and Analyze Traces with Python
```bash
# Generate 2,000 event JSONL synthetic workload trace
python3 python/collect_trace.py --output traces/workload.jsonl --events 2000

# Analyze size distributions, lifetimes, and size-class internal waste
python3 python/analyze_trace.py traces/workload.jsonl
```

---

## Documentation Index

- [Architecture Specification](ARCHITECTURE.md): Data-flow diagrams, complete codebase file map, API contracts, and invariants.
- [Metrics Specification](docs/METRICS.md): Exact mathematical formulas for waste ratios, capacity tracking, and measurement boundaries.
- [Experimental Results](docs/EXPERIMENTS.md): Methodology, hardware environment, reproducibility logs, and benchmark tables.
- [Current Limitations](docs/LIMITATIONS.md): Unsupported features, deferred phases (CUDA, LinUCB, PyTorch replacement), and risk evaluation.

---

## Current Limitations

1. **CPU Only:** Requests with `TENSOR_DEVICE_CUDA` are cleanly rejected (`TENSOR_STATUS_UNSUPPORTED_DEVICE`). CUDA stream synchronization and device pointers are scheduled for Phase 5.
2. **Learned Predictor:** The runtime currently uses the deterministic baseline. Machine learning models and ONNX inference are scheduled for Phase 3.
3. **Pluggable Allocator:** Provides a standalone C API; not a drop-in replacement for PyTorch's internal allocator.
