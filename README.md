# tensor_alloc

`tensor_alloc` is a high-performance, userspace memory allocator designed specifically for tensor and machine learning computation workloads. Rather than treating memory requests as opaque byte counts, `tensor_alloc` accepts structured tensor metadata (dimensions, rank, strides, data type, device, lifetime hint, and authoritative byte count), enabling fine-grained size-classing, 64-byte vector alignment, userspace pool reuse, and comprehensive memory metrics tracking.

## Current Milestone Status: Phase 1, Phase 2 & Phase 3 Complete

- **Platform:** macOS (Apple Silicon arm64) and POSIX-compliant systems.
- **Backend:** CPU-only (`TENSOR_DEVICE_CPU`). Guaranteed 64-byte alignment (`TENSOR_ALLOC_ALIGNMENT`).
- **Policy:** Deterministic power-of-two size class baseline + C++ Learned Decision Tree Predictor (`models/learned_predictor.json`) with automatic fallback to baseline on mismatch or safety checks.
- **Safety:** Explicit metadata validation, integer overflow protection, allocation ownership registry, double-free detection, and predictor capacity invariants ($\text{capacity} \ge \text{requested}$).
- **Metrics & Tracing:** Thread-safe metrics tracking, peak watermarks, and runtime JSONL event recording (`tensor_trace_start` / `tensor_trace_stop`).
- **Trace Pipeline:** Python multi-workload trace generator (Transformer, ConvNet, Mixed), multidimensional trace analyzer, and split-by-run label generator for ML training.
- **Learned Predictor:** Offline CART decision tree trainer with Gini impurity splitting, multi-policy evaluation harness, and sub-microsecond C++ tree inference engine.
- **Testing:** 100% passing test suite across 11 test targets in Debug, Release, and AddressSanitizer/UndefinedBehaviorSanitizer builds.

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
Compare `sys_malloc`, `tensor_alloc (baseline)`, and `tensor_alloc (learned ML predictor)` across synthetic machine learning workload patterns:
```bash
./build_release/allocator_microbench
```

#### Locally Measured Results (Apple M-series arm64, macOS Darwin 24.6.0, AppleClang 17.0.0):
| Workload Pattern | Allocator | Total Operations | Throughput | Latency (p50) | Latency (p99) | Pool Reuse % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Repeated Same Size (4KB)** | `sys_malloc` | 100,000 | 46.92M ops/s | 0.0 ns | 42.0 ns | 0.0% |
| **Repeated Same Size (4KB)** | `tensor_alloc(base)` | 100,000 | 16.05M ops/s | 83.0 ns | 84.0 ns | **100.0%** |
| **Repeated Same Size (4KB)** | `tensor_alloc(ml)` | 100,000 | 16.04M ops/s | 83.0 ns | 84.0 ns | **100.0%** |
| **Mixed Sizes (64B – 1MB)** | `sys_malloc` | 60,000 | 54.45M ops/s | 0.0 ns | 42.0 ns | 0.0% |
| **Mixed Sizes (64B – 1MB)** | `tensor_alloc(base)` | 60,000 | 15.47M ops/s | 83.0 ns | 84.0 ns | **100.0%** |
| **Mixed Sizes (64B – 1MB)** | `tensor_alloc(ml)` | 60,000 | 16.87M ops/s | 83.0 ns | 84.0 ns | **100.0%** |
| **Bursty (Batch 100)** | `sys_malloc` | 50,000 | 32.67M ops/s | 41.0 ns | 42.0 ns | 0.0% |
| **Bursty (Batch 100)** | `tensor_alloc(base)` | 50,000 | 15.85M ops/s | 42.0 ns | 84.0 ns | **99.6%** |
| **Bursty (Batch 100)** | `tensor_alloc(ml)` | 50,000 | 15.31M ops/s | 42.0 ns | 84.0 ns | **99.6%** |

### Run Trace Replay
Replay a sequence of tensor allocations and deallocations:
```bash
./build_release/replay_trace 50000
```
*Result: Replayed 60,392 events in 13.04 ms (4.63M ops/sec) with 65.57% pool reuse rate.*

### Trace Collection, ML Model Training & Evaluation
```bash
# 1. Generate 2,000-event JSONL synthetic workload trace
python3 python/collect_trace.py --output traces/workload.jsonl --events 2000

# 2. Analyze size distributions, lifetimes, and size-class internal waste
python3 python/analyze_trace.py traces/workload.jsonl

# 3. Extract features and build labeled ML dataset
python3 python/build_labels.py --input traces/workload.jsonl --output traces/labeled_dataset.json

# 4. Train CART decision tree predictor model
python3 python/train_predictor.py --dataset traces/labeled_dataset.json --output models/learned_predictor.json

# 5. Evaluate learned predictor against baseline, heuristic, and oracle
python3 python/evaluate_predictor.py --dataset traces/labeled_dataset.json --model models/learned_predictor.json
```

---

## Documentation Index

- [Architecture Specification](ARCHITECTURE.md): Data-flow diagrams, complete codebase file map, API contracts, and invariants.
- [Metrics Specification](docs/METRICS.md): Exact mathematical formulas for waste ratios, capacity tracking, and measurement boundaries.
- [Model Specification & Schema](models/README.md): Decision tree architecture, feature vector mapping, and inference engine.
- [Experimental Results](docs/EXPERIMENTS.md): Methodology, hardware environment, reproducibility logs, and benchmark tables.
- [Current Limitations](docs/LIMITATIONS.md): Unsupported features, deferred phases (CUDA, LinUCB, PyTorch replacement), and risk evaluation.

---

## Current Limitations

1. **CPU Only:** Requests with `TENSOR_DEVICE_CUDA` are cleanly rejected (`TENSOR_STATUS_UNSUPPORTED_DEVICE`). CUDA stream synchronization and device pointers are scheduled for Phase 5.
2. **Learned Predictor Deployment:** The sub-microsecond C++ decision tree engine (`models/learned_predictor.json`) is fully integrated and tested in runtime. Online reinforcement learning / LinUCB contextual bandit updates are scheduled for Phase 4.
3. **Pluggable Allocator:** Provides a standalone C API; not a drop-in replacement for PyTorch's internal allocator.
