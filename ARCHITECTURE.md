# tensor_alloc — System Architecture and Codebase Specification

> **Milestone Status:** Phase 1 (CPU MVP), Phase 2 (Trace Collection, Analysis & Labeling Pipeline), and Phase 3 (Learned Predictor & Runtime Integration) Complete and Tested.

---

## 1. System Overview and Data Flow

`tensor_alloc` is a userspace memory allocator designed specifically for tensor and machine learning computation workloads. In contrast to standard system allocators that only receive raw byte counts, `tensor_alloc` receives structured tensor metadata (shape, rank, strides, data type, device target, lifetime hint, and requested byte count).

```text
                     +---------------------------------------+
                     |         Application / Benchmark       |
                     +---------------------------------------+
                                         |
                                         v
                         tensor_malloc(req, &out_ptr)
                                         |
                                         v
                     +---------------------------------------+
                     |        Request Validation Module      |
                     |       (src/request_validation.cpp)   |
                     +---------------------------------------+
                       | Checks integer overflow, rank,
                       | dtype, CPU device, zero-byte rules
                                         v
                     +---------------------------------------+
                     |        Size Class Selection Policy    |
                     |      (src/predictor_fallback.cpp)     |
                     +---------------------------------------+
                       | Predictor abstraction queried;
                       | Validates capacity >= requested;
                       | Safely falls back to baseline class
                                         v
                     +---------------------------------------+
                     |         CPU Memory Pool Backend       |
                     |          (src/cpu_backend.cpp)        |
                     +---------------------------------------+
                       | Checks free-list for matching class;
                       | Reuses existing 64-byte aligned block;
                       | Or calls posix_memalign for new block
                                         v
                     +---------------------------------------+
                     |        Allocation Registry Module     |
                     |      (src/allocation_registry.cpp)    |
                     +---------------------------------------+
                       | Records pointer, size class, bytes;
                       | Prevents unknown / double-free bugs
                                         v
                     +---------------------------------------+
                     |           Metrics Subsystem           |
                     |           (src/metrics.cpp)           |
                     +---------------------------------------+
                       | Updates live requested, reserved,
                       | peak watermarks, reuse count
                                         v
                            Aligned Pointer to Caller
```

### Deallocation Flow (`tensor_free`):
```text
                          tensor_free(ptr)
                                 |
                                 v
                     [ Pointer is NULL? ] ---> Returns TENSOR_STATUS_SUCCESS (No-op)
                                 |
                                 v
                     +---------------------------------------+
                     |        Allocation Registry Lookup     |
                     |      (src/allocation_registry.cpp)    |
                     +---------------------------------------+
                       | Checks if live: retrieves metadata
                       | Checks if in freed history: DOUBLE_FREE
                       | If unrecognized: NOT_FOUND
                                 v
                     +---------------------------------------+
                     |         CPU Memory Pool Backend       |
                     |          (src/cpu_backend.cpp)        |
                     +---------------------------------------+
                       | Returns block to size-class free list
                       | Or calls free() if large allocation
                                 v
                     +---------------------------------------+
                     |           Metrics Subsystem           |
                     |           (src/metrics.cpp)           |
                     +---------------------------------------+
                       | Decrements current requested/reserved
                       | Increments free_count
```

---

## 2. Complete Repository Tree

```text
tensor_alloc/
├── CMakeLists.txt                 # Root CMake configuration (C++17, ASan/UBSan, CTest)
├── README.md                      # Quick-start guide, build instructions, benchmark summary
├── ARCHITECTURE.md                # System architecture, file map, API contracts
├── TENSOR_ALLOC_CODING_PROMPT.md  # Core project requirements and design intent
├── include/
│   └── tensor_alloc.h             # Stable C-compatible public API declarations (malloc, free, trace)
├── src/
│   ├── api.cpp                    # Public API orchestration (tensor_malloc, tensor_free, metrics, trace)
│   ├── request_validation.h       # Validation declarations & overflow helpers
│   ├── request_validation.cpp     # Request metadata verification & arithmetic overflow checks
│   ├── size_class.h               # Size class definitions (21 power-of-two classes, large path)
│   ├── size_class.cpp             # Size class lookup & rounding logic
│   ├── cpu_backend.h              # CPU backend memory pool declarations
│   ├── cpu_backend.cpp            # Pool free-lists, 64-byte posix_memalign, and large allocations
│   ├── allocation_registry.h      # Allocation ownership registry declarations
│   ├── allocation_registry.cpp    # Live pointer tracking, double-free FIFO history, thread locking
│   ├── metrics.h                  # Metrics tracking hooks and snapshot declarations
│   ├── metrics.cpp                # Thread-safe counter updates, peak tracking, ratios, summaries
│   ├── trace_recorder.h           # Runtime trace recording declarations
│   ├── trace_recorder.cpp         # High-resolution JSONL event logging for allocations and frees
│   ├── predictor_interface.h      # IPredictor abstract interface & safe validation wrapper
│   ├── predictor_fallback.cpp     # Deterministic baseline predictor implementation
│   ├── learned_predictor.h        # Tree-based learned predictor class & feature extraction
│   └── learned_predictor.cpp      # JSON tree deserialization, feature computation & inference
├── tests/
│   ├── test_helpers.h             # TEST_ASSERT macro (always evaluated regardless of NDEBUG)
│   ├── test_size_class.cpp        # Size class lookup, boundaries, powers of 2, large path
│   ├── test_validation.cpp        # Request validation, null checks, dtype/device, overflow
│   ├── test_api.cpp               # End-to-end tensor_malloc/free, metrics, shutdown
│   ├── test_alignment.cpp         # 64-byte alignment verification across size classes
│   ├── test_reuse.cpp             # Free-list pool reuse verification & reuse counter checks
│   ├── test_invalid_inputs.cpp    # Double-free, unknown pointer, zero-byte, CUDA rejection
│   ├── test_multithreading.cpp    # Concurrent thread-safety stress test across 8 threads
│   ├── test_trace_recorder.cpp    # Verification of C++ runtime trace recording
│   ├── test_trace_pipeline.py     # End-to-end Python pipeline test (collect, analyze, replay, label)
│   ├── test_learned_predictor.cpp # Verification of C++ learned predictor loader & inference
│   └── test_phase3_pipeline.py    # Verification of train/eval Python pipeline & artifact generation
├── benchmarks/
│   ├── allocator_microbench.cpp   # Microbenchmark comparing sys_malloc vs tensor_alloc (baseline & ML)
│   └── replay_trace.cpp           # JSONL trace replay benchmark harness with sys_malloc comparison
├── python/
│   ├── collect_trace.py           # Multi-workload trace generator (Transformer, ConvNet, Mixed; JSONL/CSV)
│   ├── analyze_trace.py           # Multi-metric trace analyzer (percentiles, concurrency, waste, comparison)
│   ├── build_labels.py            # Feature extractor and objective labeler with split-by-run partitioning
│   ├── train_predictor.py         # Offline decision tree trainer (Gini impurity, CART splits, JSON export)
│   └── evaluate_predictor.py      # Multi-policy offline evaluation (baseline vs heuristic vs learned vs oracle)
├── traces/
│   ├── README.md                  # Trace schema documentation (v1.1.0) and semantics
│   ├── transformer_trace.jsonl    # Transformer model step allocation trace
│   ├── convnet_trace.jsonl        # ConvNet multi-stage allocation trace
│   └── labeled_dataset.json       # Labeled ML dataset partitioned by run_id
├── models/
│   ├── README.md                  # Model artifact schema and performance characteristics
│   └── learned_predictor.json     # Trained 11-node decision tree for size class prediction
├── examples/
│   └── basic_usage.cpp            # Complete 3D tensor allocation and pool reuse example
└── docs/
    ├── METRICS.md                 # Metric formulas, definitions, and measurement boundaries
    ├── EXPERIMENTS.md             # Benchmark methodologies and measured hardware results
    └── LIMITATIONS.md             # Current limitations, constraints, and risk analysis
```

---

## 3. Module Responsibilities and Key Functions

### 3.1 Public Interface (`include/tensor_alloc.h`, `src/api.cpp`)
- `tensor_malloc(const tensor_alloc_request_t* req, void** out_ptr)`:
  Validates metadata, requests a size class from the predictor/fallback policy, obtains 64-byte aligned memory from the CPU pool, registers pointer ownership, records metrics, and returns status.
- `tensor_free(void* ptr)`:
  Safely accepts `NULL` as a no-op; validates pointer ownership in registry (rejecting unknown pointers and detecting double frees); returns memory to pool; updates metrics.
- `tensor_get_metrics(void)` / `tensor_print_metrics(FILE* stream)`:
  Provides atomic snapshots of allocator metrics.
- `tensor_shutdown(void)`:
  Drains all cached pool blocks back to the OS and resets internal tracking structures.

### 3.2 Request Validation (`src/request_validation.h/.cpp`)
- `validate_tensor_request(...)`:
  Guarantees that `req != NULL`, `out_ptr != NULL`, `requested_bytes > 0`, `device == TENSOR_DEVICE_CPU`, `dtype` is supported, `ndim <= 8`, dimensions are non-negative, and no integer multiplication overflows occur.
- `safe_mul_size(...)`, `safe_add_size(...)`:
  Overflow-safe arithmetic operations using `SIZE_MAX` boundaries.

### 3.3 Size Class System (`src/size_class.h/.cpp`)
- Predefines 21 geometric size classes from 64 B ($2^6$) to 64 MB ($2^{26}$).
- `size_class_lookup(size_t bytes)`: Maps requested bytes to the smallest size class where `capacity >= bytes`. Requests $> 64\text{ MB}$ return `TENSOR_SIZE_CLASS_LARGE`.
- `size_class_round_large(size_t bytes)`: Rounds large allocations up to the nearest 64-byte cache line boundary.

### 3.4 Memory Engine (`src/cpu_backend.h/.cpp`)
- `cpu_backend_alloc(...)`:
  Maintains separate LIFO free lists for each size class. If a free block exists, it is popped and reused immediately. Otherwise, allocates new memory via `posix_memalign(&ptr, 64, capacity)`.
- `cpu_backend_free(...)`:
  Pushes freed blocks back onto their corresponding class free list for immediate reuse, or calls `free()` for large allocations.
- `cpu_backend_destroy(...)`:
  Frees all idling blocks across all pool lists back to the OS.

### 3.5 Ownership Registry (`src/allocation_registry.h/.cpp`)
- `registry_insert(...)`:
  Records live allocations in a hash table (`ptr -> allocation_record_t`).
- `registry_remove(...)`:
  Removes live allocation. If found, logs pointer in a 100,000-element FIFO history to differentiate future double frees from completely unknown pointers.
- Thread-safe via `std::mutex`.

### 3.6 Predictor Subsystem (`src/predictor_interface.h`, `src/predictor_fallback.cpp`)
- `IPredictor`: Pure abstract interface for future ML-based size-class recommendation models.
- `predict_size_class_safe(...)`: Queries predictor, validates that recommendation satisfies `capacity >= requested_bytes`, and safely triggers the deterministic baseline fallback if validation fails.

### 3.7 Learned Predictor Engine (`src/learned_predictor.h/.cpp`, `python/train_predictor.py`, `python/evaluate_predictor.py`)
- `LearnedPredictor`: C++ runtime implementation of `IPredictor` loading trained JSON decision tree models.
  - Implements sub-microsecond recursive tree traversal without third-party dependencies.
  - Feature extraction maps incoming `tensor_alloc_request_t` to the canonical 11-dimensional feature vector.
  - Returns recommended `size_class_t` or delegates to deterministic baseline.
- `python/train_predictor.py`: Pure standard library CART decision tree trainer with Gini impurity splitting, cost-complexity pruning, and clean JSON export.
- `python/evaluate_predictor.py`: Multi-policy benchmark evaluator comparing baseline, heuristic, learned decision tree, and oracle on held-out test splits.

---

## 4. Invariants and Safety Guarantees

1. **Alignment Guarantee:** Every successful allocation returns a memory address where `reinterpret_cast<uintptr_t>(ptr) % 64 == 0`.
2. **Capacity Invariant:** For every allocation, $\text{reserved capacity} \ge \text{requested bytes}$.
3. **No Pointer-Address Class Guessing:** The size class and reserved capacity of a block are **never** guessed from pointer bit patterns; they are retrieved directly from the verified allocation registry.
4. **Zero-Byte Invariant:** Allocating 0 bytes is strictly disallowed and returns `TENSOR_STATUS_INVALID_ARGUMENT`.
5. **Double-Free Safety:** Deallocating an already-freed pointer returns `TENSOR_STATUS_DOUBLE_FREE` without corrupting heap metadata.
6. **Unknown Pointer Safety:** Deallocating arbitrary heap or stack pointers returns `TENSOR_STATUS_NOT_FOUND`.
7. **Thread Safety:** All public APIs and internal pools are protected against race conditions across concurrent threads.
8. **Predictor Safety Fallback Invariant:** If a machine-learned predictor suggests a size class whose capacity is smaller than `requested_bytes` (or if an inference failure occurs), `predict_size_class_safe` unconditionally intercepts the recommendation, increments `fallback_count`, and assigns the verified deterministic baseline size class. Memory safety is never sacrificed for learned recommendations.
