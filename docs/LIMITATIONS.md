# tensor_alloc — Current Limitations and Known Risks

This document records the design constraints, unimplemented features, and known architectural risks of `tensor_alloc` as of Milestone Phase 3 (Learned Predictor & Runtime Integration).

---

## 1. Scope and Platform Limitations

- **CPU-Only Milestone:**
  Only `TENSOR_DEVICE_CPU` is currently supported. Allocations requesting `TENSOR_DEVICE_CUDA` are rejected immediately with `TENSOR_STATUS_UNSUPPORTED_DEVICE`. CUDA stream synchronization, device pointer ownership, and unified memory semantics are deferred to Phase 5.
- **Custom C API vs. Framework Integration:**
  `tensor_alloc` exposes a standalone C-compatible API (`tensor_malloc`, `tensor_free`). It is **not** a drop-in replacement for PyTorch's native allocator (`c10::cuda::CUDACachingAllocator` or default CPU allocator). Integrating into PyTorch's pluggable allocator interface requires custom PyTorch C++ bindings and is deferred.
- **Phase 2 & Phase 3 Complete:**
  Trace recording, pipeline analysis, split-by-run dataset generation, offline CART decision tree training (`train_predictor.py`), multi-policy offline evaluation (`evaluate_predictor.py`), sub-microsecond C++ tree inference (`LearnedPredictor`), and runtime safety fallback wrappers are fully implemented, benchmarked, and verified.
- **No Third-Party Python Dependencies:**
  To guarantee zero environmental dependencies and deterministic portability across fresh host environments without pip packages, the tree model training and evaluation are implemented in standard Python (`python3`). Full neural network architectures / ONNX runtime inference can be integrated when external C++ runtimes are linked.
- **No Online Bandit (Phase 4):**
  LinUCB contextual bandit online exploration and live policy updates are scheduled for Phase 4. Currently, the tree model is trained offline and executed deterministically at runtime.

---

## 2. Memory Backend Constraints

- **No Active Memory Trimming:**
  Allocated memory blocks returned to pools via `tensor_free` are preserved in userspace free lists for immediate reuse. They remain mapped in the process address space until `tensor_shutdown()` is explicitly called. Kernel page discarding (`madvise(MADV_DONTNEED)` / `madvise(MADV_FREE)`) is not yet implemented.
- **Strided Layout Constraints:**
  Strides are interpreted strictly in **elements** and must be non-negative. Requests with negative strides (which require base address displacement) or strides causing integer overflow are rejected with `TENSOR_STATUS_INVALID_ARGUMENT` or `TENSOR_STATUS_OVERFLOW`.
- **Lock Contention Under Extreme Thread Counts:**
  Thread safety is enforced via mutex locks on the allocation registry, size-class pools, and metrics. While tested and verified for 8 concurrent threads, high-core-count workloads (e.g., 64+ threads) may experience mutex contention. Thread-local caching (`tcache`) is a candidate for future phases.

---

## 3. Risk Assessment

| Risk Area | Severity | Current Mitigation | Future Planned Work |
| :--- | :--- | :--- | :--- |
| **Out-of-Bounds Memory Access** | High | Validation checks all multiplications for integer overflow (`src/request_validation.cpp`); enforces 64-byte alignment. | Add hardware guard pages in debug mode. |
| **Double-Free Memory Corruption** | High | Bounded FIFO history (`src/allocation_registry.cpp`) detects double frees and returns `TENSOR_STATUS_DOUBLE_FREE`. | Maintain persistent tombstone bitmaps. |
| **Predictor Misprediction** | Medium | `predict_size_class_safe` always validates that predicted capacity $\ge$ requested bytes, safely falling back to baseline. | Offline evaluation before runtime activation. |
| **Heap Bloat from Idle Pools** | Low | Explicit `tensor_shutdown()` and `cpu_backend_destroy()` APIs drain all pooled memory back to OS. | Add watermark-based periodic pool compaction. |
