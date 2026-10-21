# tensor_alloc — Metrics Specification and Formulation

This document defines the metrics collected, calculated, and reported by `tensor_alloc`.

---

## 1. Metric Definitions and Mathematical Formulas

### 1.1 Operation Counters

- **Allocation Count (`alloc_count`)**:
  Total number of successful allocation calls handled by `tensor_malloc`.
  $$\text{alloc\_count} = \sum \mathbf{1}_{\{\text{tensor\_malloc returns SUCCESS}\}}$$

- **Deallocation Count (`free_count`)**:
  Total number of successful free calls handled by `tensor_free`.
  $$\text{free\_count} = \sum \mathbf{1}_{\{\text{tensor\_free returns SUCCESS}\}}$$

- **Allocation Failure Count (`alloc_failure_count`)**:
  Total number of failed allocation requests (e.g., validation failure, out-of-memory, or unsupported device/dtype).
  $$\text{alloc\_failure\_count} = \sum \mathbf{1}_{\{\text{tensor\_malloc returns ERROR}\}}$$

### 1.2 Memory Volumes

- **Requested Bytes (`current_requested_bytes`, $B_{\text{req}}$)**:
  Total bytes actively requested by all currently live, un-freed allocations.
  $$B_{\text{req}}(t) = \sum_{a \in \text{Live}(t)} \text{requested\_bytes}(a)$$

- **Reserved Bytes (`current_reserved_bytes`, $B_{\text{res}}$)**:
  Total bytes of capacity allocated from the backend pools or operating system to house all currently live allocations. Because each allocation is placed into a size class whose capacity $C(a) \ge \text{requested\_bytes}(a)$, $B_{\text{res}}(t) \ge B_{\text{req}}(t)$ always holds.
  $$B_{\text{res}}(t) = \sum_{a \in \text{Live}(t)} \text{capacity}(a)$$

- **Peak Requested Bytes (`peak_requested_bytes`)**:
  The high-water mark of live requested bytes over the lifetime of the process.
  $$\text{Peak}_{\text{req}} = \max_{t} B_{\text{req}}(t)$$

- **Peak Reserved Bytes (`peak_reserved_bytes`)**:
  The high-water mark of live reserved capacity bytes over the lifetime of the process.
  $$\text{Peak}_{\text{res}} = \max_{t} B_{\text{res}}(t)$$

### 1.3 Memory Waste and Ratios

- **Internal Waste ($W_{\text{internal}}$)**:
  Memory allocated to a block but unused by the application because the size-class capacity exceeds the requested bytes:
  $$W_{\text{internal}}(t) = B_{\text{res}}(t) - B_{\text{req}}(t)$$

- **Internal Waste Ratio ($R_{\text{waste}}$)**:
  The proportion of active reserved memory consumed by size-class rounding:
  $$R_{\text{waste}}(t) = \begin{cases} \frac{B_{\text{res}}(t) - B_{\text{req}}(t)}{B_{\text{res}}(t)} & \text{if } B_{\text{res}}(t) > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

- **Distinction from External Fragmentation**:
  - *Internal Fragmentation / Waste* occurs inside an allocated block when the assigned size class is larger than requested. `tensor_alloc` measures this directly and precisely.
  - *External Fragmentation* refers to unusable free memory scattered between allocated blocks or idle in free lists that cannot satisfy an incoming request. Idle blocks in free lists are tracked separately via `cpu_backend_idle_block_count()`.

### 1.4 Pool Reuse and Predictor Metrics

- **Reuse Count (`reuse_count`)**:
  Number of `tensor_malloc` invocations fulfilled by popping an existing aligned block from a pool's free list without invoking `posix_memalign`.

- **Reuse Rate ($R_{\text{reuse}}$)**:
  $$\text{reuse\_rate} = \begin{cases} \frac{\text{reuse\_count}}{\text{alloc\_count}} & \text{if } \text{alloc\_count} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

- **Predictor Fallback Count (`fallback_count`)**:
  Number of times an active predictor recommended an invalid size class, out-of-range class, or a class with capacity smaller than `requested_bytes`, triggering the safe deterministic fallback.

### 1.5 Temporal Concurrency and Trace Metrics

- **Instantaneous Active Count ($N_{\text{active}}(t)$)**:
  Number of allocations that have been issued at or before timestamp $t$ and not yet freed:
  $$N_{\text{active}}(t) = |\{ a \mid t_{\text{alloc}}(a) \le t < t_{\text{free}}(a) \}|$$

- **Peak Concurrency ($N_{\text{peak}}$)**:
  $$N_{\text{peak}} = \max_{t} N_{\text{active}}(t)$$

- **Buffer Lifetime ($\Delta t(a)$)**:
  $$\Delta t(a) = t_{\text{free}}(a) - t_{\text{alloc}}(a)$$
  Measured in nanoseconds in raw traces, reported in milliseconds during analysis. Temporary activation buffers typically exhibit $\Delta t \le 1.0\text{ ms}$, whereas persistent weight buffers span the entire run duration.

---

## 2. Measurement Boundaries and Overhead

### 2.1 Thread Safety and Lock Scope
Metrics updates occur inside a lightweight mutex lock (`std::lock_guard<std::mutex>`) in `src/metrics.cpp`. The update consists solely of integer additions and comparisons, taking under 10 nanoseconds per call.

### 2.2 Disabling Metrics
To measure allocator latency strictly without metrics overhead during benchmarking:
```c
metrics_set_enabled(false);
```
When disabled, atomic load checks bypass all lock acquisition and counter mutations.
