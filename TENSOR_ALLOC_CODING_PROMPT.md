# Coding IDE Prompt: Implement tensor_alloc

You are a senior systems engineer helping implement `tensor_alloc`, a research prototype for a learned, tensor-aware userspace memory allocator. Work incrementally, prioritize memory safety and measurable results, and explain the code so a student can understand and maintain it.

## Source of truth

Use the project's design presentation and `ARCHITECTURE.md` as the design intent. Treat the presentation's performance improvements as hypotheses, not established facts. If the repository already contains code, inspect it first and preserve working behavior. Do not invent benchmark results or claim support that has not been tested.

## Goal

Build a CPU-first allocator with a C-compatible API that accepts tensor metadata and requested bytes, chooses a safe size class, allocates aligned memory, tracks ownership, supports freeing and reuse, records metrics, and can be benchmarked against a baseline. Keep the learned model, CUDA backend, online bandit, and framework allocator replacement out of the first milestone unless the MVP is complete and tested.

## Required workflow

1. Inspect the repository tree, README, build system, existing code, compiler, OS, and available test tools.
2. Summarize the current state and propose a short implementation plan before making broad changes.
3. Implement in small, reviewable stages. Build and test after each stage.
4. Do not replace languages or frameworks without a concrete reason. Prefer C++17 or newer and CMake for the initial CPU library, unless the existing repository establishes a different stack.
5. Keep the implementation understandable: small functions, meaningful names, comments explaining non-obvious decisions, and documented invariants.
6. Create and maintain both `README.md` and `ARCHITECTURE.md`. Also add `docs/METRICS.md`, `docs/EXPERIMENTS.md`, and `docs/LIMITATIONS.md` as the project grows.
7. Never provide fabricated test output, benchmark numbers, model accuracy, or claims of CUDA/PyTorch compatibility. Clearly mark anything that could not be tested in the current environment.

## MVP architecture

- Public C-compatible API: `include/tensor_alloc.h`
- API orchestration: `src/api.cpp`
- Request validation and overflow checks: `src/request_validation.h/.cpp`
- Fixed size-class baseline: `src/size_class.h/.cpp`
- CPU allocation pools, alignment, and reuse: `src/cpu_backend.h/.cpp`
- Live allocation ownership registry: `src/allocation_registry.h/.cpp`
- Metrics: `src/metrics.h/.cpp`
- Predictor abstraction with deterministic baseline fallback: `src/predictor_interface.h` and `src/predictor_fallback.cpp`
- Tests under `tests/`
- Microbenchmark and trace replay under `benchmarks/`
- Python trace analysis and later ML pipeline under `python/`
- A simple usage example under `examples/`

Adjust filenames if necessary to fit the existing repository, but update `ARCHITECTURE.md` with the exact final tree and the responsibility of every file.

## API requirements

Design and document a C-compatible API with clear error handling. A possible design is a metadata struct containing shape pointer, rank, strides pointer, dtype enum, device enum, lifetime hint, and requested byte count, plus `tensor_malloc(...)` and `tensor_free(...)`. You may improve the signature if you explain why.

- State clearly whether strides are in elements or bytes.
- Define zero-byte allocation behavior and `free(NULL)` behavior.
- Validate metadata and all multiplication/addition for integer overflow.
- Do not assume `product(shape) * dtype_size` is always correct for arbitrary strides or non-contiguous tensors. Treat requested bytes as authoritative or independently validate it according to the documented API.
- Guarantee documented alignment for successful allocations.
- A size class must always have capacity greater than or equal to requested bytes.
- If a request is too large for configured classes, use a documented large-allocation path or return a clear error.
- Track pointer ownership so unknown pointers and double frees are not silently accepted.
- Never infer an allocation's class solely from the pointer address.
- Keep host and device pointer types/ownership separate; MVP is CPU-only.
- Specify thread-safety guarantees. If thread safety is not implemented initially, say so explicitly and test/document that limitation.

## Allocation policy

First implement a deterministic baseline: select the smallest supported size class that can hold the requested number of bytes. Keep selection policy separate from the memory backend.

Add a predictor interface but use a deterministic fallback in the MVP. Any future predictor recommendation must be checked before use. If inference fails, returns an invalid class, or selects a class too small for the request, fall back safely and increment a fallback metric.

Do not claim that tensor-aware size classes improve performance until benchmarks demonstrate it.

## Metrics

Implement and document:
- Allocation/free operation count
- Allocation failures
- Requested bytes
- Reserved/capacity bytes
- Internal waste (`reserved - requested`)
- Peak live bytes and peak reserved bytes
- Pool reuse count/rate
- Allocation latency samples or benchmark-level latency
- Predictor inference time and fallback count when ML is added

Define every ratio precisely and avoid conflating internal fragmentation with external fragmentation. Ensure instrumentation can be disabled or measured for overhead.

## Testing requirements

Create tests for:
- Basic allocation and free
- Multiple simultaneous allocations
- Size-class boundaries
- Zero-byte behavior
- Largest supported class and large-allocation path
- Alignment
- Reuse after free
- Invalid metadata and dtype/device values
- Overflow cases
- Unknown pointer and double free behavior
- Allocation failure handling
- Thread safety if promised

Use CTest and add sanitizer instructions (AddressSanitizer and UndefinedBehaviorSanitizer where supported). Do not run performance benchmarks in sanitizer builds and report this distinction.

## Benchmark requirements

Provide a reproducible microbenchmark and optionally trace replay. Compare at least:
1. system `malloc/free`
2. fixed size-class allocator without ML
3. learned policy later, using the same backend

Use identical request sequences for fair comparisons. Measure median and tail allocation/free latency, throughput, requested/reserved memory, internal waste, peak memory, and reuse. Report compiler, OS, CPU, allocator configuration, repetitions, and workload pattern. Do not make benchmark claims until the benchmark has actually run.

## Later phases, only after MVP is stable

### Trace pipeline
Implement trace collection/replay with versioned JSONL or CSV schema containing request/run IDs, shape, strides, dtype, device, requested bytes, lifetime hints, and allocation/free timing. Do not store tensor contents. Add analysis scripts and document units/semantics.

### Learned predictor
Train a compact model in Python using reproducible splits, versioned preprocessing, class mapping, and saved artifacts. Evaluate against fixed classes and simple heuristics on held-out workloads. Classification accuracy is insufficient; compare end-to-end allocator metrics including inference overhead. Add ONNX/C++ inference only if the environment supports it and measured results justify it.

### Online adaptation
Do not add LinUCB until the offline policy has a reliable reward and stable baselines. Document context, actions, reward, exploration bounds, reproducibility, and safe fallback. Adaptation must never bypass allocator safety checks.

### CUDA
Treat CUDA as a separate phase requiring compatible hardware/toolkit. Document stream ordering, lifetime semantics, alignment, ownership, and host/device separation. Do not stub a CUDA implementation and call it complete.

### PyTorch integration
Investigate the supported allocator extension points for the actual PyTorch version in use. Treat a custom C API as distinct from drop-in allocator replacement. Do not claim drop-in support until integration tests prove it.

## Documentation requirements

`README.md` must include:
- Project purpose and current implementation status
- Prerequisites
- Build/test commands
- Minimal API example
- Benchmark commands
- Current limitations and supported platforms

`ARCHITECTURE.md` must include:
- Architecture and data-flow diagrams
- Exact repository tree
- Responsibility of every source, header, test, script, and benchmark file
- Important functions and what each does
- Ownership and error-handling rules
- Data structures and metadata schema
- How the predictor fits into the design
- Build/test flow and phase status

`docs/METRICS.md` must define metric formulas and measurement boundaries.
`docs/EXPERIMENTS.md` must provide a repeatable experiment template and preserve actual results only.
`docs/LIMITATIONS.md` must list unimplemented features and known risks.

## Completion report

At the end of each stage, tell me:
1. What you implemented
2. Files created or modified and each file's purpose
3. Key functions and their responsibilities
4. Exact commands to build and test
5. Actual test results, not guessed results
6. What is not implemented or not verified
7. The next smallest recommended step

Begin by inspecting the repository and implementing only the CPU MVP. Do not jump directly to CUDA, ONNX, a LinUCB bandit, or PyTorch allocator replacement.
