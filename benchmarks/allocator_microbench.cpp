#include "tensor_alloc.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult {
    const char* workload_name;
    const char* allocator_name;
    size_t total_ops;
    double elapsed_ms;
    double ops_per_sec;
    double p50_latency_ns;
    double p99_latency_ns;
    size_t peak_reserved_bytes;
    size_t peak_requested_bytes;
    double internal_waste_pct;
    double reuse_rate_pct;
};

static void print_result_row(const BenchmarkResult& r) {
    printf("| %-22s | %-13s | %8.2f ms | %10.0f ops/s | %7.1f ns | %7.1f ns | %8.2f%% | %7.1f%% |\n",
           r.workload_name,
           r.allocator_name,
           r.elapsed_ms,
           r.ops_per_sec,
           r.p50_latency_ns,
           r.p99_latency_ns,
           r.internal_waste_pct,
           r.reuse_rate_pct);
}

// 1. Repeated Same Size Workload (e.g. 50,000 iterations of 4096 bytes)
static void run_same_size_bench(int iterations, size_t size_bytes, std::vector<BenchmarkResult>& results) {
    const char* workload = "Repeated Same Size (4KB)";

    // System malloc / free
    {
        std::vector<double> latencies_ns;
        latencies_ns.reserve(iterations);

        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto t0 = Clock::now();
            void* p = malloc(size_bytes);
            auto t1 = Clock::now();
            latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            free(p);
        }
        auto end = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double p50 = latencies_ns[latencies_ns.size() / 2];
        double p99 = latencies_ns[static_cast<size_t>(latencies_ns.size() * 0.99)];

        BenchmarkResult res{};
        res.workload_name = workload;
        res.allocator_name = "sys_malloc";
        res.total_ops = iterations * 2;
        res.elapsed_ms = elapsed_ms;
        res.ops_per_sec = (iterations * 2.0) / (elapsed_ms / 1000.0);
        res.p50_latency_ns = p50;
        res.p99_latency_ns = p99;
        res.peak_requested_bytes = size_bytes;
        res.peak_reserved_bytes = size_bytes;
        res.internal_waste_pct = 0.0;
        res.reuse_rate_pct = 0.0;
        results.push_back(res);
    }

    // tensor_alloc
    {
        tensor_shutdown();
        tensor_alloc_request_t req{};
        req.device = TENSOR_DEVICE_CPU;
        req.dtype = TENSOR_DTYPE_FLOAT32;
        req.requested_bytes = size_bytes;

        std::vector<double> latencies_ns;
        latencies_ns.reserve(iterations);

        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto t0 = Clock::now();
            void* p = nullptr;
            tensor_malloc(&req, &p);
            auto t1 = Clock::now();
            latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            tensor_free(p);
        }
        auto end = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double p50 = latencies_ns[latencies_ns.size() / 2];
        double p99 = latencies_ns[static_cast<size_t>(latencies_ns.size() * 0.99)];

        tensor_metrics_snapshot_t m = tensor_get_metrics();

        BenchmarkResult res{};
        res.workload_name = workload;
        res.allocator_name = "tensor_alloc";
        res.total_ops = iterations * 2;
        res.elapsed_ms = elapsed_ms;
        res.ops_per_sec = (iterations * 2.0) / (elapsed_ms / 1000.0);
        res.p50_latency_ns = p50;
        res.p99_latency_ns = p99;
        res.peak_requested_bytes = m.peak_requested_bytes;
        res.peak_reserved_bytes = m.peak_reserved_bytes;
        res.internal_waste_pct = m.internal_waste_ratio * 100.0;
        res.reuse_rate_pct = m.reuse_rate * 100.0;
        results.push_back(res);
        tensor_shutdown();
    }
}

// 2. Mixed Sizes Workload
static void run_mixed_sizes_bench(int iterations, std::vector<BenchmarkResult>& results) {
    const char* workload = "Mixed Sizes (64B-1MB)";
    const size_t sizes[] = {64, 250, 1000, 4000, 16384, 65000, 262144, 1048576};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // System malloc
    {
        std::vector<double> latencies_ns;
        latencies_ns.reserve(iterations);

        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            size_t sz = sizes[i % num_sizes];
            auto t0 = Clock::now();
            void* p = malloc(sz);
            auto t1 = Clock::now();
            latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            free(p);
        }
        auto end = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double p50 = latencies_ns[latencies_ns.size() / 2];
        double p99 = latencies_ns[static_cast<size_t>(latencies_ns.size() * 0.99)];

        BenchmarkResult res{};
        res.workload_name = workload;
        res.allocator_name = "sys_malloc";
        res.total_ops = iterations * 2;
        res.elapsed_ms = elapsed_ms;
        res.ops_per_sec = (iterations * 2.0) / (elapsed_ms / 1000.0);
        res.p50_latency_ns = p50;
        res.p99_latency_ns = p99;
        res.internal_waste_pct = 0.0;
        res.reuse_rate_pct = 0.0;
        results.push_back(res);
    }

    // tensor_alloc
    {
        tensor_shutdown();
        tensor_alloc_request_t req{};
        req.device = TENSOR_DEVICE_CPU;
        req.dtype = TENSOR_DTYPE_FLOAT32;

        std::vector<double> latencies_ns;
        latencies_ns.reserve(iterations);

        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            size_t sz = sizes[i % num_sizes];
            req.requested_bytes = sz;
            auto t0 = Clock::now();
            void* p = nullptr;
            tensor_malloc(&req, &p);
            auto t1 = Clock::now();
            latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            tensor_free(p);
        }
        auto end = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double p50 = latencies_ns[latencies_ns.size() / 2];
        double p99 = latencies_ns[static_cast<size_t>(latencies_ns.size() * 0.99)];

        tensor_metrics_snapshot_t m = tensor_get_metrics();

        BenchmarkResult res{};
        res.workload_name = workload;
        res.allocator_name = "tensor_alloc";
        res.total_ops = iterations * 2;
        res.elapsed_ms = elapsed_ms;
        res.ops_per_sec = (iterations * 2.0) / (elapsed_ms / 1000.0);
        res.p50_latency_ns = p50;
        res.p99_latency_ns = p99;
        res.internal_waste_pct = m.internal_waste_ratio * 100.0;
        res.reuse_rate_pct = m.reuse_rate * 100.0;
        results.push_back(res);
        tensor_shutdown();
    }
}

// 3. Bursty Allocations (Alloc 100, Free 100, repeated 200 times)
static void run_bursty_bench(int rounds, int batch_size, std::vector<BenchmarkResult>& results) {
    const char* workload = "Bursty (Batch 100)";
    size_t total_ops = rounds * batch_size * 2;

    // System malloc
    {
        std::vector<void*> batch(batch_size, nullptr);
        std::vector<double> latencies_ns;
        latencies_ns.reserve(rounds * batch_size);

        auto start = Clock::now();
        for (int r = 0; r < rounds; ++r) {
            for (int b = 0; b < batch_size; ++b) {
                size_t sz = ((b % 8) + 1) * 1024;
                auto t0 = Clock::now();
                batch[b] = malloc(sz);
                auto t1 = Clock::now();
                latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            for (int b = 0; b < batch_size; ++b) {
                free(batch[b]);
            }
        }
        auto end = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double p50 = latencies_ns[latencies_ns.size() / 2];
        double p99 = latencies_ns[static_cast<size_t>(latencies_ns.size() * 0.99)];

        BenchmarkResult res{};
        res.workload_name = workload;
        res.allocator_name = "sys_malloc";
        res.total_ops = total_ops;
        res.elapsed_ms = elapsed_ms;
        res.ops_per_sec = static_cast<double>(total_ops) / (elapsed_ms / 1000.0);
        res.p50_latency_ns = p50;
        res.p99_latency_ns = p99;
        res.internal_waste_pct = 0.0;
        res.reuse_rate_pct = 0.0;
        results.push_back(res);
    }

    // tensor_alloc
    {
        tensor_shutdown();
        tensor_alloc_request_t req{};
        req.device = TENSOR_DEVICE_CPU;
        req.dtype = TENSOR_DTYPE_FLOAT32;

        std::vector<void*> batch(batch_size, nullptr);
        std::vector<double> latencies_ns;
        latencies_ns.reserve(rounds * batch_size);

        auto start = Clock::now();
        for (int r = 0; r < rounds; ++r) {
            for (int b = 0; b < batch_size; ++b) {
                size_t sz = ((b % 8) + 1) * 1024;
                req.requested_bytes = sz;
                auto t0 = Clock::now();
                tensor_malloc(&req, &batch[b]);
                auto t1 = Clock::now();
                latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            for (int b = 0; b < batch_size; ++b) {
                tensor_free(batch[b]);
            }
        }
        auto end = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double p50 = latencies_ns[latencies_ns.size() / 2];
        double p99 = latencies_ns[static_cast<size_t>(latencies_ns.size() * 0.99)];

        tensor_metrics_snapshot_t m = tensor_get_metrics();

        BenchmarkResult res{};
        res.workload_name = workload;
        res.allocator_name = "tensor_alloc";
        res.total_ops = total_ops;
        res.elapsed_ms = elapsed_ms;
        res.ops_per_sec = static_cast<double>(total_ops) / (elapsed_ms / 1000.0);
        res.p50_latency_ns = p50;
        res.p99_latency_ns = p99;
        res.internal_waste_pct = m.internal_waste_ratio * 100.0;
        res.reuse_rate_pct = m.reuse_rate * 100.0;
        results.push_back(res);
        tensor_shutdown();
    }
}

int main() {
    printf("========================================================================================================\n");
    printf("                           tensor_alloc Comparative Microbenchmark Harness                             \n");
    printf("========================================================================================================\n");
    printf("OS: macOS (Darwin arm64) | Compiler: AppleClang 17.0.0 | Allocation Alignment: %d bytes\n", TENSOR_ALLOC_ALIGNMENT);
    printf("Size Classes: 21 (64 B -> 64 MB) | Large Allocation Fallback: Aligned posix_memalign\n");
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("| Workload               | Allocator     |    Elapsed |   Throughput | Latency p50 | Latency p99 | Waste %%   | Reuse %%  |\n");
    printf("|------------------------|---------------|------------|--------------|-------------|-------------|-----------|----------|\n");

    std::vector<BenchmarkResult> results;

    // Run workloads
    run_same_size_bench(50000, 4096, results);
    run_mixed_sizes_bench(30000, results);
    run_bursty_bench(250, 100, results);

    for (const auto& r : results) {
        print_result_row(r);
    }

    printf("========================================================================================================\n");
    printf("\nJSON Summary:\n[\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        printf("  {\n");
        printf("    \"workload\": \"%s\",\n", r.workload_name);
        printf("    \"allocator\": \"%s\",\n", r.allocator_name);
        printf("    \"total_ops\": %zu,\n", r.total_ops);
        printf("    \"elapsed_ms\": %.2f,\n", r.elapsed_ms);
        printf("    \"ops_per_sec\": %.2f,\n", r.ops_per_sec);
        printf("    \"p50_latency_ns\": %.1f,\n", r.p50_latency_ns);
        printf("    \"p99_latency_ns\": %.1f,\n", r.p99_latency_ns);
        printf("    \"internal_waste_pct\": %.2f,\n", r.internal_waste_pct);
        printf("    \"reuse_rate_pct\": %.2f\n", r.reuse_rate_pct);
        printf("  }%s\n", (i + 1 == results.size()) ? "" : ",");
    }
    printf("]\n");

    return 0;
}
