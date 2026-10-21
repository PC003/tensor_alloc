#include "tensor_alloc.h"
#include "test_helpers.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>

static void worker_func(int thread_id, int iterations) {
    tensor_alloc_request_t req{};
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_INT32;
    req.ndim = 0;
    req.shape = nullptr;
    req.strides = nullptr;

    for (int iter = 0; iter < iterations; ++iter) {
        // Vary sizes across different size classes
        size_t size = ((iter % 16) + 1) * 256;
        req.requested_bytes = size;

        void* ptr = nullptr;
        tensor_status_t status = tensor_malloc(&req, &ptr);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
        TEST_ASSERT(ptr != nullptr);
        TEST_ASSERT(reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALLOC_ALIGNMENT == 0);

        // Write and read pattern
        int* int_arr = static_cast<int*>(ptr);
        size_t num_ints = size / sizeof(int);
        for (size_t k = 0; k < num_ints; ++k) {
            int_arr[k] = (thread_id << 16) | (int)(k & 0xFFFF);
        }
        for (size_t k = 0; k < num_ints; ++k) {
            int expected = (thread_id << 16) | (int)(k & 0xFFFF);
            TEST_ASSERT(int_arr[k] == expected);
        }

        status = tensor_free(ptr);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    }
}

int main() {
    printf("[test_multithreading] Starting concurrent multi-threaded stress test...\n");
    tensor_shutdown();

    const int NUM_THREADS = 8;
    const int ITERS_PER_THREAD = 500;

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker_func, t, ITERS_PER_THREAD);
    }

    for (auto& th : threads) {
        th.join();
    }

    tensor_metrics_snapshot_t metrics = tensor_get_metrics();
    uint64_t expected_total_ops = NUM_THREADS * ITERS_PER_THREAD;
    TEST_ASSERT(metrics.alloc_count == expected_total_ops);
    TEST_ASSERT(metrics.free_count == expected_total_ops);
    TEST_ASSERT(metrics.alloc_failure_count == 0);
    TEST_ASSERT(metrics.current_requested_bytes == 0);
    TEST_ASSERT(metrics.current_reserved_bytes == 0);
    TEST_ASSERT(metrics.reuse_count > 0);

    printf("[test_multithreading] Completed %llu thread-safe operations successfully!\n",
           (unsigned long long)expected_total_ops);

    tensor_shutdown();
    return 0;
}
