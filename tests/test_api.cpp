#include "tensor_alloc.h"
#include "test_helpers.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    printf("[test_api] Starting public API tests...\n");
    tensor_shutdown(); // Start clean

    // 1. Basic allocation and free
    int64_t shape[2] = {32, 64}; // 2048 float32 elements = 8192 bytes
    tensor_alloc_request_t req{};
    req.shape = shape;
    req.ndim = 2;
    req.strides = nullptr;
    req.dtype = TENSOR_DTYPE_FLOAT32;
    req.device = TENSOR_DEVICE_CPU;
    req.lifetime_hint = TENSOR_LIFETIME_TEMPORARY;
    req.requested_bytes = 8192;

    void* ptr = nullptr;
    tensor_status_t status = tensor_malloc(&req, &ptr);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr != nullptr);
    TEST_ASSERT(reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALLOC_ALIGNMENT == 0);

    // Verify writeable
    memset(ptr, 0x42, req.requested_bytes);

    // Check metrics
    tensor_metrics_snapshot_t metrics = tensor_get_metrics();
    TEST_ASSERT(metrics.alloc_count == 1);
    TEST_ASSERT(metrics.free_count == 0);
    TEST_ASSERT(metrics.current_requested_bytes == 8192);
    TEST_ASSERT(metrics.current_reserved_bytes >= 8192);

    // Free the allocation
    status = tensor_free(ptr);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);

    metrics = tensor_get_metrics();
    TEST_ASSERT(metrics.alloc_count == 1);
    TEST_ASSERT(metrics.free_count == 1);
    TEST_ASSERT(metrics.current_requested_bytes == 0);
    TEST_ASSERT(metrics.current_reserved_bytes == 0);

    // 2. free(NULL) behavior
    TEST_ASSERT(tensor_free(nullptr) == TENSOR_STATUS_SUCCESS);

    // 3. Multiple simultaneous allocations
    const int N = 50;
    std::vector<void*> ptrs(N, nullptr);
    for (int i = 0; i < N; ++i) {
        req.requested_bytes = (i + 1) * 128;
        req.ndim = 0;
        req.shape = nullptr;
        status = tensor_malloc(&req, &ptrs[i]);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
        TEST_ASSERT(ptrs[i] != nullptr);
        memset(ptrs[i], i & 0xFF, req.requested_bytes);
    }

    metrics = tensor_get_metrics();
    TEST_ASSERT(metrics.alloc_count == 1 + N);
    TEST_ASSERT(metrics.free_count == 1);
    TEST_ASSERT(metrics.current_requested_bytes > 0);

    // Free all
    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(tensor_free(ptrs[i]) == TENSOR_STATUS_SUCCESS);
    }

    metrics = tensor_get_metrics();
    TEST_ASSERT(metrics.current_requested_bytes == 0);
    TEST_ASSERT(metrics.current_reserved_bytes == 0);

    // 4. Clean shutdown
    tensor_shutdown();
    metrics = tensor_get_metrics();
    TEST_ASSERT(metrics.alloc_count == 0);
    TEST_ASSERT(metrics.free_count == 0);

    printf("[test_api] All public API tests passed!\n");
    return 0;
}
