#include "tensor_alloc.h"
#include "test_helpers.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    printf("[test_reuse] Starting pool reuse tests...\n");
    tensor_shutdown();

    tensor_alloc_request_t req{};
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_FLOAT32;
    req.ndim = 0;
    req.shape = nullptr;
    req.strides = nullptr;
    req.requested_bytes = 4096;

    // 1. First allocation: should be fresh from OS
    void* ptr1 = nullptr;
    tensor_status_t status = tensor_malloc(&req, &ptr1);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr1 != nullptr);
    memset(ptr1, 0x11, req.requested_bytes);

    tensor_metrics_snapshot_t s1 = tensor_get_metrics();
    TEST_ASSERT(s1.alloc_count == 1);
    TEST_ASSERT(s1.reuse_count == 0);

    // Free back to pool
    status = tensor_free(ptr1);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);

    // 2. Second allocation of identical size: must reuse freed block from pool!
    void* ptr2 = nullptr;
    status = tensor_malloc(&req, &ptr2);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr2 == ptr1); // Address re-used from LIFO free list

    tensor_metrics_snapshot_t s2 = tensor_get_metrics();
    TEST_ASSERT(s2.alloc_count == 2);
    TEST_ASSERT(s2.reuse_count == 1);
    TEST_ASSERT(s2.reuse_rate == 0.5);

    // Overwrite with new data to confirm memory safety
    memset(ptr2, 0x22, req.requested_bytes);
    uint8_t* b2 = static_cast<uint8_t*>(ptr2);
    TEST_ASSERT(b2[0] == 0x22);
    TEST_ASSERT(b2[4095] == 0x22);

    status = tensor_free(ptr2);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);

    // 3. Multi-block batch reuse
    const int BATCH = 20;
    std::vector<void*> batch(BATCH, nullptr);
    for (int i = 0; i < BATCH; ++i) {
        req.requested_bytes = 1024;
        status = tensor_malloc(&req, &batch[i]);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
        TEST_ASSERT(batch[i] != nullptr);
    }

    // Free entire batch
    for (int i = 0; i < BATCH; ++i) {
        status = tensor_free(batch[i]);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    }

    uint64_t prev_reuse = tensor_get_metrics().reuse_count;

    // Re-allocate entire batch: all should be reused
    for (int i = 0; i < BATCH; ++i) {
        status = tensor_malloc(&req, &batch[i]);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
        TEST_ASSERT(batch[i] != nullptr);
    }

    tensor_metrics_snapshot_t s3 = tensor_get_metrics();
    TEST_ASSERT(s3.reuse_count == prev_reuse + BATCH);

    for (int i = 0; i < BATCH; ++i) {
        status = tensor_free(batch[i]);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    }

    tensor_shutdown();
    printf("[test_reuse] All pool reuse tests passed!\n");
    return 0;
}
