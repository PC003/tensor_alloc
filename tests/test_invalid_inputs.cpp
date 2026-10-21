#include "tensor_alloc.h"
#include "test_helpers.h"
#include <cstdio>
#include <cstdlib>

int main() {
    printf("[test_invalid_inputs] Starting invalid input and ownership tests...\n");
    tensor_shutdown();

    tensor_alloc_request_t req{};
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_FLOAT32;
    req.requested_bytes = 2048;

    void* ptr = nullptr;
    tensor_status_t status = tensor_malloc(&req, &ptr);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr != nullptr);

    // 1. Double free test
    TEST_ASSERT(tensor_free(ptr) == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(tensor_free(ptr) == TENSOR_STATUS_DOUBLE_FREE);

    // 2. Unknown pointer test
    int stack_var = 123;
    TEST_ASSERT(tensor_free(&stack_var) == TENSOR_STATUS_NOT_FOUND);

    void* raw_sys_ptr = malloc(256);
    TEST_ASSERT(tensor_free(raw_sys_ptr) == TENSOR_STATUS_NOT_FOUND);
    free(raw_sys_ptr);

    // 3. Free NULL test (must be safe no-op)
    TEST_ASSERT(tensor_free(nullptr) == TENSOR_STATUS_SUCCESS);

    // 4. Zero byte allocation rejection
    req.requested_bytes = 0;
    void* zero_ptr = nullptr;
    TEST_ASSERT(tensor_malloc(&req, &zero_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);
    TEST_ASSERT(zero_ptr == nullptr);

    // 5. CUDA device rejection in CPU MVP
    req.requested_bytes = 1024;
    req.device = TENSOR_DEVICE_CUDA;
    void* cuda_ptr = nullptr;
    TEST_ASSERT(tensor_malloc(&req, &cuda_ptr) == TENSOR_STATUS_UNSUPPORTED_DEVICE);
    TEST_ASSERT(cuda_ptr == nullptr);

    // 6. Unknown dtype rejection
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_UNKNOWN;
    void* bad_dtype_ptr = nullptr;
    TEST_ASSERT(tensor_malloc(&req, &bad_dtype_ptr) == TENSOR_STATUS_UNSUPPORTED_DTYPE);
    TEST_ASSERT(bad_dtype_ptr == nullptr);

    tensor_shutdown();
    printf("[test_invalid_inputs] All invalid input tests passed!\n");
    return 0;
}
