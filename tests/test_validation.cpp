#include "request_validation.h"
#include "test_helpers.h"
#include <cstdio>
#include <limits.h>

int main() {
    printf("[test_validation] Starting validation tests...\n");

    void* dummy_ptr = nullptr;
    tensor_alloc_request_t req{};
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_FLOAT32;
    req.requested_bytes = 1024;
    req.ndim = 0;
    req.shape = nullptr;
    req.strides = nullptr;

    // 1. Null checks
    TEST_ASSERT(validate_tensor_request(nullptr, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);
    TEST_ASSERT(validate_tensor_request(&req, nullptr) == TENSOR_STATUS_INVALID_ARGUMENT);

    // 2. Zero-byte request
    req.requested_bytes = 0;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);
    req.requested_bytes = 1024;

    // 3. Device validation
    req.device = TENSOR_DEVICE_CUDA;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_UNSUPPORTED_DEVICE);
    req.device = TENSOR_DEVICE_UNKNOWN;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_UNSUPPORTED_DEVICE);
    req.device = TENSOR_DEVICE_CPU;

    // 4. Dtype validation
    req.dtype = TENSOR_DTYPE_UNKNOWN;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_UNSUPPORTED_DTYPE);
    req.dtype = static_cast<tensor_dtype_t>(999);
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_UNSUPPORTED_DTYPE);
    req.dtype = TENSOR_DTYPE_FLOAT32;

    // 5. Rank / Shape validation
    req.ndim = TENSOR_MAX_RANK + 1;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);

    req.ndim = 2;
    req.shape = nullptr; // ndim > 0 but shape is null
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);

    int64_t valid_shape[2] = {16, 16}; // 256 float32 elements = 1024 bytes
    req.shape = valid_shape;
    req.requested_bytes = 1024;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_SUCCESS);

    // Negative dimension
    int64_t neg_shape[2] = {16, -5};
    req.shape = neg_shape;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);
    req.shape = valid_shape;

    // 6. Contiguous requested_bytes check (requested_bytes < min_bytes)
    req.requested_bytes = 512; // Required is 1024 bytes
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);
    req.requested_bytes = 2048; // Generous requested_bytes >= 1024 is valid
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_SUCCESS);

    // 7. Overflow checks
    int64_t overflow_shape[2] = {static_cast<int64_t>(1ULL << 32), static_cast<int64_t>(1ULL << 32)};
    req.shape = overflow_shape;
    req.requested_bytes = 1024;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_OVERFLOW);

    // 8. Strides validation
    req.shape = valid_shape; // {16, 16}
    int64_t strided[2] = {32, 1}; // Dim 0 has stride 32 elements.
    // Max offset = (16 - 1) * 32 + (16 - 1) * 1 = 15 * 32 + 15 = 480 + 15 = 495 elements.
    // Minimum bytes = (495 + 1) * 4 = 1984 bytes.
    req.strides = strided;
    req.requested_bytes = 1000; // Too small for strided span (needs 1984)
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);

    req.requested_bytes = 2000; // Sufficient
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_SUCCESS);

    // Negative stride
    int64_t neg_strides[2] = {-1, 1};
    req.strides = neg_strides;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_INVALID_ARGUMENT);

    // Stride multiplication overflow
    int64_t of_strides[2] = {static_cast<int64_t>(SIZE_MAX / 2), 1};
    req.strides = of_strides;
    TEST_ASSERT(validate_tensor_request(&req, &dummy_ptr) == TENSOR_STATUS_OVERFLOW);

    printf("[test_validation] All validation tests passed!\n");
    return 0;
}
