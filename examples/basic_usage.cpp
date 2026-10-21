#include "tensor_alloc.h"
#include <cstdio>
#include <cstdlib>

int main() {
    printf("=== tensor_alloc: Basic Usage Example ===\n\n");

    // 1. Prepare tensor metadata request: 3D float32 tensor of shape [16, 32, 64]
    int64_t shape[3] = {16, 32, 64};
    size_t num_elements = 16 * 32 * 64; // 32,768 elements
    size_t byte_count = num_elements * sizeof(float); // 131,072 bytes (128 KB)

    tensor_alloc_request_t req{};
    req.shape = shape;
    req.ndim = 3;
    req.strides = nullptr; // Standard C-contiguous
    req.dtype = TENSOR_DTYPE_FLOAT32;
    req.device = TENSOR_DEVICE_CPU;
    req.lifetime_hint = TENSOR_LIFETIME_TEMPORARY;
    req.requested_bytes = byte_count;

    printf("Requesting allocation for 3D tensor: shape [16, 32, 64], float32 (%zu bytes)...\n", byte_count);

    // 2. Allocate memory via public C API
    void* tensor_data = nullptr;
    tensor_status_t status = tensor_malloc(&req, &tensor_data);

    if (status != TENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "Error allocating tensor: %s\n", tensor_status_str(status));
        return 1;
    }

    printf("Successfully allocated pointer: %p\n", tensor_data);
    printf("Alignment check (multiple of 64): %s (address %% 64 = %zu)\n",
           ((uintptr_t)tensor_data % TENSOR_ALLOC_ALIGNMENT == 0) ? "PASS" : "FAIL",
           (uintptr_t)tensor_data % TENSOR_ALLOC_ALIGNMENT);

    // 3. Use memory
    float* float_ptr = static_cast<float*>(tensor_data);
    for (size_t i = 0; i < num_elements; ++i) {
        float_ptr[i] = 1.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        sum += float_ptr[i];
    }
    printf("Verification sum across %zu elements: %.1f (expected: %.1f)\n\n",
           num_elements, sum, (float)num_elements);

    // 4. Inspect allocator metrics
    printf("Metrics after first allocation:\n");
    tensor_print_metrics(stdout);

    // 5. Free tensor
    printf("\nFreeing tensor pointer...\n");
    status = tensor_free(tensor_data);
    if (status != TENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "Error freeing tensor: %s\n", tensor_status_str(status));
        return 1;
    }

    // 6. Demonstrate pool reuse: allocate another tensor of same size
    printf("Allocating second tensor of identical size to demonstrate reuse...\n");
    void* second_ptr = nullptr;
    tensor_malloc(&req, &second_ptr);
    printf("Second allocated pointer: %p (reused identical block: %s)\n\n",
           second_ptr, (second_ptr == tensor_data) ? "YES" : "NO");

    printf("Metrics after second allocation:\n");
    tensor_print_metrics(stdout);

    // Clean up
    tensor_free(second_ptr);
    tensor_shutdown();

    printf("\nExample completed successfully!\n");
    return 0;
}
