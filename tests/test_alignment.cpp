#include "tensor_alloc.h"
#include "test_helpers.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    printf("[test_alignment] Starting alignment tests...\n");
    tensor_shutdown();

    // Verify 64-byte alignment across varied request sizes
    size_t test_sizes[] = {
        64, 65, 128, 250, 512, 1000, 4096, 8192, 65536,
        1048576, 2097152, 67108864, // 64 MB (maximum pooled class)
        67108864 + 128,             // Large allocation path
        100 * 1024 * 1024           // 100 MB large allocation path
    };

    size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    std::vector<void*> allocated(num_tests, nullptr);

    tensor_alloc_request_t req{};
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_UINT8;
    req.ndim = 0;
    req.shape = nullptr;
    req.strides = nullptr;

    for (size_t i = 0; i < num_tests; ++i) {
        req.requested_bytes = test_sizes[i];
        tensor_status_t status = tensor_malloc(&req, &allocated[i]);
        TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
        TEST_ASSERT(allocated[i] != nullptr);

        // Strict 64-byte alignment check
        uintptr_t addr = reinterpret_cast<uintptr_t>(allocated[i]);
        TEST_ASSERT(addr % TENSOR_ALLOC_ALIGNMENT == 0);

        // Verify boundary bytes are writable
        uint8_t* byte_ptr = static_cast<uint8_t*>(allocated[i]);
        byte_ptr[0] = 0xAA;
        byte_ptr[test_sizes[i] - 1] = 0x55;
        TEST_ASSERT(byte_ptr[0] == 0xAA);
        TEST_ASSERT(byte_ptr[test_sizes[i] - 1] == 0x55);
    }

    for (size_t i = 0; i < num_tests; ++i) {
        TEST_ASSERT(tensor_free(allocated[i]) == TENSOR_STATUS_SUCCESS);
    }

    tensor_shutdown();
    printf("[test_alignment] All alignment tests passed!\n");
    return 0;
}
