#include "tensor_alloc.h"
#include "test_helpers.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>

int main() {
    printf("[test_trace_recorder] Starting runtime trace recorder tests...\n");
    tensor_shutdown();

    const char* trace_path = "test_runtime_trace.jsonl";

    // 1. Initial state
    TEST_ASSERT(!tensor_trace_is_active());

    // 2. Start trace recording
    tensor_status_t status = tensor_trace_start(trace_path);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(tensor_trace_is_active());

    // 3. Perform allocations
    int64_t shape1[2] = {32, 64};
    tensor_alloc_request_t req1{};
    req1.shape = shape1;
    req1.ndim = 2;
    req1.dtype = TENSOR_DTYPE_FLOAT32;
    req1.device = TENSOR_DEVICE_CPU;
    req1.lifetime_hint = TENSOR_LIFETIME_TEMPORARY;
    req1.requested_bytes = 32 * 64 * sizeof(float); // 8192 bytes

    void* ptr1 = nullptr;
    status = tensor_malloc(&req1, &ptr1);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr1 != nullptr);

    int64_t shape2[1] = {1024};
    tensor_alloc_request_t req2{};
    req2.shape = shape2;
    req2.ndim = 1;
    req2.dtype = TENSOR_DTYPE_INT32;
    req2.device = TENSOR_DEVICE_CPU;
    req2.lifetime_hint = TENSOR_LIFETIME_PERSISTENT;
    req2.requested_bytes = 1024 * sizeof(int); // 4096 bytes

    void* ptr2 = nullptr;
    status = tensor_malloc(&req2, &ptr2);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr2 != nullptr);

    // 4. Free allocations
    status = tensor_free(ptr1);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);

    status = tensor_free(ptr2);
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);

    // 5. Stop trace recording
    status = tensor_trace_stop();
    TEST_ASSERT(status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(!tensor_trace_is_active());

    // 6. Verify contents of recorded trace file
    std::ifstream infile(trace_path);
    TEST_ASSERT(infile.is_open());

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    infile.close();

    // Expect 5 lines: Header + 2 Allocs + 2 Frees
    TEST_ASSERT(lines.size() == 5);
    TEST_ASSERT(lines[0].find("\"schema_version\":\"1.0.0\"") != std::string::npos);
    TEST_ASSERT(lines[1].find("\"event_type\":\"alloc\"") != std::string::npos);
    TEST_ASSERT(lines[1].find("\"shape\":[32,64]") != std::string::npos);
    TEST_ASSERT(lines[1].find("\"dtype\":\"float32\"") != std::string::npos);
    TEST_ASSERT(lines[2].find("\"event_type\":\"alloc\"") != std::string::npos);
    TEST_ASSERT(lines[2].find("\"shape\":[1024]") != std::string::npos);
    TEST_ASSERT(lines[2].find("\"dtype\":\"int32\"") != std::string::npos);
    TEST_ASSERT(lines[3].find("\"event_type\":\"free\"") != std::string::npos);
    TEST_ASSERT(lines[4].find("\"event_type\":\"free\"") != std::string::npos);

    // Clean up temporary trace file
    remove(trace_path);
    tensor_shutdown();

    printf("[test_trace_recorder] All runtime trace recorder tests passed!\n");
    return 0;
}
