#include "tensor_alloc.h"
#include "request_validation.h"
#include "size_class.h"
#include "predictor_interface.h"
#include "cpu_backend.h"
#include "allocation_registry.h"
#include "metrics.h"
#include <atomic>

static std::atomic<uint64_t> g_sequence_id(1);

const char* tensor_status_str(tensor_status_t status) {
    switch (status) {
        case TENSOR_STATUS_SUCCESS:
            return "SUCCESS";
        case TENSOR_STATUS_INVALID_ARGUMENT:
            return "INVALID_ARGUMENT: Malformed metadata, null pointer, or zero byte request";
        case TENSOR_STATUS_UNSUPPORTED_DEVICE:
            return "UNSUPPORTED_DEVICE: Only TENSOR_DEVICE_CPU is supported in current milestone";
        case TENSOR_STATUS_UNSUPPORTED_DTYPE:
            return "UNSUPPORTED_DTYPE: Unknown or unhandled tensor data type";
        case TENSOR_STATUS_OVERFLOW:
            return "OVERFLOW: Integer overflow occurred during shape or stride calculation";
        case TENSOR_STATUS_OUT_OF_MEMORY:
            return "OUT_OF_MEMORY: Backend allocation failed to obtain memory";
        case TENSOR_STATUS_NOT_FOUND:
            return "NOT_FOUND: Pointer not recognized as an active allocation";
        case TENSOR_STATUS_DOUBLE_FREE:
            return "DOUBLE_FREE: Pointer was previously deallocated";
        default:
            return "UNKNOWN_STATUS";
    }
}

tensor_status_t tensor_malloc(const tensor_alloc_request_t* req, void** out_ptr) {
    // 1. Validate request
    tensor_status_t status = validate_tensor_request(req, out_ptr);
    if (status != TENSOR_STATUS_SUCCESS) {
        if (out_ptr) *out_ptr = nullptr;
        metrics_record_alloc_failure();
        return status;
    }

    // 2. Select size class (predictor with safe deterministic fallback)
    bool used_fallback = false;
    int size_class = predict_size_class_safe(req, &used_fallback);

    // 3. Obtain aligned memory from backend pool
    void* mem = nullptr;
    size_t capacity = 0;
    bool reused = false;
    status = cpu_backend_alloc(size_class, req->requested_bytes, &mem, &capacity, &reused);
    if (status != TENSOR_STATUS_SUCCESS) {
        metrics_record_alloc_failure();
        *out_ptr = nullptr;
        return status;
    }

    // 4. Register live allocation ownership
    allocation_record_t record;
    record.ptr = mem;
    record.requested_bytes = req->requested_bytes;
    record.reserved_bytes = capacity;
    record.size_class = size_class;
    record.device = req->device;
    record.is_large = (size_class == TENSOR_SIZE_CLASS_LARGE);
    record.sequence_id = g_sequence_id.fetch_add(1, std::memory_order_relaxed);

    if (!registry_insert(&record)) {
        // Double registration error: reclaim and report OOM
        cpu_backend_free(mem, size_class, capacity);
        metrics_record_alloc_failure();
        *out_ptr = nullptr;
        return TENSOR_STATUS_OUT_OF_MEMORY;
    }

    // 5. Update metrics
    metrics_record_alloc(req->requested_bytes, capacity, reused, used_fallback);

    // 6. Return aligned memory pointer
    *out_ptr = mem;
    return TENSOR_STATUS_SUCCESS;
}

tensor_status_t tensor_free(void* ptr) {
    // Freeing NULL is a safe no-op per contract
    if (!ptr) {
        return TENSOR_STATUS_SUCCESS;
    }

    // 1. Remove from registry and check ownership
    allocation_record_t record{};
    tensor_status_t status = registry_remove(ptr, &record);
    if (status != TENSOR_STATUS_SUCCESS) {
        // Returns TENSOR_STATUS_DOUBLE_FREE or TENSOR_STATUS_NOT_FOUND
        return status;
    }

    // 2. Return block to backend pool
    status = cpu_backend_free(ptr, record.size_class, record.reserved_bytes);

    // 3. Record metrics
    metrics_record_free(record.requested_bytes, record.reserved_bytes);

    return status;
}

tensor_metrics_snapshot_t tensor_get_metrics(void) {
    return metrics_get_snapshot();
}

void tensor_reset_metrics(void) {
    metrics_reset();
}

void tensor_print_metrics(FILE* stream) {
    metrics_print(stream);
}

void tensor_shutdown(void) {
    cpu_backend_destroy();
    registry_clear();
    metrics_reset();
}
