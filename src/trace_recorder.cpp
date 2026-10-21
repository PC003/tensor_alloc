#include "trace_recorder.h"
#include <cstdio>
#include <chrono>
#include <mutex>
#include <atomic>

static std::mutex g_trace_mutex;
static FILE* g_trace_file = nullptr;
static std::chrono::high_resolution_clock::time_point g_trace_start_time;

static const char* dtype_to_str(tensor_dtype_t dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT32: return "float32";
        case TENSOR_DTYPE_FLOAT64: return "float64";
        case TENSOR_DTYPE_FLOAT16: return "float16";
        case TENSOR_DTYPE_BFLOAT16: return "bfloat16";
        case TENSOR_DTYPE_INT32: return "int32";
        case TENSOR_DTYPE_INT64: return "int64";
        case TENSOR_DTYPE_INT16: return "int16";
        case TENSOR_DTYPE_INT8: return "int8";
        case TENSOR_DTYPE_UINT8: return "uint8";
        case TENSOR_DTYPE_BOOL: return "bool";
        case TENSOR_DTYPE_UNKNOWN:
        default: return "unknown";
    }
}

static const char* lifetime_to_str(tensor_lifetime_hint_t hint) {
    switch (hint) {
        case TENSOR_LIFETIME_TEMPORARY: return "temporary";
        case TENSOR_LIFETIME_PERSISTENT: return "persistent";
        case TENSOR_LIFETIME_UNKNOWN:
        default: return "unknown";
    }
}

tensor_status_t trace_recorder_start(const char* filepath) {
    if (!filepath) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(g_trace_mutex);
    if (g_trace_file) {
        fclose(g_trace_file);
        g_trace_file = nullptr;
    }

    g_trace_file = fopen(filepath, "w");
    if (!g_trace_file) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    g_trace_start_time = std::chrono::high_resolution_clock::now();

    // Write trace header
    fprintf(g_trace_file, "{\"schema_version\":\"1.0.0\",\"trace_type\":\"runtime_recorded_workload\"}\n");
    fflush(g_trace_file);

    return TENSOR_STATUS_SUCCESS;
}

tensor_status_t trace_recorder_stop(void) {
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    if (g_trace_file) {
        fflush(g_trace_file);
        fclose(g_trace_file);
        g_trace_file = nullptr;
    }
    return TENSOR_STATUS_SUCCESS;
}

bool trace_recorder_is_active(void) {
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    return (g_trace_file != nullptr);
}

void trace_record_alloc(uint64_t request_id,
                        const tensor_alloc_request_t* req,
                        int size_class,
                        size_t reserved_bytes) {
    if (!req) return;

    std::lock_guard<std::mutex> lock(g_trace_mutex);
    if (!g_trace_file) return;

    auto now = std::chrono::high_resolution_clock::now();
    int64_t ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_trace_start_time).count();

    fprintf(g_trace_file,
            "{\"event_type\":\"alloc\",\"request_id\":%llu,\"ndim\":%zu,\"requested_bytes\":%zu,\"reserved_bytes\":%zu,\"size_class\":%d,\"dtype\":\"%s\",\"device\":\"cpu\",\"lifetime_hint\":\"%s\",\"timestamp_ns\":%lld",
            (unsigned long long)request_id,
            req->ndim,
            req->requested_bytes,
            reserved_bytes,
            size_class,
            dtype_to_str(req->dtype),
            lifetime_to_str(req->lifetime_hint),
            (long long)ts_ns);

    // Write shape array if present
    if (req->ndim > 0 && req->shape) {
        fprintf(g_trace_file, ",\"shape\":[");
        for (size_t i = 0; i < req->ndim; ++i) {
            fprintf(g_trace_file, "%lld%s", (long long)req->shape[i], (i + 1 < req->ndim) ? "," : "");
        }
        fprintf(g_trace_file, "]");
    }

    // Write strides array if present
    if (req->ndim > 0 && req->strides) {
        fprintf(g_trace_file, ",\"strides\":[");
        for (size_t i = 0; i < req->ndim; ++i) {
            fprintf(g_trace_file, "%lld%s", (long long)req->strides[i], (i + 1 < req->ndim) ? "," : "");
        }
        fprintf(g_trace_file, "]");
    }

    fprintf(g_trace_file, "}\n");
}

void trace_record_free(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(g_trace_mutex);
    if (!g_trace_file) return;

    auto now = std::chrono::high_resolution_clock::now();
    int64_t ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_trace_start_time).count();

    fprintf(g_trace_file,
            "{\"event_type\":\"free\",\"request_id\":%llu,\"timestamp_ns\":%lld}\n",
            (unsigned long long)request_id,
            (long long)ts_ns);
}
