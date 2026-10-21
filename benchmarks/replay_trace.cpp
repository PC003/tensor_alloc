#include "tensor_alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <chrono>

struct ParsedTraceEvent {
    uint64_t req_id;
    bool is_alloc;
    size_t requested_bytes;
    tensor_dtype_t dtype;
};

static tensor_dtype_t parse_dtype(const std::string& str) {
    if (str.find("float32") != std::string::npos) return TENSOR_DTYPE_FLOAT32;
    if (str.find("float64") != std::string::npos) return TENSOR_DTYPE_FLOAT64;
    if (str.find("float16") != std::string::npos) return TENSOR_DTYPE_FLOAT16;
    if (str.find("bfloat16") != std::string::npos) return TENSOR_DTYPE_BFLOAT16;
    if (str.find("int32") != std::string::npos) return TENSOR_DTYPE_INT32;
    if (str.find("int64") != std::string::npos) return TENSOR_DTYPE_INT64;
    if (str.find("int8") != std::string::npos) return TENSOR_DTYPE_INT8;
    if (str.find("uint8") != std::string::npos) return TENSOR_DTYPE_UINT8;
    if (str.find("bool") != std::string::npos) return TENSOR_DTYPE_BOOL;
    return TENSOR_DTYPE_FLOAT32;
}

static bool extract_field(const std::string& line, const std::string& key, std::string* out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = line.find(pattern);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size()) return false;
    if (line[pos] == '"') {
        pos++;
        size_t end = line.find('"', pos);
        if (end == std::string::npos) return false;
        if (out) *out = line.substr(pos, end - pos);
    } else {
        size_t end = pos;
        while (end < line.size() && line[end] != ',' && line[end] != '}' && line[end] != ' ' && line[end] != '\r' && line[end] != '\n') {
            end++;
        }
        if (out) *out = line.substr(pos, end - pos);
    }
    return true;
}

// Parse a JSONL trace file into a vector of TraceEvents
static std::vector<ParsedTraceEvent> load_trace_file(const char* filepath) {
    std::vector<ParsedTraceEvent> events;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open trace file: %s\n", filepath);
        return events;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.find("\"schema_version\"") != std::string::npos) {
            continue;
        }

        std::string ev_type;
        if (!extract_field(line, "event_type", &ev_type)) {
            continue;
        }

        std::string req_id_str;
        if (!extract_field(line, "request_id", &req_id_str)) {
            continue;
        }
        uint64_t req_id = std::stoull(req_id_str);

        if (ev_type == "alloc") {
            std::string bytes_str;
            size_t bytes = 0;
            if (extract_field(line, "requested_bytes", &bytes_str)) {
                bytes = std::stoull(bytes_str);
            }

            std::string dtype_str;
            tensor_dtype_t dtype = TENSOR_DTYPE_FLOAT32;
            if (extract_field(line, "dtype", &dtype_str)) {
                dtype = parse_dtype(dtype_str);
            }

            if (bytes > 0) {
                events.push_back({req_id, true, bytes, dtype});
            }
        } else if (ev_type == "free") {
            events.push_back({req_id, false, 0, TENSOR_DTYPE_FLOAT32});
        }
    }

    return events;
}

// Generate reproducible synthetic events if no trace file is passed
static std::vector<ParsedTraceEvent> generate_fallback_trace(size_t num_events) {
    std::vector<ParsedTraceEvent> events;
    events.reserve(num_events);

    std::vector<uint64_t> live_ids;
    uint64_t next_id = 1;

    for (size_t i = 0; i < num_events; ++i) {
        bool do_alloc = live_ids.empty() || ((rand() % 100) < 60);

        if (do_alloc) {
            uint64_t id = next_id++;
            size_t bytes = ((rand() % 32) + 1) * 1024;
            events.push_back({id, true, bytes, TENSOR_DTYPE_FLOAT32});
            live_ids.push_back(id);
        } else {
            size_t idx = rand() % live_ids.size();
            uint64_t id = live_ids[idx];
            live_ids.erase(live_ids.begin() + idx);
            events.push_back({id, false, 0, TENSOR_DTYPE_FLOAT32});
        }
    }

    for (uint64_t id : live_ids) {
        events.push_back({id, false, 0, TENSOR_DTYPE_FLOAT32});
    }

    return events;
}

int main(int argc, char** argv) {
    printf("====================================================================================\n");
    printf("                    tensor_alloc Trace Replay & Benchmark Harness                   \n");
    printf("====================================================================================\n");

    std::vector<ParsedTraceEvent> trace;
    const char* trace_source = "synthetic fallback";

    if (argc > 1 && argv[1][0] != '-') {
        trace = load_trace_file(argv[1]);
        if (!trace.empty()) {
            trace_source = argv[1];
        }
    }

    if (trace.empty()) {
        srand(42);
        trace = generate_fallback_trace(30000);
    }

    printf("Loaded trace from: %s (%zu events)\n", trace_source, trace.size());
    printf("------------------------------------------------------------------------------------\n");

    // 1. Replay using sys_malloc
    double sys_elapsed_ms = 0.0;
    double sys_ops_sec = 0.0;
    {
        std::unordered_map<uint64_t, void*> active_allocs;
        active_allocs.reserve(4096);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto& ev : trace) {
            if (ev.is_alloc) {
                void* p = malloc(ev.requested_bytes);
                active_allocs[ev.req_id] = p;
            } else {
                auto it = active_allocs.find(ev.req_id);
                if (it != active_allocs.end()) {
                    free(it->second);
                    active_allocs.erase(it);
                }
            }
        }
        for (auto& kv : active_allocs) {
            free(kv.second);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        sys_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sys_ops_sec = static_cast<double>(trace.size()) / (sys_elapsed_ms / 1000.0);
    }

    // 2. Replay using tensor_alloc
    double tensor_elapsed_ms = 0.0;
    double tensor_ops_sec = 0.0;
    tensor_metrics_snapshot_t m{};
    {
        tensor_shutdown();
        std::unordered_map<uint64_t, void*> active_allocs;
        active_allocs.reserve(4096);

        tensor_alloc_request_t req{};
        req.device = TENSOR_DEVICE_CPU;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto& ev : trace) {
            if (ev.is_alloc) {
                req.dtype = ev.dtype;
                req.requested_bytes = ev.requested_bytes;
                void* p = nullptr;
                tensor_status_t status = tensor_malloc(&req, &p);
                if (status == TENSOR_STATUS_SUCCESS) {
                    active_allocs[ev.req_id] = p;
                }
            } else {
                auto it = active_allocs.find(ev.req_id);
                if (it != active_allocs.end()) {
                    tensor_free(it->second);
                    active_allocs.erase(it);
                }
            }
        }
        for (auto& kv : active_allocs) {
            tensor_free(kv.second);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        tensor_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        tensor_ops_sec = static_cast<double>(trace.size()) / (tensor_elapsed_ms / 1000.0);
        m = tensor_get_metrics();
        tensor_shutdown();
    }

    printf("| Allocator     | Replay Elapsed |     Throughput | Peak Live Req | Peak Live Res | Pool Reuse %% |\n");
    printf("|---------------|----------------|----------------|---------------|---------------|--------------|\n");
    printf("| sys_malloc    |    %8.2f ms | %11.0f ops/s |             - |             - |         0.0%% |\n",
           sys_elapsed_ms, sys_ops_sec);
    printf("| tensor_alloc  |    %8.2f ms | %11.0f ops/s |  %9zu B |  %9zu B |       %5.1f%% |\n",
           tensor_elapsed_ms, tensor_ops_sec, m.peak_requested_bytes, m.peak_reserved_bytes, m.reuse_rate * 100.0);
    printf("------------------------------------------------------------------------------------\n");
    printf("Pool Reuse Count: %llu out of %llu allocations (%.2f%% reuse rate)\n",
           (unsigned long long)m.reuse_count, (unsigned long long)m.alloc_count, m.reuse_rate * 100.0);
    printf("Predictor Fallbacks: %llu\n", (unsigned long long)m.fallback_count);
    printf("====================================================================================\n");

    return 0;
}
