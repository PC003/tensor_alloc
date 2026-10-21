#include "tensor_alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>

struct TraceEvent {
    uint64_t req_id;
    bool is_alloc;
    size_t bytes;
    tensor_dtype_t dtype;
};

// Generates a reproducible synthetic trace for replay benchmarking
static std::vector<TraceEvent> generate_synthetic_trace(size_t num_events) {
    std::vector<TraceEvent> events;
    events.reserve(num_events);

    std::vector<uint64_t> live_ids;
    uint64_t next_id = 1;

    for (size_t i = 0; i < num_events; ++i) {
        bool do_alloc = live_ids.empty() || ((rand() % 100) < 60); // 60% alloc, 40% free

        if (do_alloc) {
            uint64_t id = next_id++;
            size_t bytes = ((rand() % 32) + 1) * 512; // 512B to 16KB
            events.push_back({id, true, bytes, TENSOR_DTYPE_FLOAT32});
            live_ids.push_back(id);
        } else {
            size_t idx = rand() % live_ids.size();
            uint64_t id = live_ids[idx];
            live_ids.erase(live_ids.begin() + idx);
            events.push_back({id, false, 0, TENSOR_DTYPE_FLOAT32});
        }
    }

    // Free any remaining live allocations
    for (uint64_t id : live_ids) {
        events.push_back({id, false, 0, TENSOR_DTYPE_FLOAT32});
    }

    return events;
}

int main(int argc, char** argv) {
    printf("[replay_trace] Trace Replay Benchmarking Harness\n");

    size_t event_count = 20000;
    if (argc > 1) {
        event_count = static_cast<size_t>(atoi(argv[1]));
        if (event_count == 0) event_count = 20000;
    }

    srand(42); // Deterministic seed
    std::vector<TraceEvent> trace = generate_synthetic_trace(event_count);
    printf("Replaying trace with %zu events...\n", trace.size());

    tensor_shutdown();
    std::unordered_map<uint64_t, void*> active_allocations;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (const auto& ev : trace) {
        if (ev.is_alloc) {
            tensor_alloc_request_t req{};
            req.device = TENSOR_DEVICE_CPU;
            req.dtype = ev.dtype;
            req.requested_bytes = ev.bytes;
            void* ptr = nullptr;
            tensor_status_t status = tensor_malloc(&req, &ptr);
            if (status == TENSOR_STATUS_SUCCESS) {
                active_allocations[ev.req_id] = ptr;
            }
        } else {
            auto it = active_allocations.find(ev.req_id);
            if (it != active_allocations.end()) {
                tensor_free(it->second);
                active_allocations.erase(it);
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("Replay finished in %.2f ms (%.0f ops/sec)\n",
           ms, (trace.size() / (ms / 1000.0)));

    tensor_print_metrics(stdout);
    tensor_shutdown();
    return 0;
}
