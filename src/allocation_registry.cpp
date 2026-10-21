#include "allocation_registry.h"
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <mutex>

static const size_t kMaxFreedHistory = 100000;

static std::mutex g_registry_mutex;
static std::unordered_map<void*, allocation_record_t> g_live_allocations;
static std::unordered_set<void*> g_freed_history;
static std::deque<void*> g_freed_fifo;

void registry_init(void) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_live_allocations.clear();
    g_freed_history.clear();
    g_freed_fifo.clear();
}

bool registry_insert(const allocation_record_t* record) {
    if (!record || !record->ptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_registry_mutex);

    // If pointer was previously freed and is now reused, remove from freed history
    if (g_freed_history.find(record->ptr) != g_freed_history.end()) {
        g_freed_history.erase(record->ptr);
    }

    auto res = g_live_allocations.emplace(record->ptr, *record);
    return res.second;
}

tensor_status_t registry_remove(void* ptr, allocation_record_t* out_record) {
    if (!ptr) {
        return TENSOR_STATUS_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(g_registry_mutex);

    auto it = g_live_allocations.find(ptr);
    if (it != g_live_allocations.end()) {
        if (out_record) {
            *out_record = it->second;
        }
        g_live_allocations.erase(it);

        // Record in freed history
        g_freed_history.insert(ptr);
        g_freed_fifo.push_back(ptr);

        if (g_freed_fifo.size() > kMaxFreedHistory) {
            void* oldest = g_freed_fifo.front();
            g_freed_fifo.pop_front();
            g_freed_history.erase(oldest);
        }

        return TENSOR_STATUS_SUCCESS;
    }

    // Check if pointer is in freed history (double free)
    if (g_freed_history.find(ptr) != g_freed_history.end()) {
        return TENSOR_STATUS_DOUBLE_FREE;
    }

    // Pointer was never registered
    return TENSOR_STATUS_NOT_FOUND;
}

bool registry_is_live(const void* ptr) {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    return g_live_allocations.find(const_cast<void*>(ptr)) != g_live_allocations.end();
}

void registry_clear(void) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_live_allocations.clear();
    g_freed_history.clear();
    g_freed_fifo.clear();
}

size_t registry_live_count(void) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    return g_live_allocations.size();
}
