#include "metrics.h"
#include <mutex>
#include <atomic>

static std::atomic<bool> g_metrics_enabled(true);
static std::mutex g_metrics_mutex;

static uint64_t g_alloc_count = 0;
static uint64_t g_free_count = 0;
static uint64_t g_alloc_failure_count = 0;
static size_t g_current_requested = 0;
static size_t g_peak_requested = 0;
static size_t g_current_reserved = 0;
static size_t g_peak_reserved = 0;
static uint64_t g_reuse_count = 0;
static uint64_t g_fallback_count = 0;

void metrics_set_enabled(bool enabled) {
    g_metrics_enabled.store(enabled, std::memory_order_release);
}

bool metrics_is_enabled(void) {
    return g_metrics_enabled.load(std::memory_order_acquire);
}

void metrics_record_alloc(size_t requested_bytes,
                          size_t reserved_bytes,
                          bool reused,
                          bool fallback) {
    if (!g_metrics_enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    g_alloc_count++;
    if (reused) {
        g_reuse_count++;
    }
    if (fallback) {
        g_fallback_count++;
    }

    g_current_requested += requested_bytes;
    if (g_current_requested > g_peak_requested) {
        g_peak_requested = g_current_requested;
    }

    g_current_reserved += reserved_bytes;
    if (g_current_reserved > g_peak_reserved) {
        g_peak_reserved = g_current_reserved;
    }
}

void metrics_record_free(size_t requested_bytes, size_t reserved_bytes) {
    if (!g_metrics_enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    g_free_count++;

    if (g_current_requested >= requested_bytes) {
        g_current_requested -= requested_bytes;
    } else {
        g_current_requested = 0;
    }

    if (g_current_reserved >= reserved_bytes) {
        g_current_reserved -= reserved_bytes;
    } else {
        g_current_reserved = 0;
    }
}

void metrics_record_alloc_failure(void) {
    if (!g_metrics_enabled.load(std::memory_order_relaxed)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    g_alloc_failure_count++;
}

void metrics_reset(void) {
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    g_alloc_count = 0;
    g_free_count = 0;
    g_alloc_failure_count = 0;
    g_current_requested = 0;
    g_peak_requested = 0;
    g_current_reserved = 0;
    g_peak_reserved = 0;
    g_reuse_count = 0;
    g_fallback_count = 0;
}

tensor_metrics_snapshot_t metrics_get_snapshot(void) {
    std::lock_guard<std::mutex> lock(g_metrics_mutex);
    tensor_metrics_snapshot_t snap;
    snap.alloc_count = g_alloc_count;
    snap.free_count = g_free_count;
    snap.alloc_failure_count = g_alloc_failure_count;
    snap.current_requested_bytes = g_current_requested;
    snap.peak_requested_bytes = g_peak_requested;
    snap.current_reserved_bytes = g_current_reserved;
    snap.peak_reserved_bytes = g_peak_reserved;
    snap.reuse_count = g_reuse_count;
    snap.fallback_count = g_fallback_count;

    if (g_current_reserved > 0 && g_current_reserved >= g_current_requested) {
        snap.internal_waste_ratio = static_cast<double>(g_current_reserved - g_current_requested) /
                                    static_cast<double>(g_current_reserved);
    } else {
        snap.internal_waste_ratio = 0.0;
    }

    if (g_alloc_count > 0) {
        snap.reuse_rate = static_cast<double>(g_reuse_count) / static_cast<double>(g_alloc_count);
    } else {
        snap.reuse_rate = 0.0;
    }

    return snap;
}

void metrics_print(FILE* stream) {
    if (!stream) stream = stdout;
    tensor_metrics_snapshot_t s = metrics_get_snapshot();
    fprintf(stream, "--- tensor_alloc Metrics Summary ---\n");
    fprintf(stream, "Allocations:          %llu\n", (unsigned long long)s.alloc_count);
    fprintf(stream, "Frees:                %llu\n", (unsigned long long)s.free_count);
    fprintf(stream, "Failures:             %llu\n", (unsigned long long)s.alloc_failure_count);
    fprintf(stream, "Reuses:               %llu (%.2f%%)\n", (unsigned long long)s.reuse_count, s.reuse_rate * 100.0);
    fprintf(stream, "Predictor Fallbacks:  %llu\n", (unsigned long long)s.fallback_count);
    fprintf(stream, "Current Requested:    %zu bytes\n", s.current_requested_bytes);
    fprintf(stream, "Current Reserved:     %zu bytes\n", s.current_reserved_bytes);
    fprintf(stream, "Peak Requested:       %zu bytes\n", s.peak_requested_bytes);
    fprintf(stream, "Peak Reserved:        %zu bytes\n", s.peak_reserved_bytes);
    fprintf(stream, "Internal Waste Ratio: %.4f (%.2f%%)\n", s.internal_waste_ratio, s.internal_waste_ratio * 100.0);
    fprintf(stream, "------------------------------------\n");
}
