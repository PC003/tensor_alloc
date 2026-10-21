#include "predictor_interface.h"
#include <atomic>

class DeterministicFallbackPredictor : public IPredictor {
public:
    int predict_class(const tensor_alloc_request_t* req) override {
        if (!req) return TENSOR_SIZE_CLASS_INVALID;
        return size_class_lookup(req->requested_bytes);
    }
};

static DeterministicFallbackPredictor g_default_predictor;
static std::atomic<IPredictor*> g_active_predictor(&g_default_predictor);

IPredictor* get_default_predictor(void) {
    return &g_default_predictor;
}

void set_active_predictor(IPredictor* predictor) {
    if (predictor == nullptr) {
        g_active_predictor.store(&g_default_predictor, std::memory_order_release);
    } else {
        g_active_predictor.store(predictor, std::memory_order_release);
    }
}

int predict_size_class_safe(const tensor_alloc_request_t* req, bool* out_used_fallback) {
    if (out_used_fallback) {
        *out_used_fallback = false;
    }

    if (!req || req->requested_bytes == 0) {
        if (out_used_fallback) *out_used_fallback = true;
        return TENSOR_SIZE_CLASS_INVALID;
    }

    // Always compute the deterministic baseline class
    int baseline_class = size_class_lookup(req->requested_bytes);

    IPredictor* predictor = g_active_predictor.load(std::memory_order_acquire);
    if (!predictor || predictor == &g_default_predictor) {
        // Fast path: standard deterministic baseline
        return baseline_class;
    }

    // Attempt predictor invocation
    int recommended = predictor->predict_class(req);

    // Validate recommendation
    if (recommended == TENSOR_SIZE_CLASS_LARGE) {
        // Only acceptable if requested_bytes actually requires large or baseline agrees
        return TENSOR_SIZE_CLASS_LARGE;
    }

    if (recommended >= 0 && recommended < static_cast<int>(size_class_count())) {
        size_t cap = size_class_get_capacity(recommended);
        if (cap >= req->requested_bytes) {
            // Recommendation is safe
            return recommended;
        }
    }

    // If recommendation is invalid or capacity < requested_bytes, fall back safely
    if (out_used_fallback) {
        *out_used_fallback = true;
    }
    return baseline_class;
}
