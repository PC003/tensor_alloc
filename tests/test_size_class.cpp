#include "size_class.h"
#include "predictor_interface.h"
#include "test_helpers.h"

class MockBadPredictor : public IPredictor {
public:
    int mode = 0;
    int predict_class(const tensor_alloc_request_t*) override {
        if (mode == 0) {
            // Return class 0 (64 bytes) regardless of request size
            return 0;
        } else if (mode == 1) {
            // Return out-of-range class index
            return 999;
        } else {
            // Return negative invalid
            return -5;
        }
    }
};

int main() {
    printf("[test_size_class] Starting size class tests...\n");

    // 1. Zero-byte request
    TEST_ASSERT(size_class_lookup(0) == TENSOR_SIZE_CLASS_INVALID);

    // 2. Exact powers of two
    TEST_ASSERT(size_class_lookup(64) == 0);
    TEST_ASSERT(size_class_get_capacity(0) == 64);

    TEST_ASSERT(size_class_lookup(128) == 1);
    TEST_ASSERT(size_class_get_capacity(1) == 128);

    TEST_ASSERT(size_class_lookup(1024) == 4);
    TEST_ASSERT(size_class_get_capacity(4) == 1024);

    TEST_ASSERT(size_class_lookup(67108864) == 20); // 64 MB
    TEST_ASSERT(size_class_get_capacity(20) == 67108864);

    // 3. Boundary & off-by-one checks
    TEST_ASSERT(size_class_lookup(65) == 1);    // Needs 128B class
    TEST_ASSERT(size_class_lookup(127) == 1);   // Needs 128B class
    TEST_ASSERT(size_class_lookup(129) == 2);   // Needs 256B class
    TEST_ASSERT(size_class_lookup(1023) == 4);  // Needs 1024B class
    TEST_ASSERT(size_class_lookup(1025) == 5);  // Needs 2048B class

    // Invariant: capacity >= requested_bytes for all pooled requests
    for (size_t bytes = 1; bytes <= 100000; bytes += 337) {
        int sc = size_class_lookup(bytes);
        TEST_ASSERT(sc >= 0 && sc < (int)size_class_count());
        TEST_ASSERT(size_class_get_capacity(sc) >= bytes);
    }

    // 4. Large allocation path (> 64 MB)
    TEST_ASSERT(size_class_lookup(67108865) == TENSOR_SIZE_CLASS_LARGE);
    TEST_ASSERT(size_class_lookup(100 * 1024 * 1024) == TENSOR_SIZE_CLASS_LARGE);

    // Large allocation rounding (64-byte alignment)
    TEST_ASSERT(size_class_round_large(65) == 128);
    TEST_ASSERT(size_class_round_large(64) == 64);
    TEST_ASSERT(size_class_round_large(1000) == 1024);

    // 5. Predictor fallback safety
    MockBadPredictor bad_pred;
    set_active_predictor(&bad_pred);

    tensor_alloc_request_t req{};
    req.requested_bytes = 4096; // Should normally map to class 6
    req.device = TENSOR_DEVICE_CPU;
    req.dtype = TENSOR_DTYPE_FLOAT32;

    bool fallback_triggered = false;

    // Test mode 0: predictor returns class 0 (64 bytes, too small for 4096)
    bad_pred.mode = 0;
    int chosen_sc = predict_size_class_safe(&req, &fallback_triggered);
    TEST_ASSERT(fallback_triggered == true);
    TEST_ASSERT(chosen_sc == 6); // Safely fell back to class 6
    TEST_ASSERT(size_class_get_capacity(chosen_sc) >= req.requested_bytes);

    // Test mode 1: predictor returns out of range 999
    bad_pred.mode = 1;
    fallback_triggered = false;
    chosen_sc = predict_size_class_safe(&req, &fallback_triggered);
    TEST_ASSERT(fallback_triggered == true);
    TEST_ASSERT(chosen_sc == 6);

    // Test mode 2: predictor returns negative invalid
    bad_pred.mode = 2;
    fallback_triggered = false;
    chosen_sc = predict_size_class_safe(&req, &fallback_triggered);
    TEST_ASSERT(fallback_triggered == true);
    TEST_ASSERT(chosen_sc == 6);

    // Reset predictor back to default
    set_active_predictor(nullptr);
    fallback_triggered = false;
    chosen_sc = predict_size_class_safe(&req, &fallback_triggered);
    TEST_ASSERT(fallback_triggered == false);
    TEST_ASSERT(chosen_sc == 6);

    printf("[test_size_class] All size class and fallback tests passed!\n");
    return 0;
}
