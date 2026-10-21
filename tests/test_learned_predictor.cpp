#include "tensor_alloc.h"
#include "predictor_interface.h"
#include "learned_predictor.h"
#include "size_class.h"
#include "test_helpers.h"
#include <cstdio>
#include <chrono>
#include <fstream>

int main() {
    printf("[test_learned_predictor] Starting C++ learned predictor tests...\n");
    tensor_shutdown();

    // 1. Initial unloaded state
    TEST_ASSERT(!learned_predictor_is_loaded());
    TEST_ASSERT(learned_predictor_get() == nullptr);

    // 2. Reject missing file
    TEST_ASSERT(learned_predictor_load("non_existent_file.json") == TENSOR_STATUS_INVALID_ARGUMENT);

    // 3. Load valid model artifact
    const char* model_path = "models/learned_predictor.json";
    std::ifstream check_model(model_path);
    if (!check_model.is_open()) {
        model_path = "../models/learned_predictor.json";
    } else {
        check_model.close();
    }

    tensor_status_t load_status = learned_predictor_load(model_path);
    TEST_ASSERT(load_status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(learned_predictor_is_loaded());

    IPredictor* predictor = learned_predictor_get();
    TEST_ASSERT(predictor != nullptr);

    // 4. Test predictions on in-distribution request (e.g., q_proj: [16, 12, 128, 64], float16, temporary)
    int64_t shape_in_dist[4] = {16, 12, 128, 64}; // 3,145,728 bytes (3 MB)
    tensor_alloc_request_t req_in_dist{};
    req_in_dist.ndim = 4;
    req_in_dist.shape = shape_in_dist;
    req_in_dist.strides = nullptr;
    req_in_dist.dtype = TENSOR_DTYPE_FLOAT16;
    req_in_dist.device = TENSOR_DEVICE_CPU;
    req_in_dist.lifetime_hint = TENSOR_LIFETIME_TEMPORARY;
    req_in_dist.requested_bytes = 16 * 12 * 128 * 64 * sizeof(uint16_t);

    int pred_class = predictor->predict_class(&req_in_dist);
    TEST_ASSERT(pred_class >= 0 && pred_class < (int)size_class_count());
    TEST_ASSERT(size_class_get_capacity(pred_class) >= req_in_dist.requested_bytes);

    // 5. Test inference speed in C++
    const int BENCH_ITERS = 100000;
    auto t0 = std::chrono::high_resolution_clock::now();
    int dummy = 0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        dummy += predictor->predict_class(&req_in_dist);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    double ns_per_inference = total_ns / BENCH_ITERS;

    printf("C++ Model Inference Latency: %.1f ns per prediction (%.0f predictions/sec)\n",
           ns_per_inference, 1e9 / ns_per_inference);
    (void)dummy;

    // 6. Test in-distribution allocation with active learned predictor
    set_active_predictor(predictor);

    void* ptr1 = nullptr;
    tensor_status_t alloc_status = tensor_malloc(&req_in_dist, &ptr1);
    TEST_ASSERT(alloc_status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr1 != nullptr);
    TEST_ASSERT(reinterpret_cast<uintptr_t>(ptr1) % TENSOR_ALLOC_ALIGNMENT == 0);

    tensor_metrics_snapshot_t m1 = tensor_get_metrics();
    TEST_ASSERT(m1.alloc_count == 1);
    TEST_ASSERT(m1.fallback_count == 0); // Correct prediction, no fallback needed

    TEST_ASSERT(tensor_free(ptr1) == TENSOR_STATUS_SUCCESS);

    // 7. Test out-of-distribution request triggering safe deterministic fallback
    // A request where the raw model predicts too-small class 10 (64 KB) for a 128 KB request
    int64_t shape_ood[3] = {16, 32, 64}; // 131,072 bytes (128 KB)
    tensor_alloc_request_t req_ood{};
    req_ood.ndim = 3;
    req_ood.shape = shape_ood;
    req_ood.strides = nullptr;
    req_ood.dtype = TENSOR_DTYPE_FLOAT32;
    req_ood.device = TENSOR_DEVICE_CPU;
    req_ood.lifetime_hint = TENSOR_LIFETIME_TEMPORARY;
    req_ood.requested_bytes = 131072;

    bool used_fallback = false;
    int safe_class = predict_size_class_safe(&req_ood, &used_fallback);
    TEST_ASSERT(used_fallback == true); // Safety wrapper caught capacity mismatch!
    TEST_ASSERT(size_class_get_capacity(safe_class) >= req_ood.requested_bytes);

    void* ptr2 = nullptr;
    alloc_status = tensor_malloc(&req_ood, &ptr2);
    TEST_ASSERT(alloc_status == TENSOR_STATUS_SUCCESS);
    TEST_ASSERT(ptr2 != nullptr);

    tensor_metrics_snapshot_t m2 = tensor_get_metrics();
    TEST_ASSERT(m2.alloc_count == 2);
    TEST_ASSERT(m2.fallback_count == 1); // Safe fallback counted!

    TEST_ASSERT(tensor_free(ptr2) == TENSOR_STATUS_SUCCESS);

    // Reset predictor back to default deterministic
    set_active_predictor(nullptr);
    learned_predictor_destroy();
    TEST_ASSERT(!learned_predictor_is_loaded());

    tensor_shutdown();
    printf("[test_learned_predictor] All C++ learned predictor tests passed!\n");
    return 0;
}
