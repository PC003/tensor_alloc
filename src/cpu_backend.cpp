#include "cpu_backend.h"
#include <stdlib.h>
#include <vector>
#include <mutex>

struct SizeClassPool {
    std::mutex mtx;
    std::vector<void*> free_list;
};

static SizeClassPool g_pools[TENSOR_NUM_SIZE_CLASSES];

void cpu_backend_init(void) {
    for (int i = 0; i < TENSOR_NUM_SIZE_CLASSES; ++i) {
        std::lock_guard<std::mutex> lock(g_pools[i].mtx);
        for (void* p : g_pools[i].free_list) {
            free(p);
        }
        g_pools[i].free_list.clear();
    }
}

tensor_status_t cpu_backend_alloc(int size_class,
                                  size_t required_bytes,
                                  void** out_ptr,
                                  size_t* out_capacity,
                                  bool* out_was_reused) {
    if (!out_ptr || !out_capacity || !out_was_reused) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    *out_ptr = nullptr;
    *out_capacity = 0;
    *out_was_reused = false;

    // Handle large allocation
    if (size_class == TENSOR_SIZE_CLASS_LARGE) {
        size_t cap = size_class_round_large(required_bytes);
        if (cap == 0) {
            return TENSOR_STATUS_OVERFLOW;
        }

        void* mem = nullptr;
        int res = posix_memalign(&mem, TENSOR_ALLOC_ALIGNMENT, cap);
        if (res != 0 || mem == nullptr) {
            return TENSOR_STATUS_OUT_OF_MEMORY;
        }

        *out_ptr = mem;
        *out_capacity = cap;
        *out_was_reused = false;
        return TENSOR_STATUS_SUCCESS;
    }

    // Standard pooled size class
    if (size_class < 0 || size_class >= TENSOR_NUM_SIZE_CLASSES) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    size_t cap = size_class_get_capacity(size_class);
    if (cap < required_bytes) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    SizeClassPool& pool = g_pools[size_class];
    {
        std::lock_guard<std::mutex> lock(pool.mtx);
        if (!pool.free_list.empty()) {
            *out_ptr = pool.free_list.back();
            pool.free_list.pop_back();
            *out_capacity = cap;
            *out_was_reused = true;
            return TENSOR_STATUS_SUCCESS;
        }
    }

    // Free list empty: allocate new block from OS with required alignment
    void* mem = nullptr;
    int res = posix_memalign(&mem, TENSOR_ALLOC_ALIGNMENT, cap);
    if (res != 0 || mem == nullptr) {
        return TENSOR_STATUS_OUT_OF_MEMORY;
    }

    *out_ptr = mem;
    *out_capacity = cap;
    *out_was_reused = false;
    return TENSOR_STATUS_SUCCESS;
}

tensor_status_t cpu_backend_free(void* ptr, int size_class, size_t reserved_bytes) {
    (void)reserved_bytes;
    if (!ptr) {
        return TENSOR_STATUS_SUCCESS;
    }

    if (size_class == TENSOR_SIZE_CLASS_LARGE) {
        free(ptr);
        return TENSOR_STATUS_SUCCESS;
    }

    if (size_class >= 0 && size_class < TENSOR_NUM_SIZE_CLASSES) {
        SizeClassPool& pool = g_pools[size_class];
        std::lock_guard<std::mutex> lock(pool.mtx);
        pool.free_list.push_back(ptr);
        return TENSOR_STATUS_SUCCESS;
    }

    // Unknown size class fallback
    free(ptr);
    return TENSOR_STATUS_SUCCESS;
}

void cpu_backend_destroy(void) {
    for (int i = 0; i < TENSOR_NUM_SIZE_CLASSES; ++i) {
        std::lock_guard<std::mutex> lock(g_pools[i].mtx);
        for (void* p : g_pools[i].free_list) {
            free(p);
        }
        g_pools[i].free_list.clear();
    }
}

size_t cpu_backend_idle_block_count(void) {
    size_t count = 0;
    for (int i = 0; i < TENSOR_NUM_SIZE_CLASSES; ++i) {
        std::lock_guard<std::mutex> lock(g_pools[i].mtx);
        count += g_pools[i].free_list.size();
    }
    return count;
}
