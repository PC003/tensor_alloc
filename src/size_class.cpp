#include "size_class.h"
#include <limits.h>

// 21 size classes from 64 B to 64 MB (powers of 2)
static const size_t kSizeClassCapacities[TENSOR_NUM_SIZE_CLASSES] = {
    64ULL,                // Class 0
    128ULL,               // Class 1
    256ULL,               // Class 2
    512ULL,               // Class 3
    1024ULL,              // Class 4: 1 KB
    2048ULL,              // Class 5: 2 KB
    4096ULL,              // Class 6: 4 KB
    8192ULL,              // Class 7: 8 KB
    16384ULL,             // Class 8: 16 KB
    32768ULL,             // Class 9: 32 KB
    65536ULL,             // Class 10: 64 KB
    131072ULL,            // Class 11: 128 KB
    262144ULL,            // Class 12: 256 KB
    524288ULL,            // Class 13: 512 KB
    1048576ULL,           // Class 14: 1 MB
    2097152ULL,           // Class 15: 2 MB
    4194304ULL,           // Class 16: 4 MB
    8388608ULL,           // Class 17: 8 MB
    16777216ULL,          // Class 18: 16 MB
    33554432ULL,          // Class 19: 32 MB
    67108864ULL           // Class 20: 64 MB
};

size_t size_class_count(void) {
    return TENSOR_NUM_SIZE_CLASSES;
}

size_t size_class_get_capacity(int class_idx) {
    if (class_idx < 0 || class_idx >= TENSOR_NUM_SIZE_CLASSES) {
        return 0;
    }
    return kSizeClassCapacities[class_idx];
}

int size_class_lookup(size_t requested_bytes) {
    if (requested_bytes == 0) {
        return TENSOR_SIZE_CLASS_INVALID;
    }

    if (requested_bytes > TENSOR_MAX_POOLED_SIZE) {
        return TENSOR_SIZE_CLASS_LARGE;
    }

    // Binary search or linear search over the 21 classes
    for (int i = 0; i < TENSOR_NUM_SIZE_CLASSES; ++i) {
        if (kSizeClassCapacities[i] >= requested_bytes) {
            return i;
        }
    }

    return TENSOR_SIZE_CLASS_LARGE;
}

size_t size_class_round_large(size_t requested_bytes) {
    if (requested_bytes == 0) {
        return 0;
    }
    const size_t align = TENSOR_ALLOC_ALIGNMENT;
    // Round up to nearest multiple of alignment: (bytes + align - 1) & ~(align - 1)
    if (requested_bytes > SIZE_MAX - (align - 1)) {
        return 0; // Overflow
    }
    return (requested_bytes + (align - 1)) & ~(align - 1);
}
