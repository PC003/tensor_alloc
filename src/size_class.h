#ifndef TENSOR_SIZE_CLASS_H
#define TENSOR_SIZE_CLASS_H

#include <stddef.h>
#include <stdbool.h>
#include "tensor_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TENSOR_NUM_SIZE_CLASSES 21
#define TENSOR_MAX_POOLED_SIZE (64 * 1024 * 1024) /* 64 MB */
#define TENSOR_SIZE_CLASS_LARGE (-1)
#define TENSOR_SIZE_CLASS_INVALID (-2)

/**
 * Determine the smallest fixed size-class index capable of holding requested_bytes.
 *
 * @param requested_bytes Number of bytes required.
 * @return Size class index [0, TENSOR_NUM_SIZE_CLASSES - 1],
 *         or TENSOR_SIZE_CLASS_LARGE if > TENSOR_MAX_POOLED_SIZE,
 *         or TENSOR_SIZE_CLASS_INVALID if requested_bytes == 0.
 */
int size_class_lookup(size_t requested_bytes);

/**
 * Retrieve the capacity in bytes for a given size class index.
 *
 * @param class_idx Index in [0, TENSOR_NUM_SIZE_CLASSES - 1].
 * @return Capacity in bytes, or 0 if class_idx is out of range.
 */
size_t size_class_get_capacity(int class_idx);

/**
 * Total number of predefined fixed size classes.
 */
size_t size_class_count(void);

/**
 * Calculate capacity for a large allocation by rounding up to TENSOR_ALLOC_ALIGNMENT.
 * Returns 0 if overflow occurs.
 */
size_t size_class_round_large(size_t requested_bytes);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_SIZE_CLASS_H */
