#ifndef TENSOR_CPU_BACKEND_H
#define TENSOR_CPU_BACKEND_H

#include "tensor_alloc.h"
#include "size_class.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize CPU backend memory pools.
 */
void cpu_backend_init(void);

/**
 * Allocate a block of memory from the CPU backend.
 *
 * Guarantees:
 * - Returned pointer is 64-byte aligned.
 * - If size_class is valid [0, TENSOR_NUM_SIZE_CLASSES - 1], attempts pool reuse first.
 * - If size_class is TENSOR_SIZE_CLASS_LARGE, allocates directly via aligned system allocation.
 *
 * @param size_class     Target size class index or TENSOR_SIZE_CLASS_LARGE.
 * @param required_bytes Required size (used to compute capacity if large).
 * @param out_ptr        Receives allocated aligned pointer.
 * @param out_capacity   Receives the exact reserved capacity bytes.
 * @param out_was_reused Receives true if block was obtained from pool free list.
 * @return TENSOR_STATUS_SUCCESS or TENSOR_STATUS_OUT_OF_MEMORY.
 */
tensor_status_t cpu_backend_alloc(int size_class,
                                  size_t required_bytes,
                                  void** out_ptr,
                                  size_t* out_capacity,
                                  bool* out_was_reused);

/**
 * Return a block to the CPU backend.
 *
 * @param ptr            Pointer previously returned by cpu_backend_alloc.
 * @param size_class     Size class associated with this block.
 * @param reserved_bytes Reserved capacity of the block.
 * @return TENSOR_STATUS_SUCCESS.
 */
tensor_status_t cpu_backend_free(void* ptr, int size_class, size_t reserved_bytes);

/**
 * Release all pooled/cached memory blocks in the free lists back to the OS.
 */
void cpu_backend_destroy(void);

/**
 * Get total number of blocks currently idling in pool free lists.
 */
size_t cpu_backend_idle_block_count(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_CPU_BACKEND_H */
