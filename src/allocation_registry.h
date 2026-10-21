#ifndef TENSOR_ALLOCATION_REGISTRY_H
#define TENSOR_ALLOCATION_REGISTRY_H

#include "tensor_alloc.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* ptr;
    size_t requested_bytes;
    size_t reserved_bytes;
    int size_class;
    tensor_device_t device;
    bool is_large;
    uint64_t sequence_id;
} allocation_record_t;

/**
 * Initialize allocation registry.
 */
void registry_init(void);

/**
 * Register an active allocation.
 *
 * @param record Information about the live block.
 * @return true if successfully inserted, false if pointer already registered.
 */
bool registry_insert(const allocation_record_t* record);

/**
 * Look up and remove an active allocation for freeing.
 *
 * @param ptr Pointer to look up.
 * @param out_record If found, copied here (can be NULL).
 * @return TENSOR_STATUS_SUCCESS if found and removed,
 *         TENSOR_STATUS_DOUBLE_FREE if pointer was previously freed,
 *         TENSOR_STATUS_NOT_FOUND if pointer was never known.
 */
tensor_status_t registry_remove(void* ptr, allocation_record_t* out_record);

/**
 * Check if a pointer is currently registered as live.
 */
bool registry_is_live(const void* ptr);

/**
 * Clear all records from registry and history (called during shutdown).
 */
void registry_clear(void);

/**
 * Return total number of live registered allocations.
 */
size_t registry_live_count(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_ALLOCATION_REGISTRY_H */
