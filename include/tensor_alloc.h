#ifndef TENSOR_ALLOC_H
#define TENSOR_ALLOC_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Alignment guaranteed for all valid allocations (64 bytes, cache-line aligned).
 */
#define TENSOR_ALLOC_ALIGNMENT 64

/**
 * Maximum supported tensor rank/dimension count.
 */
#define TENSOR_MAX_RANK 8

/**
 * Return / error status codes for tensor_alloc operations.
 */
typedef enum {
    TENSOR_STATUS_SUCCESS = 0,
    TENSOR_STATUS_INVALID_ARGUMENT = 1,
    TENSOR_STATUS_UNSUPPORTED_DEVICE = 2,
    TENSOR_STATUS_UNSUPPORTED_DTYPE = 3,
    TENSOR_STATUS_OVERFLOW = 4,
    TENSOR_STATUS_OUT_OF_MEMORY = 5,
    TENSOR_STATUS_NOT_FOUND = 6,     /* Pointer not recognized / not allocated by tensor_alloc */
    TENSOR_STATUS_DOUBLE_FREE = 7    /* Pointer was previously freed */
} tensor_status_t;

/**
 * Supported tensor data types.
 */
typedef enum {
    TENSOR_DTYPE_UNKNOWN = 0,
    TENSOR_DTYPE_FLOAT32 = 1,
    TENSOR_DTYPE_FLOAT64 = 2,
    TENSOR_DTYPE_FLOAT16 = 3,
    TENSOR_DTYPE_BFLOAT16 = 4,
    TENSOR_DTYPE_INT32 = 5,
    TENSOR_DTYPE_INT64 = 6,
    TENSOR_DTYPE_INT16 = 7,
    TENSOR_DTYPE_INT8 = 8,
    TENSOR_DTYPE_UINT8 = 9,
    TENSOR_DTYPE_BOOL = 10
} tensor_dtype_t;

/**
 * Target device backends.
 */
typedef enum {
    TENSOR_DEVICE_CPU = 0,
    TENSOR_DEVICE_CUDA = 1,          /* Phase 5; rejected in CPU-only MVP */
    TENSOR_DEVICE_UNKNOWN = 2
} tensor_device_t;

/**
 * Lifetime hints for pool placement / tiering.
 */
typedef enum {
    TENSOR_LIFETIME_UNKNOWN = 0,
    TENSOR_LIFETIME_TEMPORARY = 1,   /* Intermediate activation / short-lived */
    TENSOR_LIFETIME_PERSISTENT = 2   /* Model weights / optimizer states / long-lived */
} tensor_lifetime_hint_t;

/**
 * Tensor allocation request metadata.
 *
 * Rules:
 * - requested_bytes MUST be > 0. A request of 0 bytes yields TENSOR_STATUS_INVALID_ARGUMENT.
 * - requested_bytes is authoritative for allocation size.
 * - If shape != NULL and ndim > 0, strides are interpreted in ELEMENTS (not bytes).
 *   If strides == NULL, standard row-major C-contiguous layout is assumed.
 * - If contiguous shape is specified, requested_bytes must be >= product(shape) * dtype_size.
 */
typedef struct {
    const int64_t* shape;                 /**< Array of dimension sizes (length ndim) */
    size_t ndim;                          /**< Tensor rank (0 to TENSOR_MAX_RANK) */
    const int64_t* strides;               /**< Strides in elements, or NULL if contiguous */
    tensor_dtype_t dtype;                 /**< Element data type */
    tensor_device_t device;               /**< Device backend (must be TENSOR_DEVICE_CPU in MVP) */
    tensor_lifetime_hint_t lifetime_hint; /**< Allocation lifetime hint */
    size_t requested_bytes;               /**< Authoritative requested byte count */
} tensor_alloc_request_t;

/**
 * Point-in-time metrics snapshot.
 */
typedef struct {
    uint64_t alloc_count;
    uint64_t free_count;
    uint64_t alloc_failure_count;
    size_t current_requested_bytes;
    size_t peak_requested_bytes;
    size_t current_reserved_bytes;
    size_t peak_reserved_bytes;
    uint64_t reuse_count;
    uint64_t fallback_count;
    double internal_waste_ratio;          /**< (reserved - requested) / reserved */
    double reuse_rate;                    /**< reuse_count / alloc_count */
} tensor_metrics_snapshot_t;

/**
 * Allocate aligned memory for a tensor request.
 *
 * @param req     Pointer to validated tensor request metadata (must not be NULL).
 * @param out_ptr Pointer to write the resulting aligned memory address (must not be NULL).
 * @return TENSOR_STATUS_SUCCESS on success, or specific error status code.
 */
tensor_status_t tensor_malloc(const tensor_alloc_request_t* req, void** out_ptr);

/**
 * Free memory previously allocated by tensor_malloc.
 *
 * Contract:
 * - Freeing NULL safely returns TENSOR_STATUS_SUCCESS without error.
 * - Freeing an unrecognized pointer returns TENSOR_STATUS_NOT_FOUND.
 * - Double-freeing a previously freed pointer returns TENSOR_STATUS_DOUBLE_FREE.
 *
 * @param ptr Pointer previously returned by tensor_malloc, or NULL.
 * @return TENSOR_STATUS_SUCCESS on success, or error status code.
 */
tensor_status_t tensor_free(void* ptr);

/**
 * Return human-readable string description for a status code.
 */
const char* tensor_status_str(tensor_status_t status);

/**
 * Return byte size for a given tensor dtype, or 0 if unknown/invalid.
 */
size_t tensor_dtype_size(tensor_dtype_t dtype);

/**
 * Retrieve current metrics snapshot.
 */
tensor_metrics_snapshot_t tensor_get_metrics(void);

/**
 * Reset all metric counters to zero.
 */
void tensor_reset_metrics(void);

/**
 * Print formatted metrics summary to the specified stream.
 */
void tensor_print_metrics(FILE* stream);

/**
 * Start recording allocation and free events to a JSONL trace file.
 *
 * @param filepath Path to output file (must not be NULL).
 * @return TENSOR_STATUS_SUCCESS on success, or error status code.
 */
tensor_status_t tensor_trace_start(const char* filepath);

/**
 * Stop active trace recording and flush pending records to disk.
 */
tensor_status_t tensor_trace_stop(void);

/**
 * Check if trace recording is currently active.
 */
bool tensor_trace_is_active(void);

/**
 * Clean up all allocated memory pools and clear internal registries.
 * Typically called at program termination or before running test suites.
 */
void tensor_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_ALLOC_H */
