#ifndef TENSOR_METRICS_H
#define TENSOR_METRICS_H

#include "tensor_alloc.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enable or disable metrics tracking (default is enabled).
 */
void metrics_set_enabled(bool enabled);

/**
 * Check if metrics tracking is currently enabled.
 */
bool metrics_is_enabled(void);

/**
 * Record a successful allocation.
 *
 * @param requested_bytes Number of bytes requested by user.
 * @param reserved_bytes  Exact capacity allocated/reserved.
 * @param reused          True if block was reused from pool.
 * @param fallback        True if deterministic fallback was triggered.
 */
void metrics_record_alloc(size_t requested_bytes,
                          size_t reserved_bytes,
                          bool reused,
                          bool fallback);

/**
 * Record a deallocation.
 *
 * @param requested_bytes User bytes freed.
 * @param reserved_bytes  Reserved capacity returned to pool.
 */
void metrics_record_free(size_t requested_bytes, size_t reserved_bytes);

/**
 * Record an allocation failure.
 */
void metrics_record_alloc_failure(void);

/**
 * Reset all metrics counters.
 */
void metrics_reset(void);

/**
 * Return current metrics snapshot.
 */
tensor_metrics_snapshot_t metrics_get_snapshot(void);

/**
 * Print formatted metrics summary to output stream.
 */
void metrics_print(FILE* stream);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_METRICS_H */
