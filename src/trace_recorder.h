#ifndef TENSOR_TRACE_RECORDER_H
#define TENSOR_TRACE_RECORDER_H

#include "tensor_alloc.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Begin recording trace events to the specified file path.
 */
tensor_status_t trace_recorder_start(const char* filepath);

/**
 * Stop active trace recording and close the file.
 */
tensor_status_t trace_recorder_stop(void);

/**
 * Query whether the trace recorder is active.
 */
bool trace_recorder_is_active(void);

/**
 * Record an allocation event to the active trace file.
 */
void trace_record_alloc(uint64_t request_id,
                        const tensor_alloc_request_t* req,
                        int size_class,
                        size_t reserved_bytes);

/**
 * Record a free event to the active trace file.
 */
void trace_record_free(uint64_t request_id);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_TRACE_RECORDER_H */
