#ifndef TENSOR_REQUEST_VALIDATION_H
#define TENSOR_REQUEST_VALIDATION_H

#include "tensor_alloc.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Validate a tensor allocation request.
 *
 * Checks:
 * - req and out_ptr are non-null.
 * - requested_bytes > 0.
 * - device is TENSOR_DEVICE_CPU.
 * - dtype is a recognized enum.
 * - ndim <= TENSOR_MAX_RANK.
 * - If ndim > 0, shape is non-null and all dims are >= 0.
 * - Integer overflow checks during element count and stride calculation.
 * - requested_bytes is at least the minimum bytes required by shape and dtype.
 *
 * @param req     Request pointer to validate.
 * @param out_ptr Destination pointer address to validate.
 * @return TENSOR_STATUS_SUCCESS or corresponding error code.
 */
tensor_status_t validate_tensor_request(const tensor_alloc_request_t* req, void** out_ptr);

/**
 * Check if multiplication of two size_t integers would overflow.
 * If safe, writes (a * b) to *out and returns true.
 * If overflow occurs, returns false.
 */
bool safe_mul_size(size_t a, size_t b, size_t* out);

/**
 * Check if addition of two size_t integers would overflow.
 * If safe, writes (a + b) to *out and returns true.
 * If overflow occurs, returns false.
 */
bool safe_add_size(size_t a, size_t b, size_t* out);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_REQUEST_VALIDATION_H */
