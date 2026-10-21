#include "request_validation.h"
#include <limits.h>

bool safe_mul_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > SIZE_MAX / b) {
        return false;
    }
    *out = a * b;
    return true;
}

bool safe_add_size(size_t a, size_t b, size_t* out) {
    if (SIZE_MAX - a < b) {
        return false;
    }
    *out = a + b;
    return true;
}

size_t tensor_dtype_size(tensor_dtype_t dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT32:
        case TENSOR_DTYPE_INT32:
            return 4;
        case TENSOR_DTYPE_FLOAT64:
        case TENSOR_DTYPE_INT64:
            return 8;
        case TENSOR_DTYPE_FLOAT16:
        case TENSOR_DTYPE_BFLOAT16:
        case TENSOR_DTYPE_INT16:
            return 2;
        case TENSOR_DTYPE_INT8:
        case TENSOR_DTYPE_UINT8:
        case TENSOR_DTYPE_BOOL:
            return 1;
        case TENSOR_DTYPE_UNKNOWN:
        default:
            return 0;
    }
}

tensor_status_t validate_tensor_request(const tensor_alloc_request_t* req, void** out_ptr) {
    if (req == NULL || out_ptr == NULL) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    // Zero-byte allocation is explicitly rejected by the API contract
    if (req->requested_bytes == 0) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    // Only CPU device is supported in Phase 1 MVP
    if (req->device != TENSOR_DEVICE_CPU) {
        return TENSOR_STATUS_UNSUPPORTED_DEVICE;
    }

    // Validate dtype
    size_t elem_size = tensor_dtype_size(req->dtype);
    if (elem_size == 0) {
        return TENSOR_STATUS_UNSUPPORTED_DTYPE;
    }

    // Validate rank
    if (req->ndim > TENSOR_MAX_RANK) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    // If tensor has dimensions specified
    if (req->ndim > 0) {
        if (req->shape == NULL) {
            return TENSOR_STATUS_INVALID_ARGUMENT;
        }

        // Check for non-negative dimensions
        for (size_t i = 0; i < req->ndim; ++i) {
            if (req->shape[i] < 0) {
                return TENSOR_STATUS_INVALID_ARGUMENT;
            }
        }

        // If strides are NULL, assume standard C-contiguous layout
        if (req->strides == NULL) {
            size_t num_elements = 1;
            for (size_t i = 0; i < req->ndim; ++i) {
                if (!safe_mul_size(num_elements, (size_t)req->shape[i], &num_elements)) {
                    return TENSOR_STATUS_OVERFLOW;
                }
            }

            size_t min_bytes = 0;
            if (!safe_mul_size(num_elements, elem_size, &min_bytes)) {
                return TENSOR_STATUS_OVERFLOW;
            }

            // Requested bytes must be at least large enough for the tensor contents
            if (req->requested_bytes < min_bytes) {
                return TENSOR_STATUS_INVALID_ARGUMENT;
            }
        } else {
            // Strides are provided in element units
            size_t max_offset = 0;
            for (size_t i = 0; i < req->ndim; ++i) {
                if (req->strides[i] < 0) {
                    // Negative strides without explicit base offset not supported in MVP
                    return TENSOR_STATUS_INVALID_ARGUMENT;
                }
                if (req->shape[i] > 1) {
                    size_t dim_extent = (size_t)(req->shape[i] - 1);
                    size_t stride_val = (size_t)req->strides[i];
                    size_t dim_offset = 0;
                    if (!safe_mul_size(dim_extent, stride_val, &dim_offset)) {
                        return TENSOR_STATUS_OVERFLOW;
                    }
                    if (!safe_add_size(max_offset, dim_offset, &max_offset)) {
                        return TENSOR_STATUS_OVERFLOW;
                    }
                }
            }

            // Minimum spanned elements = max_offset + 1 (unless empty tensor)
            bool is_empty = false;
            for (size_t i = 0; i < req->ndim; ++i) {
                if (req->shape[i] == 0) {
                    is_empty = true;
                    break;
                }
            }

            if (!is_empty) {
                size_t spanned_elements = 0;
                if (!safe_add_size(max_offset, 1, &spanned_elements)) {
                    return TENSOR_STATUS_OVERFLOW;
                }
                size_t min_bytes = 0;
                if (!safe_mul_size(spanned_elements, elem_size, &min_bytes)) {
                    return TENSOR_STATUS_OVERFLOW;
                }
                if (req->requested_bytes < min_bytes) {
                    return TENSOR_STATUS_INVALID_ARGUMENT;
                }
            }
        }
    }

    return TENSOR_STATUS_SUCCESS;
}
