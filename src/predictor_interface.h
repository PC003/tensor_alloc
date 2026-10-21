#ifndef TENSOR_PREDICTOR_INTERFACE_H
#define TENSOR_PREDICTOR_INTERFACE_H

#include "tensor_alloc.h"
#include "size_class.h"
#include <stdbool.h>

#ifdef __cplusplus
class IPredictor {
public:
    virtual ~IPredictor() = default;

    /**
     * Recommend a size class index for the given tensor request.
     *
     * @param req The validated tensor request metadata.
     * @return Recommended size class index, or TENSOR_SIZE_CLASS_LARGE,
     *         or TENSOR_SIZE_CLASS_INVALID on prediction error.
     */
    virtual int predict_class(const tensor_alloc_request_t* req) = 0;
};

extern "C" {
#else
typedef struct IPredictor IPredictor;
#endif

/**
 * Obtain a pointer to the global default deterministic fallback predictor.
 */
IPredictor* get_default_predictor(void);

/**
 * Set the active predictor (passing NULL resets to the default deterministic fallback).
 */
void set_active_predictor(IPredictor* predictor);

/**
 * Query the active predictor with safe deterministic validation and fallback.
 *
 * Guarantees:
 * - Returned class capacity is ALWAYS >= req->requested_bytes.
 * - If predictor returns an invalid class, a too-small class, or throws an error,
 *   the baseline deterministic size class is used, and *out_used_fallback is set to true.
 *
 * @param req                The validated tensor request.
 * @param out_used_fallback  Set to true if predictor failed or recommended invalid class.
 * @return Guaranteed safe size class index (or TENSOR_SIZE_CLASS_LARGE).
 */
int predict_size_class_safe(const tensor_alloc_request_t* req, bool* out_used_fallback);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_PREDICTOR_INTERFACE_H */
