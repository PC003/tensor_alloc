#ifndef TENSOR_LEARNED_PREDICTOR_H
#define TENSOR_LEARNED_PREDICTOR_H

#include "tensor_alloc.h"
#include "predictor_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a trained decision tree predictor from a JSON model artifact.
 *
 * @param json_filepath Path to models/learned_predictor.json.
 * @return TENSOR_STATUS_SUCCESS on success, or error status code.
 */
tensor_status_t learned_predictor_load(const char* json_filepath);

/**
 * Retrieve the active learned predictor instance (implements IPredictor).
 * Returns NULL if no model is loaded.
 */
IPredictor* learned_predictor_get(void);

/**
 * Destroy the loaded predictor and free its internal tree nodes.
 */
void learned_predictor_destroy(void);

/**
 * Check if a learned predictor model is currently loaded.
 */
bool learned_predictor_is_loaded(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_LEARNED_PREDICTOR_H */
