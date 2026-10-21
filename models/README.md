# models/

This directory is reserved for trained learned predictor artifacts (Phase 3).

## Status: CPU MVP (Phase 1)
In the current milestone (Phase 1), learned predictors and ONNX runtime models are deferred in accordance with the project guidelines:
- The allocator uses a deterministic fixed size-class baseline (`src/predictor_fallback.cpp`) that maps `requested_bytes` to the smallest power-of-two size class capable of holding the request.
- The predictor abstraction (`IPredictor` in `src/predictor_interface.h`) is established and verified.
- When an ML model is added in Phase 3, any invalid prediction or capacity violation automatically and safely falls back to the deterministic baseline via `predict_size_class_safe()`.
- Future model artifacts should be saved here alongside training logs, preprocessing configurations, feature scalers, and class mappings.
