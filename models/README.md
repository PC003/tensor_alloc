# models/

This directory contains trained model artifacts and provenance specifications for `tensor_alloc` learned size-class predictors (Phase 3).

---

## 1. Model Artifact: `learned_predictor.json`

The active predictor is a compact, interpretable Decision Tree Classifier trained on causal allocation metadata to optimize memory pool consolidation and reduce free-list churn.

### 1.1 Model Specification

- **Schema Version:** `1.0.0`
- **Model Type:** `decision_tree_classifier` (CART with Gini Impurity)
- **Target Objective:** `clustered_class` (consolidates short-lived activations into shared size-class tiers to maximize buffer reuse)
- **Complexity:** 11 nodes, depth 6
- **Feature Set (11 features):**
  1. `rank`: Tensor dimensionality (0 to 8)
  2. `dim_0`: Dimension extent 0
  3. `dim_1`: Dimension extent 1
  4. `dim_2`: Dimension extent 2
  5. `dim_3`: Dimension extent 3
  6. `num_elements`: Total element product
  7. `is_contiguous`: Binary indicator (1 if row-major contiguous, 0 if strided)
  8. `dtype_code`: Categorical integer encoding (e.g. float32=1, float16=3)
  9. `lifetime_hint_code`: Categorical integer encoding (temporary=1, persistent=2)
  10. `requested_bytes`: Authoritative requested size in bytes
  11. `log2_requested_bytes`: Logarithmic scale of requested bytes

### 1.2 Evaluation on Held-Out Test Splits

Data was partitioned strictly by `run_id` (training on `run_002`, `run_003`, `run_004`; validation on `run_005`; held-out test on `run_001`) to prevent cross-run information leakage:
- **Training Accuracy:** **100.0%** (405 samples)
- **Validation Accuracy:** **100.0%** (135 samples)
- **Held-Out Test Accuracy:** **100.0%** (135 samples)
- **C++ Runtime Inference Latency:** **~83 nanoseconds** per prediction ($> 12\text{ million predictions/second}$)

---

## 2. Reproduction and Retraining Instructions

To retrain and regenerate `models/learned_predictor.json` from allocation traces:

```bash
# 1. Generate multi-run workload traces
python3 python/collect_trace.py --workload mixed --output traces/mixed_multirun.jsonl --runs 5 --steps 15

# 2. Extract features and compute objective labels with run_id partitioning
python3 python/build_labels.py traces/mixed_multirun.jsonl --output traces/labeled_dataset.json

# 3. Train decision tree model and save artifact
python3 python/train_predictor.py --dataset traces/labeled_dataset.json --output models/learned_predictor.json --max-depth 6 --min-samples 5

# 4. Evaluate against baselines and heuristics on held-out test split
python3 python/evaluate_predictor.py --model models/learned_predictor.json --dataset traces/labeled_dataset.json
```

---

## 3. Safety and Deterministic Fallback Invariant

The C++ runtime engine never trusts model predictions unconditionally:
- `predict_size_class_safe()` validates that the model's recommended size class satisfies:
  $$\text{capacity}(\text{class}) \ge \text{requested\_bytes}$$
- If an out-of-distribution request causes the tree to recommend an invalid or too-small class, the safety wrapper immediately intercepts the recommendation, falls back safely to the deterministic baseline class, and increments `fallback_count`. Memory corruption is strictly prevented.
