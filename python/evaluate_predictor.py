#!/usr/bin/env python3
"""
evaluate_predictor.py: Comprehensive evaluation of learned predictor against
fixed baselines and heuristics on held-out test workloads.

Measures:
- Target objective accuracy and match rate
- Simulated memory waste (reserved vs. requested)
- Pool consolidation and simulated buffer reuse
- Model inference latency benchmark (nanoseconds per inference)
"""

import argparse
import json
import time

FIXED_CLASSES = [64 * (2 ** i) for i in range(21)] # 64 B to 64 MB

def get_baseline_class(requested_bytes):
    for idx, c in enumerate(FIXED_CLASSES):
        if c >= requested_bytes:
            return idx, c
    return -1, ((requested_bytes + 63) // 64) * 64

def get_heuristic_class(requested_bytes, hint_code):
    base_idx, base_cap = get_baseline_class(requested_bytes)
    if base_idx == -1:
        return base_idx, base_cap
    # If temporary activation (hint_code == 1), round up to even-indexed class
    if hint_code == 1 and base_idx % 2 != 0:
        if base_idx + 1 < len(FIXED_CLASSES):
            return base_idx + 1, FIXED_CLASSES[base_idx + 1]
    return base_idx, base_cap

def predict_tree(node_dict, features):
    if node_dict["is_leaf"]:
        return node_dict["prediction"]
    f_idx = node_dict["feature_idx"]
    thresh = node_dict["threshold"]
    if features[f_idx] <= thresh:
        return predict_tree(node_dict["left"], features)
    else:
        return predict_tree(node_dict["right"], features)

def evaluate_policy(name, class_predictions, records, target_key):
    total_requested = 0
    total_reserved = 0
    total_waste = 0
    correct = 0
    classes_used = set()

    for i, r in enumerate(records):
        pred_sc = class_predictions[i]
        req_bytes = r["features"][9] # requested_bytes feature index
        target_sc = r["labels"][target_key]

        if pred_sc == target_sc:
            correct += 1

        if pred_sc >= 0 and pred_sc < len(FIXED_CLASSES):
            cap = FIXED_CLASSES[pred_sc]
        else:
            cap = ((req_bytes + 63) // 64) * 64

        classes_used.add(pred_sc)
        total_requested += req_bytes
        total_reserved += cap
        total_waste += (cap - req_bytes)

    waste_pct = (total_waste / total_reserved * 100.0) if total_reserved > 0 else 0.0
    acc_pct = (correct / len(records) * 100.0) if records else 0.0

    return {
        "policy": name,
        "accuracy_pct": acc_pct,
        "total_requested_mb": total_requested / (1024**2),
        "total_reserved_mb": total_reserved / (1024**2),
        "waste_pct": waste_pct,
        "distinct_classes": len(classes_used)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate learned size-class predictor against baselines.")
    parser.add_argument("--model", "-m", default="models/learned_predictor.json", help="Path to learned model artifact")
    parser.add_argument("--dataset", "-d", default="traces/labeled_dataset.json", help="Path to labeled dataset JSON")
    args = parser.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)

    target_key = model.get("target_objective", "clustered_class")
    test_records = ds.get("test", [])
    if not test_records:
        test_records = ds.get("val", [])

    print("=" * 80)
    print("             Learned Predictor vs. Baselines Evaluation Report                ")
    print("=" * 80)
    print(f"Model Artifact:       {args.model}")
    print(f"Target Objective:     {target_key}")
    print(f"Evaluation Samples:   {len(test_records)} (held-out test runs: {', '.join(ds['split_summary']['test_runs'])})")

    # 1. Predictions under Fixed Power-of-Two Baseline
    baseline_preds = []
    for r in test_records:
        sc, _ = get_baseline_class(r["features"][9])
        baseline_preds.append(sc)

    # 2. Predictions under Heuristic Policy
    heuristic_preds = []
    for r in test_records:
        sc, _ = get_heuristic_class(r["features"][9], r["features"][8])
        heuristic_preds.append(sc)

    # 3. Predictions under Learned Decision Tree
    tree_dict = model["tree"]
    learned_preds = []

    # Measure pure inference latency over 50,000 iterations
    t0 = time.time_ns()
    repeat = 200
    for _ in range(repeat):
        for r in test_records:
            pred = predict_tree(tree_dict, r["features"])
            if _ == 0:
                learned_preds.append(pred)
    total_ns = time.time_ns() - t0
    avg_inference_ns = total_ns / (repeat * len(test_records))

    # Evaluate all three policies
    results = [
        evaluate_policy("Fixed Power-of-2 Baseline", baseline_preds, test_records, target_key),
        evaluate_policy("Static Heuristic Policy", heuristic_preds, test_records, target_key),
        evaluate_policy("Learned Decision Tree", learned_preds, test_records, target_key)
    ]

    print("\n--------------------------------------------------------------------------------")
    print("| Policy                     | Accuracy | Total Req MB | Total Res MB | Waste % | Pools |")
    print("|----------------------------|----------|--------------|--------------|---------|-------|")
    for r in results:
        print(f"| {r['policy']:26s} |  {r['accuracy_pct']:5.1f}%  |  {r['total_requested_mb']:8.2f} MB |  {r['total_reserved_mb']:8.2f} MB | {r['waste_pct']:6.2f}% |   {r['distinct_classes']:2d}  |")
    print("--------------------------------------------------------------------------------")

    print(f"\nInference Performance:")
    print(f"  Average Model Inference Latency: {avg_inference_ns:.1f} ns ({avg_inference_ns / 1000.0:.3f} µs)")
    print(f"  Throughput:                      {1e9 / avg_inference_ns:,.0f} predictions/sec")
    print("=" * 80)

if __name__ == "__main__":
    main()
