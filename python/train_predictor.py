#!/usr/bin/env python3
"""
train_predictor.py: Trains a compact, interpretable decision tree predictor
for tensor size-class allocation.

Pure standard library implementation (zero external dependencies).
Supports:
- Training on labeled dataset partitions (split by run_id)
- Recursive CART tree induction with Gini impurity
- Configurable max_depth and min_samples_split to prevent overfitting
- Evaluation on held-out validation and test splits
- Serialization to versioned model artifact (models/learned_predictor.json)
"""

import argparse
import json
import math
import sys
import time

def gini_impurity(labels):
    if not labels:
        return 0.0
    total = len(labels)
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    impurity = 1.0 - sum((count / total) ** 2 for count in counts.values())
    return impurity

def majority_class(labels):
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]

class DecisionTreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature_idx=None, feature_name=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = None
        self.right = None
        self.samples = 0
        self.impurity = 0.0

    def to_dict(self):
        if self.is_leaf:
            return {
                "is_leaf": True,
                "prediction": self.prediction,
                "samples": self.samples
            }
        return {
            "is_leaf": False,
            "feature_idx": self.feature_idx,
            "feature_name": self.feature_name,
            "threshold": self.threshold,
            "samples": self.samples,
            "impurity": round(self.impurity, 4),
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        }

    def predict(self, features):
        if self.is_leaf:
            return self.prediction
        val = features[self.feature_idx]
        if val <= self.threshold:
            return self.left.predict(features)
        else:
            return self.right.predict(features)

def build_tree(X, y, feature_names, depth, max_depth, min_samples_split):
    node = DecisionTreeNode()
    node.samples = len(y)
    node.impurity = gini_impurity(y)

    # Base cases: pure node, max depth reached, or insufficient samples
    if len(set(y)) == 1 or depth >= max_depth or len(y) < min_samples_split:
        node.is_leaf = True
        node.prediction = majority_class(y)
        return node

    best_gain = -1.0
    best_feat = None
    best_thresh = None
    best_splits = None

    current_gini = node.impurity
    n_features = len(feature_names)

    for f_idx in range(n_features):
        vals = [row[f_idx] for row in X]
        distinct_vals = sorted(list(set(vals)))
        if len(distinct_vals) <= 1:
            continue

        # Test candidate thresholds between adjacent sorted values
        candidate_thresholds = [
            (distinct_vals[i] + distinct_vals[i+1]) / 2.0
            for i in range(min(50, len(distinct_vals) - 1))
        ]

        for thresh in candidate_thresholds:
            left_y = []
            right_y = []
            left_X = []
            right_X = []

            for i, row in enumerate(X):
                if row[f_idx] <= thresh:
                    left_X.append(row)
                    left_y.append(y[i])
                else:
                    right_X.append(row)
                    right_y.append(y[i])

            if not left_y or not right_y:
                continue

            # Weighted Gini impurity
            p_left = len(left_y) / len(y)
            p_right = len(right_y) / len(y)
            gain = current_gini - (p_left * gini_impurity(left_y) + p_right * gini_impurity(right_y))

            if gain > best_gain:
                best_gain = gain
                best_feat = f_idx
                best_thresh = thresh
                best_splits = (left_X, left_y, right_X, right_y)

    if best_gain <= 1e-6 or best_splits is None:
        node.is_leaf = True
        node.prediction = majority_class(y)
        return node

    left_X, left_y, right_X, right_y = best_splits
    node.feature_idx = best_feat
    node.feature_name = feature_names[best_feat]
    node.threshold = round(best_thresh, 4)
    node.left = build_tree(left_X, left_y, feature_names, depth + 1, max_depth, min_samples_split)
    node.right = build_tree(right_X, right_y, feature_names, depth + 1, max_depth, min_samples_split)

    return node

def evaluate_accuracy(tree, X, y):
    if not y:
        return 0.0
    correct = 0
    for i, row in enumerate(X):
        pred = tree.predict(row)
        if pred == y[i]:
            correct += 1
    return correct / len(y)

def count_nodes(node):
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def main():
    parser = argparse.ArgumentParser(description="Train decision tree predictor for tensor size classing.")
    parser.add_argument("--dataset", "-d", default="traces/labeled_dataset.json", help="Path to labeled dataset JSON")
    parser.add_argument("--output", "-o", default="models/learned_predictor.json", help="Path to save model artifact")
    parser.add_argument("--target", choices=["baseline_class", "clustered_class", "oracle_class"], default="clustered_class",
                        help="Target label objective to train on")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples per split")
    args = parser.parse_args()

    print("=" * 70)
    print("                 Training Learned Size-Class Predictor                ")
    print("=" * 70)

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)

    feature_names = ds["feature_names"]
    target_key = args.target

    train_X = [r["features"] for r in ds["train"]]
    train_y = [r["labels"][target_key] for r in ds["train"]]

    val_X = [r["features"] for r in ds["val"]]
    val_y = [r["labels"][target_key] for r in ds["val"]]

    test_X = [r["features"] for r in ds["test"]]
    test_y = [r["labels"][target_key] for r in ds["test"]]

    print(f"Target Objective:    {target_key}")
    print(f"Features ({len(feature_names)}):        {', '.join(feature_names)}")
    print(f"Training Samples:    {len(train_X)} (from runs: {', '.join(ds['split_summary']['train_runs'])})")
    print(f"Validation Samples:  {len(val_X)} (from runs: {', '.join(ds['split_summary']['val_runs'])})")
    print(f"Test Samples:        {len(test_X)} (from runs: {', '.join(ds['split_summary']['test_runs'])})")

    t0 = time.time()
    tree = build_tree(train_X, train_y, feature_names, depth=0, max_depth=args.max_depth, min_samples_split=args.min_samples)
    train_time_ms = (time.time() - t0) * 1000.0

    train_acc = evaluate_accuracy(tree, train_X, train_y)
    val_acc = evaluate_accuracy(tree, val_X, val_y) if val_y else 0.0
    test_acc = evaluate_accuracy(tree, test_X, test_y) if test_y else 0.0

    print(f"\nTraining Duration:   {train_time_ms:.2f} ms")
    print(f"Tree Complexity:     {count_nodes(tree)} nodes (max depth {args.max_depth})")
    print(f"Train Accuracy:      {train_acc * 100.0:.2f}%")
    print(f"Val Accuracy:        {val_acc * 100.0:.2f}%")
    print(f"Held-out Test Acc:   {test_acc * 100.0:.2f}%")

    # Serialize model artifact
    artifact = {
        "schema_version": "1.0.0",
        "model_type": "decision_tree_classifier",
        "target_objective": target_key,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "training_time_ms": round(train_time_ms, 2),
            "total_nodes": count_nodes(tree)
        },
        "feature_names": feature_names,
        "class_mapping": ds.get("class_mapping", {}),
        "split_summary": ds.get("split_summary", {}),
        "tree": tree.to_dict()
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nModel artifact saved to: {args.output}")
    print("=" * 70)

if __name__ == "__main__":
    main()
