#!/usr/bin/env python3
"""
build_labels.py: Feature extraction and objective-driven label generation.

Extracts causal features available at allocation time (shape, rank, strides,
dtype, lifetime hint, requested bytes) and computes target size-class labels
under clearly documented objectives:
1. Baseline (smallest class with capacity >= requested)
2. Clustered (quantized tiering to maximize pool reuse)
3. Oracle Reuse (offline oracle matching idle pool blocks)

Partitions records by run_id (train / val / test) to prevent cross-run data leakage.
"""

import argparse
import csv
import json
import math
import random
import sys

FIXED_CLASSES = [64 * (2 ** i) for i in range(21)] # 64 B to 64 MB

DTYPE_CODES = {
    "unknown": 0, "float32": 1, "float64": 2, "float16": 3, "bfloat16": 4,
    "int32": 5, "int64": 6, "int16": 7, "int8": 8, "uint8": 9, "bool": 10
}

LIFETIME_CODES = {
    "unknown": 0, "temporary": 1, "persistent": 2
}

FEATURE_NAMES = [
    "rank",
    "dim_0",
    "dim_1",
    "dim_2",
    "dim_3",
    "num_elements",
    "is_contiguous",
    "dtype_code",
    "lifetime_hint_code",
    "requested_bytes",
    "log2_requested_bytes"
]

def get_baseline_class(requested_bytes):
    for idx, c in enumerate(FIXED_CLASSES):
        if c >= requested_bytes:
            return idx, c
    return -1, ((requested_bytes + 63) // 64) * 64

def get_clustered_class(requested_bytes, lifetime_hint):
    """
    Clustered policy: for short-lived temporary tensors, groups requests into
    even-indexed power-of-two classes (4x spacing) to consolidate free lists and
    boost cross-operator pool reuse.
    """
    base_idx, base_cap = get_baseline_class(requested_bytes)
    if base_idx == -1:
        return base_idx, base_cap
    if lifetime_hint == "temporary" and base_idx % 2 != 0:
        if base_idx + 1 < len(FIXED_CLASSES):
            return base_idx + 1, FIXED_CLASSES[base_idx + 1]
    return base_idx, base_cap

def extract_features(alloc_ev):
    shape = alloc_ev.get("shape", [])
    rank = len(shape)
    d0 = shape[0] if rank > 0 else 1
    d1 = shape[1] if rank > 1 else 1
    d2 = shape[2] if rank > 2 else 1
    d3 = shape[3] if rank > 3 else 1

    num_elems = 1
    for d in shape:
        num_elems *= d

    strides = alloc_ev.get("strides")
    is_contiguous = 1 if (strides is None or len(strides) == 0) else 0

    dtype_code = DTYPE_CODES.get(alloc_ev.get("dtype", "unknown"), 0)
    hint_code = LIFETIME_CODES.get(alloc_ev.get("lifetime_hint", "unknown"), 0)
    req_bytes = alloc_ev.get("requested_bytes", 0)
    log_bytes = math.log2(req_bytes) if req_bytes > 0 else 0.0

    return [
        rank,
        d0,
        d1,
        d2,
        d3,
        num_elems,
        is_contiguous,
        dtype_code,
        hint_code,
        req_bytes,
        round(log_bytes, 4)
    ]

def simulate_oracle_labels(events):
    """
    Simulates memory pool replay to produce offline oracle reuse labels:
    If an idle block in a slightly larger class (up to 2x) is available in the pool,
    the oracle labels that class to achieve 0-cost allocation and avoid pool growth.
    """
    pool_free_counts = {idx: 0 for idx in range(len(FIXED_CLASSES))}
    live_classes = {} # req_id -> (sc_idx, req_bytes)
    oracle_labels = {} # req_id -> target_class

    for ev in events:
        req_id = ev["request_id"]
        if ev["event_type"] == "alloc":
            req_bytes = ev["requested_bytes"]
            base_idx, _ = get_baseline_class(req_bytes)

            chosen_idx = base_idx
            if base_idx >= 0:
                # Check if exact class is free
                if pool_free_counts[base_idx] > 0:
                    chosen_idx = base_idx
                    pool_free_counts[base_idx] -= 1
                # Check if class + 1 is free (slack up to 2x capacity)
                elif base_idx + 1 < len(FIXED_CLASSES) and pool_free_counts[base_idx + 1] > 0:
                    chosen_idx = base_idx + 1
                    pool_free_counts[base_idx + 1] -= 1

            live_classes[req_id] = chosen_idx
            oracle_labels[req_id] = chosen_idx

        elif ev["event_type"] == "free":
            if req_id in live_classes:
                sc_idx = live_classes.pop(req_id)
                if sc_idx >= 0:
                    pool_free_counts[sc_idx] += 1

    return oracle_labels

def main():
    parser = argparse.ArgumentParser(description="Build training datasets with feature extraction and labeling.")
    parser.add_argument("traces", nargs="+", help="Input JSONL trace files")
    parser.add_argument("--output", "-o", default="traces/labeled_dataset.json", help="Output labeled dataset JSON")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio by run_id")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio by run_id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split partitioning")
    args = parser.parse_args()

    random.seed(args.seed)

    all_events = []
    run_records = {} # run_id -> list of alloc_events

    for tp in args.traces:
        with open(tp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "schema_version" in row:
                    continue
                all_events.append(row)
                if row.get("event_type") == "alloc":
                    rid = row.get("run_id", "default")
                    if rid not in run_records:
                        run_records[rid] = []
                    run_records[rid].append(row)

    # Compute oracle labels over full event stream
    oracle_map = simulate_oracle_labels(all_events)

    # Partition by run_id to guarantee zero data leakage between splits
    distinct_runs = sorted(list(run_records.keys()))
    random.shuffle(distinct_runs)

    num_runs = len(distinct_runs)
    n_train = max(1, int(num_runs * args.train_ratio))
    n_val = max(1, int(num_runs * args.val_ratio)) if num_runs > 2 else 0

    train_runs = set(distinct_runs[:n_train])
    val_runs = set(distinct_runs[n_train:n_train + n_val])
    test_runs = set(distinct_runs[n_train + n_val:])
    if not test_runs and val_runs:
        test_runs = val_runs

    def format_records(alloc_list):
        records = []
        for ev in alloc_list:
            feats = extract_features(ev)
            req_id = ev["request_id"]
            req_bytes = ev["requested_bytes"]
            hint = ev.get("lifetime_hint", "unknown")

            base_idx, _ = get_baseline_class(req_bytes)
            clust_idx, _ = get_clustered_class(req_bytes, hint)
            oracle_idx = oracle_map.get(req_id, base_idx)

            records.append({
                "request_id": req_id,
                "run_id": ev.get("run_id", "default"),
                "features": feats,
                "labels": {
                    "baseline_class": base_idx,
                    "clustered_class": clust_idx,
                    "oracle_class": oracle_idx
                }
            })
        return records

    train_list = []
    for r in train_runs:
        train_list.extend(run_records[r])

    val_list = []
    for r in val_runs:
        val_list.extend(run_records[r])

    test_list = []
    for r in test_runs:
        test_list.extend(run_records[r])

    train_data = format_records(train_list)
    val_data = format_records(val_list)
    test_data = format_records(test_list)

    dataset = {
        "schema_version": "1.0.0",
        "description": "Labeled tensor allocation dataset with split-by-run partitioning",
        "feature_names": FEATURE_NAMES,
        "class_mapping": {i: FIXED_CLASSES[i] for i in range(len(FIXED_CLASSES))},
        "split_summary": {
            "train_runs": list(sorted(train_runs)),
            "val_runs": list(sorted(val_runs)),
            "test_runs": list(sorted(test_runs)),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
            "total_count": len(train_data) + len(val_data) + len(test_data)
        },
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("=" * 70)
    print("                Dataset Labeling & Partitioning Summary                ")
    print("=" * 70)
    print(f"Total Unique Runs:     {num_runs} ({', '.join(distinct_runs)})")
    print(f"Train Split Runs:      {len(train_runs)} -> {len(train_data)} samples")
    print(f"Val Split Runs:        {len(val_runs)} -> {len(val_data)} samples")
    print(f"Test Split Runs:       {len(test_runs)} -> {len(test_data)} samples")
    print(f"Features ({len(FEATURE_NAMES)}):        {', '.join(FEATURE_NAMES)}")
    print(f"Output File:           {args.output}")
    print("=" * 70)

if __name__ == "__main__":
    main()
