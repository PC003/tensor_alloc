#!/usr/bin/env python3
"""
collect_trace.py: Synthetic trace generator for tensor memory allocation workloads.

Emits JSONL records representing tensor allocation and deallocation events.
Schema Version: 1.0.0
Units: requested_bytes in bytes, timestamps in integer nanoseconds from trace start.
"""

import argparse
import json
import random
import sys
import time

SCHEMA_VERSION = "1.0.0"

DTYPE_SIZES = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "int64": 8,
    "int8": 1,
    "uint8": 1,
    "bool": 1
}

COMMON_SHAPES = [
    ([32, 128, 768], "float32"),       # Transformer hidden state (BERT-base)
    ([1, 3, 224, 224], "float32"),     # Vision batch input (ResNet)
    ([64, 3072], "float32"),           # MLP intermediate projection
    ([16, 8, 128, 64], "float16"),     # Multi-head attention Q/K/V
    ([128, 512], "float32"),           # Embedding lookup batch
    ([32, 32], "float32"),             # Small matrix multiplication
    ([4, 16, 256], "int32"),           # Token ID tensors
]

def generate_trace(output_file, num_events, seed=42):
    random.seed(seed)
    start_ns = time.time_ns()

    live_tensors = []
    next_req_id = 1

    with open(output_file, "w", encoding="utf-8") as f:
        # Write metadata header
        header = {
            "schema_version": SCHEMA_VERSION,
            "trace_type": "synthetic_ml_workload",
            "seed": seed,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_target_events": num_events
        }
        f.write(json.dumps(header) + "\n")

        events_written = 0
        for _ in range(num_events):
            # 60% alloc, 40% free when live tensors exist
            should_alloc = len(live_tensors) == 0 or (random.random() < 0.6)

            if should_alloc:
                req_id = next_req_id
                next_req_id += 1

                shape, dtype = random.choice(COMMON_SHAPES)
                # Compute contiguous strides
                elem_size = DTYPE_SIZES[dtype]
                strides = []
                stride = 1
                for dim in reversed(shape):
                    strides.insert(0, stride)
                    stride *= dim
                total_elements = stride
                requested_bytes = total_elements * elem_size

                lifetime = random.choice(["temporary", "temporary", "persistent"])
                now_ns = time.time_ns() - start_ns

                record = {
                    "event_type": "alloc",
                    "request_id": req_id,
                    "shape": shape,
                    "strides": strides,
                    "dtype": dtype,
                    "device": "cpu",
                    "requested_bytes": requested_bytes,
                    "lifetime_hint": lifetime,
                    "timestamp_ns": now_ns
                }
                f.write(json.dumps(record) + "\n")
                live_tensors.append(record)
                events_written += 1
            else:
                idx = random.randrange(len(live_tensors))
                allocated = live_tensors.pop(idx)
                now_ns = time.time_ns() - start_ns

                record = {
                    "event_type": "free",
                    "request_id": allocated["request_id"],
                    "timestamp_ns": now_ns
                }
                f.write(json.dumps(record) + "\n")
                events_written += 1

        # Drain remaining live allocations
        for allocated in live_tensors:
            now_ns = time.time_ns() - start_ns
            record = {
                "event_type": "free",
                "request_id": allocated["request_id"],
                "timestamp_ns": now_ns
            }
            f.write(json.dumps(record) + "\n")
            events_written += 1

    print(f"Generated {events_written} trace records into '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic tensor allocation trace.")
    parser.add_argument("--output", "-o", default="traces/synthetic_trace.jsonl", help="Output path")
    parser.add_argument("--events", "-n", type=int, default=5000, help="Number of trace events")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_trace(args.output, args.events, args.seed)
