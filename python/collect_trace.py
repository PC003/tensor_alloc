#!/usr/bin/env python3
"""
collect_trace.py: Comprehensive trace collection and generation for ML workloads.

Generates versioned JSONL or CSV allocation traces representing:
1. Transformer (e.g. BERT/LLaMA attention & MLP activations)
2. ConvNet (e.g. ResNet multi-stage feature maps & residual additions)
3. Synthetic mixed distributions

Schema Version: 1.1.0
Units: requested_bytes in bytes, timestamps in integer nanoseconds from trace start.
"""

import argparse
import csv
import json
import random
import time

SCHEMA_VERSION = "1.1.0"

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

def generate_transformer_step(batch_size, seq_len, hidden_dim, num_heads, ffn_dim, run_id, next_id_start, start_ns):
    """
    Simulates one transformer layer forward pass allocation sequence:
    Q, K, V projections -> Attention scores -> Softmax -> Context -> FFN1 -> FFN2
    """
    events = []
    head_dim = hidden_dim // num_heads
    req_id = next_id_start

    # Layer activations sequence: (name, shape, dtype, lifetime_hint)
    patterns = [
        # Q, K, V projections
        ("q_proj", [batch_size, num_heads, seq_len, head_dim], "float16", "temporary"),
        ("k_proj", [batch_size, num_heads, seq_len, head_dim], "float16", "temporary"),
        ("v_proj", [batch_size, num_heads, seq_len, head_dim], "float16", "temporary"),
        # Attention score matrix [batch, heads, seq, seq]
        ("attn_scores", [batch_size, num_heads, seq_len, seq_len], "float32", "temporary"),
        # Softmax probabilities
        ("attn_probs", [batch_size, num_heads, seq_len, seq_len], "float16", "temporary"),
        # Attention output context
        ("attn_context", [batch_size, seq_len, hidden_dim], "float16", "temporary"),
        # FFN intermediate expansion
        ("ffn_expanded", [batch_size, seq_len, ffn_dim], "float16", "temporary"),
        # FFN output projection
        ("ffn_output", [batch_size, seq_len, hidden_dim], "float16", "temporary"),
        # Layer norm / residual
        ("residual_accum", [batch_size, seq_len, hidden_dim], "float32", "persistent"),
    ]

    active = []
    for name, shape, dtype, lifetime in patterns:
        elem_size = DTYPE_SIZES[dtype]
        strides = []
        stride = 1
        for d in reversed(shape):
            strides.insert(0, stride)
            stride *= d
        bytes_needed = stride * elem_size

        now_ns = time.time_ns() - start_ns
        alloc_ev = {
            "event_type": "alloc",
            "run_id": run_id,
            "request_id": req_id,
            "name": name,
            "shape": shape,
            "ndim": len(shape),
            "strides": strides,
            "dtype": dtype,
            "device": "cpu",
            "requested_bytes": bytes_needed,
            "lifetime_hint": lifetime,
            "timestamp_ns": now_ns
        }
        events.append(alloc_ev)
        active.append(alloc_ev)
        req_id += 1

    # Free temporary buffers after computation
    for alloc_ev in active:
        if alloc_ev["lifetime_hint"] == "temporary":
            now_ns = time.time_ns() - start_ns
            free_ev = {
                "event_type": "free",
                "run_id": run_id,
                "request_id": alloc_ev["request_id"],
                "timestamp_ns": now_ns
            }
            events.append(free_ev)

    return events, req_id, [a for a in active if a["lifetime_hint"] == "persistent"]

def generate_convnet_step(batch_size, run_id, next_id_start, start_ns):
    """
    Simulates a ConvNet (ResNet-style) forward stage sequence:
    Input -> Conv1 (64ch) -> Conv2 (128ch) -> Conv3 (256ch) -> Conv4 (512ch) -> Pool -> FC
    """
    events = []
    req_id = next_id_start

    stages = [
        ("stage1_conv", [batch_size, 64, 56, 56], "float32", "temporary"),
        ("stage1_res",  [batch_size, 64, 56, 56], "float32", "persistent"),
        ("stage2_conv", [batch_size, 128, 28, 28], "float32", "temporary"),
        ("stage2_res",  [batch_size, 128, 28, 28], "float32", "persistent"),
        ("stage3_conv", [batch_size, 256, 14, 14], "float32", "temporary"),
        ("stage3_res",  [batch_size, 256, 14, 14], "float32", "persistent"),
        ("stage4_conv", [batch_size, 512, 7, 7], "float32", "temporary"),
        ("stage4_pool", [batch_size, 512, 1, 1], "float32", "temporary"),
        ("classifier_logits", [batch_size, 1000], "float32", "temporary")
    ]

    active = []
    for name, shape, dtype, lifetime in stages:
        elem_size = DTYPE_SIZES[dtype]
        strides = []
        stride = 1
        for d in reversed(shape):
            strides.insert(0, stride)
            stride *= d
        bytes_needed = stride * elem_size

        now_ns = time.time_ns() - start_ns
        alloc_ev = {
            "event_type": "alloc",
            "run_id": run_id,
            "request_id": req_id,
            "name": name,
            "shape": shape,
            "ndim": len(shape),
            "strides": strides,
            "dtype": dtype,
            "device": "cpu",
            "requested_bytes": bytes_needed,
            "lifetime_hint": lifetime,
            "timestamp_ns": now_ns
        }
        events.append(alloc_ev)
        active.append(alloc_ev)
        req_id += 1

    for alloc_ev in active:
        if alloc_ev["lifetime_hint"] == "temporary":
            now_ns = time.time_ns() - start_ns
            free_ev = {
                "event_type": "free",
                "run_id": run_id,
                "request_id": alloc_ev["request_id"],
                "timestamp_ns": now_ns
            }
            events.append(free_ev)

    return events, req_id, [a for a in active if a["lifetime_hint"] == "persistent"]

def main():
    parser = argparse.ArgumentParser(description="Collect and generate structured tensor allocation traces.")
    parser.add_argument("--workload", choices=["transformer", "convnet", "mixed"], default="transformer")
    parser.add_argument("--output", "-o", default="traces/workload.jsonl", help="Output file path (.jsonl or .csv)")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output file format")
    parser.add_argument("--runs", type=int, default=3, help="Number of distinct runs to simulate")
    parser.add_argument("--steps", type=int, default=50, help="Number of model iteration steps per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    start_ns = time.time_ns()

    all_events = []
    header = {
        "schema_version": SCHEMA_VERSION,
        "trace_type": f"{args.workload}_ml_workload",
        "num_runs": args.runs,
        "steps_per_run": args.steps,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed
    }

    req_id_counter = 1
    for r in range(1, args.runs + 1):
        run_id = f"run_{r:03d}"
        persistent_pool = []

        for s in range(args.steps):
            if args.workload == "transformer":
                evs, req_id_counter, pers = generate_transformer_step(
                    batch_size=16, seq_len=128, hidden_dim=768, num_heads=12, ffn_dim=3072,
                    run_id=run_id, next_id_start=req_id_counter, start_ns=start_ns)
            elif args.workload == "convnet":
                evs, req_id_counter, pers = generate_convnet_step(
                    batch_size=32, run_id=run_id, next_id_start=req_id_counter, start_ns=start_ns)
            else:
                # Mixed: interleave transformer and convnet steps
                if s % 2 == 0:
                    evs, req_id_counter, pers = generate_transformer_step(
                        batch_size=8, seq_len=64, hidden_dim=512, num_heads=8, ffn_dim=2048,
                        run_id=run_id, next_id_start=req_id_counter, start_ns=start_ns)
                else:
                    evs, req_id_counter, pers = generate_convnet_step(
                        batch_size=16, run_id=run_id, next_id_start=req_id_counter, start_ns=start_ns)

            all_events.extend(evs)
            persistent_pool.extend(pers)

        # Free persistent buffers at the end of the run
        for p in persistent_pool:
            now_ns = time.time_ns() - start_ns
            all_events.append({
                "event_type": "free",
                "run_id": run_id,
                "request_id": p["request_id"],
                "timestamp_ns": now_ns
            })

    # Save to file
    if args.format == "csv" or args.output.endswith(".csv"):
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["event_type", "run_id", "request_id", "name", "shape", "ndim", "dtype", "device", "requested_bytes", "lifetime_hint", "timestamp_ns"])
            for ev in all_events:
                shape_str = ";".join(str(x) for x in ev.get("shape", [])) if "shape" in ev else ""
                writer.writerow([
                    ev["event_type"],
                    ev["run_id"],
                    ev["request_id"],
                    ev.get("name", ""),
                    shape_str,
                    ev.get("ndim", 0),
                    ev.get("dtype", ""),
                    ev.get("device", ""),
                    ev.get("requested_bytes", 0),
                    ev.get("lifetime_hint", ""),
                    ev["timestamp_ns"]
                ])
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for ev in all_events:
                f.write(json.dumps(ev) + "\n")

    print(f"Generated {len(all_events)} events across {args.runs} runs into '{args.output}'.")

if __name__ == "__main__":
    main()
