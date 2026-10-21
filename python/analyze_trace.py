#!/usr/bin/env python3
"""
analyze_trace.py: Summarizes tensor allocation traces.

Computes:
- Request size distributions (percentiles, min, max, mean)
- Internal waste under fixed power-of-two size classes
- Tensor lifetime statistics (duration between alloc and free)
- Data type and lifetime hint frequencies
"""

import argparse
import json
import statistics
import sys

FIXED_CLASSES = [64 * (2 ** i) for i in range(21)] # 64 B to 64 MB

def get_size_class(bytes_needed):
    for c in FIXED_CLASSES:
        if c >= bytes_needed:
            return c
    return ((bytes_needed + 63) // 64) * 64 # Large allocation aligned

def analyze_trace(trace_path):
    allocs = {}
    requested_sizes = []
    lifetimes_ns = []
    dtypes = {}
    hints = {}
    total_waste = 0
    total_reserved = 0
    total_requested = 0

    with open(trace_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "schema_version" in record:
                continue # Header

            ev = record.get("event_type")
            req_id = record.get("request_id")
            ts = record.get("timestamp_ns", 0)

            if ev == "alloc":
                bytes_req = record["requested_bytes"]
                dt = record.get("dtype", "unknown")
                hint = record.get("lifetime_hint", "unknown")

                allocs[req_id] = (bytes_req, ts)
                requested_sizes.append(bytes_req)
                dtypes[dt] = dtypes.get(dt, 0) + 1
                hints[hint] = hints.get(hint, 0) + 1

                cap = get_size_class(bytes_req)
                total_requested += bytes_req
                total_reserved += cap
                total_waste += (cap - bytes_req)

            elif ev == "free":
                if req_id in allocs:
                    bytes_req, alloc_ts = allocs.pop(req_id)
                    lifetimes_ns.append(ts - alloc_ts)

    if not requested_sizes:
        print("No allocation events found in trace.")
        return

    print("=" * 60)
    print(f"  Trace Analysis Summary: {trace_path}")
    print("=" * 60)
    print(f"Total Allocations:     {len(requested_sizes)}")
    print(f"Total Deallocations:   {len(lifetimes_ns)}")
    print(f"Total Requested Bytes: {total_requested:,} bytes ({total_requested / (1024**2):.2f} MB)")
    print(f"Total Reserved Bytes:  {total_reserved:,} bytes ({total_reserved / (1024**2):.2f} MB)")
    waste_pct = (total_waste / total_reserved * 100.0) if total_reserved > 0 else 0.0
    print(f"Total Internal Waste:  {total_waste:,} bytes ({waste_pct:.2f}%)")

    print("\n--- Request Size Distribution (Bytes) ---")
    requested_sizes.sort()
    print(f"  Min:    {requested_sizes[0]:,}")
    print(f"  p25:    {int(statistics.quantiles(requested_sizes, n=4)[0]):,}")
    print(f"  p50:    {int(statistics.median(requested_sizes)):,}")
    print(f"  p75:    {int(statistics.quantiles(requested_sizes, n=4)[2]):,}")
    print(f"  p95:    {int(statistics.quantiles(requested_sizes, n=20)[18]):,}")
    print(f"  p99:    {int(statistics.quantiles(requested_sizes, n=100)[98]):,}")
    print(f"  Max:    {requested_sizes[-1]:,}")
    print(f"  Mean:   {int(statistics.mean(requested_sizes)):,}")

    if lifetimes_ns:
        lifetimes_ms = [t / 1e6 for t in lifetimes_ns]
        lifetimes_ms.sort()
        print("\n--- Lifetime Duration Distribution (Milliseconds) ---")
        print(f"  Min:    {lifetimes_ms[0]:.3f} ms")
        print(f"  Median: {statistics.median(lifetimes_ms):.3f} ms")
        print(f"  Max:    {lifetimes_ms[-1]:.3f} ms")

    print("\n--- Data Type Frequencies ---")
    for dt, count in sorted(dtypes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dt:12s}: {count:5d} ({count / len(requested_sizes) * 100.1:.1f}%)")

    print("\n--- Lifetime Hint Frequencies ---")
    for hint, count in sorted(hints.items(), key=lambda x: x[1], reverse=True):
        print(f"  {hint:12s}: {count:5d} ({count / len(requested_sizes) * 100.1:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a JSONL allocation trace.")
    parser.add_argument("trace", help="Path to JSONL trace file")
    args = parser.parse_args()

    analyze_trace(args.trace)
