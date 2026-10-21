#!/usr/bin/env python3
"""
analyze_trace.py: Comprehensive trace analysis tool.

Computes:
- Request size distribution percentiles (p10, p25, p50, p75, p90, p95, p99, min, max, mean)
- Temporal concurrency (peak live allocations, peak live memory volume)
- Lifetime duration distribution (ms)
- Size-class utilization and internal waste per class
- Workload drift across runs or between multiple trace files
"""

import argparse
import csv
import json
import statistics
import sys

FIXED_CLASSES = [64 * (2 ** i) for i in range(21)] # 64 B to 64 MB

def get_size_class(bytes_needed):
    for idx, c in enumerate(FIXED_CLASSES):
        if c >= bytes_needed:
            return idx, c
    # Large allocation
    return -1, ((bytes_needed + 63) // 64) * 64

def load_events(filepath):
    events = []
    if filepath.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shape = [int(x) for x in row["shape"].split(";")] if row.get("shape") else []
                ev = {
                    "event_type": row["event_type"],
                    "run_id": row.get("run_id", "default"),
                    "request_id": int(row["request_id"]),
                    "requested_bytes": int(row["requested_bytes"]) if row.get("requested_bytes") else 0,
                    "dtype": row.get("dtype", "unknown"),
                    "lifetime_hint": row.get("lifetime_hint", "unknown"),
                    "timestamp_ns": int(row.get("timestamp_ns", 0)),
                    "shape": shape
                }
                events.append(ev)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "schema_version" in row:
                    continue
                events.append(row)
    return events

def analyze_single_trace(events, trace_name):
    allocs = {}
    requested_sizes = []
    lifetimes_ns = []
    dtypes = {}
    hints = {}
    runs = set()

    # Size class tracking: class_idx -> {count, requested, reserved, waste}
    sc_stats = {}

    # Concurrency tracking
    current_live_allocs = 0
    peak_live_allocs = 0
    current_live_bytes = 0
    peak_live_bytes = 0

    total_requested = 0
    total_reserved = 0
    total_waste = 0

    for ev in events:
        run_id = ev.get("run_id", "default")
        runs.add(run_id)
        req_id = ev["request_id"]
        ts = ev.get("timestamp_ns", 0)

        if ev["event_type"] == "alloc":
            bytes_req = ev["requested_bytes"]
            dt = ev.get("dtype", "unknown")
            hint = ev.get("lifetime_hint", "unknown")

            allocs[req_id] = (bytes_req, ts)
            requested_sizes.append(bytes_req)
            dtypes[dt] = dtypes.get(dt, 0) + 1
            hints[hint] = hints.get(hint, 0) + 1

            sc_idx, cap = get_size_class(bytes_req)
            waste = cap - bytes_req

            total_requested += bytes_req
            total_reserved += cap
            total_waste += waste

            if sc_idx not in sc_stats:
                sc_stats[sc_idx] = {"count": 0, "requested": 0, "reserved": 0, "waste": 0, "cap": cap}
            sc_stats[sc_idx]["count"] += 1
            sc_stats[sc_idx]["requested"] += bytes_req
            sc_stats[sc_idx]["reserved"] += cap
            sc_stats[sc_idx]["waste"] += waste

            current_live_allocs += 1
            if current_live_allocs > peak_live_allocs:
                peak_live_allocs = current_live_allocs

            current_live_bytes += bytes_req
            if current_live_bytes > peak_live_bytes:
                peak_live_bytes = current_live_bytes

        elif ev["event_type"] == "free":
            if req_id in allocs:
                bytes_req, alloc_ts = allocs.pop(req_id)
                lifetimes_ns.append(ts - alloc_ts)
                current_live_allocs -= 1
                current_live_bytes -= bytes_req

    print("=" * 80)
    print(f"  Trace Analysis Report: {trace_name}")
    print("=" * 80)
    print(f"Total Allocations:       {len(requested_sizes):,}")
    print(f"Total Frees:             {len(lifetimes_ns):,}")
    print(f"Distinct Runs:           {len(runs)} ({', '.join(sorted(runs))})")
    print(f"Total Volume Requested:  {total_requested:,} B ({total_requested / (1024**2):.2f} MB)")
    print(f"Total Volume Reserved:   {total_reserved:,} B ({total_reserved / (1024**2):.2f} MB)")
    waste_pct = (total_waste / total_reserved * 100.0) if total_reserved > 0 else 0.0
    print(f"Internal Fragmentation:  {total_waste:,} B ({waste_pct:.2f}% waste)")
    print(f"Peak Live Allocations:   {peak_live_allocs:,} buffers")
    print(f"Peak Live Memory:        {peak_live_bytes:,} B ({peak_live_bytes / (1024**2):.2f} MB)")

    if requested_sizes:
        requested_sizes.sort()
        print("\n--- Request Size Percentiles (Bytes) ---")
        q = statistics.quantiles(requested_sizes, n=100) if len(requested_sizes) >= 100 else [requested_sizes[len(requested_sizes)//2]] * 100
        print(f"  Min:  {requested_sizes[0]:,}")
        print(f"  p10:  {int(q[9]):,}")
        print(f"  p25:  {int(q[24]):,}")
        print(f"  p50:  {int(statistics.median(requested_sizes)):,}")
        print(f"  p75:  {int(q[74]):,}")
        print(f"  p90:  {int(q[89]):,}")
        print(f"  p99:  {int(q[98]):,}")
        print(f"  Max:  {requested_sizes[-1]:,}")
        print(f"  Mean: {int(statistics.mean(requested_sizes)):,}")

    if lifetimes_ns:
        lifetimes_ms = [t / 1e6 for t in lifetimes_ns]
        lifetimes_ms.sort()
        print("\n--- Lifetime Durations (Milliseconds) ---")
        print(f"  Min:    {lifetimes_ms[0]:.3f} ms")
        print(f"  Median: {statistics.median(lifetimes_ms):.3f} ms")
        print(f"  Max:    {lifetimes_ms[-1]:.3f} ms")

    print("\n--- Size Class Distribution & Internal Waste ---")
    print("| Class ID | Class Cap | Alloc Count |   Total Req |   Total Res | Waste % |")
    print("|----------|-----------|-------------|-------------|-------------|---------|")
    for sc_idx in sorted(sc_stats.keys()):
        st = sc_stats[sc_idx]
        cap_str = f"{st['cap']:,} B" if sc_idx >= 0 else "Large (Aligned)"
        class_id_str = f"Class {sc_idx:02d}" if sc_idx >= 0 else "LARGE"
        w_pct = (st["waste"] / st["reserved"] * 100.0) if st["reserved"] > 0 else 0.0
        print(f"| {class_id_str:8s} | {cap_str:>9s} | {st['count']:11d} | {st['requested']:9,d} B | {st['reserved']:9,d} B | {w_pct:6.2f}% |")

    print("\n--- Data Type Frequencies ---")
    for dt, count in sorted(dtypes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dt:12s}: {count:5d} ({count / len(requested_sizes) * 100.0:.1f}%)")

    print("\n--- Lifetime Hint Frequencies ---")
    for hint, count in sorted(hints.items(), key=lambda x: x[1], reverse=True):
        print(f"  {hint:12s}: {count:5d} ({count / len(requested_sizes) * 100.0:.1f}%)")
    print("=" * 80 + "\n")

    return {
        "total_allocs": len(requested_sizes),
        "total_requested": total_requested,
        "total_reserved": total_reserved,
        "waste_pct": waste_pct,
        "peak_live_bytes": peak_live_bytes
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze allocation traces and compute distributions.")
    parser.add_argument("traces", nargs="+", help="One or more trace files (.jsonl or .csv)")
    parser.add_argument("--compare", action="store_true", help="Compare multiple traces side-by-side")
    args = parser.parse_args()

    summaries = []
    for tp in args.traces:
        events = load_events(tp)
        s = analyze_single_trace(events, tp)
        summaries.append((tp, s))

    if args.compare and len(summaries) > 1:
        print("=" * 70)
        print("                   Workload Comparison Summary                        ")
        print("=" * 70)
        print("| Trace File                   | Allocs | Total Req MB | Peak MB | Waste % |")
        print("|------------------------------|--------|--------------|---------|---------|")
        for tp, s in summaries:
            fname = tp.split("/")[-1]
            req_mb = s["total_requested"] / (1024**2)
            peak_mb = s["peak_live_bytes"] / (1024**2)
            print(f"| {fname:28s} | {s['total_allocs']:6d} | {req_mb:10.2f} MB | {peak_mb:6.2f} MB | {s['waste_pct']:6.2f}% |")
        print("=" * 70)

if __name__ == "__main__":
    main()
