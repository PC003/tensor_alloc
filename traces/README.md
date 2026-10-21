# traces/

This directory contains recorded and synthetic workload traces in JSON Lines (`.jsonl`) format.

## Schema Version 1.0.0

Each trace file begins with an optional metadata header:
```json
{"schema_version": "1.0.0", "trace_type": "synthetic_ml_workload", "seed": 42, "created_at_utc": "2026-10-21T05:00:00Z", "total_target_events": 5000}
```

Followed by event records:

### Allocation Event (`"event_type": "alloc"`)
```json
{
  "event_type": "alloc",
  "request_id": 42,
  "shape": [32, 128, 768],
  "strides": [98304, 768, 1],
  "dtype": "float32",
  "device": "cpu",
  "requested_bytes": 12582912,
  "lifetime_hint": "temporary",
  "timestamp_ns": 12458900
}
```

### Deallocation Event (`"event_type": "free"`)
```json
{
  "event_type": "free",
  "request_id": 42,
  "timestamp_ns": 15682100
}
```

## Units and Semantics
- `requested_bytes`: Authoritative requested size in bytes.
- `strides`: Measured in **elements** (matching NumPy and PyTorch layout convention).
- `timestamp_ns`: Relative nanoseconds elapsed from start of trace execution.
- Tensor buffer contents are **never** stored to avoid unnecessary I/O and privacy leakage.
