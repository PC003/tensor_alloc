#!/usr/bin/env python3
"""
test_trace_pipeline.py: Automated end-to-end testing for Phase 2 trace collection,
analysis, replay, and label generation.
"""

import os
import subprocess
import json
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT_DIR)
    if res.returncode != 0:
        print(f"[FAIL] Command failed: {cmd}\nOutput:\n{res.stdout}\nError:\n{res.stderr}")
        sys.exit(1)
    return res.stdout

def main():
    os.chdir(ROOT_DIR)
    print("[test_trace_pipeline] Running end-to-end Python trace pipeline tests...")

    test_trace_jsonl = "traces/test_temp_pipeline.jsonl"
    test_trace_csv = "traces/test_temp_pipeline.csv"
    test_dataset_json = "traces/test_temp_dataset.json"

    # 1. Test trace generation in JSONL format
    run_cmd(f"python3 python/collect_trace.py --workload transformer --output {test_trace_jsonl} --runs 3 --steps 10")
    assert os.path.exists(test_trace_jsonl), "JSONL trace was not created"

    # 2. Test trace generation in CSV format
    run_cmd(f"python3 python/collect_trace.py --workload convnet --output {test_trace_csv} --format csv --runs 2 --steps 10")
    assert os.path.exists(test_trace_csv), "CSV trace was not created"

    # 3. Test trace analysis on JSONL
    out = run_cmd(f"python3 python/analyze_trace.py {test_trace_jsonl}")
    assert "Trace Analysis Report" in out, "Analysis report missing from output"
    assert "Internal Fragmentation" in out, "Fragmentation missing from output"

    # 4. Test trace analysis on CSV
    out_csv = run_cmd(f"python3 python/analyze_trace.py {test_trace_csv}")
    assert "Trace Analysis Report" in out_csv, "CSV analysis report missing"

    # 5. Test trace comparison
    comp_out = run_cmd(f"python3 python/analyze_trace.py --compare {test_trace_jsonl} {test_trace_csv}")
    assert "Workload Comparison Summary" in comp_out, "Comparison summary table missing"

    # 6. Test label generation with train/val/test split
    run_cmd(f"python3 python/build_labels.py {test_trace_jsonl} --output {test_dataset_json}")
    assert os.path.exists(test_dataset_json), "Labeled dataset JSON missing"

    with open(test_dataset_json, "r", encoding="utf-8") as f:
        ds = json.load(f)

    assert "train" in ds and "val" in ds and "test" in ds, "Missing splits in dataset"
    assert len(ds["train"]) > 0, "Train split is empty"
    assert len(ds["feature_names"]) == 11, "Feature names count mismatch"

    # Verify split independence (no run_id overlap between train and test)
    train_runs = set(x["run_id"] for x in ds["train"])
    test_runs = set(x["run_id"] for x in ds["test"])
    if len(ds["split_summary"]["train_runs"]) > 1:
        assert train_runs.isdisjoint(test_runs), "Data leakage detected: train and test share run_ids!"

    # 7. Test C++ replay on the generated JSONL trace
    replay_out = run_cmd(f"./build_release/replay_trace {test_trace_jsonl}")
    assert "Pool Reuse Count" in replay_out, "Replay did not output pool reuse metrics"

    # Clean up test artifacts
    for f in [test_trace_jsonl, test_trace_csv, test_dataset_json]:
        if os.path.exists(f):
            os.remove(f)

    print("[test_trace_pipeline] All Python trace pipeline and replay tests passed successfully!")

if __name__ == "__main__":
    main()
