#!/usr/bin/env python3
"""
test_phase3_pipeline.py: Automated tests for Phase 3 training and evaluation pipeline.
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
    print("[test_phase3_pipeline] Running Phase 3 model training & evaluation pipeline tests...")

    test_model_json = "models/test_temp_model.json"

    # 1. Test training on existing labeled dataset
    out_train = run_cmd(f"python3 python/train_predictor.py --dataset traces/labeled_dataset.json --output {test_model_json} --max-depth 5 --min-samples 3")
    assert "Training Duration" in out_train, "Training output missing duration"
    assert "Train Accuracy" in out_train, "Training output missing accuracy"
    assert os.path.exists(test_model_json), "Model JSON artifact was not created"

    # 2. Verify model artifact structure
    with open(test_model_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["schema_version"] == "1.0.0", "Model schema version mismatch"
    assert "tree" in m, "Tree structure missing in artifact"
    assert "metrics" in m, "Metrics missing in artifact"
    assert m["metrics"]["train_accuracy"] >= 0.90, "Train accuracy unexpectedly low"

    # 3. Test model evaluation script
    out_eval = run_cmd(f"python3 python/evaluate_predictor.py --model {test_model_json} --dataset traces/labeled_dataset.json")
    assert "Learned Predictor vs. Baselines Evaluation Report" in out_eval, "Evaluation header missing"
    assert "Inference Performance" in out_eval, "Inference latency missing"
    assert "Fixed Power-of-2 Baseline" in out_eval, "Baseline policy missing"
    assert "Learned Decision Tree" in out_eval, "Learned policy missing"

    # Clean up test artifact
    if os.path.exists(test_model_json):
        os.remove(test_model_json)

    print("[test_phase3_pipeline] All Phase 3 model training and evaluation tests passed successfully!")

if __name__ == "__main__":
    main()
