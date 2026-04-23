"""
run_all.py
----------
Master runner: executes the full analytics pipeline end-to-end.
Run this single script to reproduce all results.

Usage: python python/analysis/run_all.py
"""

import subprocess
import sys
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script: str, label: str):
    print(f"\n{'─'*55}")
    print(f"  ▶ {label}")
    print(f"{'─'*55}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, script)],
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ Done in {elapsed:.1f}s")
    else:
        print(f"  ❌ Failed (exit code {result.returncode})")
    return result.returncode


def main():
    print("\n" + "=" * 55)
    print("  E-COMMERCE REVENUE INTELLIGENCE PLATFORM")
    print("  Full Pipeline Run")
    print("=" * 55)

    steps = [
        ("python/etl/generate_data.py",             "Step 1/5 — Generate synthetic data"),
        ("python/etl/pipeline.py",                  "Step 2/5 — Run ETL pipeline"),
        ("python/analysis/reconciliation_engine.py","Step 3/5 — Financial reconciliation"),
        ("python/analysis/marketing_roi.py",        "Step 4/5 — Marketing ROI analysis"),
        ("python/analysis/cohort_analysis.py",      "Step 5a  — Cohort & retention"),
        ("python/analysis/ab_testing.py",           "Step 5b  — A/B test evaluation"),
        ("python/analysis/anomaly_detection.py",    "Step 5c  — Fraud & anomaly detection"),
    ]

    failures = 0
    for script, label in steps:
        code = run(script, label)
        if code != 0:
            failures += 1

    print("\n" + "=" * 55)
    if failures == 0:
        print("  🎉 All steps completed successfully!")
        print("  📁 All outputs written to: data/exports/")
    else:
        print(f"  ⚠️  {failures} step(s) failed. Check logs above.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
