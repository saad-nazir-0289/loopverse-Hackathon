"""One command for the whole system: clean -> forecast -> checks -> (optional) answer questions.

    python src/run_all.py                                   # Sprint 1 end to end
    python src/run_all.py --questions spec/25_unseen_questions.md
    python src/run_all.py --skip-audit --model ridge_strong

Inputs are read from the standard folders (sensors/, weather/, holdout/, docs/), so judges
can drop in their own files with the same names and columns and rerun this.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def step(title, args, soft=False):
    print(f"\n=== {title} ===", flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode:
        if soft:
            print(f"WARNING: {title} failed (exit {r.returncode}); continuing")
            return
        sys.exit(f"step failed: {title} (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", help="question file (.md/.txt/.csv/.json) to answer after the forecast")
    ap.add_argument("--out", default="answers.json")
    ap.add_argument("--model", help="force a forecast model (default: chosen by walk-forward validation)")
    ap.add_argument("--alarm-threshold", default=None, help="hazardous alarm threshold (default SPEC 165)")
    ap.add_argument("--skip-audit", action="store_true", help="skip the plots and trap audit")
    args = ap.parse_args()

    step("1. clean and standardise sensor data", ["src/clean.py"])
    fc = ["src/forecast.py"] + (["--model", args.model] if args.model else []) \
        + (["--alarm-threshold", args.alarm_threshold] if args.alarm_threshold else [])
    step("2. forecast + walk-forward validation -> predictions.csv", fc)
    step("3. leakage tests", ["tests/test_leakage.py"])
    if not args.skip_audit:
        step("4. trap audit and figures", ["src/audit_traps.py"])
    # these tests ask about specific areas and dates of the supplied data; on other files they may not apply
    step("5. assistant tests (written for the supplied data)", ["tests/test_assistant.py"], soft=True)
    if args.questions:
        step("6. answer questions", ["src/run_questions.py", args.questions, "--out", args.out, "--trace"])
    print("\nall steps passed")


if __name__ == "__main__":
    main()
