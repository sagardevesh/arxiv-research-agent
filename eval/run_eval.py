"""CLI entry point for the RAGAS eval gate.

Exit code 0 = all metrics above threshold (CI passes).
Exit code 1 = one or more metrics below threshold (CI blocks Docker build).

Usage:
    python -m eval.run_eval
"""

import sys
from eval.harness import check_thresholds, run_eval

if __name__ == "__main__":
    print("Running RAGAS evaluation...\n")
    scores = run_eval()
    print("\nResults:")
    passed = check_thresholds(scores)
    print()
    if passed:
        print("Eval PASSED — all metrics above threshold.")
        sys.exit(0)
    else:
        print("Eval FAILED — one or more metrics below threshold.")
        sys.exit(1)
