"""CI eval gate. Runs a fast ~10-case smoke eval and fails the build if a metric
drops more than --max-drop below the committed baseline. This is the eval-from-day-one
rule so regressions can't sneak in. Called from .github/workflows/ci.yml:

    python -m src.eval.smoke_gate --baseline data/eval/baseline.json --max-drop 5.0
"""

from __future__ import annotations

import argparse
import sys


def load_baseline(path: str) -> dict[str, float]:
    """Load committed baseline metrics: {metric_name: value}."""
    raise NotImplementedError("week 4 wed: read baseline.json")


def run_smoke_metrics(*, deterministic_only: bool) -> dict[str, float]:
    """Run the ~10-case smoke set; return {metric_name: value}."""
    raise NotImplementedError("week 4 wed: run smoke subset")


def check_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    max_drop: float,
) -> list[str]:
    """Return a list of failure messages (empty means pass). A failure is any metric
    where baseline - current > max_drop. Pure function, so I unit-test the boundary
    (drop == max_drop passes; drop > max_drop fails).
    """
    raise NotImplementedError("week 4 wed: compare current vs baseline")


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Returns 0 on pass, 1 on regression (so CI fails the build)."""
    parser = argparse.ArgumentParser(description="CI eval gate")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--max-drop", type=float, default=5.0)
    parser.parse_args(argv)
    raise NotImplementedError("week 4 wed: wire load/run/check; return exit code")


if __name__ == "__main__":
    sys.exit(main())
