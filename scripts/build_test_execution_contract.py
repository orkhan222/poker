from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.test_execution_contract import write_test_execution_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the test execution transparency contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "test_execution_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "test_execution_contract.md", type=Path)
    parser.add_argument("--full-pytest-status", default="TIMEOUT", choices=("PASS", "FAIL", "TIMEOUT", "NOT_RUN"))
    parser.add_argument("--full-pytest-timeout-seconds", default=124, type=int)
    parser.add_argument("--critical-tests-status", default="PASS", choices=("PASS", "FAIL", "NOT_RUN"))
    parser.add_argument("--critical-tests-passed", default=26, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_test_execution_contract(
        args.project_root,
        args.out,
        args.markdown_out,
        full_pytest_status=args.full_pytest_status,
        full_pytest_timeout_seconds=args.full_pytest_timeout_seconds,
        critical_tests_status=args.critical_tests_status,
        critical_tests_passed=args.critical_tests_passed,
    )
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "boundary": payload["boundary"],
                "full_pytest_status": payload["full_pytest"]["status"],
                "critical_validation_status": payload["critical_validation"]["status"],
                "delivery_verifier_status": payload["delivery_verifier"]["status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
