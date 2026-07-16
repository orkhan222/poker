from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.acceptance_criteria import (
    AcceptanceCriteria,
    build_acceptance_metrics,
    evaluate_acceptance_criteria,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate numeric delivery acceptance criteria")
    parser.add_argument("--metrics", type=Path, default=None, help="JSON file with latency/invalid-action/validation/reproducibility metrics")
    parser.add_argument("--out", type=Path, default=Path("reports/acceptance_criteria.json"))
    parser.add_argument("--smoke", action="store_true", help="Use deterministic contract smoke observations when no metrics file exists")
    parser.add_argument("--latency-p95-ms-max", type=float, default=150.0)
    parser.add_argument("--latency-p99-ms-max", type=float, default=300.0)
    parser.add_argument("--invalid-action-rate-max", type=float, default=0.0)
    parser.add_argument("--validation-pass-rate-min", type=float, default=1.0)
    parser.add_argument("--reproducibility-pass-rate-min", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    criteria = AcceptanceCriteria(
        latency_p95_ms_max=args.latency_p95_ms_max,
        latency_p99_ms_max=args.latency_p99_ms_max,
        invalid_action_rate_max=args.invalid_action_rate_max,
        validation_pass_rate_min=args.validation_pass_rate_min,
        reproducibility_pass_rate_min=args.reproducibility_pass_rate_min,
    )
    if args.metrics and args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    elif args.smoke:
        metrics = smoke_metrics()
    else:
        raise SystemExit("Provide --metrics or use --smoke for a deterministic contract smoke check.")

    report = evaluate_acceptance_criteria(metrics, criteria)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


def smoke_metrics() -> dict:
    return build_acceptance_metrics(
        latencies_ms=[18.0, 21.0, 25.0, 28.0, 32.0],
        prediction_payloads=[
            {"action": "call", "legal_actions": ["fold", "call", "raise", "all_in"]},
            {"action": "check", "legal_actions": ["check", "bet", "all_in"]},
        ],
        validation_checks=[True, {"name": "schema", "status": "PASS"}],
        reproducibility_checks=[True, {"name": "seeded_rl_episode", "passed": True, "hash_mismatch": False}],
    )


if __name__ == "__main__":
    main()
