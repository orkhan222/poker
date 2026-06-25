from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_decision_gate import build_llm_decision_gate, write_llm_decision_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the independent LLM decision-model acceptance gate")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--holdout-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--min-examples", default=20, type=int)
    parser.add_argument("--min-macro-f1", default=0.40, type=float)
    parser.add_argument("--min-schema-valid-rate", default=0.95, type=float)
    parser.add_argument("--min-legal-action-rate", default=0.99, type=float)
    parser.add_argument("--max-average-latency-ms", default=5000.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_llm_decision_gate(
        args.benchmark,
        args.holdout_report,
        min_examples=args.min_examples,
        min_macro_f1=args.min_macro_f1,
        min_schema_valid_rate=args.min_schema_valid_rate,
        min_legal_action_rate=args.min_legal_action_rate,
        max_average_latency_ms=args.max_average_latency_ms,
    )
    write_llm_decision_gate(payload, args.out, args.report_out)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_context_mode": payload["selected_context_mode"],
                "failed_checks": payload["failed_checks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
