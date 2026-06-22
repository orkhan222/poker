from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_decision_context import write_decision_context_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM decision context contract")
    parser.add_argument("--out", default=ROOT / "reports" / "llm_decision_context.json", type=Path)
    parser.add_argument("--report-out", default=ROOT / "reports" / "llm_decision_context.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_decision_context_report(args.out, args.report_out)
    records = payload["prompt_records"]
    print(
        json.dumps(
            {
                "context_modes": len(payload["supported_context_modes"]),
                "default_context_mode": payload["default_context_mode"],
                "prompt_records": len(records),
                "rules_grounded_records": sum(1 for item in records if item["contains_rules"]),
                "full_context_records": sum(1 for item in records if item["contains_strategy_guidelines"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
