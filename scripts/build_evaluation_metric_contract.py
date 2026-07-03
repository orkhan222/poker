from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluation_metric_contract import write_evaluation_metric_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the production evaluation metric contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "evaluation_metric_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "evaluation_metric_contract.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_evaluation_metric_contract(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "boundary": payload["boundary"],
                "final_metric_bundle_passed": payload["final_metric_bundle_passed"],
                "final_strategy_quality_claim_allowed": payload["final_strategy_quality_claim_allowed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
