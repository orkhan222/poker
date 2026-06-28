from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.challenger_strategy_quality import write_challenger_strategy_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build challenger strategy-quality claim boundary")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "challenger_strategy_quality.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "challenger_strategy_quality.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_challenger_strategy_quality(args.project_root, args.out, args.markdown_out)
    boundary = payload["strategy_quality_boundary"]
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "strategy_quality_status": boundary["status"],
                "final_production_strategy_quality_claim_allowed": boundary[
                    "final_production_strategy_quality_claim_allowed"
                ],
                "challenger_gate_status": boundary["challenger_gate_status"],
                "raw_production_gate_status": boundary["raw_production_gate_status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
