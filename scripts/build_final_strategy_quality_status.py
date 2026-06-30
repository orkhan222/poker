from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.final_strategy_quality_status import write_final_strategy_quality_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final production-level strategy quality boundary")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "final_strategy_quality_status.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "final_strategy_quality_status.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_final_strategy_quality_status(args.project_root, args.out, args.markdown_out)
    boundary = payload["final_strategy_quality_boundary"]
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "software_delivery_ready": payload["delivery_boundary"]["software_delivery_ready"],
                "final_strategy_quality_status": boundary["status"],
                "final_production_strategy_quality_approved": boundary[
                    "final_production_strategy_quality_approved"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
