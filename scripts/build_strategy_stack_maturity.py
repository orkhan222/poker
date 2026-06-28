from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.strategy_stack_maturity import write_strategy_stack_maturity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strategy stack maturity and deployment boundary contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "strategy_stack_maturity.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "strategy_stack_maturity.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_strategy_stack_maturity(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "current_strategy_stack": payload["current_strategy_stack"]["status"],
                "final_engine_status": payload["final_engine_boundary"]["status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
