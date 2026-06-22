from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.production_approval import write_production_approval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build production approval contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "production_approval.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "production_approval.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_production_approval(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "delivery_ready": payload["delivery_ready"],
                "deployed_strategy_stack": payload["deployed_strategy_stack"]["status"],
                "raw_supervised_model": payload["raw_supervised_model"]["standalone_status"],
                "deployment_blockers": payload["risk_position"]["deployment_blockers"],
                "component_risks": payload["risk_position"]["component_risks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
