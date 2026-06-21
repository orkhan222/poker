from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.model_risk_register import write_model_risk_register


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model risk register")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "model_risk_register.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "model_risk_register.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_model_risk_register(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "deployed_strategy_stack_status": payload["deployed_strategy_stack_status"],
                "raw_supervised_model_status": payload["raw_supervised_model_status"],
                "open_component_risks": payload["risk_summary"]["component_risks"],
                "deployment_blockers": payload["risk_summary"]["deployment_blockers"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
