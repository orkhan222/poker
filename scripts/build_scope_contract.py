from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.scope_contract import write_scope_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DOCX/PDF scope contract report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "scope_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "scope_contract.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_scope_contract(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "delivery_status": payload["delivery_status"],
                "strategy_policy_status": payload["strategy_policy_status"],
                "deployed_strategy_gate_status": payload["deployed_strategy_gate_status"],
                "raw_supervised_model_status": payload["raw_supervised_model_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
