from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.actions_dataset_export_contract import write_actions_dataset_export_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build actions.csv future dataset export contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "actions_dataset_export_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "actions_dataset_export_contract.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_actions_dataset_export_contract(
        args.project_root,
        args.out,
        args.markdown_out,
    )
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "status": payload["status"],
                "required_explicit_fields": payload["required_explicit_fields"],
                "current_delivery_blocker": payload["current_delivery_boundary"]["current_delivery_blocker"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
