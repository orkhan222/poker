from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.normalized_action_contract import write_normalized_action_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the normalized action schema contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "normalized_action_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "normalized_action_contract.md", type=Path)
    parser.add_argument("--max-rows", default=5000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_normalized_action_contract(
        args.project_root,
        args.out,
        args.markdown_out,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "normalized_action_status": payload["normalized_action_status"],
                "canonical_actions": payload["canonical_actions"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
