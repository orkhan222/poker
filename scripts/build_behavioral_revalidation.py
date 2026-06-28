from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.behavioral_revalidation import write_behavioral_revalidation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build human-likeness and action-distribution revalidation contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "behavioral_revalidation.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "behavioral_revalidation.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_behavioral_revalidation(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "human_likeness_status": payload["current_validation_scope"]["human_likeness_status"],
                "action_distribution_status": payload["current_validation_scope"]["action_distribution_status"],
                "revalidation_required": payload["revalidation_boundary"]["larger_clean_real_gameplay_revalidation_required"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
