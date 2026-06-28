from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.qlora_next_stage import write_qlora_next_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the QLoRA next-stage boundary contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "qlora_next_stage.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "qlora_next_stage.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_qlora_next_stage(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "stage_status": payload["stage_boundary"]["stage_status"],
                "fine_tuning_completed": payload["stage_boundary"]["fine_tuning_completed"],
                "production_approved": payload["stage_boundary"]["production_approved"],
                "current_delivery_blocker": payload["stage_boundary"]["current_delivery_blocker"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
