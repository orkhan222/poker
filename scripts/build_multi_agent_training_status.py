from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.multi_agent_training_status import write_multi_agent_training_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-agent training completion boundary report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "multi_agent_training_status.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "multi_agent_training_status.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_multi_agent_training_status(args.project_root, args.out, args.markdown_out)
    boundary = payload["training_boundary"]
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "delivery_validation_status": boundary["delivery_validation_status"],
                "full_production_scale_multi_agent_training_status": boundary[
                    "full_production_scale_multi_agent_training_status"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
