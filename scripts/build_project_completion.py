from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.project_completion import write_project_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build project completion contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "project_completion.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "project_completion.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_project_completion(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "delivery_verification": payload["delivery_status"]["delivery_verification"],
                "deployed_strategy_stack": payload["delivery_status"]["deployed_strategy_stack"],
                "phase_3": payload["phase_completion"]["phase_3_evaluation"]["status"],
                "phase_4": payload["phase_completion"]["phase_4_deployment"]["status"],
                "raw_supervised_model": payload["known_boundary"]["raw_supervised_model_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
