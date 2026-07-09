from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.rl_delivery_boundary import write_rl_delivery_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RL delivery and strategy-claim boundary")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "rl_delivery_boundary.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "rl_delivery_boundary.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_rl_delivery_boundary(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "boundary": payload["boundary"],
                "current_delivery_blocker": payload["current_delivery_blocker"],
                "self_play_win_rate_claim_allowed": payload["claim_permissions"][
                    "self_play_win_rate_claim_allowed"
                ],
                "production_strategy_quality_claim_allowed": payload["claim_permissions"][
                    "production_strategy_quality_claim_allowed"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
