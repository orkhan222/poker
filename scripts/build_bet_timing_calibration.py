from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.bet_timing_calibration import write_bet_timing_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bet-sizing and timing calibration boundary")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "bet_timing_calibration.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "bet_timing_calibration.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_bet_timing_calibration(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "implementation_status": payload["current_delivery_scope"]["implementation_status"],
                "calibration_status": payload["calibration_boundary"]["status"],
                "requires_more_labels": payload["calibration_boundary"]["requires_more_real_player_behavior_labels"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
