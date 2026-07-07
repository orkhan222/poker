from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.open_spiel_claim_contract import write_open_spiel_claim_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OpenSpiel/RL self-play claim contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "open_spiel_claim_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "open_spiel_claim_contract.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_open_spiel_claim_contract(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "code_status": payload["code_status"],
                "self_play_win_rate_claim_allowed": payload["self_play_win_rate_claim_allowed"],
                "current_delivery_blocker": payload["current_delivery_blocker"],
                "model_quality_risk": payload["model_quality_risk"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
