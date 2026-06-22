from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.client_handoff import write_client_handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build client handoff statement")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "client_handoff.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "client_handoff.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_client_handoff(args.project_root, args.out, args.markdown_out)
    position = payload["technical_position"]
    print(
        json.dumps(
            {
                "handoff_status": payload["handoff_status"],
                "service_delivery": position["service_delivery"],
                "deployed_strategy_stack": position["deployed_strategy_stack"],
                "raw_supervised_model_runtime": position["raw_supervised_model_runtime"],
                "raw_supervised_model_standalone": position["raw_supervised_model_standalone"],
                "production_blocker": position["production_blocker"],
                "component_risk": position["component_risk"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
