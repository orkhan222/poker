from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.final_delivery_acceptance import write_final_delivery_acceptance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final delivery acceptance contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "final_delivery_acceptance.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "final_delivery_acceptance.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_final_delivery_acceptance(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "final_status": payload["final_status"],
                "service_delivery": payload["acceptance_summary"]["service_delivery"],
                "deployed_strategy_stack": payload["acceptance_summary"]["deployed_strategy_stack"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
