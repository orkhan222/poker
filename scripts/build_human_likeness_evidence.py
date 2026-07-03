from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.human_likeness_evidence import write_human_likeness_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the human-likeness evidence contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "human_likeness_evidence.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "human_likeness_evidence.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_human_likeness_evidence(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "boundary": payload["boundary"],
                "human_likeness_fully_proven": payload["human_likeness_fully_proven"],
                "final_human_likeness_claim_allowed": payload["final_human_likeness_claim_allowed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
