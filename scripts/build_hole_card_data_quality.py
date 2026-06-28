from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.hole_card_data_quality import write_hole_card_data_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hole-card data-quality limitation contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "hole_card_data_quality.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "hole_card_data_quality.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_hole_card_data_quality(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "mitigation_status": payload["mitigation_boundary"]["mitigation_status"],
                "limitation_status": payload["upstream_data_quality_boundary"]["limitation_status"],
                "upstream_resolved": payload["upstream_data_quality_boundary"]["upstream_data_quality_issue_resolved"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
