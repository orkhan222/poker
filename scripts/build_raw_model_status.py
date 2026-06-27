from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.raw_model_status import write_raw_model_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw supervised model status contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "raw_model_status.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "raw_model_status.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_raw_model_status(args.project_root, args.out, args.markdown_out)
    raw = payload["raw_supervised_model"]
    print(
        json.dumps(
            {
                "runtime_status": raw["runtime_status"],
                "standalone_status": raw["standalone_status"],
                "quality_gate_status": raw["quality_gate_status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
