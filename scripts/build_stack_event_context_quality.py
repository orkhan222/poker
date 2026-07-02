from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.stack_event_context_quality import write_stack_event_context_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stack_events.csv decision-context quality contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "stack_event_context_quality.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "stack_event_context_quality.md", type=Path)
    parser.add_argument("--max-examples", default=5000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_stack_event_context_quality(
        args.project_root,
        args.out,
        args.markdown_out,
        max_examples=args.max_examples,
    )
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "raw_stack_event_status": payload["raw_stack_event_boundary"]["status"],
                "derived_context_status": payload["derived_context_mitigation"]["status"],
                "training_feature_audit": payload["training_feature_audit"]["status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
