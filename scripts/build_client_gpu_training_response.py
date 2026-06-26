from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.client_gpu_training_response import write_client_gpu_training_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build client-facing A100/H100 training response")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "client_gpu_training_response.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "client_gpu_training_response.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_client_gpu_training_response(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "subject": payload["subject"],
                "training_status": payload["current_delivery_training"]["training_status"],
                "delivery_status": payload["current_delivery_training"]["delivery_status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()