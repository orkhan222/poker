from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.behavioral_revalidation_proof import write_behavioral_revalidation_proof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build executable proof for behavioral revalidation boundary")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "behavioral_revalidation_proof.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "behavioral_revalidation_proof.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_behavioral_revalidation_proof(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "proof_status": payload["proof_status"],
                "proof_cases": len(payload["proof_cases"]),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
