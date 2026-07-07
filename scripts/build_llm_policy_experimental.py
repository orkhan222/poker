from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_policy_experimental import write_experimental_llm_policy_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the experimental LLM policy adapter contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "llm_policy_experimental.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "llm_policy_experimental.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_experimental_llm_policy_contract(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "status": payload["status"],
                "production_policy_approved": payload["production_policy_approved"],
                "autonomous_policy_claim_allowed": payload["autonomous_policy_claim_allowed"],
                "served_by_predict_endpoint": payload["served_by_predict_endpoint"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
