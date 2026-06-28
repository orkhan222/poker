from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_role_boundary import write_llm_role_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LLM role boundary contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "llm_role_boundary.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "llm_role_boundary.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_llm_role_boundary(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "llm_role": payload["current_llm_role"]["status"],
                "autonomous_llm_status": payload["autonomous_llm_agent_boundary"]["status"],
                "autonomous_llm_claim_allowed": payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
