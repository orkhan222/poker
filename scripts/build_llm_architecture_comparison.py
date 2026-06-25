from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_architecture_comparison import (
    build_llm_architecture_comparison,
    write_llm_architecture_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare measured LLM decision architectures")
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--candidate-ranker", required=True, type=Path)
    parser.add_argument("--generation-gate", required=True, type=Path)
    parser.add_argument("--candidate-gate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_llm_architecture_comparison(
        args.generation,
        args.candidate_ranker,
        args.generation_gate,
        args.candidate_gate,
    )
    write_llm_architecture_comparison(payload, args.out, args.report_out)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "recommended_architecture": payload["recommended_architecture"],
                "production_approved": payload["production_approved"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
