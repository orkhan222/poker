from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.data_leakage_contract import write_data_leakage_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build outcome-field data-leakage contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "data_leakage_contract.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "data_leakage_contract.md", type=Path)
    parser.add_argument("--max-examples", default=5000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_data_leakage_contract(
        args.project_root,
        args.out,
        args.markdown_out,
        max_examples=args.max_examples,
    )
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "training_feature_audit": payload["feature_name_audit"]["status"],
                "model_artifact_audit": payload["model_artifact_audit"]["status"],
                "source_usage_audit": payload["source_usage_audit"]["status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
