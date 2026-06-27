from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.raw_model_challenger import (
    build_existing_model_challenger_report,
    train_raw_model_challengers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and gate raw supervised policy challengers")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--dataset", default=ROOT / "data", type=Path)
    parser.add_argument("--report-out", default=ROOT / "reports" / "raw_model_challenger.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "raw_model_challenger.md", type=Path)
    parser.add_argument("--model-dir", default=ROOT / "models" / "raw_challengers", type=Path)
    parser.add_argument("--audit-report", default=ROOT / "reports" / "dataset_audit.json", type=Path)
    parser.add_argument("--max-examples", default=50000, type=int)
    parser.add_argument("--valid-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--promote-model-out", default=None, type=Path)
    parser.add_argument(
        "--evaluate-existing-only",
        action="store_true",
        help="Evaluate the existing raw model under the challenger contract without training new artifacts.",
    )
    parser.add_argument("--existing-model", default=ROOT / "models" / "poker_policy.joblib", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluate_existing_only:
        payload = build_existing_model_challenger_report(
            project_root=args.project_root,
            model_path=args.existing_model,
            dataset=args.dataset,
            report_out=args.report_out,
            markdown_out=args.markdown_out,
            audit_report=args.audit_report,
            max_examples=args.max_examples,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
    else:
        payload = train_raw_model_challengers(
            project_root=args.project_root,
            dataset=args.dataset,
            report_out=args.report_out,
            markdown_out=args.markdown_out,
            model_dir=args.model_dir,
            audit_report=args.audit_report,
            max_examples=args.max_examples,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            promote_model_out=args.promote_model_out,
        )

    best = payload.get("best_candidate") or {}
    gate = best.get("gate") or {}
    print(
        json.dumps(
            {
                "standalone_status": payload.get("standalone_status"),
                "approved_as_standalone_policy": payload.get("approved_as_standalone_policy"),
                "best_candidate": best.get("name"),
                "candidate_gate": gate.get("status"),
                "failed_gates": gate.get("failed_gates"),
                "report_out": str(args.report_out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
