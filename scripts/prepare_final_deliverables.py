from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.deliverables import (
    build_smoke_reports,
    describe_final_deliverables_contract,
    validate_final_deliverables,
    write_api_docs,
    write_delivery_report,
)
from poker_agent.legacy_reports import build_legacy_delivery_reports
from poker_agent.project_scope import write_project_scope_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate final poker-agent delivery artifacts")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--out", default=Path("reports/final_deliverables.json"), type=Path)
    parser.add_argument("--docs-out", default=Path("docs/API_CONTRACT.md"), type=Path)
    parser.add_argument("--delivery-report-out", default=Path("reports/delivery_report.md"), type=Path)
    parser.add_argument("--contract-out", default=None, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Regenerate deterministic smoke reports before validation")
    return parser.parse_args()


def resolve(root: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else root / value


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = resolve(root, args.out)
    docs_out = resolve(root, args.docs_out)
    delivery_report_out = resolve(root, args.delivery_report_out)
    contract_out = resolve(root, args.contract_out)

    if args.smoke:
        write_project_scope_reports(root)
        build_smoke_reports(root)
        build_legacy_delivery_reports(root, overwrite=False)
        if docs_out != root / "docs" / "API_CONTRACT.md":
            write_api_docs(root, docs_out)
    else:
        write_api_docs(root, docs_out)

    manifest = validate_final_deliverables(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_delivery_report(delivery_report_out, manifest)

    if contract_out is not None:
        contract_out.parent.mkdir(parents=True, exist_ok=True)
        contract_out.write_text(
            json.dumps(describe_final_deliverables_contract(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
