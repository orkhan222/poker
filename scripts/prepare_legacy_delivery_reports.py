from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.legacy_reports import (
    build_legacy_delivery_reports,
    describe_legacy_reports_contract,
    validate_legacy_delivery_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic compatibility reports for legacy delivery checks")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--out", default=Path("reports/legacy_delivery_reports.json"), type=Path)
    parser.add_argument("--contract-out", default=None, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else root / value


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    build_legacy_delivery_reports(root, overwrite=args.overwrite)
    report = validate_legacy_delivery_reports(root)

    out = resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    contract_out = resolve(root, args.contract_out)
    if contract_out is not None:
        contract_out.parent.mkdir(parents=True, exist_ok=True)
        contract_out.write_text(
            json.dumps(describe_legacy_reports_contract(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
