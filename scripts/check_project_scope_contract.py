from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.project_scope import describe_project_scope_contract, validate_project_scope, write_project_scope_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the machine-readable poker project scope contract")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--out", default=Path("reports/project_scope_contract.json"), type=Path)
    parser.add_argument("--docs-out", default=Path("docs/PROJECT_SCOPE_CONTRACT.md"), type=Path)
    parser.add_argument("--contract-out", default=None, type=Path)
    parser.add_argument("--smoke", action="store_true")
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
    contract_out = resolve(root, args.contract_out)
    outputs = write_project_scope_reports(root, out=out, docs_out=docs_out)
    validation = validate_project_scope(root)

    if contract_out is not None:
        contract_out.parent.mkdir(parents=True, exist_ok=True)
        contract_out.write_text(
            json.dumps(describe_project_scope_contract(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps({"status": validation["status"], "outputs": {key: str(path) for key, path in outputs.items()}}, sort_keys=True))
    if validation["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
