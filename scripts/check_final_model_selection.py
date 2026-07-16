from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.final_model_selection import (
    describe_final_model_selection,
    final_model_selection_status,
    write_final_model_selection_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit and validate the QwenPoker final model selection contract")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--out", default=Path("reports/final_model_selection.json"), type=Path)
    parser.add_argument("--docs-out", default=Path("docs/FINAL_MODEL_SELECTION.md"), type=Path)
    parser.add_argument("--contract-out", default=None, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = resolve(root, args.out)
    docs_out = resolve(root, args.docs_out)
    contract_out = resolve(root, args.contract_out)

    outputs = write_final_model_selection_reports(root, out=out, docs_out=docs_out)
    status = final_model_selection_status()
    if contract_out is not None:
        contract_out.parent.mkdir(parents=True, exist_ok=True)
        contract_out.write_text(json.dumps(describe_final_model_selection(), indent=2, sort_keys=True), encoding="utf-8")

    payload = {"status": status["status"], "outputs": {key: str(value) for key, value in outputs.items()}}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
