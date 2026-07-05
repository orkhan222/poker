from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.scenario_sanity import validate_scenario_sanity, write_scenario_sanity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted poker scenario sanity report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--model", default=None, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "scenario_sanity.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "scenario_sanity.md", type=Path)
    parser.add_argument("--fail-on-sanity-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_scenario_sanity(
        args.project_root,
        args.out,
        args.markdown_out,
        model_path=args.model,
    )
    errors = validate_scenario_sanity(payload)
    summary = {
        "overall_status": payload["overall_status"],
        "scenario_count": payload["scenario_count"],
        "passed_scenarios": payload["passed_scenarios"],
        "failed_scenarios": payload["failed_scenarios"],
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors and args.fail_on_sanity_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
