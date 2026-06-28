from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.production_runtime_monitoring import write_production_runtime_monitoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the production runtime monitoring contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "production_runtime_monitoring.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "production_runtime_monitoring.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_production_runtime_monitoring(args.project_root, args.out, args.markdown_out)
    boundary = payload["runtime_observability_boundary"]
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "status": boundary["status"],
                "monitoring_required": boundary["monitoring_required_for_real_traffic"],
                "rollback_required": boundary["rollback_rules_required_for_real_traffic"],
                "drift_tracking_required": boundary["live_drift_tracking_required_for_real_traffic"],
                "current_delivery_blocker": boundary["current_delivery_blocker"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
