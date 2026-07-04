from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.phase2_selection_comparison import write_phase2_selection_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict Phase 2 common-condition selection contract")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "phase2_selection_comparison.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "phase2_selection_comparison.md", type=Path)
    args = parser.parse_args()

    payload = write_phase2_selection_comparison(args.project_root, args.out, args.markdown_out)
    gate = payload["comparison_gate"]
    print(f"phase2_selection_status={payload['overall_status']}")
    print(f"current_delivery_architecture={gate['selected_for_current_delivery']}")
    print(f"final_selection_claim_allowed={gate['final_selection_claim_allowed']}")


if __name__ == "__main__":
    main()
