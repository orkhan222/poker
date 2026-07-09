from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.open_spiel_claim_readiness import write_open_spiel_claim_readiness
from poker_agent.rl_training_evidence_gate import DEFAULT_POLICY_UPDATE_ALGORITHM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OpenSpiel/RL claim-readiness preflight report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "open_spiel_claim_readiness.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "open_spiel_claim_readiness.md", type=Path)
    parser.add_argument("--agent-a-model-path", default="models/phase1_llm_policy_a.joblib")
    parser.add_argument("--agent-b-model-path", default="models/phase1_llm_policy_b.joblib")
    parser.add_argument("--episodes", default=5000, type=int)
    parser.add_argument("--independent-seed-count", default=5, type=int)
    parser.add_argument("--policy-update-training-completed", action="store_true")
    parser.add_argument("--policy-update-algorithm", default=DEFAULT_POLICY_UPDATE_ALGORITHM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_open_spiel_claim_readiness(
        args.project_root,
        args.out,
        args.markdown_out,
        agent_a_model_path=args.agent_a_model_path,
        agent_b_model_path=args.agent_b_model_path,
        episodes=args.episodes,
        independent_seed_count=args.independent_seed_count,
        policy_update_training_completed=args.policy_update_training_completed,
        policy_update_algorithm=args.policy_update_algorithm,
    )
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "claim_ready": payload["claim_ready"],
                "missing_requirements": payload["missing_requirements"],
                "current_delivery_blocker": payload["current_delivery_blocker"],
                "model_quality_risk": payload["model_quality_risk"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
