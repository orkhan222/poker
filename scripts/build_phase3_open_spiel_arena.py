from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.open_spiel_llm_arena import ArenaRunConfig, write_phase3_open_spiel_arena_report
from poker_agent.rl_training_evidence_gate import (
    DEFAULT_POLICY_UPDATE_ALGORITHM,
    MINIMUM_RL_LONG_RUN_EPISODES,
    MINIMUM_RL_STABILITY_SEEDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 3 OpenSpiel LLM-vs-LLM arena report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "phase3_open_spiel_arena.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "phase3_open_spiel_arena.md", type=Path)
    parser.add_argument("--game-name", default="kuhn_poker")
    parser.add_argument("--episodes", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--independent-seed-count", default=1, type=int)
    parser.add_argument("--max-steps-per-episode", default=256, type=int)
    parser.add_argument("--agent-a-name", default="phase1_llm_agent_a")
    parser.add_argument("--agent-b-name", default="phase1_llm_agent_b")
    parser.add_argument("--agent-a-source", default="phase1_trained_llm_policy_a")
    parser.add_argument("--agent-b-source", default="phase1_trained_llm_policy_b")
    parser.add_argument("--agent-a-model-path", default=None)
    parser.add_argument("--agent-b-model-path", default=None)
    parser.add_argument(
        "--phase1-adapters-ready",
        action="store_true",
        help="Allow measured arena execution only when both Phase 1 policy adapters are supplied.",
    )
    parser.add_argument(
        "--run-if-available",
        action="store_true",
        help=(
            "Run measured OpenSpiel episodes only if pyspiel is installed and "
            "--phase1-adapters-ready with both adapter paths is supplied."
        ),
    )
    parser.add_argument(
        "--policy-update-training-completed",
        action="store_true",
        help=(
            "Mark the report as policy-update training only after a real RL training loop "
            "has been executed and reviewed. This is not enabled by smoke runs."
        ),
    )
    parser.add_argument(
        "--policy-update-algorithm",
        default=DEFAULT_POLICY_UPDATE_ALGORITHM,
        help="Policy-update algorithm used for the Phase 3 RL run. PPO or an equivalent update is required.",
    )
    parser.add_argument(
        "--claim-mode",
        action="store_true",
        help=(
            "Fail the command unless the run is eligible to open measured self-play and "
            "production strategy-quality claims."
        ),
    )
    return parser.parse_args()


def validate_claim_mode_args(args: argparse.Namespace, project_root: Path) -> list[str]:
    violations: list[str] = []
    if not args.run_if_available:
        violations.append("claim_mode_requires_run_if_available")
    if not args.phase1_adapters_ready:
        violations.append("claim_mode_requires_phase1_adapters_ready")
    if not args.agent_a_model_path:
        violations.append("claim_mode_requires_agent_a_model_path")
    if not args.agent_b_model_path:
        violations.append("claim_mode_requires_agent_b_model_path")
    for field in ("agent_a_model_path", "agent_b_model_path"):
        raw_path = getattr(args, field)
        if raw_path:
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else project_root / candidate
            if not resolved.exists():
                violations.append(f"claim_mode_missing_{field}")
    if int(args.episodes) < MINIMUM_RL_LONG_RUN_EPISODES:
        violations.append("claim_mode_requires_5000_or_more_episodes")
    if int(args.independent_seed_count) < MINIMUM_RL_STABILITY_SEEDS:
        violations.append("claim_mode_requires_5_or_more_independent_seeds")
    if not args.policy_update_training_completed:
        violations.append("claim_mode_requires_policy_update_training_completed")
    algorithm = str(args.policy_update_algorithm or "").upper().replace("-", "_").replace(" ", "_")
    if "PPO" not in algorithm and "EQUIVALENT" not in algorithm:
        violations.append("claim_mode_requires_ppo_or_equivalent_policy_update")
    return violations


def main() -> None:
    args = parse_args()
    if args.claim_mode:
        violations = validate_claim_mode_args(args, args.project_root)
        if violations:
            raise SystemExit(
                "OpenSpiel/RL claim mode is not eligible. Missing: "
                + ", ".join(sorted(violations))
            )
    config = ArenaRunConfig(
        game_name=args.game_name,
        episodes=args.episodes,
        seed=args.seed,
        independent_seed_count=args.independent_seed_count,
        max_steps_per_episode=args.max_steps_per_episode,
        agent_a_name=args.agent_a_name,
        agent_b_name=args.agent_b_name,
        agent_a_source=args.agent_a_source,
        agent_b_source=args.agent_b_source,
        agent_a_model_path=args.agent_a_model_path,
        agent_b_model_path=args.agent_b_model_path,
        phase1_adapters_ready=args.phase1_adapters_ready,
        policy_update_training_completed=args.policy_update_training_completed,
        policy_update_algorithm=args.policy_update_algorithm,
    )
    payload = write_phase3_open_spiel_arena_report(
        args.project_root,
        args.out,
        args.markdown_out,
        config=config,
        run_if_available=args.run_if_available,
    )
    if args.claim_mode and not payload["rl_training_proof_boundary"]["measured_win_rate_claim_allowed"]:
        missing = payload["rl_training_proof_boundary"].get("missing_requirements") or []
        raise SystemExit(
            "OpenSpiel/RL claim mode finished without claim eligibility. Missing: "
            + ", ".join(missing)
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "agent_only_table": payload["arena_contract"]["agent_only_table"],
                "game_name": payload["environment"]["game_name"],
                "rl_training_proof": payload["rl_training_proof_boundary"]["status"],
                "win_rate_claim_allowed": payload["rl_training_proof_boundary"]["measured_win_rate_claim_allowed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
