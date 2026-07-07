from __future__ import annotations

from typing import Any


RL_TRAINING_EVIDENCE_GATE_VERSION = "2026-07-07"
RL_TRAINING_PROOF_PENDING_STATUS = "TRAINING_PROOF_NOT_COMPLETED"
RL_TRAINING_PROOF_COMPLETED_STATUS = "TRAINING_PROOF_COMPLETED"
MINIMUM_RL_STABILITY_SEEDS = 5
MINIMUM_RL_LONG_RUN_EPISODES = 5_000

REQUIRED_RL_EVIDENCE = (
    "real_open_spiel_runtime",
    "agent_only_arena",
    "two_phase1_trained_policy_artifacts",
    "long_run_simulation_volume",
    "seed_stability",
    "policy_update_training",
)


def build_rl_training_evidence_gate(
    *,
    real_open_spiel_runtime_available: bool,
    phase1_trained_policy_artifacts_attached: bool,
    agent_only_table_verified: bool,
    episodes: int,
    independent_seed_count: int,
    policy_update_training_completed: bool,
) -> dict[str, Any]:
    seed_stability_evaluated = int(independent_seed_count) >= MINIMUM_RL_STABILITY_SEEDS
    long_run_completed = int(episodes) >= MINIMUM_RL_LONG_RUN_EPISODES
    measured_win_rate_claim_allowed = all(
        (
            real_open_spiel_runtime_available,
            phase1_trained_policy_artifacts_attached,
            agent_only_table_verified,
            seed_stability_evaluated,
            long_run_completed,
            policy_update_training_completed,
        )
    )
    missing_requirements = _missing_requirements(
        real_open_spiel_runtime_available=real_open_spiel_runtime_available,
        phase1_trained_policy_artifacts_attached=phase1_trained_policy_artifacts_attached,
        agent_only_table_verified=agent_only_table_verified,
        seed_stability_evaluated=seed_stability_evaluated,
        long_run_completed=long_run_completed,
        policy_update_training_completed=policy_update_training_completed,
    )
    return {
        "version": RL_TRAINING_EVIDENCE_GATE_VERSION,
        "gate_name": "phase3_open_spiel_rl_training_evidence_gate",
        "required_evidence": list(REQUIRED_RL_EVIDENCE),
        "status": (
            RL_TRAINING_PROOF_COMPLETED_STATUS
            if measured_win_rate_claim_allowed
            else RL_TRAINING_PROOF_PENDING_STATUS
        ),
        "real_open_spiel_runtime_required": True,
        "real_open_spiel_runtime_available": bool(real_open_spiel_runtime_available),
        "phase1_trained_policy_artifacts_required": True,
        "phase1_trained_policy_artifacts_attached": bool(phase1_trained_policy_artifacts_attached),
        "agent_only_table_required": True,
        "agent_only_table_verified": bool(agent_only_table_verified),
        "seed_stability_required": True,
        "minimum_independent_seeds": MINIMUM_RL_STABILITY_SEEDS,
        "independent_seed_count": int(independent_seed_count),
        "seed_stability_evaluated": seed_stability_evaluated,
        "long_run_required": True,
        "minimum_long_run_episodes": MINIMUM_RL_LONG_RUN_EPISODES,
        "episodes": int(episodes),
        "long_run_completed": long_run_completed,
        "policy_update_training_required": True,
        "policy_update_training_completed": bool(policy_update_training_completed),
        "measured_win_rate_claim_allowed": measured_win_rate_claim_allowed,
        "current_delivery_blocker": False,
        "model_quality_risk": not measured_win_rate_claim_allowed,
        "missing_requirements": missing_requirements,
        "allowed_current_claim": (
            "The Phase 3 agent-only OpenSpiel arena code is ready for a measured run when the "
            "runtime, Phase 1 adapters, seed-stability run, and long-run training profile are available."
        ),
        "blocked_claim": (
            "Do not claim RL win-rate or production strategy quality from Phase 3 until real OpenSpiel "
            "runtime execution, two trained Phase 1 policy artifacts, seed stability, long-run volume, "
            "and policy-update training are all complete."
        ),
    }


def validate_rl_training_evidence_gate(gate: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    expected = build_rl_training_evidence_gate(
        real_open_spiel_runtime_available=gate.get("real_open_spiel_runtime_available") is True,
        phase1_trained_policy_artifacts_attached=gate.get("phase1_trained_policy_artifacts_attached") is True,
        agent_only_table_verified=gate.get("agent_only_table_verified") is True,
        episodes=int(gate.get("episodes") or 0),
        independent_seed_count=int(gate.get("independent_seed_count") or 0),
        policy_update_training_completed=gate.get("policy_update_training_completed") is True,
    )

    if gate.get("gate_name") != "phase3_open_spiel_rl_training_evidence_gate":
        violations.append("rl_training_evidence_gate_name_must_be_explicit")
    if set(gate.get("required_evidence") or []) != set(REQUIRED_RL_EVIDENCE):
        violations.append("rl_training_required_evidence_must_be_complete")
    if gate.get("real_open_spiel_runtime_required") is not True:
        violations.append("rl_training_proof_must_require_real_open_spiel_runtime")
    if gate.get("phase1_trained_policy_artifacts_required") is not True:
        violations.append("rl_training_proof_must_require_two_phase1_adapters")
    if gate.get("agent_only_table_required") is not True:
        violations.append("rl_training_proof_must_require_agent_only_table")
    if gate.get("seed_stability_required") is not True:
        violations.append("rl_training_proof_must_require_seed_stability")
    if int(gate.get("minimum_independent_seeds") or 0) < MINIMUM_RL_STABILITY_SEEDS:
        violations.append("rl_training_proof_seed_threshold_too_low")
    if gate.get("long_run_required") is not True:
        violations.append("rl_training_proof_must_require_long_run")
    if int(gate.get("minimum_long_run_episodes") or 0) < MINIMUM_RL_LONG_RUN_EPISODES:
        violations.append("rl_training_proof_long_run_threshold_too_low")
    if gate.get("policy_update_training_required") is not True:
        violations.append("rl_training_proof_must_require_policy_update_training")
    if gate.get("current_delivery_blocker") is not False:
        violations.append("rl_training_proof_gap_must_not_block_current_delivery")
    if gate.get("seed_stability_evaluated") is not expected["seed_stability_evaluated"]:
        violations.append("rl_training_seed_stability_status_must_match_seed_count")
    if gate.get("long_run_completed") is not expected["long_run_completed"]:
        violations.append("rl_training_long_run_status_must_match_episode_count")
    if gate.get("missing_requirements") != expected["missing_requirements"]:
        violations.append("rl_training_missing_requirements_must_match_failed_gates")
    if gate.get("measured_win_rate_claim_allowed") is not expected["measured_win_rate_claim_allowed"]:
        violations.append("rl_win_rate_claim_requires_runtime_adapters_seed_stability_long_run_and_training")
    if gate.get("status") != expected["status"]:
        violations.append("rl_training_proof_status_must_match_evidence_gate")
    if expected["measured_win_rate_claim_allowed"] is False and gate.get("model_quality_risk") is not True:
        violations.append("incomplete_rl_training_proof_must_remain_model_quality_risk")
    if expected["measured_win_rate_claim_allowed"] is True and gate.get("model_quality_risk") is not False:
        violations.append("completed_rl_training_proof_must_clear_model_quality_risk")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def _missing_requirements(
    *,
    real_open_spiel_runtime_available: bool,
    phase1_trained_policy_artifacts_attached: bool,
    agent_only_table_verified: bool,
    seed_stability_evaluated: bool,
    long_run_completed: bool,
    policy_update_training_completed: bool,
) -> list[str]:
    missing_requirements = []
    if not real_open_spiel_runtime_available:
        missing_requirements.append("real_open_spiel_runtime")
    if not phase1_trained_policy_artifacts_attached:
        missing_requirements.append("two_phase1_trained_policy_artifacts")
    if not agent_only_table_verified:
        missing_requirements.append("agent_only_table")
    if not seed_stability_evaluated:
        missing_requirements.append("seed_stability")
    if not long_run_completed:
        missing_requirements.append("long_run_training_volume")
    if not policy_update_training_completed:
        missing_requirements.append("policy_update_training")
    return missing_requirements
