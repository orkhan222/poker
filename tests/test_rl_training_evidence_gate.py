from __future__ import annotations

from poker_agent.rl_training_evidence_gate import (
    MINIMUM_RL_LONG_RUN_EPISODES,
    MINIMUM_RL_STABILITY_SEEDS,
    REQUIRED_RL_EVIDENCE,
    build_rl_training_evidence_gate,
    validate_rl_training_evidence_gate,
)


def test_rl_training_evidence_gate_blocks_self_play_claim_without_required_evidence() -> None:
    gate = build_rl_training_evidence_gate(
        real_open_spiel_runtime_available=False,
        phase1_trained_policy_artifacts_attached=False,
        agent_only_table_verified=True,
        episodes=256,
        independent_seed_count=1,
        policy_update_training_completed=False,
    )

    assert gate["status"] == "TRAINING_PROOF_NOT_COMPLETED"
    assert gate["measured_win_rate_claim_allowed"] is False
    assert gate["current_delivery_blocker"] is False
    assert gate["model_quality_risk"] is True
    assert gate["required_phase1_trained_policy_artifact_count"] == 2
    assert gate["phase1_trained_policy_artifact_count"] == 0
    assert gate["policy_update_algorithm_required"] == "PPO_OR_EQUIVALENT"
    assert gate["ppo_policy_update_required"] is True
    assert gate["ppo_or_equivalent_policy_update_completed"] is False
    assert set(gate["required_evidence"]) == set(REQUIRED_RL_EVIDENCE)
    assert gate["missing_requirements"] == [
        "real_open_spiel_runtime",
        "two_phase1_trained_policy_artifacts",
        "seed_stability",
        "long_run_training_volume",
        "policy_update_training",
    ]
    assert validate_rl_training_evidence_gate(gate)["status"] == "PASS"


def test_rl_training_evidence_gate_allows_claim_only_when_all_gates_pass() -> None:
    gate = build_rl_training_evidence_gate(
        real_open_spiel_runtime_available=True,
        phase1_trained_policy_artifacts_attached=True,
        agent_only_table_verified=True,
        episodes=MINIMUM_RL_LONG_RUN_EPISODES,
        independent_seed_count=MINIMUM_RL_STABILITY_SEEDS,
        policy_update_training_completed=True,
    )

    assert gate["status"] == "TRAINING_PROOF_COMPLETED"
    assert gate["measured_win_rate_claim_allowed"] is True
    assert gate["phase1_trained_policy_artifact_count"] == 2
    assert gate["ppo_or_equivalent_policy_update_completed"] is True
    assert gate["missing_requirements"] == []
    assert gate["model_quality_risk"] is False
    assert validate_rl_training_evidence_gate(gate)["status"] == "PASS"


def test_rl_training_evidence_gate_rejects_false_win_rate_claim() -> None:
    gate = build_rl_training_evidence_gate(
        real_open_spiel_runtime_available=False,
        phase1_trained_policy_artifacts_attached=False,
        agent_only_table_verified=True,
        episodes=256,
        independent_seed_count=1,
        policy_update_training_completed=False,
    )
    gate["measured_win_rate_claim_allowed"] = True
    gate["status"] = "TRAINING_PROOF_COMPLETED"
    gate["model_quality_risk"] = False

    validation = validate_rl_training_evidence_gate(gate)

    assert validation["status"] == "FAIL"
    assert "rl_win_rate_claim_requires_runtime_adapters_seed_stability_long_run_and_training" in validation[
        "violations"
    ]
    assert "rl_training_proof_status_must_match_evidence_gate" in validation["violations"]
    assert "incomplete_rl_training_proof_must_remain_model_quality_risk" in validation["violations"]


def test_rl_training_evidence_gate_requires_exactly_two_policy_artifacts() -> None:
    gate = build_rl_training_evidence_gate(
        real_open_spiel_runtime_available=True,
        phase1_trained_policy_artifacts_attached=True,
        phase1_trained_policy_artifact_count=1,
        agent_only_table_verified=True,
        episodes=MINIMUM_RL_LONG_RUN_EPISODES,
        independent_seed_count=MINIMUM_RL_STABILITY_SEEDS,
        policy_update_training_completed=True,
    )

    assert gate["status"] == "TRAINING_PROOF_NOT_COMPLETED"
    assert gate["measured_win_rate_claim_allowed"] is False
    assert gate["phase1_trained_policy_artifacts_attached"] is False
    assert gate["missing_requirements"] == ["two_phase1_trained_policy_artifacts"]
    assert validate_rl_training_evidence_gate(gate)["status"] == "PASS"


def test_rl_training_evidence_gate_requires_ppo_or_equivalent_update() -> None:
    gate = build_rl_training_evidence_gate(
        real_open_spiel_runtime_available=True,
        phase1_trained_policy_artifacts_attached=True,
        agent_only_table_verified=True,
        episodes=MINIMUM_RL_LONG_RUN_EPISODES,
        independent_seed_count=MINIMUM_RL_STABILITY_SEEDS,
        policy_update_training_completed=True,
        policy_update_algorithm="supervised_only",
    )

    assert gate["status"] == "TRAINING_PROOF_NOT_COMPLETED"
    assert gate["policy_update_algorithm_supported"] is False
    assert gate["ppo_or_equivalent_policy_update_completed"] is False
    assert gate["missing_requirements"] == ["policy_update_training"]
    assert validate_rl_training_evidence_gate(gate)["status"] == "PASS"
