from __future__ import annotations

import pytest

from poker_agent.open_spiel_llm_arena import (
    ArenaRunConfig,
    FixedOpenSpielPolicy,
    OpenSpielAgentOnlyArena,
    OpenSpielArenaError,
    RUNTIME_PENDING_STATUS,
    build_phase3_open_spiel_arena_report,
    prediction_request_from_open_spiel_state,
    validate_phase3_open_spiel_arena_report,
)


class FakeOpenSpielGame:
    def num_players(self) -> int:
        return 2

    def new_initial_state(self) -> "FakeOpenSpielState":
        return FakeOpenSpielState()


class FakeOpenSpielState:
    def __init__(self) -> None:
        self.step = 0
        self.history: list[tuple[int, int]] = []

    def is_terminal(self) -> bool:
        return self.step >= 2

    def is_chance_node(self) -> bool:
        return False

    def current_player(self) -> int:
        return self.step

    def legal_actions(self, player_id: int) -> list[int]:
        if player_id == 0:
            return [0, 1]
        if player_id == 1:
            return [2, 3]
        return []

    def action_to_string(self, player_id: int, action_id: int) -> str:
        labels = {
            0: "Check",
            1: "Bet 1",
            2: "Fold",
            3: "Call",
        }
        return labels[action_id]

    def apply_action(self, action_id: int) -> None:
        self.history.append((self.step, action_id))
        self.step += 1

    def returns(self) -> list[float]:
        actions = [action_id for _, action_id in self.history]
        if actions == [1, 2]:
            return [1.0, -1.0]
        if actions == [1, 3]:
            return [0.0, 0.0]
        return [-0.25, 0.25]

    def history_str(self) -> str:
        return ",".join(str(action_id) for _, action_id in self.history)


def test_open_spiel_arena_requires_one_agent_per_seat() -> None:
    with pytest.raises(OpenSpielArenaError, match="one policy per seat"):
        OpenSpielAgentOnlyArena(
            FakeOpenSpielGame(),
            (FixedOpenSpielPolicy("agent-a", ("bet",)),),
        )


def test_open_spiel_arena_runs_agent_only_table() -> None:
    arena = OpenSpielAgentOnlyArena(
        FakeOpenSpielGame(),
        (
            FixedOpenSpielPolicy("phase1_llm_agent_a", ("bet", "check")),
            FixedOpenSpielPolicy("phase1_llm_agent_b", ("fold", "call")),
        ),
        seed=7,
        game_name="fake_kuhn_poker",
    )

    payload = arena.run(episodes=8, max_steps_per_episode=8)

    assert payload["status"] == "COMPLETED"
    assert payload["arena_contract"]["agent_only_table"] is True
    assert payload["arena_contract"]["human_players_present"] is False
    assert payload["arena_contract"]["fixed_scripted_opponents_present"] is False
    assert payload["arena_contract"]["all_seats_controlled_by_agents"] is True
    assert payload["quality_boundary"]["is_reinforcement_learning_stage"] is True
    assert payload["quality_boundary"]["phase1_policy_adapters_ready"] is True
    assert payload["quality_boundary"]["metrics_claim_allowed"] is True
    assert payload["rl_training_proof_boundary"]["status"] == "TRAINING_PROOF_NOT_COMPLETED"
    assert payload["rl_training_proof_boundary"]["gate_name"] == "phase3_open_spiel_rl_training_evidence_gate"
    assert payload["rl_training_proof_boundary"]["measured_win_rate_claim_allowed"] is False
    assert set(payload["rl_training_proof_boundary"]["required_evidence"]) == {
        "real_open_spiel_runtime",
        "agent_only_arena",
        "two_phase1_trained_policy_artifacts",
        "long_run_simulation_volume",
        "seed_stability",
        "policy_update_training",
    }
    assert payload["rl_training_proof_boundary"]["policy_update_training_completed"] is False
    assert payload["rl_training_proof_boundary"]["seed_stability_evaluated"] is False
    assert payload["rl_training_proof_boundary"]["long_run_completed"] is False
    assert payload["rl_training_proof_boundary"]["model_quality_risk"] is True
    assert all(case["result"] == "PASS" for case in payload["proof_cases"])
    assert payload["invariants"]["status"] == "PASS"
    assert payload["overall_status"] == "PASS"
    assert payload["metrics"]["per_agent"]["0"]["episodes"] == 8
    assert payload["metrics"]["total_actions"] == 16


def test_open_spiel_state_maps_to_structured_prediction_request() -> None:
    state = FakeOpenSpielState()

    request = prediction_request_from_open_spiel_state(state, player_id=0)

    assert request.position == "P0"
    assert request.street == "preflop"
    assert request.to_call == 0.0
    assert request.player_count == 2


def test_pending_report_does_not_claim_measured_metrics() -> None:
    payload = build_phase3_open_spiel_arena_report(
        project_root=__import__("pathlib").Path("."),
        config=ArenaRunConfig(episodes=4),
        run_if_available=False,
    )

    assert payload["status"] == RUNTIME_PENDING_STATUS
    assert payload["overall_status"] == "PASS"
    assert payload["invariants"]["status"] == "PASS"
    assert payload["arena_contract"]["agent_only_table"] is True
    assert payload["runtime_boundary"]["run_if_available_required_for_metrics"] is True
    assert payload["runtime_boundary"]["phase1_adapters_required_for_metrics"] is True
    assert payload["quality_boundary"]["metrics_claim_allowed"] is False
    assert payload["rl_training_proof_boundary"]["measured_win_rate_claim_allowed"] is False
    assert payload["rl_training_proof_boundary"]["gate_name"] == "phase3_open_spiel_rl_training_evidence_gate"
    assert payload["rl_training_proof_boundary"]["real_open_spiel_runtime_available"] in {True, False}
    assert payload["rl_training_proof_boundary"]["phase1_trained_policy_artifacts_attached"] is False
    assert payload["rl_training_proof_boundary"]["current_delivery_blocker"] is False
    assert payload["rl_training_proof_boundary"]["model_quality_risk"] is True
    assert all(case["result"] == "PASS" for case in payload["proof_cases"])
    assert "metrics" not in payload


def test_pending_report_with_metrics_fails_invariant() -> None:
    payload = build_phase3_open_spiel_arena_report(
        project_root=__import__("pathlib").Path("."),
        config=ArenaRunConfig(episodes=4),
        run_if_available=False,
    )
    payload["metrics"] = {"fake": 1.0}

    validation = validate_phase3_open_spiel_arena_report(payload)

    assert validation["status"] == "FAIL"
    assert "pending_open_spiel_report_must_not_include_measured_metrics" in validation["violations"]


def test_phase3_blocks_rl_win_rate_claim_without_training_proof() -> None:
    payload = build_phase3_open_spiel_arena_report(
        project_root=__import__("pathlib").Path("."),
        config=ArenaRunConfig(episodes=4),
        run_if_available=False,
    )
    proof = payload["rl_training_proof_boundary"]
    proof["measured_win_rate_claim_allowed"] = True
    proof["status"] = "TRAINING_PROOF_COMPLETED"

    validation = validate_phase3_open_spiel_arena_report(payload)

    assert validation["status"] == "FAIL"
    assert "rl_win_rate_claim_requires_runtime_adapters_seed_stability_long_run_and_training" in validation["violations"]
    assert "rl_training_proof_cannot_be_completed_without_all_gates" in validation["violations"]


def test_run_if_available_without_phase1_adapters_stays_pending() -> None:
    payload = build_phase3_open_spiel_arena_report(
        project_root=__import__("pathlib").Path("."),
        config=ArenaRunConfig(episodes=4, phase1_adapters_ready=False),
        run_if_available=True,
    )

    assert payload["status"] == RUNTIME_PENDING_STATUS
    assert payload["quality_boundary"]["metrics_claim_allowed"] is False
    assert payload["rl_training_proof_boundary"]["measured_win_rate_claim_allowed"] is False
    assert payload["invariants"]["status"] == "PASS"
