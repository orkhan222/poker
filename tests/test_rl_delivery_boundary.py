from __future__ import annotations

import json
from pathlib import Path

from poker_agent.rl_delivery_boundary import (
    build_rl_delivery_boundary,
    build_rl_delivery_boundary_from_claim,
    validate_rl_delivery_boundary,
    write_rl_delivery_boundary,
)
from poker_agent.rl_training_evidence_gate import REQUIRED_RL_EVIDENCE


def _pending_open_spiel_claim_contract() -> dict[str, object]:
    return {
        "overall_status": "PASS",
        "training_proof_completed": False,
        "self_play_win_rate_claim_allowed": False,
        "current_delivery_blocker": False,
        "model_quality_risk": True,
        "required_evidence_before_self_play_claim": list(REQUIRED_RL_EVIDENCE),
        "missing_requirements": [
            "real_open_spiel_runtime",
            "two_phase1_trained_policy_artifacts",
            "seed_stability",
            "long_run_training_volume",
            "policy_update_training",
        ],
        "real_open_spiel_runtime_available": False,
        "phase1_trained_policy_artifact_count": 0,
        "seed_stability_evaluated": False,
        "long_run_completed": False,
        "ppo_or_equivalent_policy_update_completed": False,
    }


def test_rl_delivery_boundary_keeps_delivery_ready_but_blocks_strategy_claims() -> None:
    payload = build_rl_delivery_boundary_from_claim(_pending_open_spiel_claim_contract())

    assert payload["overall_status"] == "PASS"
    assert payload["current_delivery_blocker"] is False
    assert payload["delivery_scope"]["service_delivery_blocked_by_rl_training_gap"] is False
    assert payload["claim_permissions"]["delivery_readiness_claim_allowed"] is True
    assert payload["claim_permissions"]["self_play_win_rate_claim_allowed"] is False
    assert payload["claim_permissions"]["production_strategy_quality_claim_allowed"] is False
    assert payload["model_quality_risk"] is True
    assert set(payload["rl_training_proof"]["required_evidence"]) == set(REQUIRED_RL_EVIDENCE)


def test_rl_delivery_boundary_rejects_false_self_play_claim_without_rl_proof() -> None:
    payload = build_rl_delivery_boundary_from_claim(_pending_open_spiel_claim_contract())
    payload["claim_permissions"]["self_play_win_rate_claim_allowed"] = True

    validation = validate_rl_delivery_boundary(payload)

    assert validation["status"] == "FAIL"
    assert "self_play_win_rate_claim_requires_completed_rl_training_proof" in validation["violations"]


def test_rl_delivery_boundary_rejects_false_production_strategy_claim_without_self_play_evidence() -> None:
    payload = build_rl_delivery_boundary_from_claim(_pending_open_spiel_claim_contract())
    payload["claim_permissions"]["production_strategy_quality_claim_allowed"] = True

    validation = validate_rl_delivery_boundary(payload)

    assert validation["status"] == "FAIL"
    assert (
        "production_strategy_quality_claim_requires_self_play_win_rate_claim"
        in validation["violations"]
    )


def test_rl_delivery_boundary_rejects_turning_rl_gap_into_delivery_blocker() -> None:
    payload = build_rl_delivery_boundary_from_claim(_pending_open_spiel_claim_contract())
    payload["current_delivery_blocker"] = True
    payload["delivery_scope"]["service_delivery_blocked_by_rl_training_gap"] = True

    validation = validate_rl_delivery_boundary(payload)

    assert validation["status"] == "FAIL"
    assert "rl_training_gap_must_not_block_service_delivery" in validation["violations"]
    assert "rl_delivery_boundary_must_not_be_current_delivery_blocker" in validation["violations"]


def test_rl_delivery_boundary_allows_claim_after_complete_rl_evidence() -> None:
    source = _pending_open_spiel_claim_contract()
    source.update(
        {
            "training_proof_completed": True,
            "self_play_win_rate_claim_allowed": True,
            "model_quality_risk": False,
            "missing_requirements": [],
            "real_open_spiel_runtime_available": True,
            "phase1_trained_policy_artifact_count": 2,
            "seed_stability_evaluated": True,
            "long_run_completed": True,
            "ppo_or_equivalent_policy_update_completed": True,
        }
    )

    payload = build_rl_delivery_boundary_from_claim(source)

    assert payload["overall_status"] == "PASS"
    assert payload["claim_permissions"]["self_play_win_rate_claim_allowed"] is True
    assert payload["claim_permissions"]["production_strategy_quality_claim_allowed"] is True
    assert payload["model_quality_risk"] is False


def test_rl_delivery_boundary_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "open_spiel_claim_contract.json").write_text(
        json.dumps(_pending_open_spiel_claim_contract()),
        encoding="utf-8",
    )
    json_out = reports / "rl_delivery_boundary.json"
    markdown_out = reports / "rl_delivery_boundary.md"

    payload = write_rl_delivery_boundary(tmp_path, json_out, markdown_out)
    rebuilt = build_rl_delivery_boundary(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert rebuilt["overall_status"] == "PASS"
    assert json.loads(json_out.read_text(encoding="utf-8"))["gate_name"] == (
        "rl_delivery_vs_strategy_claim_boundary"
    )
    assert "Self-play win-rate claim allowed" in markdown_out.read_text(encoding="utf-8")
