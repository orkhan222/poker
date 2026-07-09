from __future__ import annotations

from poker_agent.delivery_strategy_boundary import (
    CLAIM_APPROVED_STATUS,
    CLAIM_BLOCKED_STATUS,
    COMPETITIVE_CLAIM_BLOCKED_STATUS,
    REQUIRED_STRATEGY_APPROVAL_GATES,
    build_delivery_strategy_boundary,
    is_delivery_ready_strategy_claim_blocked,
    validate_delivery_strategy_boundary,
)
from poker_agent.strategy_metric_gate import REQUIRED_PRODUCTION_METRICS


def test_delivery_ready_keeps_final_strategy_claim_blocked_until_metric_bundle_passes() -> None:
    payload = build_delivery_strategy_boundary(
        acceptance_summary={
            "service_delivery": "READY",
            "deployed_strategy_stack": "APPROVED",
        },
        evaluation_metric_coverage={
            "final_metric_bundle_passed": False,
            "final_strategy_quality_claim_allowed": False,
        },
    )

    assert payload["status"] == CLAIM_BLOCKED_STATUS
    assert payload["software_delivery_ready"] is True
    assert payload["deployment_ready"] is True
    assert payload["deployment_ready_does_not_imply_strategy_approved"] is True
    assert payload["strategy_approved"] is False
    assert payload["competitive_poker_agent_claim_allowed"] is False
    assert payload["competitive_poker_agent_claim_state"] == COMPETITIVE_CLAIM_BLOCKED_STATUS
    assert payload["current_delivery_blocker"] is False
    assert payload["final_metric_bundle_passed"] is False
    assert payload["final_strategy_quality_claim_allowed"] is False
    assert payload["strategy_hardening_complete"] is False
    assert payload["model_quality_risk"] is True
    assert set(payload["required_metric_bundle"]) == set(REQUIRED_PRODUCTION_METRICS)
    assert set(payload["required_strategy_approval_gates"]) == set(REQUIRED_STRATEGY_APPROVAL_GATES)
    assert set(payload["required_before_competitive_claim"]) == set(REQUIRED_STRATEGY_APPROVAL_GATES)
    assert payload["approval_separation"]["deployment_ready_does_not_imply_strategy_approved"] is True
    assert payload["approval_separation"]["fastapi_docker_predict_are_delivery_evidence_only"] is True
    assert payload["invariants"]["status"] == "PASS"
    assert is_delivery_ready_strategy_claim_blocked(payload) is True


def test_delivery_strategy_boundary_rejects_forced_claim_without_full_metric_bundle() -> None:
    payload = build_delivery_strategy_boundary(
        acceptance_summary={
            "service_delivery": "READY",
            "deployed_strategy_stack": "APPROVED",
        },
        evaluation_metric_coverage={
            "final_metric_bundle_passed": False,
            "final_strategy_quality_claim_allowed": True,
        },
    )

    assert payload["final_strategy_quality_claim_allowed"] is False
    assert payload["metric_gate_allows_claim"] is True
    assert payload["status"] == CLAIM_BLOCKED_STATUS
    assert payload["invariants"]["status"] == "PASS"

    tampered = dict(payload)
    tampered["final_strategy_quality_claim_allowed"] = True
    tampered["strategy_approved"] = True
    tampered["competitive_poker_agent_claim_allowed"] = True
    tampered["status"] = CLAIM_APPROVED_STATUS
    invariants = validate_delivery_strategy_boundary(tampered)

    assert invariants["status"] == "FAIL"
    assert "final_strategy_claim_cannot_open_without_full_metric_bundle" in invariants["violations"]
    assert "final_strategy_claim_cannot_open_without_strategy_hardening" in invariants["violations"]
    assert "strategy_claim_status_must_stay_blocked_until_metric_bundle_passes" in invariants["violations"]


def test_delivery_strategy_boundary_keeps_claim_blocked_until_strategy_hardening_is_complete() -> None:
    payload = build_delivery_strategy_boundary(
        acceptance_summary={
            "service_delivery": "READY",
            "deployed_strategy_stack": "APPROVED",
        },
        evaluation_metric_coverage={
            "final_metric_bundle_passed": True,
            "final_strategy_quality_claim_allowed": True,
        },
    )

    assert payload["status"] == CLAIM_BLOCKED_STATUS
    assert payload["software_delivery_ready"] is True
    assert payload["deployment_ready"] is True
    assert payload["final_metric_bundle_passed"] is True
    assert payload["metric_gate_allows_claim"] is True
    assert payload["strategy_hardening_complete"] is False
    assert payload["final_strategy_quality_claim_allowed"] is False
    assert payload["strategy_approved"] is False
    assert payload["competitive_poker_agent_claim_allowed"] is False
    assert payload["current_delivery_blocker"] is False
    assert payload["model_quality_risk"] is True
    assert payload["invariants"]["status"] == "PASS"


def test_missing_delivery_readiness_is_delivery_blocker_even_if_metric_gate_passes() -> None:
    payload = build_delivery_strategy_boundary(
        acceptance_summary={
            "service_delivery": "NOT_READY",
            "deployed_strategy_stack": "APPROVED",
        },
        evaluation_metric_coverage={
            "final_metric_bundle_passed": True,
            "final_strategy_quality_claim_allowed": True,
        },
    )

    assert payload["software_delivery_ready"] is False
    assert payload["current_delivery_blocker"] is True
    assert payload["final_strategy_quality_claim_allowed"] is False
    assert payload["strategy_approved"] is False
    assert payload["competitive_poker_agent_claim_allowed"] is False
    assert payload["invariants"]["status"] == "PASS"
