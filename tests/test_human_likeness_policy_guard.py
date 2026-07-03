from __future__ import annotations

from poker_agent.human_likeness_policy_guard import (
    CLAIM_DECISION_BLOCKED,
    FULL_HUMAN_LIKENESS_CLAIM,
    evaluate_full_human_likeness_claim,
    is_full_human_likeness_claim_blocked,
)


def test_policy_guard_rejects_action_distribution_only_human_likeness_claim() -> None:
    evidence = {
        "overall_status": "PASS",
        "invariants": {"status": "PASS"},
        "current_scope_action_distribution_passed": True,
        "final_human_likeness_claim_allowed": False,
        "behavior_dimensions": {
            "action_distribution": {
                "current_status": "PASS",
                "current_scope_metric": "js_divergence",
                "current_scope_value": 0.0026,
            }
        },
    }

    decision = evaluate_full_human_likeness_claim(evidence)

    assert decision["claim"] == FULL_HUMAN_LIKENESS_CLAIM
    assert decision["decision"] == CLAIM_DECISION_BLOCKED
    assert decision["claim_allowed"] is False
    assert decision["human_likeness_fully_proven"] is False
    assert decision["action_distribution_only_proof_rejected"] is True
    assert decision["current_scope_action_distribution_passed"] is True
    assert decision["evidence_requirements"]["action_distribution"]["currently_sufficient_for_final_claim"] is False
    assert decision["evidence_requirements"]["bet_sizing"]["currently_sufficient_for_final_claim"] is False
    assert decision["evidence_requirements"]["timing"]["currently_sufficient_for_final_claim"] is False
    assert decision["evidence_requirements"]["position_based_behavior"]["currently_sufficient_for_final_claim"] is False
    assert decision["evidence_requirements"]["street_level_strategy"]["currently_sufficient_for_final_claim"] is False
    assert is_full_human_likeness_claim_blocked(decision) is True


def test_policy_guard_detects_tampered_full_claim_approval() -> None:
    decision = evaluate_full_human_likeness_claim(
        {
            "overall_status": "PASS",
            "invariants": {"status": "PASS"},
            "current_scope_action_distribution_passed": True,
            "final_human_likeness_claim_allowed": False,
            "behavior_dimensions": {},
        }
    )
    decision["decision"] = "APPROVED"
    decision["claim_allowed"] = True

    assert is_full_human_likeness_claim_blocked(decision) is False
