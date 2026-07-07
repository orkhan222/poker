from __future__ import annotations

from poker_agent.human_likeness_policy_guard import (
    CLAIM_DECISION_APPROVED,
    CLAIM_DECISION_BLOCKED,
    FINAL_CLAIM_BLOCKING_REASONS,
    FULL_HUMAN_LIKENESS_CLAIM,
    evaluate_full_human_likeness_claim,
    is_full_human_likeness_claim_approved,
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


def test_policy_guard_rejects_any_single_dimension_as_final_proof() -> None:
    decision = evaluate_full_human_likeness_claim(
        {
            "overall_status": "PASS",
            "invariants": {"status": "PASS"},
            "current_scope_action_distribution_passed": True,
            "final_human_likeness_claim_allowed": False,
            "behavior_dimensions": {},
        }
    )

    for dimension in (
        "action_distribution",
        "bet_sizing",
        "timing",
        "position_based_behavior",
        "street_level_strategy",
    ):
        candidate = {
            **decision,
            "evidence_requirements": {
                key: value.copy() for key, value in decision["evidence_requirements"].items()
            },
        }
        candidate["evidence_requirements"][dimension]["currently_sufficient_for_final_claim"] = True

        assert is_full_human_likeness_claim_approved(candidate) is False
        assert is_full_human_likeness_claim_blocked(candidate) is True


def test_policy_guard_requires_complete_blocking_contract() -> None:
    decision = evaluate_full_human_likeness_claim(
        {
            "overall_status": "PASS",
            "invariants": {"status": "PASS"},
            "current_scope_action_distribution_passed": True,
            "final_human_likeness_claim_allowed": False,
            "behavior_dimensions": {},
        }
    )

    missing_reason = {**decision, "blocking_reasons": list(FINAL_CLAIM_BLOCKING_REASONS[:-1])}
    missing_dimension = {
        **decision,
        "required_evidence_dimensions": [
            "action_distribution",
            "bet_sizing",
            "timing",
            "position_based_behavior",
        ],
    }
    source_claim_unblocked = {
        **decision,
        "source_evidence": {
            **decision["source_evidence"],
            "final_human_likeness_claim_allowed": True,
        },
    }

    assert is_full_human_likeness_claim_blocked(missing_reason) is False
    assert is_full_human_likeness_claim_blocked(missing_dimension) is False
    assert is_full_human_likeness_claim_blocked(source_claim_unblocked) is False


def test_policy_guard_approves_only_complete_reviewed_behavior_bundle() -> None:
    evidence = _complete_reviewed_human_likeness_evidence()

    decision = evaluate_full_human_likeness_claim(evidence)

    assert decision["decision"] == CLAIM_DECISION_APPROVED
    assert decision["claim_allowed"] is True
    assert decision["human_likeness_fully_proven"] is True
    assert decision["action_distribution_only_proof_rejected"] is True
    assert decision["blocking_reasons"] == []
    assert decision["blocked_final_claim"] is None
    assert is_full_human_likeness_claim_approved(decision) is True
    assert is_full_human_likeness_claim_blocked(decision) is False

    for requirement in decision["evidence_requirements"].values():
        assert requirement["reviewed_evidence"] is True
        assert requirement["final_proof_allowed"] is True
        assert requirement["currently_sufficient_for_final_claim"] is True


def test_policy_guard_blocks_if_one_reviewed_dimension_is_missing() -> None:
    evidence = _complete_reviewed_human_likeness_evidence()
    evidence["final_human_likeness_claim_allowed"] = False
    evidence["behavior_dimensions"]["timing"]["reviewed_evidence"] = False

    decision = evaluate_full_human_likeness_claim(evidence)

    assert decision["decision"] == CLAIM_DECISION_BLOCKED
    assert decision["claim_allowed"] is False
    assert decision["human_likeness_fully_proven"] is False
    assert decision["evidence_requirements"]["timing"]["currently_sufficient_for_final_claim"] is False
    assert is_full_human_likeness_claim_approved(decision) is False
    assert is_full_human_likeness_claim_blocked(decision) is True


def test_policy_guard_rejects_inconsistent_source_final_claim() -> None:
    evidence = _complete_reviewed_human_likeness_evidence()
    evidence["behavior_dimensions"]["timing"]["reviewed_evidence"] = False

    decision = evaluate_full_human_likeness_claim(evidence)

    assert decision["decision"] == CLAIM_DECISION_BLOCKED
    assert is_full_human_likeness_claim_approved(decision) is False
    assert is_full_human_likeness_claim_blocked(decision) is False


def _complete_reviewed_human_likeness_evidence() -> dict:
    dimensions = {
        "action_distribution": ("js_divergence", 0.0026),
        "bet_sizing": ("bet_size_mae", 0.18),
        "timing": ("decision_time_mae", 120.0),
        "position_based_behavior": ("position_slice_similarity", 0.91),
        "street_level_strategy": ("street_slice_similarity", 0.89),
    }
    return {
        "overall_status": "PASS",
        "invariants": {"status": "PASS"},
        "current_scope_action_distribution_passed": True,
        "final_human_likeness_claim_allowed": True,
        "behavior_dimensions": {
            name: {
                "current_status": "REVIEWED_PASS",
                "current_scope_metric": metric,
                "current_scope_value": value,
                "reviewed_evidence": True,
                "final_proof_allowed": True,
                "remaining_requirement": None,
            }
            for name, (metric, value) in dimensions.items()
        },
    }
