from __future__ import annotations

from poker_agent.raw_model_challenger import (
    NOT_STANDALONE_APPROVED,
    STANDALONE_APPROVED,
    assert_challenger_report,
    evaluate_challenger_gate,
    validate_challenger_report,
)


def test_challenger_gate_blocks_dataset_audit_blockers() -> None:
    metrics = {
        "accuracy": 0.72,
        "lift_vs_majority": 0.04,
        "macro_f1": 0.56,
        "balanced_accuracy": 0.55,
        "ece_10": 0.05,
    }
    slices = {
        "observed_hole_cards": {"macro_f1": 0.58},
        "facing_bet": {"macro_f1": 0.49},
    }
    blockers = [{"severity": "blocker", "issue": "Hole-card coverage is too low."}]

    gate = evaluate_challenger_gate(metrics, slices, blockers)

    assert gate["status"] == "FAIL"
    assert "dataset_audit_blockers" in gate["failed_gates"]


def test_challenger_report_rejects_false_standalone_approval() -> None:
    payload = {
        "standalone_status": STANDALONE_APPROVED,
        "approved_as_standalone_policy": True,
        "audit": {"blocker_count": 1},
        "best_candidate": {"gate": {"status": "FAIL"}},
        "approval_boundary": {
            "existing_service_delivery_affected": False,
            "raw_model_standalone_allowed": True,
        },
    }

    violations = validate_challenger_report(payload)

    assert "standalone_approval_requires_challenger_gate_pass" in violations
    assert "standalone_approval_requires_zero_dataset_blockers" in violations
    assert "approval_boundary_cannot_allow_raw_standalone_when_gate_fails" in violations


def test_challenger_report_accepts_not_approved_component_boundary() -> None:
    payload = {
        "standalone_status": NOT_STANDALONE_APPROVED,
        "approved_as_standalone_policy": False,
        "audit": {"blocker_count": 1},
        "best_candidate": {"gate": {"status": "FAIL"}},
        "approval_boundary": {
            "existing_service_delivery_affected": False,
            "raw_model_standalone_allowed": False,
        },
    }

    assert validate_challenger_report(payload) == []
    assert_challenger_report(payload)
