from __future__ import annotations

from poker_agent.timing_evidence_guard import (
    TIMING_EVIDENCE_STATUS,
    TIMING_LABEL_BOUNDARY,
    TIMING_LABEL_QUALITY_UNCERTAIN,
    TIMING_POLICY_HEURISTIC_CALIBRATED,
    build_timing_label_quality_boundary,
    timing_evidence_allows_final_human_likeness_claim,
    validate_current_timing_evidence_boundary,
)


def _current_delivery_scope() -> dict[str, object]:
    return {
        "timing_policy_type": TIMING_POLICY_HEURISTIC_CALIBRATED,
        "real_human_timing_label_quality": TIMING_LABEL_QUALITY_UNCERTAIN,
        "real_human_timing_labels_available": False,
        "timing_human_likeness_final_proof_allowed": False,
        "timing_evidence_status": TIMING_EVIDENCE_STATUS,
    }


def test_timing_guard_accepts_current_delivery_boundary() -> None:
    boundary = build_timing_label_quality_boundary(timing_feature_available=True)

    assert boundary["boundary"] == TIMING_LABEL_BOUNDARY
    assert boundary["status"] == TIMING_LABEL_QUALITY_UNCERTAIN
    assert boundary["timing_policy_type"] == TIMING_POLICY_HEURISTIC_CALIBRATED
    assert validate_current_timing_evidence_boundary(_current_delivery_scope(), boundary) == []
    assert timing_evidence_allows_final_human_likeness_claim(boundary) is False


def test_timing_guard_blocks_wait_time_as_full_human_likeness_proof() -> None:
    boundary = build_timing_label_quality_boundary(timing_feature_available=True)
    boundary["heuristic_timing_counts_as_full_human_likeness_proof"] = True
    boundary["final_human_likeness_claim_allowed_from_timing_alone"] = True
    boundary["final_production_human_likeness_proof_allowed"] = True

    violations = validate_current_timing_evidence_boundary(_current_delivery_scope(), boundary)

    assert "heuristic_timing_must_not_count_as_full_human_likeness_proof" in violations
    assert "timing_alone_must_not_allow_final_human_likeness_claim" in violations
    assert "timing_final_production_human_likeness_proof_must_remain_blocked" in violations


def test_timing_guard_requires_complete_real_timing_label_schema() -> None:
    boundary = build_timing_label_quality_boundary(timing_feature_available=True)
    boundary["requires_real_human_timing_labels"] = False
    boundary["required_timing_label_fields"] = ["human_wait_time_ms"]

    violations = validate_current_timing_evidence_boundary(_current_delivery_scope(), boundary)

    assert "timing_boundary_must_require_real_human_timing_labels" in violations
    assert "timing_boundary_required_label_fields_must_be_complete" in violations
