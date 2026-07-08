from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TIMING_LABEL_QUALITY_UNCERTAIN = "TIMING_LABEL_QUALITY_UNCERTAIN"
TIMING_POLICY_HEURISTIC_CALIBRATED = "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED"
TIMING_LABEL_BOUNDARY = "REAL_HUMAN_TIMING_LABELS_REQUIRED_FOR_FULL_HUMAN_LIKENESS_PROOF"
TIMING_EVIDENCE_STATUS = "HEURISTIC_TIMING_ONLY_NOT_FINAL_HUMAN_LIKENESS_PROOF"
REQUIRED_TIMING_LABEL_FIELDS = (
    "decision_start_ts",
    "decision_end_ts",
    "human_wait_time_ms",
    "street",
    "position",
    "facing_bet",
    "action",
)


def build_timing_label_quality_boundary(*, timing_feature_available: bool) -> dict[str, Any]:
    return {
        "boundary": TIMING_LABEL_BOUNDARY,
        "status": TIMING_LABEL_QUALITY_UNCERTAIN,
        "timing_feature_available": timing_feature_available,
        "timing_policy_type": TIMING_POLICY_HEURISTIC_CALIBRATED,
        "real_human_timing_labels_available": False,
        "requires_real_human_timing_labels": True,
        "uses_real_human_timing_labels": False,
        "required_timing_label_fields": list(REQUIRED_TIMING_LABEL_FIELDS),
        "heuristic_timing_counts_as_full_human_likeness_proof": False,
        "final_human_likeness_claim_allowed_from_timing_alone": False,
        "final_production_human_likeness_proof_allowed": False,
        "current_delivery_blocker": False,
        "model_quality_risk": True,
        "reason": (
            "The agent returns a bounded wait_time_ms value, but without reviewed real human timing labels "
            "the timing behavior is calibrated heuristically and cannot be used as final production "
            "human-likeness evidence."
        ),
    }


def validate_current_timing_evidence_boundary(
    current_scope: Mapping[str, Any],
    timing_boundary: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if current_scope.get("timing_policy_type") != TIMING_POLICY_HEURISTIC_CALIBRATED:
        violations.append("timing_policy_type_must_remain_heuristic_or_table_tempo_calibrated")
    if current_scope.get("real_human_timing_label_quality") != TIMING_LABEL_QUALITY_UNCERTAIN:
        violations.append("real_human_timing_label_quality_must_remain_uncertain")
    if current_scope.get("real_human_timing_labels_available") is not False:
        violations.append("real_human_timing_labels_must_not_be_claimed_available")
    if current_scope.get("timing_human_likeness_final_proof_allowed") is not False:
        violations.append("timing_human_likeness_final_proof_must_be_blocked")
    if current_scope.get("timing_evidence_status") != TIMING_EVIDENCE_STATUS:
        violations.append("timing_evidence_status_must_remain_heuristic_not_final_proof")

    if timing_boundary.get("boundary") != TIMING_LABEL_BOUNDARY:
        violations.append("timing_label_boundary_must_require_real_human_timing_labels")
    if timing_boundary.get("status") != TIMING_LABEL_QUALITY_UNCERTAIN:
        violations.append("timing_label_quality_status_must_remain_uncertain")
    if timing_boundary.get("timing_feature_available") is not True:
        violations.append("timing_feature_must_remain_available")
    if timing_boundary.get("timing_policy_type") != TIMING_POLICY_HEURISTIC_CALIBRATED:
        violations.append("timing_boundary_policy_type_must_remain_heuristic_or_table_tempo_calibrated")
    if timing_boundary.get("real_human_timing_labels_available") is not False:
        violations.append("timing_boundary_must_not_claim_real_human_timing_labels_available")
    if timing_boundary.get("requires_real_human_timing_labels") is not True:
        violations.append("timing_boundary_must_require_real_human_timing_labels")
    if timing_boundary.get("uses_real_human_timing_labels") is not False:
        violations.append("timing_boundary_must_not_claim_real_human_timing_labels_are_used")
    if set(timing_boundary.get("required_timing_label_fields") or []) != set(REQUIRED_TIMING_LABEL_FIELDS):
        violations.append("timing_boundary_required_label_fields_must_be_complete")
    if timing_boundary.get("heuristic_timing_counts_as_full_human_likeness_proof") is not False:
        violations.append("heuristic_timing_must_not_count_as_full_human_likeness_proof")
    if timing_boundary.get("final_human_likeness_claim_allowed_from_timing_alone") is not False:
        violations.append("timing_alone_must_not_allow_final_human_likeness_claim")
    if timing_boundary.get("final_production_human_likeness_proof_allowed") is not False:
        violations.append("timing_final_production_human_likeness_proof_must_remain_blocked")
    if timing_boundary.get("current_delivery_blocker") is not False:
        violations.append("timing_label_gap_must_not_block_current_delivery")
    if timing_boundary.get("model_quality_risk") is not True:
        violations.append("timing_label_gap_must_remain_model_quality_risk")
    return violations


def timing_evidence_allows_final_human_likeness_claim(timing_boundary: Mapping[str, Any]) -> bool:
    return (
        timing_boundary.get("boundary") == TIMING_LABEL_BOUNDARY
        and timing_boundary.get("real_human_timing_labels_available") is True
        and timing_boundary.get("requires_real_human_timing_labels") is True
        and timing_boundary.get("uses_real_human_timing_labels") is True
        and set(timing_boundary.get("required_timing_label_fields") or []) == set(REQUIRED_TIMING_LABEL_FIELDS)
        and timing_boundary.get("heuristic_timing_counts_as_full_human_likeness_proof") is False
        and timing_boundary.get("final_human_likeness_claim_allowed_from_timing_alone") is False
        and timing_boundary.get("final_production_human_likeness_proof_allowed") is True
    )
