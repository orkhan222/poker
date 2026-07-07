from __future__ import annotations

from typing import Any

from poker_agent.human_likeness_evidence import HUMAN_LIKENESS_BOUNDARY, REQUIRED_BEHAVIOR_DIMENSIONS


FULL_HUMAN_LIKENESS_CLAIM = "FULL_HUMAN_LIKENESS"
CLAIM_DECISION_BLOCKED = "BLOCKED"
CLAIM_DECISION_APPROVED = "APPROVED"
REVIEWED_EVIDENCE_PASS_STATUSES = {"PASS", "REVIEWED_PASS", "APPROVED"}

FINAL_CLAIM_BLOCKING_REASONS = (
    "action_distribution_alone_is_not_sufficient",
    "bet_sizing_requires_reviewed_real_player_labels",
    "timing_requires_reviewed_real_human_timing_labels",
    "position_based_behavior_requires_slice_validation",
    "street_level_strategy_requires_street_specific_validation",
)


def evaluate_full_human_likeness_claim(evidence: dict[str, Any]) -> dict[str, Any]:
    fully_proven = has_complete_full_human_likeness_evidence(evidence)
    decision = CLAIM_DECISION_APPROVED if fully_proven else CLAIM_DECISION_BLOCKED
    return {
        "claim": FULL_HUMAN_LIKENESS_CLAIM,
        "decision": decision,
        "boundary": HUMAN_LIKENESS_BOUNDARY,
        "claim_allowed": fully_proven,
        "action_distribution_only_proof_rejected": True,
        "current_scope_action_distribution_passed": evidence.get("current_scope_action_distribution_passed"),
        "human_likeness_fully_proven": fully_proven,
        "current_delivery_blocker": False,
        "model_quality_risk": not fully_proven,
        "source_evidence": {
            "report": "reports/human_likeness_evidence.json",
            "overall_status": evidence.get("overall_status"),
            "invariant_status": (evidence.get("invariants") or {}).get("status"),
            "final_human_likeness_claim_allowed": evidence.get("final_human_likeness_claim_allowed"),
        },
        "required_evidence_dimensions": list(REQUIRED_BEHAVIOR_DIMENSIONS),
        "evidence_requirements": build_human_likeness_evidence_requirements(evidence),
        "blocking_reasons": [] if fully_proven else list(FINAL_CLAIM_BLOCKING_REASONS),
        "allowed_current_claim": (
            "Full human-likeness is approved from reviewed action distribution, bet sizing, timing, "
            "position-based behavior, and street-level strategy evidence."
            if fully_proven
            else "Current validation shows action-distribution similarity for the monitored delivery scope."
        ),
        "blocked_final_claim": (
            None
            if fully_proven
            else (
                "Full human-likeness is blocked until action distribution, bet sizing, timing, "
                "position-based behavior, and street-level strategy all have reviewed evidence."
            )
        ),
    }


def build_human_likeness_evidence_requirements(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dimensions = evidence.get("behavior_dimensions") or {}
    requirements: dict[str, dict[str, Any]] = {}
    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        source = dimensions.get(dimension) or {}
        has_reviewed_evidence = has_reviewed_dimension_evidence(source)
        requirements[dimension] = {
            "required_for_final_claim": True,
            "current_status": source.get("current_status"),
            "current_scope_metric": source.get("current_scope_metric"),
            "current_scope_value": source.get("current_scope_value"),
            "reviewed_evidence": source.get("reviewed_evidence") is True,
            "final_proof_allowed": source.get("final_proof_allowed") is True,
            "currently_sufficient_for_final_claim": has_reviewed_evidence,
            "remaining_requirement": source.get("remaining_requirement"),
        }
    return requirements


def has_complete_full_human_likeness_evidence(evidence: dict[str, Any]) -> bool:
    if evidence.get("overall_status") != "PASS":
        return False
    if (evidence.get("invariants") or {}).get("status") != "PASS":
        return False
    if evidence.get("current_scope_action_distribution_passed") is not True:
        return False
    if evidence.get("final_human_likeness_claim_allowed") is not True:
        return False

    dimensions = evidence.get("behavior_dimensions") or {}
    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        if not has_reviewed_dimension_evidence(dimensions.get(dimension) or {}):
            return False
    return True


def has_reviewed_dimension_evidence(dimension: dict[str, Any]) -> bool:
    return (
        dimension.get("reviewed_evidence") is True
        and dimension.get("final_proof_allowed") is True
        and dimension.get("current_status") in REVIEWED_EVIDENCE_PASS_STATUSES
    )


def is_full_human_likeness_claim_blocked(decision: dict[str, Any]) -> bool:
    requirements = decision.get("evidence_requirements") or {}
    source_evidence = decision.get("source_evidence") or {}
    if decision.get("claim") != FULL_HUMAN_LIKENESS_CLAIM:
        return False
    if decision.get("decision") != CLAIM_DECISION_BLOCKED:
        return False
    if decision.get("claim_allowed") is not False:
        return False
    if decision.get("human_likeness_fully_proven") is not False:
        return False
    if decision.get("action_distribution_only_proof_rejected") is not True:
        return False
    if set(decision.get("required_evidence_dimensions") or []) != set(REQUIRED_BEHAVIOR_DIMENSIONS):
        return False
    if set(decision.get("blocking_reasons") or []) != set(FINAL_CLAIM_BLOCKING_REASONS):
        return False
    if source_evidence.get("final_human_likeness_claim_allowed") is not False:
        return False
    all_dimensions_sufficient = True
    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        requirement = requirements.get(dimension) or {}
        if requirement.get("required_for_final_claim") is not True:
            return False
        sufficient = requirement.get("currently_sufficient_for_final_claim")
        if sufficient not in (True, False):
            return False
        all_dimensions_sufficient = all_dimensions_sufficient and sufficient
    if all_dimensions_sufficient:
        return False
    return True


def is_full_human_likeness_claim_approved(decision: dict[str, Any]) -> bool:
    requirements = decision.get("evidence_requirements") or {}
    source_evidence = decision.get("source_evidence") or {}
    if decision.get("claim") != FULL_HUMAN_LIKENESS_CLAIM:
        return False
    if decision.get("decision") != CLAIM_DECISION_APPROVED:
        return False
    if decision.get("claim_allowed") is not True:
        return False
    if decision.get("human_likeness_fully_proven") is not True:
        return False
    if decision.get("action_distribution_only_proof_rejected") is not True:
        return False
    if set(decision.get("required_evidence_dimensions") or []) != set(REQUIRED_BEHAVIOR_DIMENSIONS):
        return False
    if decision.get("blocking_reasons"):
        return False
    if source_evidence.get("overall_status") != "PASS":
        return False
    if source_evidence.get("invariant_status") != "PASS":
        return False
    if source_evidence.get("final_human_likeness_claim_allowed") is not True:
        return False
    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        requirement = requirements.get(dimension) or {}
        if requirement.get("required_for_final_claim") is not True:
            return False
        if requirement.get("reviewed_evidence") is not True:
            return False
        if requirement.get("final_proof_allowed") is not True:
            return False
        if requirement.get("currently_sufficient_for_final_claim") is not True:
            return False
    return True
