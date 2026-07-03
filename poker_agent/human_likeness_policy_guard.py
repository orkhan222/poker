from __future__ import annotations

from typing import Any

from poker_agent.human_likeness_evidence import HUMAN_LIKENESS_BOUNDARY, REQUIRED_BEHAVIOR_DIMENSIONS


FULL_HUMAN_LIKENESS_CLAIM = "FULL_HUMAN_LIKENESS"
CLAIM_DECISION_BLOCKED = "BLOCKED"

FINAL_CLAIM_BLOCKING_REASONS = (
    "action_distribution_alone_is_not_sufficient",
    "bet_sizing_requires_reviewed_real_player_labels",
    "timing_requires_reviewed_real_human_timing_labels",
    "position_based_behavior_requires_slice_validation",
    "street_level_strategy_requires_street_specific_validation",
)


def evaluate_full_human_likeness_claim(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim": FULL_HUMAN_LIKENESS_CLAIM,
        "decision": CLAIM_DECISION_BLOCKED,
        "boundary": HUMAN_LIKENESS_BOUNDARY,
        "claim_allowed": False,
        "action_distribution_only_proof_rejected": True,
        "current_scope_action_distribution_passed": evidence.get("current_scope_action_distribution_passed"),
        "human_likeness_fully_proven": False,
        "current_delivery_blocker": False,
        "model_quality_risk": True,
        "source_evidence": {
            "report": "reports/human_likeness_evidence.json",
            "overall_status": evidence.get("overall_status"),
            "invariant_status": (evidence.get("invariants") or {}).get("status"),
            "final_human_likeness_claim_allowed": evidence.get("final_human_likeness_claim_allowed"),
        },
        "required_evidence_dimensions": list(REQUIRED_BEHAVIOR_DIMENSIONS),
        "evidence_requirements": build_human_likeness_evidence_requirements(evidence),
        "blocking_reasons": list(FINAL_CLAIM_BLOCKING_REASONS),
        "allowed_current_claim": (
            "Current validation shows action-distribution similarity for the monitored delivery scope."
        ),
        "blocked_final_claim": (
            "Full human-likeness is blocked until action distribution, bet sizing, timing, "
            "position-based behavior, and street-level strategy all have reviewed evidence."
        ),
    }


def build_human_likeness_evidence_requirements(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dimensions = evidence.get("behavior_dimensions") or {}
    requirements: dict[str, dict[str, Any]] = {}
    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        source = dimensions.get(dimension) or {}
        requirements[dimension] = {
            "required_for_final_claim": True,
            "current_status": source.get("current_status"),
            "current_scope_metric": source.get("current_scope_metric"),
            "current_scope_value": source.get("current_scope_value"),
            "currently_sufficient_for_final_claim": False,
            "remaining_requirement": source.get("remaining_requirement"),
        }
    return requirements


def is_full_human_likeness_claim_blocked(decision: dict[str, Any]) -> bool:
    requirements = decision.get("evidence_requirements") or {}
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
    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        requirement = requirements.get(dimension) or {}
        if requirement.get("required_for_final_claim") is not True:
            return False
        if requirement.get("currently_sufficient_for_final_claim") is not False:
            return False
    return True
