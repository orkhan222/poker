from __future__ import annotations

from typing import Any

from poker_agent.strategy_metric_gate import REQUIRED_PRODUCTION_METRICS


BOUNDARY_VERSION = "2026-07-09"
BOUNDARY_NAME = "DELIVERY_READY_FINAL_STRATEGY_REQUIRES_FULL_METRIC_BUNDLE"
DELIVERY_READY_STATUS = "READY"
DEPLOYED_STACK_APPROVED_STATUS = "APPROVED"
CLAIM_BLOCKED_STATUS = "DELIVERY_READY_STRATEGY_QUALITY_CLAIM_BLOCKED"
CLAIM_APPROVED_STATUS = "FINAL_STRATEGY_QUALITY_CLAIM_APPROVED"


def build_delivery_strategy_boundary(
    *,
    acceptance_summary: dict[str, Any],
    evaluation_metric_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Separate software delivery readiness from final strategy-quality approval."""

    service_delivery_ready = acceptance_summary.get("service_delivery") == DELIVERY_READY_STATUS
    deployed_strategy_stack_ready = (
        acceptance_summary.get("deployed_strategy_stack") == DEPLOYED_STACK_APPROVED_STATUS
    )
    final_metric_bundle_passed = evaluation_metric_coverage.get("final_metric_bundle_passed") is True
    metric_gate_allows_claim = (
        evaluation_metric_coverage.get("final_strategy_quality_claim_allowed") is True
    )
    final_strategy_quality_claim_allowed = final_metric_bundle_passed and metric_gate_allows_claim
    current_delivery_blocker = not (service_delivery_ready and deployed_strategy_stack_ready)

    payload = {
        "version": BOUNDARY_VERSION,
        "boundary": BOUNDARY_NAME,
        "status": CLAIM_APPROVED_STATUS if final_strategy_quality_claim_allowed else CLAIM_BLOCKED_STATUS,
        "service_delivery_ready": service_delivery_ready,
        "deployed_strategy_stack_ready": deployed_strategy_stack_ready,
        "software_delivery_ready": service_delivery_ready and deployed_strategy_stack_ready,
        "final_metric_bundle_passed": final_metric_bundle_passed,
        "metric_gate_allows_claim": metric_gate_allows_claim,
        "final_strategy_quality_claim_allowed": final_strategy_quality_claim_allowed,
        "current_delivery_blocker": current_delivery_blocker,
        "model_quality_risk": not final_strategy_quality_claim_allowed,
        "required_metric_bundle": list(REQUIRED_PRODUCTION_METRICS),
        "allowed_claims": [
            "The API, Docker package, verifier, reports, and deployed strategy stack are ready for delivery.",
            "The delivered stack can be handed off with tracked model-quality risks.",
        ],
        "blocked_claims": [
            "The final production-level poker strategy quality is approved.",
            "Accuracy and cross-entropy alone are sufficient for final strategy-quality approval.",
            "The delivered service implies a final competitive poker policy claim.",
        ],
    }
    payload["invariants"] = validate_delivery_strategy_boundary(payload)
    return payload


def validate_delivery_strategy_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []

    final_metric_bundle_passed = payload.get("final_metric_bundle_passed") is True
    metric_gate_allows_claim = payload.get("metric_gate_allows_claim") is True
    final_claim_allowed = payload.get("final_strategy_quality_claim_allowed") is True
    software_delivery_ready = payload.get("software_delivery_ready") is True

    if payload.get("boundary") != BOUNDARY_NAME:
        violations.append("delivery_strategy_boundary_name_must_match_contract")
    if set(payload.get("required_metric_bundle") or []) != set(REQUIRED_PRODUCTION_METRICS):
        violations.append("delivery_strategy_boundary_must_require_full_metric_bundle")
    if final_claim_allowed and not final_metric_bundle_passed:
        violations.append("final_strategy_claim_cannot_open_without_full_metric_bundle")
    if final_claim_allowed and not metric_gate_allows_claim:
        violations.append("final_strategy_claim_cannot_open_without_metric_gate_allowance")
    if not final_metric_bundle_passed and payload.get("status") != CLAIM_BLOCKED_STATUS:
        violations.append("strategy_claim_status_must_stay_blocked_until_metric_bundle_passes")
    if not final_metric_bundle_passed and final_claim_allowed:
        violations.append("final_strategy_quality_claim_must_remain_blocked")
    if software_delivery_ready and not final_metric_bundle_passed:
        if payload.get("current_delivery_blocker") is not False:
            violations.append("incomplete_metric_bundle_must_not_block_software_delivery")
        if payload.get("model_quality_risk") is not True:
            violations.append("incomplete_metric_bundle_must_remain_model_quality_risk")
    if not software_delivery_ready and payload.get("current_delivery_blocker") is not True:
        violations.append("missing_software_delivery_readiness_must_be_delivery_blocker")
    if final_claim_allowed and payload.get("status") != CLAIM_APPROVED_STATUS:
        violations.append("approved_strategy_claim_must_use_approved_status")

    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
    }


def is_delivery_ready_strategy_claim_blocked(payload: dict[str, Any]) -> bool:
    return (
        payload.get("software_delivery_ready") is True
        and payload.get("current_delivery_blocker") is False
        and payload.get("final_metric_bundle_passed") is False
        and payload.get("final_strategy_quality_claim_allowed") is False
        and payload.get("model_quality_risk") is True
    )
