from __future__ import annotations

from typing import Any

from poker_agent.strategy_metric_gate import REQUIRED_PRODUCTION_METRICS


BOUNDARY_VERSION = "2026-07-09"
BOUNDARY_NAME = "DEPLOYMENT_READY_IS_NOT_STRATEGY_APPROVED"
DELIVERY_READY_STATUS = "READY"
DEPLOYED_STACK_APPROVED_STATUS = "APPROVED"
CLAIM_BLOCKED_STATUS = "DELIVERY_READY_STRATEGY_QUALITY_CLAIM_BLOCKED"
CLAIM_APPROVED_STATUS = "FINAL_STRATEGY_QUALITY_CLAIM_APPROVED"
COMPETITIVE_CLAIM_BLOCKED_STATUS = "BLOCKED_PENDING_MODEL_DATA_CALIBRATION_AND_MULTI_AGENT_TRAINING"

REQUIRED_STRATEGY_APPROVAL_GATES = (
    "cleaner_real_gameplay_data",
    "stronger_challenger_model",
    "calibration_gate",
    "full_multi_agent_training",
    "full_metric_bundle",
)


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
    strategy_hardening_complete = False
    final_strategy_quality_claim_allowed = (
        final_metric_bundle_passed
        and metric_gate_allows_claim
        and strategy_hardening_complete
    )
    current_delivery_blocker = not (service_delivery_ready and deployed_strategy_stack_ready)

    payload = {
        "version": BOUNDARY_VERSION,
        "boundary": BOUNDARY_NAME,
        "status": CLAIM_APPROVED_STATUS if final_strategy_quality_claim_allowed else CLAIM_BLOCKED_STATUS,
        "service_delivery_ready": service_delivery_ready,
        "deployed_strategy_stack_ready": deployed_strategy_stack_ready,
        "software_delivery_ready": service_delivery_ready and deployed_strategy_stack_ready,
        "deployment_ready": service_delivery_ready and deployed_strategy_stack_ready,
        "deployment_ready_does_not_imply_strategy_approved": True,
        "deployment_sufficient_components": {
            "fastapi_service": service_delivery_ready,
            "docker_packaging": service_delivery_ready,
            "predict_endpoint": service_delivery_ready,
            "health_endpoint": service_delivery_ready,
            "reports_and_verifier": service_delivery_ready,
        },
        "strategy_approved": final_strategy_quality_claim_allowed,
        "competitive_poker_agent_claim_allowed": final_strategy_quality_claim_allowed,
        "competitive_poker_agent_claim_state": (
            CLAIM_APPROVED_STATUS
            if final_strategy_quality_claim_allowed
            else COMPETITIVE_CLAIM_BLOCKED_STATUS
        ),
        "final_metric_bundle_passed": final_metric_bundle_passed,
        "metric_gate_allows_claim": metric_gate_allows_claim,
        "strategy_hardening_complete": strategy_hardening_complete,
        "final_strategy_quality_claim_allowed": final_strategy_quality_claim_allowed,
        "current_delivery_blocker": current_delivery_blocker,
        "model_quality_risk": not final_strategy_quality_claim_allowed,
        "required_metric_bundle": list(REQUIRED_PRODUCTION_METRICS),
        "required_strategy_approval_gates": list(REQUIRED_STRATEGY_APPROVAL_GATES),
        "required_before_competitive_claim": {
            "cleaner_real_gameplay_data": {
                "required": True,
                "status": "OPEN",
                "reason": "Hole-card coverage, explicit action context, and larger clean gameplay labels remain required before a competitive-agent claim.",
            },
            "stronger_challenger_model": {
                "required": True,
                "status": "OPEN",
                "reason": "A challenger must pass raw-model and strategy-quality gates before final strategy approval.",
            },
            "calibration_gate": {
                "required": True,
                "status": "OPEN",
                "reason": "Probability calibration, bet sizing, and timing calibration must pass on reviewed data.",
            },
            "full_multi_agent_training": {
                "required": True,
                "status": "OPEN",
                "reason": "A production-scale agent-only training/evaluation run with seed stability is required.",
            },
            "full_metric_bundle": {
                "required": True,
                "status": "OPEN" if not final_metric_bundle_passed else "PARTIAL",
                "reason": "Accuracy and cross-entropy are diagnostic only; final approval requires the full production metric bundle.",
            },
        },
        "approval_separation": {
            "deployment_ready_can_pass_without_strategy_approval": True,
            "deployment_ready_does_not_imply_strategy_approved": True,
            "fastapi_docker_predict_are_delivery_evidence_only": True,
            "competitive_claim_requires_model_data_calibration_and_training": True,
        },
        "allowed_claims": [
            "The API, Docker package, verifier, reports, and deployed strategy stack are ready for delivery.",
            "The delivered stack can be handed off with tracked model-quality risks.",
            "Deployment-ready means the service package is runnable and verifiable, not that the poker strategy is final-approved.",
        ],
        "blocked_claims": [
            "The final production-level poker strategy quality is approved.",
            "Accuracy and cross-entropy alone are sufficient for final strategy-quality approval.",
            "The delivered service implies a final competitive poker policy claim.",
            "FastAPI, Docker, or /predict availability proves competitive poker-agent quality.",
            "Deployment-ready is equivalent to strategy-approved.",
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
    deployment_components = payload.get("deployment_sufficient_components") or {}
    required_strategy_gates = payload.get("required_strategy_approval_gates") or []
    required_before_claim = payload.get("required_before_competitive_claim") or {}
    separation = payload.get("approval_separation") or {}

    if payload.get("boundary") != BOUNDARY_NAME:
        violations.append("delivery_strategy_boundary_name_must_match_contract")
    if set(payload.get("required_metric_bundle") or []) != set(REQUIRED_PRODUCTION_METRICS):
        violations.append("delivery_strategy_boundary_must_require_full_metric_bundle")
    if set(required_strategy_gates) != set(REQUIRED_STRATEGY_APPROVAL_GATES):
        violations.append("delivery_strategy_boundary_must_require_all_strategy_approval_gates")
    if set(required_before_claim) != set(REQUIRED_STRATEGY_APPROVAL_GATES):
        violations.append("delivery_strategy_boundary_must_describe_every_strategy_approval_gate")
    for gate_name in REQUIRED_STRATEGY_APPROVAL_GATES:
        gate = required_before_claim.get(gate_name) or {}
        if gate.get("required") is not True:
            violations.append(f"strategy_approval_gate_must_be_required:{gate_name}")
        if gate.get("status") not in {"OPEN", "PARTIAL", "COMPLETE"}:
            violations.append(f"strategy_approval_gate_status_must_be_known:{gate_name}")
    if payload.get("deployment_ready") is not software_delivery_ready:
        violations.append("deployment_ready_must_match_software_delivery_ready")
    if payload.get("deployment_ready_does_not_imply_strategy_approved") is not True:
        violations.append("deployment_ready_must_not_imply_strategy_approval")
    for component in ("fastapi_service", "docker_packaging", "predict_endpoint", "health_endpoint", "reports_and_verifier"):
        if software_delivery_ready and deployment_components.get(component) is not True:
            violations.append(f"deployment_component_must_be_ready:{component}")
    if separation.get("deployment_ready_can_pass_without_strategy_approval") is not True:
        violations.append("approval_separation_must_allow_delivery_without_strategy_approval")
    if separation.get("deployment_ready_does_not_imply_strategy_approved") is not True:
        violations.append("approval_separation_must_keep_deployment_and_strategy_distinct")
    if separation.get("fastapi_docker_predict_are_delivery_evidence_only") is not True:
        violations.append("fastapi_docker_predict_must_be_delivery_evidence_only")
    if separation.get("competitive_claim_requires_model_data_calibration_and_training") is not True:
        violations.append("competitive_claim_must_require_model_data_calibration_and_training")
    if final_claim_allowed and not final_metric_bundle_passed:
        violations.append("final_strategy_claim_cannot_open_without_full_metric_bundle")
    if final_claim_allowed and not metric_gate_allows_claim:
        violations.append("final_strategy_claim_cannot_open_without_metric_gate_allowance")
    if final_claim_allowed and payload.get("strategy_hardening_complete") is not True:
        violations.append("final_strategy_claim_cannot_open_without_strategy_hardening")
    if payload.get("strategy_approved") is not final_claim_allowed:
        violations.append("strategy_approved_must_match_final_strategy_claim")
    if payload.get("competitive_poker_agent_claim_allowed") is not final_claim_allowed:
        violations.append("competitive_claim_must_match_final_strategy_claim")
    expected_competitive_state = (
        CLAIM_APPROVED_STATUS if final_claim_allowed else COMPETITIVE_CLAIM_BLOCKED_STATUS
    )
    if payload.get("competitive_poker_agent_claim_state") != expected_competitive_state:
        violations.append("competitive_claim_state_must_match_strategy_approval")
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
        and payload.get("deployment_ready") is True
        and payload.get("deployment_ready_does_not_imply_strategy_approved") is True
        and payload.get("current_delivery_blocker") is False
        and payload.get("strategy_approved") is False
        and payload.get("competitive_poker_agent_claim_allowed") is False
        and payload.get("competitive_poker_agent_claim_state") == COMPETITIVE_CLAIM_BLOCKED_STATUS
        and payload.get("final_metric_bundle_passed") is False
        and payload.get("strategy_hardening_complete") is False
        and payload.get("final_strategy_quality_claim_allowed") is False
        and payload.get("model_quality_risk") is True
        and set(payload.get("required_strategy_approval_gates") or []) == set(REQUIRED_STRATEGY_APPROVAL_GATES)
    )
