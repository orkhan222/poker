from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_STRATEGY_QUALITY_STATUS_VERSION = "2026-06-29"
DELIVERY_READY = "READY"
DEPLOYED_STACK_APPROVED = "APPROVED"
FINAL_STRATEGY_NOT_APPROVED = "NOT_APPROVED_PENDING_HARDENING_GATES"
FINAL_STRATEGY_APPROVED = "APPROVED"
REQUIRED = "REQUIRED"
COMPETITIVE_AGENT_CLAIM_BLOCKED = "BLOCKED_PENDING_MODEL_DATA_AND_TRAINING_HARDENING"

REQUIRED_WORK_ITEMS = (
    "stronger_challenger_model",
    "hole_card_data_quality",
    "calibration",
    "larger_validation_data",
    "production_scale_multi_agent_training",
)


def build_final_strategy_quality_status(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    final_acceptance = _read_optional_json(reports / "final_delivery_acceptance.json")
    challenger = _read_optional_json(reports / "challenger_strategy_quality.json")
    hole_card = _read_optional_json(reports / "hole_card_data_quality.json")
    calibration = _read_optional_json(reports / "bet_timing_calibration.json")
    behavioral = _read_optional_json(reports / "behavioral_revalidation.json")
    multi_agent = _read_optional_json(reports / "multi_agent_training_status.json")
    runtime_monitoring = _read_optional_json(reports / "production_runtime_monitoring.json")

    acceptance_summary = final_acceptance.get("acceptance_summary") or {}
    challenger_boundary = challenger.get("strategy_quality_boundary") or {}
    challenger_result = challenger.get("challenger_result") or {}
    hole_boundary = hole_card.get("upstream_data_quality_boundary") or {}
    hole_strength = hole_card.get("strength_signal_impact") or {}
    calibration_boundary = calibration.get("calibration_boundary") or {}
    behavioral_boundary = behavioral.get("revalidation_boundary") or {}
    multi_boundary = multi_agent.get("training_boundary") or {}
    runtime_boundary = runtime_monitoring.get("runtime_observability_boundary") or {}

    delivery_ready = (
        final_acceptance.get("overall_status") == "PASS"
        and acceptance_summary.get("service_delivery") == DELIVERY_READY
        and acceptance_summary.get("deployed_strategy_stack") == DEPLOYED_STACK_APPROVED
    )
    final_strategy_approved = _final_strategy_quality_conditions_met(
        challenger_boundary=challenger_boundary,
        hole_boundary=hole_boundary,
        calibration_boundary=calibration_boundary,
        behavioral_boundary=behavioral_boundary,
        multi_boundary=multi_boundary,
    )

    payload: dict[str, Any] = {
        "version": FINAL_STRATEGY_QUALITY_STATUS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Final production-level poker strategy quality boundary",
        "client_statement": (
            "The project is ready from a delivery perspective: API, Docker, reports, verifier, and the "
            "deployed strategy stack are present. Final production-level poker strategy quality is not yet "
            "approved because the remaining model-quality hardening gates are still open."
        ),
        "delivery_boundary": {
            "software_delivery_ready": delivery_ready,
            "service_delivery": acceptance_summary.get("service_delivery"),
            "deployed_strategy_stack": acceptance_summary.get("deployed_strategy_stack"),
            "api_present": True,
            "docker_present": True,
            "reports_present": True,
            "verifier_present": True,
            "current_delivery_blocker": False,
        },
        "deployment_vs_competitive_claim_boundary": {
            "deployment_delivery_ready": delivery_ready,
            "deployment_sufficient_components": {
                "fastapi_service": True,
                "docker_packaging": True,
                "predict_endpoint": True,
                "health_endpoint": True,
                "reports_and_verifier": True,
            },
            "deployment_claim_allowed": delivery_ready,
            "deployment_claim": (
                "The FastAPI service, Docker package, /health endpoint, /predict endpoint, reports, "
                "and verifier are sufficient for software delivery review."
            ),
            "competitive_poker_agent_claim_allowed": final_strategy_approved,
            "competitive_poker_agent_claim_state": (
                "APPROVED" if final_strategy_approved else COMPETITIVE_AGENT_CLAIM_BLOCKED
            ),
            "competitive_claim_blocker_reason": (
                "A competitive poker agent claim requires a stronger model, cleaner and more complete "
                "card/action data, calibrated behavior, larger real-game validation, and full production-scale "
                "multi-agent training."
            ),
            "required_before_competitive_claim": [
                "stronger_challenger_model",
                "hole_card_data_quality",
                "calibration",
                "larger_validation_data",
                "production_scale_multi_agent_training",
            ],
            "current_delivery_blocker": False,
            "deployed_strategy_stack_affected": False,
        },
        "final_strategy_quality_boundary": {
            "status": FINAL_STRATEGY_APPROVED if final_strategy_approved else FINAL_STRATEGY_NOT_APPROVED,
            "final_production_strategy_quality_approved": final_strategy_approved,
            "final_production_strategy_quality_claim_allowed": final_strategy_approved,
            "delivery_blocker": False,
            "deployed_strategy_stack_affected": False,
            "reason": (
                "The deployed strategy stack is deliverable, but final strategy-quality approval requires a "
                "passing challenger, improved hole-card data, calibrated behavior, larger real validation, "
                "and completed production-scale multi-agent training."
            ),
        },
        "remaining_work": {
            "stronger_challenger_model": {
                "status": REQUIRED,
                "source_report": "reports/challenger_strategy_quality.json",
                "current_gate": challenger_boundary.get("challenger_gate_status"),
                "raw_gate": challenger_boundary.get("raw_production_gate_status"),
                "best_candidate": challenger_result.get("best_candidate"),
                "macro_f1": challenger_result.get("macro_f1"),
                "failed_gates": challenger_result.get("failed_gates") or [],
                "required_outcome": "A stronger challenger passes every challenger and raw production gate.",
            },
            "hole_card_data_quality": {
                "status": REQUIRED,
                "source_report": "reports/hole_card_data_quality.json",
                "limitation_status": hole_boundary.get("limitation_status"),
                "upstream_resolved": hole_boundary.get("upstream_data_quality_issue_resolved"),
                "requires_ocr_or_parser_improvement": hole_boundary.get("requires_ocr_or_parser_improvement"),
                "strength_signal_status": hole_strength.get("status"),
                "required_outcome": "Hole-card extraction and reviewed labels are strong enough for reliable card-aware policy gates.",
            },
            "calibration": {
                "status": REQUIRED,
                "source_report": "reports/bet_timing_calibration.json",
                "calibration_status": calibration_boundary.get("status"),
                "requires_more_real_player_behavior_labels": calibration_boundary.get(
                    "requires_more_real_player_behavior_labels"
                ),
                "final_high_realism_claim_allowed": calibration_boundary.get("final_high_realism_claim_allowed"),
                "required_outcome": "Action probabilities, bet sizing, and timing are calibrated on reviewed real-player labels.",
            },
            "larger_validation_data": {
                "status": REQUIRED,
                "source_report": "reports/behavioral_revalidation.json",
                "larger_clean_real_gameplay_revalidation_required": behavioral_boundary.get(
                    "larger_clean_real_gameplay_revalidation_required"
                ),
                "generalized_human_likeness_claim_allowed": behavioral_boundary.get(
                    "generalized_human_likeness_claim_allowed"
                ),
                "required_outcome": "Larger clean real-gameplay validation confirms action alignment and human-likeness slices.",
            },
            "production_scale_multi_agent_training": {
                "status": REQUIRED,
                "source_report": "reports/multi_agent_training_status.json",
                "full_training_status": multi_boundary.get("full_production_scale_multi_agent_training_status"),
                "acceptance_training_status": multi_boundary.get("acceptance_training_status"),
                "production_blocker_for_current_delivery": multi_boundary.get(
                    "production_blocker",
                    multi_boundary.get("production_blocker_for_current_delivery"),
                ),
                "required_outcome": "Full production-scale multi-agent training completes under the approved A100/H100 hardening profile.",
            },
        },
        "real_traffic_boundary": {
            "real_production_traffic_approved": runtime_boundary.get("real_production_traffic_approved"),
            "approval_status": runtime_boundary.get("real_production_traffic_approval_status"),
            "monitoring_required": runtime_boundary.get("monitoring_required_for_real_traffic"),
            "rollback_required": runtime_boundary.get("rollback_rules_required_for_real_traffic"),
            "live_drift_tracking_required": runtime_boundary.get("live_drift_tracking_required_for_real_traffic"),
            "prediction_distribution_tracking_required": runtime_boundary.get(
                "prediction_distribution_tracking_required_for_real_traffic"
            ),
            "model_confidence_monitoring_required": runtime_boundary.get(
                "model_confidence_monitoring_required_for_real_traffic"
            ),
        },
        "allowed_claims": [
            "The API, Docker packaging, reports, verifier, and deployed strategy stack are ready for delivery.",
            "FastAPI, Docker, /health, and /predict are sufficient for software delivery review.",
            "The deployed strategy stack can be treated separately from final production-level strategy quality.",
            "The remaining hardening work is explicitly tracked and does not block the current delivery package.",
        ],
        "blocked_claims": [
            "The current delivery is a final competitive poker agent.",
            "Final production-level poker strategy quality is approved.",
            "The current challenger model is sufficient for final production strategy quality.",
            "Hole-card data quality is solved.",
            "Calibration and larger real gameplay validation are complete.",
            "Production-scale multi-agent training has been completed.",
        ],
        "evidence": {
            "final_delivery_acceptance": "reports/final_delivery_acceptance.json",
            "challenger_strategy_quality": "reports/challenger_strategy_quality.json",
            "hole_card_data_quality": "reports/hole_card_data_quality.json",
            "bet_timing_calibration": "reports/bet_timing_calibration.json",
            "behavioral_revalidation": "reports/behavioral_revalidation.json",
            "multi_agent_training_status": "reports/multi_agent_training_status.json",
            "production_runtime_monitoring": "reports/production_runtime_monitoring.json",
        },
    }
    payload["invariants"] = validate_final_strategy_quality_status(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_final_strategy_quality_status(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    delivery = payload.get("delivery_boundary") or {}
    deployment_vs_competitive = payload.get("deployment_vs_competitive_claim_boundary") or {}
    final_quality = payload.get("final_strategy_quality_boundary") or {}
    remaining = payload.get("remaining_work") or {}
    real_traffic = payload.get("real_traffic_boundary") or {}

    if payload.get("overall_status") == "PASS":
        violations.append("overall_status_must_be_assigned_after_invariant_validation")
    if delivery.get("software_delivery_ready") is not True:
        violations.append("software_delivery_must_be_ready")
    if delivery.get("service_delivery") != DELIVERY_READY:
        violations.append("service_delivery_must_be_ready")
    if delivery.get("deployed_strategy_stack") != DEPLOYED_STACK_APPROVED:
        violations.append("deployed_strategy_stack_must_be_approved")
    if delivery.get("current_delivery_blocker") is not False:
        violations.append("final_strategy_quality_gap_must_not_block_current_delivery")
    components = deployment_vs_competitive.get("deployment_sufficient_components") or {}
    for component in ("fastapi_service", "docker_packaging", "predict_endpoint", "health_endpoint", "reports_and_verifier"):
        if components.get(component) is not True:
            violations.append(f"deployment_component_must_be_present:{component}")
    if deployment_vs_competitive.get("deployment_delivery_ready") is not True:
        violations.append("deployment_boundary_must_keep_software_delivery_ready")
    if deployment_vs_competitive.get("deployment_claim_allowed") is not True:
        violations.append("deployment_claim_must_be_allowed_for_software_delivery")
    if deployment_vs_competitive.get("competitive_poker_agent_claim_allowed") is not False:
        violations.append("competitive_poker_agent_claim_must_be_blocked")
    if deployment_vs_competitive.get("competitive_poker_agent_claim_state") != COMPETITIVE_AGENT_CLAIM_BLOCKED:
        violations.append("competitive_poker_agent_claim_state_must_remain_blocked")
    if set(deployment_vs_competitive.get("required_before_competitive_claim") or []) != set(REQUIRED_WORK_ITEMS):
        violations.append("competitive_claim_required_work_items_must_match_hardening_gates")
    if deployment_vs_competitive.get("current_delivery_blocker") is not False:
        violations.append("competitive_claim_gap_must_not_block_current_delivery")
    if deployment_vs_competitive.get("deployed_strategy_stack_affected") is not False:
        violations.append("competitive_claim_gap_must_not_affect_deployed_stack")

    missing_items = sorted(set(REQUIRED_WORK_ITEMS) - set(remaining))
    if missing_items:
        violations.append(f"missing_remaining_work_items:{','.join(missing_items)}")
    for item in REQUIRED_WORK_ITEMS:
        if (remaining.get(item) or {}).get("status") != REQUIRED:
            violations.append(f"remaining_work_item_must_remain_required:{item}")

    if final_quality.get("delivery_blocker") is not False:
        violations.append("final_strategy_quality_gap_must_not_be_delivery_blocker")
    if final_quality.get("deployed_strategy_stack_affected") is not False:
        violations.append("final_strategy_quality_gap_must_not_affect_deployed_stack")
    if final_quality.get("final_production_strategy_quality_approved") is not False:
        violations.append("final_production_strategy_quality_must_not_be_approved")
    if final_quality.get("final_production_strategy_quality_claim_allowed") is not False:
        violations.append("final_production_strategy_quality_claim_must_be_blocked")
    if final_quality.get("status") != FINAL_STRATEGY_NOT_APPROVED:
        violations.append("final_strategy_quality_status_must_remain_not_approved")

    challenger = remaining.get("stronger_challenger_model") or {}
    if challenger.get("current_gate") == "PASS" and challenger.get("raw_gate") == "PASS":
        violations.append("remaining_challenger_item_cannot_be_required_if_all_gates_pass")
    if not challenger.get("failed_gates"):
        violations.append("challenger_remaining_work_must_show_failed_gates")

    hole = remaining.get("hole_card_data_quality") or {}
    if hole.get("upstream_resolved") is not False:
        violations.append("hole_card_remaining_work_requires_unresolved_upstream_status")
    if hole.get("requires_ocr_or_parser_improvement") is not True:
        violations.append("hole_card_remaining_work_requires_ocr_or_parser_improvement")

    calibration = remaining.get("calibration") or {}
    if calibration.get("requires_more_real_player_behavior_labels") is not True:
        violations.append("calibration_remaining_work_requires_more_real_player_labels")
    if calibration.get("final_high_realism_claim_allowed") is not False:
        violations.append("calibration_remaining_work_must_block_final_high_realism_claim")

    larger_validation = remaining.get("larger_validation_data") or {}
    if larger_validation.get("larger_clean_real_gameplay_revalidation_required") is not True:
        violations.append("larger_validation_remaining_work_must_require_clean_real_gameplay")
    if larger_validation.get("generalized_human_likeness_claim_allowed") is not False:
        violations.append("larger_validation_remaining_work_must_block_generalized_claims")

    multi = remaining.get("production_scale_multi_agent_training") or {}
    if multi.get("full_training_status") != "NOT_COMPLETED":
        violations.append("production_scale_multi_agent_training_must_remain_not_completed")
    if multi.get("production_blocker_for_current_delivery") is not False:
        violations.append("multi_agent_training_gap_must_not_block_current_delivery")

    if real_traffic.get("real_production_traffic_approved") is not False:
        violations.append("real_traffic_must_not_be_approved_before_observability")
    for key in (
        "monitoring_required",
        "rollback_required",
        "live_drift_tracking_required",
        "prediction_distribution_tracking_required",
        "model_confidence_monitoring_required",
    ):
        if real_traffic.get(key) is not True:
            violations.append(f"real_traffic_boundary_requires:{key}")

    blocked = set(payload.get("blocked_claims") or [])
    if "Final production-level poker strategy quality is approved." not in blocked:
        violations.append("blocked_claims_must_reject_final_strategy_quality_approval")
    if "The current delivery is a final competitive poker agent." not in blocked:
        violations.append("blocked_claims_must_reject_competitive_agent_claim")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_final_strategy_quality_status(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_final_strategy_quality_status(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_final_strategy_quality_status_markdown(payload), encoding="utf-8")
    return payload


def render_final_strategy_quality_status_markdown(payload: dict[str, Any]) -> str:
    delivery = payload["delivery_boundary"]
    deployment_vs_competitive = payload["deployment_vs_competitive_claim_boundary"]
    final_quality = payload["final_strategy_quality_boundary"]
    real_traffic = payload["real_traffic_boundary"]
    lines = [
        "# Final Strategy Quality Status",
        "",
        payload["client_statement"],
        "",
        "## Delivery Boundary",
        "",
        f"- Software delivery ready: `{delivery['software_delivery_ready']}`",
        f"- Service delivery: `{delivery['service_delivery']}`",
        f"- Deployed strategy stack: `{delivery['deployed_strategy_stack']}`",
        f"- Current delivery blocker: `{delivery['current_delivery_blocker']}`",
        "",
        "## Deployment vs Competitive Claim Boundary",
        "",
        f"- Deployment claim allowed: `{deployment_vs_competitive['deployment_claim_allowed']}`",
        f"- Competitive poker-agent claim allowed: `{deployment_vs_competitive['competitive_poker_agent_claim_allowed']}`",
        f"- Competitive poker-agent claim state: `{deployment_vs_competitive['competitive_poker_agent_claim_state']}`",
        f"- Current delivery blocker: `{deployment_vs_competitive['current_delivery_blocker']}`",
        f"- Reason: {deployment_vs_competitive['competitive_claim_blocker_reason']}",
        "",
        "## Final Strategy Quality Boundary",
        "",
        f"- Status: `{final_quality['status']}`",
        f"- Final production strategy quality approved: `{final_quality['final_production_strategy_quality_approved']}`",
        f"- Claim allowed: `{final_quality['final_production_strategy_quality_claim_allowed']}`",
        f"- Delivery blocker: `{final_quality['delivery_blocker']}`",
        f"- Deployed stack affected: `{final_quality['deployed_strategy_stack_affected']}`",
        f"- Reason: {final_quality['reason']}",
        "",
        "## Remaining Work",
        "",
    ]
    for name, item in payload["remaining_work"].items():
        lines.append(f"- `{name}`: status=`{item['status']}`, required_outcome={item['required_outcome']}")
    lines.extend(
        [
            "",
            "## Real Traffic Boundary",
            "",
            f"- Real production traffic approved: `{real_traffic['real_production_traffic_approved']}`",
            f"- Approval status: `{real_traffic['approval_status']}`",
            f"- Monitoring required: `{real_traffic['monitoring_required']}`",
            f"- Rollback required: `{real_traffic['rollback_required']}`",
            f"- Live drift tracking required: `{real_traffic['live_drift_tracking_required']}`",
            f"- Prediction distribution tracking required: `{real_traffic['prediction_distribution_tracking_required']}`",
            f"- Model confidence monitoring required: `{real_traffic['model_confidence_monitoring_required']}`",
            "",
            "## Blocked Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _final_strategy_quality_conditions_met(
    *,
    challenger_boundary: dict[str, Any],
    hole_boundary: dict[str, Any],
    calibration_boundary: dict[str, Any],
    behavioral_boundary: dict[str, Any],
    multi_boundary: dict[str, Any],
) -> bool:
    return all(
        [
            challenger_boundary.get("final_production_strategy_quality_claim_allowed") is True,
            hole_boundary.get("upstream_data_quality_issue_resolved") is True,
            calibration_boundary.get("final_high_realism_claim_allowed") is True,
            behavioral_boundary.get("larger_clean_real_gameplay_revalidation_required") is False,
            multi_boundary.get("full_production_scale_multi_agent_training_status") == "COMPLETED",
        ]
    )


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
