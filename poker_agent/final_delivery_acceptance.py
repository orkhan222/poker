from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_DELIVERY_ACCEPTANCE_VERSION = "2026-06-28"
FINAL_STATUS = "READY_WITH_TRACKED_COMPONENT_RISKS"
SERVICE_DELIVERY_STATUS = "READY"
DEPLOYED_STACK_STATUS = "APPROVED"
RAW_MODEL_BOUNDARY = "LOADABLE_NOT_STANDALONE_APPROVED"
LLM_BOUNDARY = "CONTROLLED_LAYER_NOT_AUTONOMOUS_LLM_AGENT"
QLORA_BOUNDARY = "NEXT_STAGE_IMPROVEMENT_NOT_CURRENT_DELIVERY_BLOCKER"
RUNTIME_MONITORING_BOUNDARY = "REAL_TRAFFIC_REQUIRES_MONITORING_ROLLBACK_AND_DRIFT_TRACKING"
CHALLENGER_STRATEGY_QUALITY_BOUNDARY = "FINAL_STRATEGY_QUALITY_REQUIRES_PASSING_CHALLENGER"


def build_final_delivery_acceptance(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    production_approval = _read_optional_json(reports / "production_approval.json")
    client_handoff = _read_optional_json(reports / "client_handoff.json")
    delivery_verification = _read_optional_json(reports / "delivery_verification.json")
    llm_role = _read_optional_json(reports / "llm_role_boundary.json")
    bet_timing = _read_optional_json(reports / "bet_timing_calibration.json")
    hole_card = _read_optional_json(reports / "hole_card_data_quality.json")
    behavioral = _read_optional_json(reports / "behavioral_revalidation.json")
    strategy_maturity = _read_optional_json(reports / "strategy_stack_maturity.json")
    multi_agent = _read_optional_json(reports / "multi_agent_training_status.json")
    raw_model = _read_optional_json(reports / "raw_model_status.json")
    qlora_next_stage = _read_optional_json(reports / "qlora_next_stage.json")
    production_runtime_monitoring = _read_optional_json(reports / "production_runtime_monitoring.json")
    challenger_strategy_quality = _read_optional_json(reports / "challenger_strategy_quality.json")

    handoff_position = client_handoff.get("technical_position") or {}
    approval_raw = production_approval.get("raw_supervised_model") or {}
    llm_boundary = llm_role.get("autonomous_llm_agent_boundary") or {}
    llm_current = llm_role.get("current_llm_role") or {}
    bet_boundary = bet_timing.get("calibration_boundary") or {}
    hole_boundary = hole_card.get("upstream_data_quality_boundary") or {}
    behavioral_boundary = behavioral.get("revalidation_boundary") or {}
    maturity_current = strategy_maturity.get("current_strategy_stack") or {}
    training_boundary = multi_agent.get("training_boundary") or {}
    raw_contract = raw_model.get("raw_supervised_model") or {}
    raw_release_boundary = raw_model.get("release_boundary") or {}
    qlora_boundary = qlora_next_stage.get("stage_boundary") or {}
    qlora_targets = qlora_next_stage.get("target_use_cases") or {}
    qlora_delivery = qlora_next_stage.get("delivery_classification") or {}
    qlora_plan = qlora_next_stage.get("recommended_training_plan") or {}
    runtime_monitoring = production_runtime_monitoring.get("runtime_observability_boundary") or {}
    challenger_boundary = challenger_strategy_quality.get("strategy_quality_boundary") or {}
    challenger_result = challenger_strategy_quality.get("challenger_result") or {}

    payload: dict[str, Any] = {
        "version": FINAL_DELIVERY_ACCEPTANCE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Final delivery acceptance and non-overclaim boundary",
        "final_status": FINAL_STATUS,
        "client_ready_statement": (
            "The service and deployed strategy stack are ready for delivery with monitoring. "
            "Known limitations are tracked as component risks and are not represented as solved or production-approved beyond their validated scope."
        ),
        "acceptance_summary": {
            "service_delivery": handoff_position.get("service_delivery", "UNKNOWN"),
            "deployed_strategy_stack": handoff_position.get("deployed_strategy_stack", "UNKNOWN"),
            "handoff_status": client_handoff.get("handoff_status", "UNKNOWN"),
            "production_approval_status": production_approval.get("overall_status", "UNKNOWN"),
            "delivery_verification_status": delivery_verification.get("status", "UNKNOWN"),
            "strategy_stack_maturity_status": maturity_current.get("status", "UNKNOWN"),
        },
        "approved_runtime_boundary": {
            "service_delivery_status": SERVICE_DELIVERY_STATUS,
            "deployed_strategy_stack_status": DEPLOYED_STACK_STATUS,
            "monitoring_required": True,
            "rollback_and_component_risk_reporting_required": True,
            "production_blockers": 0,
            "runtime_monitoring_contract": runtime_monitoring.get("status"),
            "real_traffic_blocker_if_observability_disabled": runtime_monitoring.get("real_traffic_blocker_if_disabled"),
        },
        "tracked_component_risks": {
            "raw_supervised_model": {
                "boundary": RAW_MODEL_BOUNDARY,
                "runtime_status": approval_raw.get("runtime_status") or raw_contract.get("runtime_status"),
                "standalone_status": approval_raw.get("standalone_status") or raw_contract.get("standalone_status"),
                "raw_production_gate": approval_raw.get("raw_production_gate"),
                "production_blocker": raw_release_boundary.get("production_blocker", False),
                "component_risk": raw_release_boundary.get("component_risk", True),
            },
            "llm_work": {
                "boundary": LLM_BOUNDARY,
                "role": llm_current.get("status"),
                "event_normalization_implemented": (llm_current.get("event_normalization_layer") or {}).get("implemented"),
                "decision_context_implemented": (llm_current.get("decision_context_layer") or {}).get("implemented"),
                "fully_autonomous_llm_agent_present": llm_boundary.get("fully_autonomous_poker_playing_llm_agent_present"),
                "fully_autonomous_llm_agent_claim_allowed": llm_boundary.get("fully_autonomous_llm_agent_claim_allowed"),
                "production_blocker": llm_boundary.get("production_blocker_for_current_delivery", False),
            },
            "qlora_larger_llm_fine_tuning": {
                "boundary": QLORA_BOUNDARY,
                "stage_status": qlora_boundary.get("stage_status"),
                "milestone_type": qlora_boundary.get("milestone_type"),
                "fine_tuning_completed": qlora_boundary.get("fine_tuning_completed"),
                "production_approved": qlora_boundary.get("production_approved"),
                "current_delivery_blocker": qlora_boundary.get("current_delivery_blocker"),
                "delivery_blocker": qlora_boundary.get("delivery_blocker"),
                "approved_current_delivery_component": qlora_boundary.get("approved_current_delivery_component"),
                "requires_separate_approval_before_promotion": qlora_boundary.get(
                    "requires_separate_approval_before_promotion"
                ),
                "delivery_classification": qlora_delivery,
                "adapter_scope": qlora_plan.get("adapter_scope"),
                "targets": {
                    "noisy_ocr_dealer_log_normalization": (qlora_targets.get("noisy_ocr_dealer_log_normalization") or {}).get("recommended"),
                    "structured_extraction": (qlora_targets.get("structured_extraction") or {}).get("recommended"),
                    "candidate_ranking": (qlora_targets.get("candidate_ranking") or {}).get("recommended"),
                    "json_schema_compliance_improvement": (
                        qlora_targets.get("json_schema_compliance_improvement") or {}
                    ).get("recommended"),
                    "autonomous_poker_policy": (qlora_targets.get("autonomous_poker_policy") or {}).get("recommended"),
                },
                "target": (
                    "noisy OCR/dealer-log normalization, structured extraction, candidate ranking, "
                    "and JSON/schema compliance improvement"
                ),
            },
            "production_runtime_monitoring": {
                "boundary": RUNTIME_MONITORING_BOUNDARY,
                "monitoring_required_for_real_traffic": runtime_monitoring.get("monitoring_required_for_real_traffic"),
                "rollback_rules_required_for_real_traffic": runtime_monitoring.get("rollback_rules_required_for_real_traffic"),
                "live_drift_tracking_required_for_real_traffic": runtime_monitoring.get("live_drift_tracking_required_for_real_traffic"),
                "prediction_distribution_tracking_required_for_real_traffic": runtime_monitoring.get(
                    "prediction_distribution_tracking_required_for_real_traffic"
                ),
                "model_confidence_monitoring_required_for_real_traffic": runtime_monitoring.get(
                    "model_confidence_monitoring_required_for_real_traffic"
                ),
                "real_traffic_claim_allowed_without_observability": runtime_monitoring.get("real_traffic_claim_allowed_without_observability"),
                "real_production_traffic_approved": runtime_monitoring.get("real_production_traffic_approved"),
                "real_production_traffic_approval_status": runtime_monitoring.get("real_production_traffic_approval_status"),
                "real_traffic_blocker_if_disabled": runtime_monitoring.get("real_traffic_blocker_if_disabled"),
                "current_delivery_blocker": runtime_monitoring.get("current_delivery_blocker"),
            },
            "challenger_strategy_quality": {
                "boundary": CHALLENGER_STRATEGY_QUALITY_BOUNDARY,
                "final_production_strategy_quality_claim_allowed": challenger_boundary.get(
                    "final_production_strategy_quality_claim_allowed"
                ),
                "challenger_required_before_final_claim": challenger_boundary.get(
                    "challenger_required_before_final_claim"
                ),
                "challenger_trained": challenger_boundary.get("challenger_trained"),
                "challenger_compared_to_raw_model": challenger_boundary.get("challenger_compared_to_raw_model"),
                "raw_production_gate_status": challenger_boundary.get("raw_production_gate_status"),
                "challenger_gate_status": challenger_boundary.get("challenger_gate_status"),
                "best_candidate": challenger_result.get("best_candidate"),
                "macro_f1": challenger_result.get("macro_f1"),
                "failed_gates": challenger_result.get("failed_gates") or [],
                "current_delivery_blocker": challenger_boundary.get("current_delivery_blocker"),
                "deployed_strategy_stack_affected": challenger_boundary.get("deployed_strategy_stack_affected"),
            },
            "hole_card_data_quality": {
                "limitation_status": hole_boundary.get("limitation_status"),
                "upstream_resolved": hole_boundary.get("upstream_data_quality_issue_resolved"),
                "requires_ocr_or_parser_improvement": hole_boundary.get("requires_ocr_or_parser_improvement"),
                "component_risk": hole_boundary.get("component_risk"),
                "production_blocker": hole_boundary.get("production_blocker_for_current_deployment"),
            },
            "bet_timing_calibration": {
                "implementation_status": (bet_timing.get("current_delivery_scope") or {}).get("implementation_status"),
                "timing_and_bet_size_status": (bet_timing.get("current_delivery_scope") or {}).get("timing_and_bet_size_status"),
                "requires_more_real_player_behavior_labels": bet_boundary.get("requires_more_real_player_behavior_labels"),
                "final_high_realism_claim_allowed": bet_boundary.get("final_high_realism_claim_allowed"),
                "production_blocker": bet_boundary.get("production_blocker_for_current_delivery"),
            },
            "behavioral_revalidation": {
                "current_scope_claim_allowed": behavioral_boundary.get("current_scope_claim_allowed"),
                "larger_clean_real_gameplay_revalidation_required": behavioral_boundary.get("larger_clean_real_gameplay_revalidation_required"),
                "generalized_human_likeness_claim_allowed": behavioral_boundary.get("generalized_human_likeness_claim_allowed"),
                "production_blocker": behavioral_boundary.get("production_blocker"),
            },
            "multi_agent_training": {
                "acceptance_training_status": training_boundary.get("acceptance_training_status"),
                "full_production_scale_multi_agent_training_status": training_boundary.get(
                    "full_production_scale_multi_agent_training_status"
                ),
                "production_blocker": training_boundary.get("production_blocker", training_boundary.get("production_blocker_for_current_delivery", False)),
            },
        },
        "allowed_delivery_claims": [
            "The service delivery package is ready.",
            "The deployed strategy stack is approved for the delivered runtime boundary with monitoring.",
            "The LLM work is a controlled decision/context and event-normalization layer, not a fully autonomous LLM poker player.",
            "QLoRA or larger LLM fine-tuning is tracked as a next-stage improvement, not as completed production-approved work.",
            "Real-traffic rollout requires active monitoring, rollback rules, live drift tracking, prediction-distribution tracking, and model-confidence monitoring.",
            "The raw supervised model is loadable inside the approved stack but not standalone production-approved.",
            "Final production-level strategy quality is blocked until a stronger challenger passes the challenger and raw gates.",
            "Known gaps are tracked as component risks or future calibration milestones, not hidden blockers.",
        ],
        "blocked_claims": [
            "The raw supervised model is standalone production-approved.",
            "Final production-level strategy quality is approved before a stronger challenger passes all required gates.",
            "A failing challenger model is promoted as a production strategy policy.",
            "The project contains a fully autonomous poker-playing LLM agent.",
            "QLoRA or larger LLM fine-tuning has already produced a production-approved model.",
            "The service is approved for real traffic without monitoring, rollback, live drift tracking, prediction-distribution tracking, and model-confidence monitoring.",
            "Real production traffic is approved before observability is enabled.",
            "Hole-card data quality is fully solved upstream.",
            "Bet-sizing and timing are fully calibrated for all production-realism conditions.",
            "Full production-scale multi-agent training has been completed by the current acceptance run.",
            "Current validation replaces larger clean real-gameplay revalidation.",
        ],
        "evidence": {
            "production_approval": "reports/production_approval.json",
            "client_handoff": "reports/client_handoff.json",
            "delivery_verification": "reports/delivery_verification.json",
            "llm_role_boundary": "reports/llm_role_boundary.json",
            "bet_timing_calibration": "reports/bet_timing_calibration.json",
            "hole_card_data_quality": "reports/hole_card_data_quality.json",
            "behavioral_revalidation": "reports/behavioral_revalidation.json",
            "strategy_stack_maturity": "reports/strategy_stack_maturity.json",
            "multi_agent_training_status": "reports/multi_agent_training_status.json",
            "raw_model_status": "reports/raw_model_status.json",
            "qlora_next_stage": "reports/qlora_next_stage.json",
            "production_runtime_monitoring": "reports/production_runtime_monitoring.json",
            "challenger_strategy_quality": "reports/challenger_strategy_quality.json",
        },
        "next_milestones": [
            "Train and promote a stronger standalone raw supervised challenger only after it beats the current raw supervised model and passes every raw gate.",
            "Run QLoRA or larger-LLM fine-tuning as a separate next-stage milestone for noisy OCR/dealer-log normalization, structured extraction, candidate ranking, and JSON/schema compliance improvement.",
            "Enable external production telemetry storage, alerting, rollback procedures, and live drift tracking before real-traffic rollout.",
            "Collect larger reviewed real gameplay labels for timing, bet size, hole-card visibility, and action distribution slices.",
            "Run a separate full production-scale multi-agent training cycle under an approved A100/H100 cluster profile.",
            "Only promote an autonomous LLM policy if stakeholders approve a separate LLM-agent milestone and it passes independent simulation and safety gates.",
        ],
    }
    payload["invariants"] = validate_final_delivery_acceptance(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_final_delivery_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    summary = payload.get("acceptance_summary") or {}
    runtime = payload.get("approved_runtime_boundary") or {}
    risks = payload.get("tracked_component_risks") or {}
    raw = risks.get("raw_supervised_model") or {}
    llm = risks.get("llm_work") or {}
    qlora = risks.get("qlora_larger_llm_fine_tuning") or {}
    runtime_observability = risks.get("production_runtime_monitoring") or {}
    challenger = risks.get("challenger_strategy_quality") or {}
    hole = risks.get("hole_card_data_quality") or {}
    bet = risks.get("bet_timing_calibration") or {}
    behavioral = risks.get("behavioral_revalidation") or {}
    multi = risks.get("multi_agent_training") or {}

    if payload.get("final_status") != FINAL_STATUS:
        violations.append("final_status_must_be_ready_with_tracked_component_risks")
    if payload.get("overall_status") == "PASS":
        violations.append("overall_status_must_be_assigned_after_invariant_validation")
    if summary.get("service_delivery") != SERVICE_DELIVERY_STATUS:
        violations.append("service_delivery_must_be_ready")
    if summary.get("deployed_strategy_stack") != DEPLOYED_STACK_STATUS:
        violations.append("deployed_strategy_stack_must_be_approved")
    if summary.get("delivery_verification_status") != "PASS":
        violations.append("delivery_verification_must_pass")
    if runtime.get("production_blockers") != 0:
        violations.append("approved_runtime_boundary_must_have_zero_production_blockers")
    if raw.get("runtime_status") != "LOADABLE":
        violations.append("raw_model_must_be_loadable")
    if raw.get("standalone_status") != "NOT_STANDALONE_APPROVED":
        violations.append("raw_model_must_not_be_standalone_approved")
    if raw.get("component_risk") is not True or raw.get("production_blocker") is not False:
        violations.append("raw_model_must_remain_component_risk_not_blocker")
    if llm.get("role") != "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER":
        violations.append("llm_role_must_be_controlled_layer")
    if llm.get("fully_autonomous_llm_agent_present") is not False:
        violations.append("fully_autonomous_llm_agent_must_not_be_present")
    if llm.get("fully_autonomous_llm_agent_claim_allowed") is not False:
        violations.append("fully_autonomous_llm_agent_claim_must_be_blocked")
    if llm.get("production_blocker") is not False:
        violations.append("llm_role_boundary_must_not_block_delivery")
    if qlora.get("stage_status") != "NEXT_STAGE_IMPROVEMENT":
        violations.append("qlora_must_remain_next_stage_improvement")
    if qlora.get("milestone_type") != "RESEARCH_QUALITY_IMPROVEMENT_MILESTONE":
        violations.append("qlora_must_remain_research_quality_improvement_milestone")
    if qlora.get("fine_tuning_completed") is not False:
        violations.append("qlora_must_not_be_marked_completed")
    if qlora.get("production_approved") is not False:
        violations.append("qlora_must_not_be_marked_production_approved")
    if qlora.get("current_delivery_blocker") is not False:
        violations.append("qlora_must_not_block_current_delivery")
    if qlora.get("delivery_blocker") is not False:
        violations.append("qlora_delivery_blocker_must_be_false")
    if qlora.get("approved_current_delivery_component") is not False:
        violations.append("qlora_must_not_be_current_delivery_component")
    if qlora.get("requires_separate_approval_before_promotion") is not True:
        violations.append("qlora_must_require_separate_promotion_approval")
    qlora_delivery = qlora.get("delivery_classification") or {}
    if qlora_delivery.get("next_stage_research_milestone") is not True:
        violations.append("qlora_delivery_classification_must_be_next_stage_research_milestone")
    if qlora_delivery.get("current_delivery_blocker") is not False:
        violations.append("qlora_delivery_classification_must_not_block_current_delivery")
    if qlora.get("adapter_scope") != "EVENT_NORMALIZATION_STRUCTURED_EXTRACTION_AND_CANDIDATE_RANKING":
        violations.append("qlora_adapter_scope_must_remain_event_normalization_and_ranking")
    qlora_targets = qlora.get("targets") or {}
    for target in (
        "noisy_ocr_dealer_log_normalization",
        "structured_extraction",
        "candidate_ranking",
        "json_schema_compliance_improvement",
    ):
        if qlora_targets.get(target) is not True:
            violations.append(f"qlora_target_must_remain_enabled:{target}")
    if qlora_targets.get("autonomous_poker_policy") is not False:
        violations.append("qlora_must_not_target_autonomous_poker_policy")
    if runtime_observability.get("monitoring_required_for_real_traffic") is not True:
        violations.append("monitoring_must_be_required_for_real_traffic")
    if runtime_observability.get("rollback_rules_required_for_real_traffic") is not True:
        violations.append("rollback_rules_must_be_required_for_real_traffic")
    if runtime_observability.get("live_drift_tracking_required_for_real_traffic") is not True:
        violations.append("live_drift_tracking_must_be_required_for_real_traffic")
    if runtime_observability.get("prediction_distribution_tracking_required_for_real_traffic") is not True:
        violations.append("prediction_distribution_tracking_must_be_required_for_real_traffic")
    if runtime_observability.get("model_confidence_monitoring_required_for_real_traffic") is not True:
        violations.append("model_confidence_monitoring_must_be_required_for_real_traffic")
    if runtime_observability.get("real_traffic_claim_allowed_without_observability") is not False:
        violations.append("unmonitored_real_traffic_claim_must_be_blocked")
    if runtime_observability.get("real_production_traffic_approved") is not False:
        violations.append("real_production_traffic_must_not_be_approved_before_observability")
    if runtime_observability.get("real_production_traffic_approval_status") != "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED":
        violations.append("real_production_traffic_status_must_require_observability")
    if runtime_observability.get("real_traffic_blocker_if_disabled") is not True:
        violations.append("disabled_observability_must_block_real_traffic_rollout")
    if runtime_observability.get("current_delivery_blocker") is not False:
        violations.append("runtime_monitoring_contract_must_not_block_current_delivery")
    if challenger.get("challenger_required_before_final_claim") is not True:
        violations.append("challenger_must_be_required_before_final_strategy_quality_claim")
    if challenger.get("challenger_compared_to_raw_model") is not True:
        violations.append("challenger_must_be_compared_before_final_strategy_quality_claim")
    if challenger.get("final_production_strategy_quality_claim_allowed") is not False:
        violations.append("final_strategy_quality_claim_must_remain_blocked_until_challenger_passes")
    if challenger.get("current_delivery_blocker") is not False:
        violations.append("challenger_gap_must_not_block_current_delivery")
    if challenger.get("deployed_strategy_stack_affected") is not False:
        violations.append("challenger_gap_must_not_affect_deployed_strategy_stack")
    if hole.get("upstream_resolved") is not False:
        violations.append("hole_card_upstream_issue_must_not_be_marked_resolved")
    if hole.get("component_risk") is not True or hole.get("production_blocker") is not False:
        violations.append("hole_card_issue_must_remain_component_risk_not_blocker")
    if bet.get("implementation_status") != "IMPLEMENTED_AND_MEASURED":
        violations.append("bet_timing_must_be_implemented_and_measured")
    if bet.get("requires_more_real_player_behavior_labels") is not True:
        violations.append("bet_timing_must_require_more_real_player_labels_for_higher_realism")
    if bet.get("final_high_realism_claim_allowed") is not False:
        violations.append("final_high_realism_bet_timing_claim_must_be_blocked")
    if behavioral.get("larger_clean_real_gameplay_revalidation_required") is not True:
        violations.append("larger_clean_real_gameplay_revalidation_must_remain_required")
    if behavioral.get("generalized_human_likeness_claim_allowed") is not False:
        violations.append("generalized_human_likeness_claim_must_be_blocked")
    if multi.get("full_production_scale_multi_agent_training_status") != "NOT_COMPLETED":
        violations.append("full_production_scale_multi_agent_training_must_not_be_marked_completed")
    if multi.get("production_blocker") is not False:
        violations.append("multi_agent_hardening_gap_must_not_block_current_delivery")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_final_delivery_acceptance(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_final_delivery_acceptance(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_final_delivery_acceptance_markdown(payload), encoding="utf-8")
    return payload


def render_final_delivery_acceptance_markdown(payload: dict[str, Any]) -> str:
    summary = payload["acceptance_summary"]
    risks = payload["tracked_component_risks"]
    lines = [
        "# Final Delivery Acceptance Contract",
        "",
        payload["client_ready_statement"],
        "",
        "## Acceptance Summary",
        "",
        f"- Final status: `{payload['final_status']}`",
        f"- Service delivery: `{summary['service_delivery']}`",
        f"- Deployed strategy stack: `{summary['deployed_strategy_stack']}`",
        f"- Production approval: `{summary['production_approval_status']}`",
        f"- Delivery verification: `{summary['delivery_verification_status']}`",
        "",
        "## Tracked Boundaries",
        "",
        f"- Raw supervised model: `{risks['raw_supervised_model']['boundary']}`",
        f"- LLM work: `{risks['llm_work']['boundary']}`",
        f"- QLoRA/larger LLM fine-tuning: `{risks['qlora_larger_llm_fine_tuning']['stage_status']}`",
        f"- Production runtime monitoring: `{risks['production_runtime_monitoring']['boundary']}`",
        f"- Real production traffic approved: `{risks['production_runtime_monitoring']['real_production_traffic_approved']}`",
        f"- Real production traffic approval status: `{risks['production_runtime_monitoring']['real_production_traffic_approval_status']}`",
        f"- Challenger strategy quality: `{risks['challenger_strategy_quality']['boundary']}`",
        f"- Final strategy-quality claim allowed: `{risks['challenger_strategy_quality']['final_production_strategy_quality_claim_allowed']}`",
        f"- Hole-card upstream resolved: `{risks['hole_card_data_quality']['upstream_resolved']}`",
        f"- Bet/timing implementation: `{risks['bet_timing_calibration']['implementation_status']}`",
        f"- Larger gameplay revalidation required: `{risks['behavioral_revalidation']['larger_clean_real_gameplay_revalidation_required']}`",
        f"- Full production-scale multi-agent training: `{risks['multi_agent_training']['full_production_scale_multi_agent_training_status']}`",
        "",
        "## Blocked Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.extend(["", "## Next Milestones", ""])
    lines.extend(f"- {item}" for item in payload["next_milestones"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
