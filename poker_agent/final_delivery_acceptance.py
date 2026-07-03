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
ACTIONS_CONTEXT_BOUNDARY = "DERIVED_BETTING_CONTEXT_NOT_FULL_EXPLICIT_ACTION_CONTEXT"
STACK_EVENT_CONTEXT_BOUNDARY = "RAW_STACK_EVENTS_REQUIRE_DERIVED_DECISION_CONTEXT"
PHASE3_OPEN_SPIEL_RL_BOUNDARY = "OPEN_SPIEL_RL_TRAINING_PROOF_REQUIRED"
EVALUATION_METRIC_BOUNDARY = "ACCURACY_ALONE_NOT_SUFFICIENT"
TEST_EXECUTION_BOUNDARY = "FULL_PYTEST_TIMEOUT_IS_NOT_DELIVERY_APPROVAL"
HUMAN_LIKENESS_EVIDENCE_BOUNDARY = "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF"


def build_final_delivery_acceptance(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    production_approval = _read_optional_json(reports / "production_approval.json")
    client_handoff = _read_optional_json(reports / "client_handoff.json")
    delivery_verification = _read_optional_json(reports / "delivery_verification.json")
    llm_role = _read_optional_json(reports / "llm_role_boundary.json")
    bet_timing = _read_optional_json(reports / "bet_timing_calibration.json")
    hole_card = _read_optional_json(reports / "hole_card_data_quality.json")
    actions_context = _read_optional_json(reports / "actions_context_quality.json")
    stack_event_context = _read_optional_json(reports / "stack_event_context_quality.json")
    behavioral = _read_optional_json(reports / "behavioral_revalidation.json")
    strategy_maturity = _read_optional_json(reports / "strategy_stack_maturity.json")
    multi_agent = _read_optional_json(reports / "multi_agent_training_status.json")
    raw_model = _read_optional_json(reports / "raw_model_status.json")
    qlora_next_stage = _read_optional_json(reports / "qlora_next_stage.json")
    production_runtime_monitoring = _read_optional_json(reports / "production_runtime_monitoring.json")
    challenger_strategy_quality = _read_optional_json(reports / "challenger_strategy_quality.json")
    phase3_open_spiel_arena = _read_optional_json(reports / "phase3_open_spiel_arena.json")
    evaluation_metric_contract = _read_optional_json(reports / "evaluation_metric_contract.json")
    test_execution_contract = _read_optional_json(reports / "test_execution_contract.json")
    human_likeness_evidence = _read_optional_json(reports / "human_likeness_evidence.json")
    human_likeness_claim_gate = _read_optional_json(reports / "human_likeness_claim_gate.json")

    handoff_position = client_handoff.get("technical_position") or {}
    approval_raw = production_approval.get("raw_supervised_model") or {}
    llm_boundary = llm_role.get("autonomous_llm_agent_boundary") or {}
    llm_current = llm_role.get("current_llm_role") or {}
    bet_boundary = bet_timing.get("calibration_boundary") or {}
    hole_boundary = hole_card.get("upstream_data_quality_boundary") or {}
    actions_schema = actions_context.get("actions_csv_schema_audit") or {}
    actions_mitigation = actions_context.get("derived_context_mitigation") or {}
    stack_raw_boundary = stack_event_context.get("raw_stack_event_boundary") or {}
    stack_mitigation = stack_event_context.get("derived_context_mitigation") or {}
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
    phase3_rl_boundary = phase3_open_spiel_arena.get("rl_training_proof_boundary") or {}
    evaluation_metric_families = evaluation_metric_contract.get("metric_families") or {}

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
                "term_status": (llm_role.get("term_boundary") or {}).get("status"),
                "term_requires_role_specific_qualification": (llm_role.get("term_boundary") or {}).get(
                    "requires_role_specific_qualification"
                ),
                "term_must_not_imply_fully_autonomous_policy": (llm_role.get("term_boundary") or {}).get(
                    "must_not_imply_fully_autonomous_policy"
                ),
                "role_taxonomy": {
                    name: {
                        "status": role.get("status"),
                        "implemented": role.get("implemented"),
                        "production_policy_approved": role.get("production_policy_approved"),
                    }
                    for name, role in (llm_role.get("role_taxonomy") or {}).items()
                },
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
            "actions_context_quality": {
                "boundary": ACTIONS_CONTEXT_BOUNDARY,
                "explicit_context_status": actions_schema.get("explicit_context_status"),
                "missing_explicit_context_fields": actions_schema.get("missing_explicit_context_fields") or [],
                "limitation_status": actions_schema.get("limitation_status"),
                "derived_context_status": actions_mitigation.get("status"),
                "uses_target_action_amount_as_feature": actions_mitigation.get("uses_target_action_amount_as_feature"),
                "uses_future_outcome_fields": actions_mitigation.get("uses_future_outcome_fields"),
                "does_not_fully_replace_explicit_context": actions_mitigation.get(
                    "does_not_fully_replace_explicit_context"
                ),
                "current_delivery_blocker": actions_mitigation.get("current_delivery_blocker"),
                "model_quality_risk": actions_mitigation.get("model_quality_risk"),
            },
            "stack_event_context_quality": {
                "boundary": STACK_EVENT_CONTEXT_BOUNDARY,
                "raw_stack_event_status": stack_raw_boundary.get("status"),
                "raw_stack_events_are_direct_policy_features": stack_raw_boundary.get(
                    "raw_stack_events_are_direct_policy_features"
                ),
                "decision_time_derivation_required": stack_raw_boundary.get(
                    "decision_time_derivation_required"
                ),
                "target_action_stack_delta_allowed_as_feature": stack_raw_boundary.get(
                    "target_action_stack_delta_allowed_as_feature"
                ),
                "post_hand_stack_outcome_allowed_as_feature": stack_raw_boundary.get(
                    "post_hand_stack_outcome_allowed_as_feature"
                ),
                "derived_context_status": stack_mitigation.get("status"),
                "uses_target_action_stack_delta_as_feature": stack_mitigation.get(
                    "uses_target_action_stack_delta_as_feature"
                ),
                "uses_post_hand_outcome_fields": stack_mitigation.get("uses_post_hand_outcome_fields"),
                "current_delivery_blocker": stack_mitigation.get("current_delivery_blocker"),
                "model_quality_risk": stack_mitigation.get("model_quality_risk"),
            },
            "bet_timing_calibration": {
                "implementation_status": (bet_timing.get("current_delivery_scope") or {}).get("implementation_status"),
                "timing_and_bet_size_status": (bet_timing.get("current_delivery_scope") or {}).get("timing_and_bet_size_status"),
                "timing_policy_type": (bet_timing.get("current_delivery_scope") or {}).get("timing_policy_type"),
                "real_human_timing_label_quality": (bet_timing.get("current_delivery_scope") or {}).get(
                    "real_human_timing_label_quality"
                ),
                "real_human_timing_labels_available": (bet_timing.get("current_delivery_scope") or {}).get(
                    "real_human_timing_labels_available"
                ),
                "timing_human_likeness_final_proof_allowed": (bet_timing.get("current_delivery_scope") or {}).get(
                    "timing_human_likeness_final_proof_allowed"
                ),
                "requires_more_real_player_behavior_labels": bet_boundary.get("requires_more_real_player_behavior_labels"),
                "final_high_realism_claim_allowed": bet_boundary.get("final_high_realism_claim_allowed"),
                "timing_label_quality_status": (bet_timing.get("timing_label_quality_boundary") or {}).get("status"),
                "timing_label_current_delivery_blocker": (bet_timing.get("timing_label_quality_boundary") or {}).get(
                    "current_delivery_blocker"
                ),
                "timing_label_model_quality_risk": (bet_timing.get("timing_label_quality_boundary") or {}).get(
                    "model_quality_risk"
                ),
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
            "phase3_open_spiel_rl_training": {
                "boundary": PHASE3_OPEN_SPIEL_RL_BOUNDARY,
                "status": phase3_rl_boundary.get("status"),
                "real_open_spiel_runtime_required": phase3_rl_boundary.get("real_open_spiel_runtime_required"),
                "real_open_spiel_runtime_available": phase3_rl_boundary.get("real_open_spiel_runtime_available"),
                "phase1_trained_policy_artifacts_required": phase3_rl_boundary.get("phase1_trained_policy_artifacts_required"),
                "phase1_trained_policy_artifacts_attached": phase3_rl_boundary.get("phase1_trained_policy_artifacts_attached"),
                "seed_stability_required": phase3_rl_boundary.get("seed_stability_required"),
                "seed_stability_evaluated": phase3_rl_boundary.get("seed_stability_evaluated"),
                "long_run_required": phase3_rl_boundary.get("long_run_required"),
                "long_run_completed": phase3_rl_boundary.get("long_run_completed"),
                "policy_update_training_required": phase3_rl_boundary.get("policy_update_training_required"),
                "policy_update_training_completed": phase3_rl_boundary.get("policy_update_training_completed"),
                "measured_win_rate_claim_allowed": phase3_rl_boundary.get("measured_win_rate_claim_allowed"),
                "current_delivery_blocker": phase3_rl_boundary.get("current_delivery_blocker"),
                "model_quality_risk": phase3_rl_boundary.get("model_quality_risk"),
                "missing_requirements": phase3_rl_boundary.get("missing_requirements") or [],
            },
            "evaluation_metric_coverage": {
                "boundary": EVALUATION_METRIC_BOUNDARY,
                "accuracy_alone_sufficient": evaluation_metric_contract.get("accuracy_alone_sufficient"),
                "required_metric_families": evaluation_metric_contract.get("required_metric_families") or [],
                "final_metric_bundle_passed": evaluation_metric_contract.get("final_metric_bundle_passed"),
                "final_strategy_quality_claim_allowed": evaluation_metric_contract.get(
                    "final_strategy_quality_claim_allowed"
                ),
                "current_delivery_blocker": evaluation_metric_contract.get("current_delivery_blocker"),
                "model_quality_risk": evaluation_metric_contract.get("model_quality_risk"),
                "metric_families": {
                    name: {
                        "required": family.get("required"),
                        "metrics": family.get("metrics") or {},
                    }
                    for name, family in evaluation_metric_families.items()
                },
            },
            "test_execution_coverage": {
                "boundary": TEST_EXECUTION_BOUNDARY,
                "full_pytest_status": (test_execution_contract.get("full_pytest") or {}).get("status"),
                "full_pytest_used_as_delivery_approval": (test_execution_contract.get("full_pytest") or {}).get(
                    "used_as_delivery_approval"
                ),
                "critical_validation_status": (test_execution_contract.get("critical_validation") or {}).get("status"),
                "critical_tests_passed": (test_execution_contract.get("critical_validation") or {}).get(
                    "passed_tests"
                ),
                "delivery_verifier_status": (test_execution_contract.get("delivery_verifier") or {}).get("status"),
                "current_delivery_blocker": test_execution_contract.get("current_delivery_blocker"),
            },
            "human_likeness_evidence": {
                "boundary": HUMAN_LIKENESS_EVIDENCE_BOUNDARY,
                "status": human_likeness_evidence.get("status"),
                "human_likeness_fully_proven": human_likeness_evidence.get("human_likeness_fully_proven"),
                "final_human_likeness_claim_allowed": human_likeness_evidence.get(
                    "final_human_likeness_claim_allowed"
                ),
                "current_scope_action_distribution_passed": human_likeness_evidence.get(
                    "current_scope_action_distribution_passed"
                ),
                "current_delivery_blocker": human_likeness_evidence.get("current_delivery_blocker"),
                "model_quality_risk": human_likeness_evidence.get("model_quality_risk"),
                "required_behavior_dimensions": human_likeness_evidence.get("required_behavior_dimensions") or [],
                "behavior_dimensions": {
                    name: {
                        "required": dimension.get("required"),
                        "current_status": dimension.get("current_status"),
                        "final_proof_allowed": dimension.get("final_proof_allowed"),
                    }
                    for name, dimension in (human_likeness_evidence.get("behavior_dimensions") or {}).items()
                },
            },
            "human_likeness_claim_gate": {
                "boundary": HUMAN_LIKENESS_EVIDENCE_BOUNDARY,
                "claim": human_likeness_claim_gate.get("claim"),
                "decision": human_likeness_claim_gate.get("decision"),
                "claim_allowed": human_likeness_claim_gate.get("claim_allowed"),
                "human_likeness_fully_proven": human_likeness_claim_gate.get("human_likeness_fully_proven"),
                "action_distribution_only_proof_rejected": human_likeness_claim_gate.get(
                    "action_distribution_only_proof_rejected"
                ),
                "current_scope_action_distribution_passed": human_likeness_claim_gate.get(
                    "current_scope_action_distribution_passed"
                ),
                "current_delivery_blocker": human_likeness_claim_gate.get("current_delivery_blocker"),
                "model_quality_risk": human_likeness_claim_gate.get("model_quality_risk"),
                "required_evidence_dimensions": human_likeness_claim_gate.get("required_evidence_dimensions") or [],
                "evidence_requirements": {
                    name: {
                        "required_for_final_claim": requirement.get("required_for_final_claim"),
                        "current_status": requirement.get("current_status"),
                        "currently_sufficient_for_final_claim": requirement.get(
                            "currently_sufficient_for_final_claim"
                        ),
                    }
                    for name, requirement in (human_likeness_claim_gate.get("evidence_requirements") or {}).items()
                },
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
            "actions.csv is a complete explicit betting-context dataset.",
            "Derived betting context fully replaces amount, to_call, pot_before_action, min_raise, legal_actions, and action_order labels.",
            "Raw stack events are sufficient policy features without decision-time pot/effective-stack/SPR derivation.",
            "Target action stack deltas can be used as features for predicting that same action.",
            "Bet-sizing and timing are fully calibrated for all production-realism conditions.",
            "Full production-scale multi-agent training has been completed by the current acceptance run.",
            "Phase 3 OpenSpiel/RL win-rate proof is complete without real pyspiel runtime, two trained Phase 1 policy artifacts, seed stability, long-run volume, and policy-update training.",
            "Accuracy alone is sufficient for final production strategy-quality approval.",
            "The full pytest suite completed successfully when the recorded run timed out.",
            "A timed-out full pytest run is used as delivery approval evidence.",
            "Human-likeness is fully proven by action distribution alone.",
            "Bet sizing, timing, position behavior, and street-level strategy do not require separate validation.",
            "Current validation replaces larger clean real-gameplay revalidation.",
        ],
        "evidence": {
            "production_approval": "reports/production_approval.json",
            "client_handoff": "reports/client_handoff.json",
            "delivery_verification": "reports/delivery_verification.json",
            "llm_role_boundary": "reports/llm_role_boundary.json",
            "bet_timing_calibration": "reports/bet_timing_calibration.json",
            "hole_card_data_quality": "reports/hole_card_data_quality.json",
            "actions_context_quality": "reports/actions_context_quality.json",
            "stack_event_context_quality": "reports/stack_event_context_quality.json",
            "behavioral_revalidation": "reports/behavioral_revalidation.json",
            "strategy_stack_maturity": "reports/strategy_stack_maturity.json",
            "multi_agent_training_status": "reports/multi_agent_training_status.json",
            "raw_model_status": "reports/raw_model_status.json",
            "qlora_next_stage": "reports/qlora_next_stage.json",
            "production_runtime_monitoring": "reports/production_runtime_monitoring.json",
            "challenger_strategy_quality": "reports/challenger_strategy_quality.json",
            "phase3_open_spiel_arena": "reports/phase3_open_spiel_arena.json",
            "evaluation_metric_contract": "reports/evaluation_metric_contract.json",
            "test_execution_contract": "reports/test_execution_contract.json",
            "human_likeness_evidence": "reports/human_likeness_evidence.json",
            "human_likeness_claim_gate": "reports/human_likeness_claim_gate.json",
        },
        "next_milestones": [
            "Train and promote a stronger standalone raw supervised challenger only after it beats the current raw supervised model and passes every raw gate.",
            "Run QLoRA or larger-LLM fine-tuning as a separate next-stage milestone for noisy OCR/dealer-log normalization, structured extraction, candidate ranking, and JSON/schema compliance improvement.",
            "Enable external production telemetry storage, alerting, rollback procedures, and live drift tracking before real-traffic rollout.",
            "Collect larger reviewed real gameplay labels for timing, bet size, hole-card visibility, and action distribution slices.",
            "Instrument actions.csv with explicit amount, to_call, pot_before_action, min_raise, legal_actions, and action_order fields.",
            "Persist explicit pre-action pot, effective_stack, SPR, current_bet_size, and min_raise labels alongside stack event logs.",
            "Run a separate full production-scale multi-agent training cycle under an approved A100/H100 cluster profile.",
            "Execute the Phase 3 OpenSpiel RL training proof with real pyspiel runtime, two trained Phase 1 policy artifacts, at least five independent seeds, long-run volume, and policy-update training.",
            "Close the evaluation metric bundle: macro F1, balanced accuracy, calibration/ECE, action distribution, bet-size MAE, win-rate, expected value, and seed stability.",
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
    actions_context = risks.get("actions_context_quality") or {}
    stack_event_context = risks.get("stack_event_context_quality") or {}
    bet = risks.get("bet_timing_calibration") or {}
    behavioral = risks.get("behavioral_revalidation") or {}
    multi = risks.get("multi_agent_training") or {}
    phase3_rl = risks.get("phase3_open_spiel_rl_training") or {}
    evaluation_metrics = risks.get("evaluation_metric_coverage") or {}
    test_execution = risks.get("test_execution_coverage") or {}
    human_likeness = risks.get("human_likeness_evidence") or {}
    human_likeness_claim_gate = risks.get("human_likeness_claim_gate") or {}

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
    if llm.get("term_status") != "LLM_BASED_AGENT_IS_UMBRELLA_TERM":
        violations.append("llm_based_agent_term_must_remain_umbrella")
    if llm.get("term_requires_role_specific_qualification") is not True:
        violations.append("llm_based_agent_term_must_require_role_specific_qualification")
    if llm.get("term_must_not_imply_fully_autonomous_policy") is not True:
        violations.append("llm_based_agent_term_must_not_imply_autonomous_policy")
    llm_taxonomy = llm.get("role_taxonomy") or {}
    required_llm_roles = {
        "event_normalization",
        "decision_context",
        "candidate_ranking",
        "real_policy_agent",
    }
    if set(llm_taxonomy) != required_llm_roles:
        violations.append("llm_role_taxonomy_must_list_all_supported_roles")
    if (llm_taxonomy.get("event_normalization") or {}).get("status") != "CONTROLLED_COMPONENT":
        violations.append("llm_event_normalization_role_must_be_controlled_component")
    if (llm_taxonomy.get("decision_context") or {}).get("status") != "CONTROLLED_COMPONENT":
        violations.append("llm_decision_context_role_must_be_controlled_component")
    if (llm_taxonomy.get("candidate_ranking") or {}).get("status") != "RESEARCH_BASELINE_COMPONENT":
        violations.append("llm_candidate_ranking_role_must_be_research_component")
    if (llm_taxonomy.get("real_policy_agent") or {}).get("status") != "NOT_CURRENT_DELIVERY_SCOPE":
        violations.append("llm_real_policy_agent_must_remain_out_of_current_scope")
    for role_name, role_payload in llm_taxonomy.items():
        if role_payload.get("production_policy_approved") is not False:
            violations.append(f"llm_role_must_not_be_production_policy:{role_name}")
    if (llm_taxonomy.get("real_policy_agent") or {}).get("implemented") is not False:
        violations.append("llm_real_policy_agent_must_not_be_implemented")
    if llm.get("fully_autonomous_llm_agent_present") is not False:
        violations.append("fully_autonomous_llm_agent_must_not_be_present")
    if llm.get("fully_autonomous_llm_agent_claim_allowed") is not False:
        violations.append("fully_autonomous_llm_agent_claim_must_be_blocked")
    if llm.get("production_blocker") is not False:
        violations.append("llm_role_boundary_must_not_block_delivery")
    if phase3_rl.get("boundary") != PHASE3_OPEN_SPIEL_RL_BOUNDARY:
        violations.append("phase3_open_spiel_rl_boundary_must_be_present")
    if phase3_rl.get("status") != "TRAINING_PROOF_NOT_COMPLETED":
        violations.append("phase3_open_spiel_rl_training_proof_must_remain_not_completed")
    if phase3_rl.get("real_open_spiel_runtime_required") is not True:
        violations.append("phase3_rl_must_require_real_open_spiel_runtime")
    if phase3_rl.get("phase1_trained_policy_artifacts_required") is not True:
        violations.append("phase3_rl_must_require_two_phase1_trained_adapters")
    if phase3_rl.get("seed_stability_required") is not True:
        violations.append("phase3_rl_must_require_seed_stability")
    if phase3_rl.get("long_run_required") is not True:
        violations.append("phase3_rl_must_require_long_run")
    if phase3_rl.get("policy_update_training_required") is not True:
        violations.append("phase3_rl_must_require_policy_update_training")
    if phase3_rl.get("measured_win_rate_claim_allowed") is not False:
        violations.append("phase3_rl_win_rate_claim_must_remain_blocked_until_training_proof")
    if phase3_rl.get("current_delivery_blocker") is not False:
        violations.append("phase3_rl_training_gap_must_not_block_current_delivery")
    if phase3_rl.get("model_quality_risk") is not True:
        violations.append("phase3_rl_training_gap_must_remain_model_quality_risk")
    if evaluation_metrics.get("boundary") != EVALUATION_METRIC_BOUNDARY:
        violations.append("evaluation_metric_boundary_must_block_accuracy_only_approval")
    if evaluation_metrics.get("accuracy_alone_sufficient") is not False:
        violations.append("accuracy_alone_must_not_be_sufficient_for_final_acceptance")
    required_metric_families = {
        "action_classification",
        "calibration",
        "action_distribution",
        "bet_sizing",
        "simulation_return",
        "seed_stability",
    }
    if set(evaluation_metrics.get("required_metric_families") or []) != required_metric_families:
        violations.append("evaluation_metric_families_must_be_complete")
    metric_families = evaluation_metrics.get("metric_families") or {}
    for family_name in required_metric_families:
        if (metric_families.get(family_name) or {}).get("required") is not True:
            violations.append(f"evaluation_metric_family_must_be_required:{family_name}")
    if evaluation_metrics.get("final_metric_bundle_passed") is not False:
        violations.append("final_metric_bundle_must_not_be_marked_passed")
    if evaluation_metrics.get("final_strategy_quality_claim_allowed") is not False:
        violations.append("final_strategy_quality_claim_must_remain_blocked_until_metric_bundle_passes")
    if evaluation_metrics.get("current_delivery_blocker") is not False:
        violations.append("evaluation_metric_gap_must_not_block_current_delivery")
    if evaluation_metrics.get("model_quality_risk") is not True:
        violations.append("evaluation_metric_gap_must_remain_model_quality_risk")
    if test_execution.get("boundary") != TEST_EXECUTION_BOUNDARY:
        violations.append("test_execution_boundary_must_be_present")
    if test_execution.get("full_pytest_status") == "TIMEOUT" and (
        test_execution.get("full_pytest_used_as_delivery_approval") is not False
    ):
        violations.append("timed_out_full_pytest_must_not_be_delivery_approval")
    if test_execution.get("critical_validation_status") != "PASS":
        violations.append("critical_validation_must_pass_for_delivery")
    if (test_execution.get("critical_tests_passed") or 0) < 1:
        violations.append("critical_validation_must_report_passed_tests")
    if test_execution.get("delivery_verifier_status") != "PASS":
        violations.append("delivery_verifier_must_pass_for_delivery")
    if test_execution.get("current_delivery_blocker") is not False:
        violations.append("test_execution_gap_must_not_block_current_delivery")
    if human_likeness.get("boundary") != HUMAN_LIKENESS_EVIDENCE_BOUNDARY:
        violations.append("human_likeness_evidence_boundary_must_be_present")
    if human_likeness.get("status") != "NOT_FULLY_PROVEN":
        violations.append("human_likeness_status_must_remain_not_fully_proven")
    if human_likeness.get("human_likeness_fully_proven") is not False:
        violations.append("human_likeness_must_not_be_marked_fully_proven")
    if human_likeness.get("final_human_likeness_claim_allowed") is not False:
        violations.append("final_human_likeness_claim_must_remain_blocked")
    if human_likeness.get("current_scope_action_distribution_passed") is not True:
        violations.append("current_scope_action_distribution_must_pass")
    if human_likeness.get("current_delivery_blocker") is not False:
        violations.append("human_likeness_gap_must_not_block_current_delivery")
    if human_likeness.get("model_quality_risk") is not True:
        violations.append("human_likeness_gap_must_remain_model_quality_risk")
    required_behavior_dimensions = {
        "action_distribution",
        "bet_sizing",
        "timing",
        "position_based_behavior",
        "street_level_strategy",
    }
    if set(human_likeness.get("required_behavior_dimensions") or []) != required_behavior_dimensions:
        violations.append("human_likeness_required_dimensions_must_be_complete")
    behavior_dimensions = human_likeness.get("behavior_dimensions") or {}
    for dimension_name in required_behavior_dimensions:
        dimension = behavior_dimensions.get(dimension_name) or {}
        if dimension.get("required") is not True:
            violations.append(f"human_likeness_dimension_must_be_required:{dimension_name}")
        if dimension.get("final_proof_allowed") is not False:
            violations.append(f"human_likeness_dimension_final_proof_must_be_blocked:{dimension_name}")
    if human_likeness_claim_gate.get("boundary") != HUMAN_LIKENESS_EVIDENCE_BOUNDARY:
        violations.append("human_likeness_claim_gate_boundary_must_be_present")
    if human_likeness_claim_gate.get("claim") != "FULL_HUMAN_LIKENESS":
        violations.append("human_likeness_claim_gate_must_target_full_human_likeness")
    if human_likeness_claim_gate.get("decision") != "BLOCKED":
        violations.append("human_likeness_claim_gate_decision_must_remain_blocked")
    if human_likeness_claim_gate.get("claim_allowed") is not False:
        violations.append("human_likeness_claim_gate_must_not_allow_final_claim")
    if human_likeness_claim_gate.get("human_likeness_fully_proven") is not False:
        violations.append("human_likeness_claim_gate_must_not_mark_full_proof")
    if human_likeness_claim_gate.get("action_distribution_only_proof_rejected") is not True:
        violations.append("human_likeness_claim_gate_must_reject_action_distribution_only_proof")
    if human_likeness_claim_gate.get("current_scope_action_distribution_passed") is not True:
        violations.append("human_likeness_claim_gate_requires_current_scope_action_distribution_pass")
    if human_likeness_claim_gate.get("current_delivery_blocker") is not False:
        violations.append("human_likeness_claim_gate_gap_must_not_block_current_delivery")
    if human_likeness_claim_gate.get("model_quality_risk") is not True:
        violations.append("human_likeness_claim_gate_gap_must_remain_model_quality_risk")
    if set(human_likeness_claim_gate.get("required_evidence_dimensions") or []) != required_behavior_dimensions:
        violations.append("human_likeness_claim_gate_dimensions_must_be_complete")
    claim_requirements = human_likeness_claim_gate.get("evidence_requirements") or {}
    for dimension_name in required_behavior_dimensions:
        requirement = claim_requirements.get(dimension_name) or {}
        if requirement.get("required_for_final_claim") is not True:
            violations.append(f"human_likeness_claim_gate_dimension_must_be_required:{dimension_name}")
        if requirement.get("currently_sufficient_for_final_claim") is not False:
            violations.append(
                f"human_likeness_claim_gate_dimension_must_not_be_currently_sufficient:{dimension_name}"
            )
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
    if actions_context.get("explicit_context_status") != "INCOMPLETE_EXPLICIT_BETTING_CONTEXT":
        violations.append("actions_context_must_remain_marked_incomplete")
    required_action_fields = {
        "amount",
        "to_call",
        "pot_before_action",
        "min_raise",
        "legal_actions",
        "action_order",
    }
    missing_action_fields = set(actions_context.get("missing_explicit_context_fields") or [])
    if not required_action_fields.issubset(missing_action_fields):
        violations.append("actions_context_must_list_missing_explicit_fields")
    if actions_context.get("limitation_status") != "OPEN_DATASET_LIMITATION":
        violations.append("actions_context_limitation_must_remain_open")
    if actions_context.get("derived_context_status") != "IMPLEMENTED_FROM_PRE_ACTION_EVENT_STREAM":
        violations.append("actions_context_derived_mitigation_must_be_implemented")
    if actions_context.get("uses_target_action_amount_as_feature") is not False:
        violations.append("actions_context_must_not_use_target_action_amount_as_feature")
    if actions_context.get("uses_future_outcome_fields") is not False:
        violations.append("actions_context_must_not_use_future_outcome_fields")
    if actions_context.get("does_not_fully_replace_explicit_context") is not True:
        violations.append("actions_context_must_not_claim_full_replacement")
    if actions_context.get("current_delivery_blocker") is not False:
        violations.append("actions_context_gap_must_not_block_current_delivery")
    if actions_context.get("model_quality_risk") is not True:
        violations.append("actions_context_gap_must_remain_model_quality_risk")
    if stack_event_context.get("raw_stack_event_status") != "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION":
        violations.append("stack_events_must_remain_marked_as_raw_context_requiring_derivation")
    if stack_event_context.get("raw_stack_events_are_direct_policy_features") is not False:
        violations.append("raw_stack_events_must_not_be_marked_direct_policy_features")
    if stack_event_context.get("decision_time_derivation_required") is not True:
        violations.append("stack_events_must_require_decision_time_derivation")
    if stack_event_context.get("target_action_stack_delta_allowed_as_feature") is not False:
        violations.append("target_action_stack_delta_must_not_be_allowed_as_feature")
    if stack_event_context.get("post_hand_stack_outcome_allowed_as_feature") is not False:
        violations.append("post_hand_stack_outcome_must_not_be_allowed_as_feature")
    if stack_event_context.get("derived_context_status") != "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS":
        violations.append("stack_event_derived_context_must_be_implemented")
    if stack_event_context.get("uses_target_action_stack_delta_as_feature") is not False:
        violations.append("stack_event_context_must_not_use_target_action_delta")
    if stack_event_context.get("uses_post_hand_outcome_fields") is not False:
        violations.append("stack_event_context_must_not_use_post_hand_outcomes")
    if stack_event_context.get("current_delivery_blocker") is not False:
        violations.append("stack_event_context_gap_must_not_block_current_delivery")
    if stack_event_context.get("model_quality_risk") is not True:
        violations.append("stack_event_context_gap_must_remain_model_quality_risk")
    if bet.get("implementation_status") != "IMPLEMENTED_AND_MEASURED":
        violations.append("bet_timing_must_be_implemented_and_measured")
    if bet.get("timing_policy_type") != "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED":
        violations.append("timing_policy_must_remain_heuristic_or_table_tempo_calibrated")
    if bet.get("real_human_timing_label_quality") != "TIMING_LABEL_QUALITY_UNCERTAIN":
        violations.append("timing_label_quality_must_remain_uncertain")
    if bet.get("real_human_timing_labels_available") is not False:
        violations.append("real_human_timing_labels_must_not_be_claimed_available")
    if bet.get("timing_human_likeness_final_proof_allowed") is not False:
        violations.append("timing_human_likeness_final_proof_must_be_blocked")
    if bet.get("requires_more_real_player_behavior_labels") is not True:
        violations.append("bet_timing_must_require_more_real_player_labels_for_higher_realism")
    if bet.get("final_high_realism_claim_allowed") is not False:
        violations.append("final_high_realism_bet_timing_claim_must_be_blocked")
    if bet.get("timing_label_quality_status") != "TIMING_LABEL_QUALITY_UNCERTAIN":
        violations.append("timing_label_quality_boundary_must_remain_uncertain")
    if bet.get("timing_label_current_delivery_blocker") is not False:
        violations.append("timing_label_quality_gap_must_not_block_current_delivery")
    if bet.get("timing_label_model_quality_risk") is not True:
        violations.append("timing_label_quality_gap_must_remain_model_quality_risk")
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
        f"- actions.csv explicit context: `{risks['actions_context_quality']['explicit_context_status']}`",
        f"- actions.csv model-quality risk: `{risks['actions_context_quality']['model_quality_risk']}`",
        f"- stack_events decision context: `{risks['stack_event_context_quality']['derived_context_status']}`",
        f"- stack_events model-quality risk: `{risks['stack_event_context_quality']['model_quality_risk']}`",
        f"- Bet/timing implementation: `{risks['bet_timing_calibration']['implementation_status']}`",
        f"- Timing policy type: `{risks['bet_timing_calibration']['timing_policy_type']}`",
        f"- Timing label quality: `{risks['bet_timing_calibration']['real_human_timing_label_quality']}`",
        f"- Timing final human-likeness proof allowed: `{risks['bet_timing_calibration']['timing_human_likeness_final_proof_allowed']}`",
        f"- Larger gameplay revalidation required: `{risks['behavioral_revalidation']['larger_clean_real_gameplay_revalidation_required']}`",
        f"- Full production-scale multi-agent training: `{risks['multi_agent_training']['full_production_scale_multi_agent_training_status']}`",
        f"- Phase 3 OpenSpiel RL proof: `{risks['phase3_open_spiel_rl_training']['status']}`",
        f"- Phase 3 measured win-rate claim allowed: `{risks['phase3_open_spiel_rl_training']['measured_win_rate_claim_allowed']}`",
        f"- Evaluation metric boundary: `{risks['evaluation_metric_coverage']['boundary']}`",
        f"- Accuracy alone sufficient: `{risks['evaluation_metric_coverage']['accuracy_alone_sufficient']}`",
        f"- Final metric bundle passed: `{risks['evaluation_metric_coverage']['final_metric_bundle_passed']}`",
        f"- Full pytest status: `{risks['test_execution_coverage']['full_pytest_status']}`",
        f"- Full pytest used as approval: `{risks['test_execution_coverage']['full_pytest_used_as_delivery_approval']}`",
        f"- Critical validation status: `{risks['test_execution_coverage']['critical_validation_status']}`",
        f"- Delivery verifier status: `{risks['test_execution_coverage']['delivery_verifier_status']}`",
        f"- Human-likeness evidence: `{risks['human_likeness_evidence']['status']}`",
        f"- Human-likeness fully proven: `{risks['human_likeness_evidence']['human_likeness_fully_proven']}`",
        f"- Final human-likeness claim allowed: `{risks['human_likeness_evidence']['final_human_likeness_claim_allowed']}`",
        f"- Human-likeness claim gate: `{risks['human_likeness_claim_gate']['decision']}`",
        f"- Action-distribution-only proof rejected: `{risks['human_likeness_claim_gate']['action_distribution_only_proof_rejected']}`",
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
