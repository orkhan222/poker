from __future__ import annotations

import json
from pathlib import Path

from poker_agent.final_delivery_acceptance import build_final_delivery_acceptance, validate_final_delivery_acceptance


def _write_reports(reports: Path) -> None:
    reports.mkdir()
    (reports / "production_approval.json").write_text(
        json.dumps(
            {
                "overall_status": "APPROVED_WITH_COMPONENT_RISK",
                "raw_supervised_model": {
                    "runtime_status": "LOADABLE",
                    "standalone_status": "NOT_STANDALONE_APPROVED",
                    "raw_production_gate": "FAIL",
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "client_handoff.json").write_text(
        json.dumps(
            {
                "handoff_status": "READY_WITH_COMPONENT_RISK",
                "technical_position": {
                    "service_delivery": "READY",
                    "deployed_strategy_stack": "APPROVED",
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "delivery_verification.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (reports / "llm_role_boundary.json").write_text(
        json.dumps(
            {
                "current_llm_role": {
                    "status": "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER",
                    "event_normalization_layer": {"implemented": True},
                    "decision_context_layer": {"implemented": True},
                },
                "autonomous_llm_agent_boundary": {
                    "fully_autonomous_poker_playing_llm_agent_present": False,
                    "fully_autonomous_llm_agent_claim_allowed": False,
                    "production_blocker_for_current_delivery": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "qlora_next_stage.json").write_text(
        json.dumps(
            {
                "stage_boundary": {
                    "stage_status": "NEXT_STAGE_IMPROVEMENT",
                    "milestone_type": "RESEARCH_QUALITY_IMPROVEMENT_MILESTONE",
                    "fine_tuning_completed": False,
                    "production_approved": False,
                    "current_delivery_blocker": False,
                    "delivery_blocker": False,
                    "approved_current_delivery_component": False,
                    "requires_separate_approval_before_promotion": True,
                },
                "delivery_classification": {
                    "current_delivery_component": False,
                    "current_delivery_blocker": False,
                    "next_stage_research_milestone": True,
                    "promotion_requires_new_gate": True,
                },
                "recommended_training_plan": {
                    "adapter_scope": "EVENT_NORMALIZATION_STRUCTURED_EXTRACTION_AND_CANDIDATE_RANKING"
                },
                "target_use_cases": {
                    "noisy_ocr_dealer_log_normalization": {"recommended": True},
                    "structured_extraction": {"recommended": True},
                    "candidate_ranking": {"recommended": True},
                    "json_schema_compliance_improvement": {"recommended": True},
                    "autonomous_poker_policy": {"recommended": False},
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_runtime_monitoring.json").write_text(
        json.dumps(
            {
                "runtime_observability_boundary": {
                    "status": "REQUIRES_MONITORING_ROLLBACK_AND_LIVE_DRIFT_TRACKING",
                    "monitoring_required_for_real_traffic": True,
                    "rollback_rules_required_for_real_traffic": True,
                    "live_drift_tracking_required_for_real_traffic": True,
                    "prediction_distribution_tracking_required_for_real_traffic": True,
                    "model_confidence_monitoring_required_for_real_traffic": True,
                    "real_traffic_claim_allowed_without_observability": False,
                    "real_production_traffic_approved": False,
                    "real_production_traffic_approval_status": "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED",
                    "real_traffic_blocker_if_disabled": True,
                    "current_delivery_blocker": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "challenger_strategy_quality.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "strategy_quality_boundary": {
                    "status": "BLOCKED_PENDING_CHALLENGER_GATE",
                    "final_production_strategy_quality_claim_allowed": False,
                    "claim_blocked_until_challenger_passes": True,
                    "challenger_required_before_final_claim": True,
                    "challenger_trained": True,
                    "challenger_compared_to_raw_model": True,
                    "raw_production_gate_status": "FAIL",
                    "challenger_gate_status": "FAIL",
                    "current_raw_model_standalone_approved": False,
                    "raw_model_component_risk": True,
                    "current_delivery_blocker": False,
                    "deployed_strategy_stack_affected": False,
                },
                "challenger_result": {
                    "best_candidate": "extra_trees_sqrt_balanced_full",
                    "macro_f1": 0.48,
                    "failed_gates": ["macro_f1", "calibration"],
                },
                "invariants": {"status": "PASS", "violations": []},
            }
        ),
        encoding="utf-8",
    )
    (reports / "bet_timing_calibration.json").write_text(
        json.dumps(
            {
                "current_delivery_scope": {
                    "implementation_status": "IMPLEMENTED_AND_MEASURED",
                    "timing_and_bet_size_status": "PASS",
                    "timing_policy_type": "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED",
                    "real_human_timing_label_quality": "TIMING_LABEL_QUALITY_UNCERTAIN",
                    "real_human_timing_labels_available": False,
                    "timing_human_likeness_final_proof_allowed": False,
                },
                "calibration_boundary": {
                    "requires_more_real_player_behavior_labels": True,
                    "final_high_realism_claim_allowed": False,
                    "production_blocker_for_current_delivery": False,
                },
                "timing_label_quality_boundary": {
                    "status": "TIMING_LABEL_QUALITY_UNCERTAIN",
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "hole_card_data_quality.json").write_text(
        json.dumps(
            {
                "upstream_data_quality_boundary": {
                    "limitation_status": "OPEN_DATA_QUALITY_LIMITATION",
                    "upstream_data_quality_issue_resolved": False,
                    "requires_ocr_or_parser_improvement": True,
                    "component_risk": True,
                    "production_blocker_for_current_deployment": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "actions_context_quality.json").write_text(
        json.dumps(
            {
                "actions_csv_schema_audit": {
                    "explicit_context_status": "INCOMPLETE_EXPLICIT_BETTING_CONTEXT",
                    "missing_explicit_context_fields": [
                        "amount",
                        "to_call",
                        "pot_before_action",
                        "min_raise",
                        "legal_actions",
                        "action_order",
                    ],
                    "limitation_status": "OPEN_DATASET_LIMITATION",
                },
                "derived_context_mitigation": {
                    "status": "IMPLEMENTED_FROM_PRE_ACTION_EVENT_STREAM",
                    "uses_target_action_amount_as_feature": False,
                    "uses_future_outcome_fields": False,
                    "does_not_fully_replace_explicit_context": True,
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "stack_event_context_quality.json").write_text(
        json.dumps(
            {
                "raw_stack_event_boundary": {
                    "status": "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION",
                    "raw_stack_events_are_direct_policy_features": False,
                    "decision_time_derivation_required": True,
                    "target_action_stack_delta_allowed_as_feature": False,
                    "post_hand_stack_outcome_allowed_as_feature": False,
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                },
                "derived_context_mitigation": {
                    "status": "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS",
                    "uses_target_action_stack_delta_as_feature": False,
                    "uses_post_hand_outcome_fields": False,
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "behavioral_revalidation.json").write_text(
        json.dumps(
            {
                "revalidation_boundary": {
                    "current_scope_claim_allowed": True,
                    "larger_clean_real_gameplay_revalidation_required": True,
                    "generalized_human_likeness_claim_allowed": False,
                    "production_blocker": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "strategy_stack_maturity.json").write_text(
        json.dumps({"current_strategy_stack": {"status": "APPROVED_FOR_DEPLOYMENT_WITH_MONITORING"}}),
        encoding="utf-8",
    )
    (reports / "multi_agent_training_status.json").write_text(
        json.dumps(
            {
                "training_boundary": {
                    "acceptance_training_status": "COMPLETED_FOR_DELIVERY_VALIDATION",
                    "full_production_scale_multi_agent_training_status": "NOT_COMPLETED",
                    "production_blocker_for_current_delivery": False,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "raw_model_status.json").write_text(
        json.dumps(
            {
                "raw_supervised_model": {"runtime_status": "LOADABLE", "standalone_status": "NOT_STANDALONE_APPROVED"},
                "release_boundary": {"component_risk": True, "production_blocker": False},
            }
        ),
        encoding="utf-8",
    )


def test_final_delivery_acceptance_passes_with_tracked_risks(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_final_delivery_acceptance(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["final_status"] == "READY_WITH_TRACKED_COMPONENT_RISKS"
    assert payload["acceptance_summary"]["service_delivery"] == "READY"
    assert payload["acceptance_summary"]["deployed_strategy_stack"] == "APPROVED"
    assert payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_claim_allowed"] is False
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["stage_status"] == "NEXT_STAGE_IMPROVEMENT"
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["milestone_type"] == "RESEARCH_QUALITY_IMPROVEMENT_MILESTONE"
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["production_approved"] is False
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["delivery_blocker"] is False
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["approved_current_delivery_component"] is False
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["targets"]["json_schema_compliance_improvement"] is True
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["monitoring_required_for_real_traffic"] is True
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["prediction_distribution_tracking_required_for_real_traffic"] is True
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["model_confidence_monitoring_required_for_real_traffic"] is True
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["real_production_traffic_approved"] is False
    assert (
        payload["tracked_component_risks"]["production_runtime_monitoring"]["real_production_traffic_approval_status"]
        == "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED"
    )
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["real_traffic_blocker_if_disabled"] is True
    assert payload["tracked_component_risks"]["challenger_strategy_quality"]["challenger_required_before_final_claim"] is True
    assert payload["tracked_component_risks"]["challenger_strategy_quality"]["final_production_strategy_quality_claim_allowed"] is False
    assert payload["tracked_component_risks"]["raw_supervised_model"]["standalone_status"] == "NOT_STANDALONE_APPROVED"
    assert (
        payload["tracked_component_risks"]["actions_context_quality"]["explicit_context_status"]
        == "INCOMPLETE_EXPLICIT_BETTING_CONTEXT"
    )
    assert payload["tracked_component_risks"]["actions_context_quality"]["current_delivery_blocker"] is False
    assert payload["tracked_component_risks"]["actions_context_quality"]["model_quality_risk"] is True
    assert (
        payload["tracked_component_risks"]["stack_event_context_quality"]["raw_stack_event_status"]
        == "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION"
    )
    assert payload["tracked_component_risks"]["stack_event_context_quality"]["decision_time_derivation_required"] is True
    assert payload["tracked_component_risks"]["stack_event_context_quality"]["current_delivery_blocker"] is False
    assert payload["tracked_component_risks"]["stack_event_context_quality"]["model_quality_risk"] is True


def test_final_delivery_acceptance_blocks_false_claims(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_final_delivery_acceptance(tmp_path)
    payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_present"] = True
    payload["tracked_component_risks"]["raw_supervised_model"]["standalone_status"] = "STANDALONE_APPROVED"
    payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["production_approved"] = True
    payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["delivery_blocker"] = True
    payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["targets"]["json_schema_compliance_improvement"] = False
    payload["tracked_component_risks"]["production_runtime_monitoring"]["monitoring_required_for_real_traffic"] = False
    payload["tracked_component_risks"]["production_runtime_monitoring"]["prediction_distribution_tracking_required_for_real_traffic"] = False
    payload["tracked_component_risks"]["production_runtime_monitoring"]["model_confidence_monitoring_required_for_real_traffic"] = False
    payload["tracked_component_risks"]["production_runtime_monitoring"]["real_production_traffic_approved"] = True
    payload["tracked_component_risks"]["production_runtime_monitoring"]["real_production_traffic_approval_status"] = "APPROVED"
    payload["tracked_component_risks"]["challenger_strategy_quality"]["final_production_strategy_quality_claim_allowed"] = True
    payload["tracked_component_risks"]["actions_context_quality"]["explicit_context_status"] = "COMPLETE"
    payload["tracked_component_risks"]["actions_context_quality"]["does_not_fully_replace_explicit_context"] = False
    payload["tracked_component_risks"]["actions_context_quality"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["stack_event_context_quality"]["raw_stack_events_are_direct_policy_features"] = True
    payload["tracked_component_risks"]["stack_event_context_quality"]["decision_time_derivation_required"] = False
    payload["tracked_component_risks"]["stack_event_context_quality"]["target_action_stack_delta_allowed_as_feature"] = True
    payload["tracked_component_risks"]["stack_event_context_quality"]["model_quality_risk"] = False
    payload.pop("overall_status", None)

    invariants = validate_final_delivery_acceptance(payload)

    assert invariants["status"] == "FAIL"
    assert "fully_autonomous_llm_agent_must_not_be_present" in invariants["violations"]
    assert "raw_model_must_not_be_standalone_approved" in invariants["violations"]
    assert "qlora_must_not_be_marked_production_approved" in invariants["violations"]
    assert "qlora_delivery_blocker_must_be_false" in invariants["violations"]
    assert "qlora_target_must_remain_enabled:json_schema_compliance_improvement" in invariants["violations"]
    assert "monitoring_must_be_required_for_real_traffic" in invariants["violations"]
    assert "prediction_distribution_tracking_must_be_required_for_real_traffic" in invariants["violations"]
    assert "actions_context_must_remain_marked_incomplete" in invariants["violations"]
    assert "actions_context_must_not_claim_full_replacement" in invariants["violations"]
    assert "actions_context_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "raw_stack_events_must_not_be_marked_direct_policy_features" in invariants["violations"]
    assert "stack_events_must_require_decision_time_derivation" in invariants["violations"]
    assert "target_action_stack_delta_must_not_be_allowed_as_feature" in invariants["violations"]
    assert "stack_event_context_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "model_confidence_monitoring_must_be_required_for_real_traffic" in invariants["violations"]
    assert "real_production_traffic_must_not_be_approved_before_observability" in invariants["violations"]
    assert "real_production_traffic_status_must_require_observability" in invariants["violations"]
    assert "final_strategy_quality_claim_must_remain_blocked_until_challenger_passes" in invariants["violations"]


def test_final_delivery_acceptance_endpoint_returns_contract() -> None:
    from poker_agent.service import final_delivery_acceptance_json

    payload = final_delivery_acceptance_json()

    assert payload["overall_status"] == "PASS"
    assert payload["final_status"] == "READY_WITH_TRACKED_COMPONENT_RISKS"
    assert payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_claim_allowed"] is False
