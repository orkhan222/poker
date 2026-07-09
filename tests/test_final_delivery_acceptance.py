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
                "term_boundary": {
                    "status": "LLM_BASED_AGENT_IS_UMBRELLA_TERM",
                    "requires_role_specific_qualification": True,
                    "must_not_imply_fully_autonomous_policy": True,
                },
                "role_taxonomy": {
                    "event_normalization": {
                        "status": "CONTROLLED_COMPONENT",
                        "implemented": True,
                        "production_policy_approved": False,
                    },
                    "decision_context": {
                        "status": "CONTROLLED_COMPONENT",
                        "implemented": True,
                        "production_policy_approved": False,
                    },
                    "candidate_ranking": {
                        "status": "RESEARCH_BASELINE_COMPONENT",
                        "implemented": True,
                        "production_policy_approved": False,
                    },
                    "real_policy_agent": {
                        "status": "NOT_CURRENT_DELIVERY_SCOPE",
                        "implemented": False,
                        "production_policy_approved": False,
                    },
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
                    "timing_evidence_status": "HEURISTIC_TIMING_ONLY_NOT_FINAL_HUMAN_LIKENESS_PROOF",
                },
                "calibration_boundary": {
                    "requires_more_real_player_behavior_labels": True,
                    "final_high_realism_claim_allowed": False,
                    "production_blocker_for_current_delivery": False,
                },
                "timing_label_quality_boundary": {
                    "boundary": "REAL_HUMAN_TIMING_LABELS_REQUIRED_FOR_FULL_HUMAN_LIKENESS_PROOF",
                    "status": "TIMING_LABEL_QUALITY_UNCERTAIN",
                    "timing_feature_available": True,
                    "timing_policy_type": "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED",
                    "real_human_timing_labels_available": False,
                    "requires_real_human_timing_labels": True,
                    "uses_real_human_timing_labels": False,
                    "required_timing_label_fields": [
                        "decision_start_ts",
                        "decision_end_ts",
                        "human_wait_time_ms",
                        "street",
                        "position",
                        "facing_bet",
                        "action",
                    ],
                    "heuristic_timing_counts_as_full_human_likeness_proof": False,
                    "final_human_likeness_claim_allowed_from_timing_alone": False,
                    "final_production_human_likeness_proof_allowed": False,
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
    (reports / "normalized_action_contract.json").write_text(
        json.dumps(
            {
                "normalized_action_status": "IMPLEMENTED",
                "raw_action_source_status": "RAW_OCR_OR_DEALER_TEXT",
                "canonical_actions": ["fold", "call", "check", "bet", "raise", "all_in"],
                "source_field": "actions.csv::action",
                "normalized_field": "canonical_action",
                "raw_ocr_action_must_not_be_training_label": True,
                "normalization_required_before_training": True,
                "normalization_required_before_evaluation": True,
                "normalization_required_before_policy_comparison": True,
                "current_delivery_blocker": False,
                "model_quality_risk": False,
                "training_label_audit": {"status": "PASS", "invalid_labels": []},
                "noisy_action_examples": [
                    {"raw_action": "ra1se", "observed": "raise", "passed": True},
                    {"raw_action": "cail", "observed": "call", "passed": True},
                    {"raw_action": "bett", "observed": "bet", "passed": True},
                    {"raw_action": "all-in", "observed": "all_in", "passed": True},
                ],
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
                        "last_aggressor",
                        "facing_bet",
                    ],
                    "limitation_status": "OPEN_DATASET_LIMITATION",
                },
                "dataset_export_contract": {
                    "status": "EXPLICIT_BETTING_CONTEXT_REQUIRED_FOR_NEXT_DATASET_EXPORT",
                    "source_table": "actions.csv",
                    "required_explicit_fields": [
                        "amount",
                        "to_call",
                        "pot_before_action",
                        "min_raise",
                        "legal_actions",
                        "action_order",
                        "last_aggressor",
                        "facing_bet",
                    ],
                    "explicit_export_required": True,
                    "reconstructed_context_allowed_for_current_delivery": True,
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                    "must_not_use_target_row_values": True,
                    "must_not_use_future_outcome_fields": True,
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
    (reports / "phase2_selection_comparison.json").write_text(
        json.dumps(
            {
                "status": "STRICT_SELECTION_GATE_IMPLEMENTED",
                "required_candidates": [
                    "llm_decision_agent",
                    "supervised_model",
                    "rule_based_fallback",
                    "routed_policy_bundle",
                    "future_rl_agent",
                ],
                "common_holdout_contract": {"same_holdout_required": True},
                "common_simulation_contract": {"same_simulation_required": True},
                "candidates": {
                    "future_rl_agent": {"implementation_status": "NOT_AVAILABLE_YET"},
                },
                "comparison_gate": {
                    "all_required_candidates_present": True,
                    "all_candidates_compared_on_common_holdout": False,
                    "all_candidates_compared_in_common_simulation": False,
                    "all_candidate_metric_bundles_complete": False,
                    "missing_common_holdout_candidates": [
                        "llm_decision_agent",
                        "supervised_model",
                        "rule_based_fallback",
                        "routed_policy_bundle",
                        "future_rl_agent",
                    ],
                    "missing_common_simulation_candidates": [
                        "llm_decision_agent",
                        "supervised_model",
                        "rule_based_fallback",
                        "routed_policy_bundle",
                        "future_rl_agent",
                    ],
                    "missing_metric_bundle_candidates": [
                        "llm_decision_agent",
                        "supervised_model",
                        "rule_based_fallback",
                        "routed_policy_bundle",
                        "future_rl_agent",
                    ],
                    "selection_ineligible_candidates": [
                        "llm_decision_agent",
                        "supervised_model",
                        "rule_based_fallback",
                        "routed_policy_bundle",
                        "future_rl_agent",
                    ],
                    "selected_for_current_delivery": "routed_policy_bundle",
                    "final_selected_architecture": None,
                    "final_selection_claim_allowed": False,
                    "best_approach_claim_allowed": False,
                    "best_approach_claim_state": "BLOCKED_PENDING_FULL_COMMON_CONDITION_EVALUATION",
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "phase3_open_spiel_arena.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "status": "READY_PENDING_OPEN_SPIEL_RUNTIME",
                "rl_training_proof_boundary": {
                    "status": "TRAINING_PROOF_NOT_COMPLETED",
                    "real_open_spiel_runtime_required": True,
                    "real_open_spiel_runtime_available": False,
                    "phase1_trained_policy_artifacts_required": True,
                    "phase1_trained_policy_artifacts_attached": False,
                    "seed_stability_required": True,
                    "seed_stability_evaluated": False,
                    "long_run_required": True,
                    "long_run_completed": False,
                    "policy_update_training_required": True,
                    "policy_update_training_completed": False,
                    "measured_win_rate_claim_allowed": False,
                    "current_delivery_blocker": False,
                    "model_quality_risk": True,
                    "missing_requirements": [
                        "real_open_spiel_runtime",
                        "two_phase1_trained_policy_artifacts",
                        "seed_stability",
                        "long_run_training_volume",
                        "policy_update_training",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "evaluation_metric_contract.json").write_text(
        json.dumps(
            {
                "boundary": "ACCURACY_AND_CROSS_ENTROPY_NOT_SUFFICIENT",
                "accuracy_alone_sufficient": False,
                "accuracy_and_cross_entropy_sufficient": False,
                "required_metric_families": [
                    "action_classification",
                    "calibration",
                    "action_distribution",
                    "bet_sizing",
                    "simulation_return",
                    "seed_stability",
                ],
                "required_production_metrics": [
                    "accuracy",
                    "macro_f1",
                    "balanced_accuracy",
                    "confusion_matrix",
                    "calibration_ece",
                    "action_distribution_js_divergence",
                    "bet_size_mae",
                    "expected_value_delta_vs_baseline",
                    "win_rate",
                    "seed_stability",
                ],
                "diagnostic_metrics_not_sufficient_for_final_claim": ["accuracy", "cross_entropy"],
                "final_metric_bundle_passed": False,
                "final_strategy_quality_claim_allowed": False,
                "current_delivery_blocker": False,
                "model_quality_risk": True,
                "metric_families": {
                    "action_classification": {
                        "required": True,
                        "metrics": {
                            "accuracy": 0.72,
                            "macro_f1": 0.49,
                            "balanced_accuracy": 0.51,
                            "confusion_matrix": {
                                "labels": ["fold", "call", "raise"],
                                "matrix": [
                                    [42, 3, 1],
                                    [4, 21, 2],
                                    [1, 2, 16],
                                ],
                            },
                        },
                    },
                    "calibration": {"required": True, "metrics": {"ece_10": 0.18}},
                    "action_distribution": {"required": True, "metrics": {"js_divergence": 0.0026}},
                    "bet_sizing": {"required": True, "metrics": {"bet_size_mae": None}},
                    "simulation_return": {
                        "required": True,
                        "metrics": {"win_rate": 0.577, "expected_value_delta_vs_baseline": 0.82},
                    },
                    "seed_stability": {
                        "required": True,
                        "metrics": {"full_training_seed_stability_required": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "test_execution_contract.json").write_text(
        json.dumps(
            {
                "boundary": "FULL_PYTEST_TIMEOUT_IS_NOT_DELIVERY_APPROVAL",
                "full_pytest": {
                    "status": "TIMEOUT",
                    "used_as_delivery_approval": False,
                },
                "critical_validation": {
                    "status": "PASS",
                    "passed_tests": 16,
                },
                "delivery_verifier": {
                    "status": "PASS",
                },
                "current_delivery_blocker": False,
            }
        ),
        encoding="utf-8",
    )
    (reports / "human_likeness_evidence.json").write_text(
        json.dumps(
            {
                "boundary": "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF",
                "status": "NOT_FULLY_PROVEN",
                "human_likeness_fully_proven": False,
                "final_human_likeness_claim_allowed": False,
                "current_scope_action_distribution_passed": True,
                "current_delivery_blocker": False,
                "model_quality_risk": True,
                "required_behavior_dimensions": [
                    "action_distribution",
                    "bet_sizing",
                    "timing",
                    "position_based_behavior",
                    "street_level_strategy",
                ],
                "behavior_dimensions": {
                    "action_distribution": {"required": True, "current_status": "PASS", "final_proof_allowed": False},
                    "bet_sizing": {"required": True, "current_status": "PASS", "final_proof_allowed": False},
                    "timing": {"required": True, "current_status": "PASS", "final_proof_allowed": False},
                    "position_based_behavior": {
                        "required": True,
                        "current_status": "REQUIRES_SLICE_REVALIDATION",
                        "final_proof_allowed": False,
                    },
                    "street_level_strategy": {
                        "required": True,
                        "current_status": "REQUIRES_SLICE_REVALIDATION",
                        "final_proof_allowed": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "human_likeness_claim_gate.json").write_text(
        json.dumps(
            {
                "boundary": "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF",
                "claim": "FULL_HUMAN_LIKENESS",
                "decision": "BLOCKED",
                "claim_allowed": False,
                "human_likeness_fully_proven": False,
                "action_distribution_only_proof_rejected": True,
                "current_scope_action_distribution_passed": True,
                "current_delivery_blocker": False,
                "model_quality_risk": True,
                "required_evidence_dimensions": [
                    "action_distribution",
                    "bet_sizing",
                    "timing",
                    "position_based_behavior",
                    "street_level_strategy",
                ],
                "evidence_requirements": {
                    "action_distribution": {
                        "required_for_final_claim": True,
                        "current_status": "PASS",
                        "currently_sufficient_for_final_claim": False,
                    },
                    "bet_sizing": {
                        "required_for_final_claim": True,
                        "current_status": "PASS",
                        "currently_sufficient_for_final_claim": False,
                    },
                    "timing": {
                        "required_for_final_claim": True,
                        "current_status": "PASS",
                        "currently_sufficient_for_final_claim": False,
                    },
                    "position_based_behavior": {
                        "required_for_final_claim": True,
                        "current_status": "REQUIRES_SLICE_REVALIDATION",
                        "currently_sufficient_for_final_claim": False,
                    },
                    "street_level_strategy": {
                        "required_for_final_claim": True,
                        "current_status": "REQUIRES_SLICE_REVALIDATION",
                        "currently_sufficient_for_final_claim": False,
                    },
                },
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
    assert payload["tracked_component_risks"]["llm_work"]["term_status"] == "LLM_BASED_AGENT_IS_UMBRELLA_TERM"
    assert payload["tracked_component_risks"]["llm_work"]["term_must_not_imply_fully_autonomous_policy"] is True
    assert (
        payload["tracked_component_risks"]["llm_work"]["role_taxonomy"]["candidate_ranking"]["status"]
        == "RESEARCH_BASELINE_COMPONENT"
    )
    assert (
        payload["tracked_component_risks"]["llm_work"]["role_taxonomy"]["real_policy_agent"]["status"]
        == "NOT_CURRENT_DELIVERY_SCOPE"
    )
    assert payload["tracked_component_risks"]["llm_work"]["role_taxonomy"]["real_policy_agent"]["implemented"] is False
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
    assert payload["tracked_component_risks"]["normalized_action_contract"]["normalized_action_status"] == "IMPLEMENTED"
    assert (
        payload["tracked_component_risks"]["normalized_action_contract"]["raw_action_source_status"]
        == "RAW_OCR_OR_DEALER_TEXT"
    )
    assert set(payload["tracked_component_risks"]["normalized_action_contract"]["canonical_actions"]) == {
        "fold",
        "call",
        "check",
        "bet",
        "raise",
        "all_in",
    }
    assert (
        payload["tracked_component_risks"]["normalized_action_contract"]["raw_ocr_action_must_not_be_training_label"]
        is True
    )
    assert payload["tracked_component_risks"]["normalized_action_contract"]["training_label_status"] == "PASS"
    assert payload["tracked_component_risks"]["normalized_action_contract"]["invalid_training_labels"] == []
    assert (
        payload["tracked_component_risks"]["actions_context_quality"]["explicit_context_status"]
        == "INCOMPLETE_EXPLICIT_BETTING_CONTEXT"
    )
    actions_context = payload["tracked_component_risks"]["actions_context_quality"]
    assert actions_context["future_dataset_explicit_export_required"] is True
    assert actions_context["reconstructed_context_allowed_for_current_delivery"] is True
    assert set(actions_context["future_dataset_required_explicit_fields"]) == {
        "amount",
        "to_call",
        "pot_before_action",
        "min_raise",
        "legal_actions",
        "action_order",
        "last_aggressor",
        "facing_bet",
    }
    assert payload["tracked_component_risks"]["actions_context_quality"]["current_delivery_blocker"] is False
    assert payload["tracked_component_risks"]["actions_context_quality"]["model_quality_risk"] is True
    assert (
        payload["tracked_component_risks"]["stack_event_context_quality"]["raw_stack_event_status"]
        == "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION"
    )
    assert payload["tracked_component_risks"]["stack_event_context_quality"]["decision_time_derivation_required"] is True
    assert payload["tracked_component_risks"]["stack_event_context_quality"]["current_delivery_blocker"] is False
    assert payload["tracked_component_risks"]["stack_event_context_quality"]["model_quality_risk"] is True
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["status"] == "STRICT_SELECTION_GATE_IMPLEMENTED"
    assert (
        payload["tracked_component_risks"]["phase2_selection_comparison"]["selected_for_current_delivery"]
        == "routed_policy_bundle"
    )
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["final_selection_claim_allowed"] is False
    assert (
        payload["tracked_component_risks"]["phase2_selection_comparison"][
            "all_candidates_compared_on_common_holdout"
        ]
        is False
    )
    assert (
        payload["tracked_component_risks"]["phase2_selection_comparison"][
            "all_candidates_compared_in_common_simulation"
        ]
        is False
    )
    assert (
        payload["tracked_component_risks"]["phase2_selection_comparison"]["all_candidate_metric_bundles_complete"]
        is False
    )
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["best_approach_claim_allowed"] is False
    assert (
        payload["tracked_component_risks"]["phase2_selection_comparison"]["best_approach_claim_state"]
        == "BLOCKED_PENDING_FULL_COMMON_CONDITION_EVALUATION"
    )
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["missing_metric_bundle_candidates"]
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["selection_ineligible_candidates"]
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["current_delivery_blocker"] is False
    assert payload["tracked_component_risks"]["phase2_selection_comparison"]["model_quality_risk"] is True
    assert payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["status"] == "TRAINING_PROOF_NOT_COMPLETED"
    assert payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["measured_win_rate_claim_allowed"] is False
    assert payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["current_delivery_blocker"] is False
    assert payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["model_quality_risk"] is True
    assert (
        payload["tracked_component_risks"]["evaluation_metric_coverage"]["boundary"]
        == "ACCURACY_AND_CROSS_ENTROPY_NOT_SUFFICIENT"
    )
    assert payload["tracked_component_risks"]["evaluation_metric_coverage"]["accuracy_alone_sufficient"] is False
    assert (
        payload["tracked_component_risks"]["evaluation_metric_coverage"]["accuracy_and_cross_entropy_sufficient"]
        is False
    )
    assert set(payload["tracked_component_risks"]["evaluation_metric_coverage"]["required_production_metrics"]) == {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "confusion_matrix",
        "calibration_ece",
        "action_distribution_js_divergence",
        "bet_size_mae",
        "expected_value_delta_vs_baseline",
        "win_rate",
        "seed_stability",
    }
    assert payload["tracked_component_risks"]["evaluation_metric_coverage"]["final_metric_bundle_passed"] is False
    assert payload["tracked_component_risks"]["evaluation_metric_coverage"]["final_strategy_quality_claim_allowed"] is False
    assert payload["tracked_component_risks"]["evaluation_metric_coverage"]["model_quality_risk"] is True
    delivery_strategy = payload["delivery_strategy_quality_boundary"]
    assert delivery_strategy["status"] == "DELIVERY_READY_STRATEGY_QUALITY_CLAIM_BLOCKED"
    assert delivery_strategy["software_delivery_ready"] is True
    assert delivery_strategy["current_delivery_blocker"] is False
    assert delivery_strategy["final_metric_bundle_passed"] is False
    assert delivery_strategy["final_strategy_quality_claim_allowed"] is False
    assert delivery_strategy["model_quality_risk"] is True
    assert delivery_strategy["invariants"]["status"] == "PASS"
    assert payload["tracked_component_risks"]["test_execution_coverage"]["full_pytest_status"] == "TIMEOUT"
    assert payload["tracked_component_risks"]["test_execution_coverage"]["full_pytest_used_as_delivery_approval"] is False
    assert payload["tracked_component_risks"]["test_execution_coverage"]["critical_validation_status"] == "PASS"
    assert payload["tracked_component_risks"]["test_execution_coverage"]["delivery_verifier_status"] == "PASS"
    assert payload["tracked_component_risks"]["human_likeness_evidence"]["status"] == "NOT_FULLY_PROVEN"
    assert payload["tracked_component_risks"]["human_likeness_evidence"]["human_likeness_fully_proven"] is False
    assert payload["tracked_component_risks"]["human_likeness_evidence"]["final_human_likeness_claim_allowed"] is False
    assert payload["tracked_component_risks"]["human_likeness_evidence"]["current_scope_action_distribution_passed"] is True
    assert payload["tracked_component_risks"]["human_likeness_evidence"]["model_quality_risk"] is True
    assert payload["tracked_component_risks"]["human_likeness_claim_gate"]["claim"] == "FULL_HUMAN_LIKENESS"
    assert payload["tracked_component_risks"]["human_likeness_claim_gate"]["decision"] == "BLOCKED"
    assert payload["tracked_component_risks"]["human_likeness_claim_gate"]["claim_allowed"] is False
    assert payload["tracked_component_risks"]["human_likeness_claim_gate"]["human_likeness_fully_proven"] is False
    assert (
        payload["tracked_component_risks"]["human_likeness_claim_gate"]["action_distribution_only_proof_rejected"]
        is True
    )
    assert payload["tracked_component_risks"]["human_likeness_claim_gate"]["model_quality_risk"] is True


def test_final_delivery_acceptance_blocks_false_claims(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_final_delivery_acceptance(tmp_path)
    payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_present"] = True
    payload["tracked_component_risks"]["llm_work"]["term_status"] = "AUTONOMOUS_POLICY_AGENT"
    payload["tracked_component_risks"]["llm_work"]["term_must_not_imply_fully_autonomous_policy"] = False
    payload["tracked_component_risks"]["llm_work"]["role_taxonomy"]["candidate_ranking"]["production_policy_approved"] = True
    payload["tracked_component_risks"]["llm_work"]["role_taxonomy"]["real_policy_agent"]["status"] = "CURRENT_DELIVERY_SCOPE"
    payload["tracked_component_risks"]["llm_work"]["role_taxonomy"]["real_policy_agent"]["implemented"] = True
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
    payload["tracked_component_risks"]["normalized_action_contract"]["normalized_action_status"] = "MISSING"
    payload["tracked_component_risks"]["normalized_action_contract"]["raw_ocr_action_must_not_be_training_label"] = False
    payload["tracked_component_risks"]["normalized_action_contract"]["normalization_required_before_training"] = False
    payload["tracked_component_risks"]["normalized_action_contract"]["model_quality_risk"] = True
    payload["tracked_component_risks"]["normalized_action_contract"]["training_label_status"] = "FAIL"
    payload["tracked_component_risks"]["normalized_action_contract"]["invalid_training_labels"] = ["ra1se"]
    payload["tracked_component_risks"]["normalized_action_contract"]["noisy_action_examples"][0]["passed"] = False
    payload["tracked_component_risks"]["actions_context_quality"]["explicit_context_status"] = "COMPLETE"
    payload["tracked_component_risks"]["actions_context_quality"]["does_not_fully_replace_explicit_context"] = False
    payload["tracked_component_risks"]["actions_context_quality"]["future_dataset_explicit_export_required"] = False
    payload["tracked_component_risks"]["actions_context_quality"]["reconstructed_context_allowed_for_current_delivery"] = False
    payload["tracked_component_risks"]["actions_context_quality"]["future_dataset_required_explicit_fields"] = ["to_call"]
    payload["tracked_component_risks"]["actions_context_quality"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["stack_event_context_quality"]["raw_stack_events_are_direct_policy_features"] = True
    payload["tracked_component_risks"]["stack_event_context_quality"]["decision_time_derivation_required"] = False
    payload["tracked_component_risks"]["stack_event_context_quality"]["target_action_stack_delta_allowed_as_feature"] = True
    payload["tracked_component_risks"]["stack_event_context_quality"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["phase2_selection_comparison"]["all_candidates_compared_on_common_holdout"] = True
    payload["tracked_component_risks"]["phase2_selection_comparison"]["all_candidates_compared_in_common_simulation"] = True
    payload["tracked_component_risks"]["phase2_selection_comparison"]["all_candidate_metric_bundles_complete"] = True
    payload["tracked_component_risks"]["phase2_selection_comparison"]["missing_common_holdout_candidates"] = []
    payload["tracked_component_risks"]["phase2_selection_comparison"]["missing_common_simulation_candidates"] = []
    payload["tracked_component_risks"]["phase2_selection_comparison"]["missing_metric_bundle_candidates"] = []
    payload["tracked_component_risks"]["phase2_selection_comparison"]["selection_ineligible_candidates"] = []
    payload["tracked_component_risks"]["phase2_selection_comparison"]["selected_for_current_delivery"] = "llm_decision_agent"
    payload["tracked_component_risks"]["phase2_selection_comparison"]["final_selected_architecture"] = "llm_decision_agent"
    payload["tracked_component_risks"]["phase2_selection_comparison"]["future_rl_agent_status"] = "AVAILABLE"
    payload["tracked_component_risks"]["phase2_selection_comparison"]["final_selection_claim_allowed"] = True
    payload["tracked_component_risks"]["phase2_selection_comparison"]["best_approach_claim_allowed"] = True
    payload["tracked_component_risks"]["phase2_selection_comparison"]["best_approach_claim_state"] = "ALLOWED"
    payload["tracked_component_risks"]["phase2_selection_comparison"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["status"] = "TRAINING_PROOF_COMPLETED"
    payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["measured_win_rate_claim_allowed"] = True
    payload["tracked_component_risks"]["phase3_open_spiel_rl_training"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["evaluation_metric_coverage"]["accuracy_alone_sufficient"] = True
    payload["tracked_component_risks"]["evaluation_metric_coverage"]["accuracy_and_cross_entropy_sufficient"] = True
    payload["tracked_component_risks"]["evaluation_metric_coverage"]["final_metric_bundle_passed"] = True
    payload["tracked_component_risks"]["evaluation_metric_coverage"]["final_strategy_quality_claim_allowed"] = True
    payload["tracked_component_risks"]["evaluation_metric_coverage"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["test_execution_coverage"]["full_pytest_used_as_delivery_approval"] = True
    payload["tracked_component_risks"]["test_execution_coverage"]["critical_validation_status"] = "FAIL"
    payload["tracked_component_risks"]["test_execution_coverage"]["delivery_verifier_status"] = "FAIL"
    payload["tracked_component_risks"]["test_execution_coverage"]["current_delivery_blocker"] = True
    payload["tracked_component_risks"]["human_likeness_evidence"]["human_likeness_fully_proven"] = True
    payload["tracked_component_risks"]["human_likeness_evidence"]["final_human_likeness_claim_allowed"] = True
    payload["tracked_component_risks"]["human_likeness_evidence"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["human_likeness_evidence"]["behavior_dimensions"]["bet_sizing"][
        "final_proof_allowed"
    ] = True
    payload["tracked_component_risks"]["human_likeness_evidence"]["behavior_dimensions"]["position_based_behavior"][
        "required"
    ] = False
    payload["tracked_component_risks"]["human_likeness_claim_gate"]["decision"] = "APPROVED"
    payload["tracked_component_risks"]["human_likeness_claim_gate"]["claim_allowed"] = True
    payload["tracked_component_risks"]["human_likeness_claim_gate"]["human_likeness_fully_proven"] = True
    payload["tracked_component_risks"]["human_likeness_claim_gate"]["action_distribution_only_proof_rejected"] = False
    payload["tracked_component_risks"]["human_likeness_claim_gate"]["model_quality_risk"] = False
    payload["tracked_component_risks"]["human_likeness_claim_gate"]["evidence_requirements"]["timing"][
        "currently_sufficient_for_final_claim"
    ] = True
    payload.pop("overall_status", None)

    invariants = validate_final_delivery_acceptance(payload)

    assert invariants["status"] == "FAIL"
    assert "fully_autonomous_llm_agent_must_not_be_present" in invariants["violations"]
    assert "llm_based_agent_term_must_remain_umbrella" in invariants["violations"]
    assert "llm_based_agent_term_must_not_imply_autonomous_policy" in invariants["violations"]
    assert "llm_role_must_not_be_production_policy:candidate_ranking" in invariants["violations"]
    assert "llm_real_policy_agent_must_remain_out_of_current_scope" in invariants["violations"]
    assert "raw_model_must_not_be_standalone_approved" in invariants["violations"]
    assert "qlora_must_not_be_marked_production_approved" in invariants["violations"]
    assert "qlora_delivery_blocker_must_be_false" in invariants["violations"]
    assert "qlora_target_must_remain_enabled:json_schema_compliance_improvement" in invariants["violations"]
    assert "monitoring_must_be_required_for_real_traffic" in invariants["violations"]
    assert "prediction_distribution_tracking_must_be_required_for_real_traffic" in invariants["violations"]
    assert "actions_context_must_remain_marked_incomplete" in invariants["violations"]
    assert "actions_context_must_not_claim_full_replacement" in invariants["violations"]
    assert "future_actions_dataset_explicit_export_required_must_be_true" in invariants["violations"]
    assert "future_actions_dataset_required_fields_must_match_contract" in invariants["violations"]
    assert "reconstructed_actions_context_must_remain_allowed_for_current_delivery" in invariants["violations"]
    assert "actions_context_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "raw_stack_events_must_not_be_marked_direct_policy_features" in invariants["violations"]
    assert "stack_events_must_require_decision_time_derivation" in invariants["violations"]
    assert "target_action_stack_delta_must_not_be_allowed_as_feature" in invariants["violations"]
    assert "stack_event_context_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "phase2_selection_common_holdout_must_not_be_marked_complete_yet" in invariants["violations"]
    assert "phase2_selection_common_simulation_must_not_be_marked_complete_yet" in invariants["violations"]
    assert "phase2_selection_metric_bundles_must_not_be_marked_complete_yet" in invariants["violations"]
    assert "phase2_selection_missing_common_holdout_candidates_must_be_listed" in invariants["violations"]
    assert "phase2_selection_missing_common_simulation_candidates_must_be_listed" in invariants["violations"]
    assert "phase2_selection_missing_metric_bundle_candidates_must_be_listed" in invariants["violations"]
    assert "phase2_selection_ineligible_candidates_must_be_listed" in invariants["violations"]
    assert "phase2_selection_current_delivery_architecture_must_be_routed_bundle" in invariants["violations"]
    assert "phase2_selection_final_architecture_must_not_be_selected_yet" in invariants["violations"]
    assert "phase2_selection_future_rl_must_not_be_claimed_available" in invariants["violations"]
    assert "phase2_selection_final_claim_must_be_blocked_until_common_conditions" in invariants["violations"]
    assert "phase2_selection_best_approach_claim_must_be_blocked_until_common_conditions" in invariants["violations"]
    assert "phase2_selection_best_approach_claim_state_must_be_blocked" in invariants["violations"]
    assert "phase2_selection_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "phase3_open_spiel_rl_training_proof_must_remain_not_completed" in invariants["violations"]
    assert "phase3_rl_win_rate_claim_must_remain_blocked_until_training_proof" in invariants["violations"]
    assert "phase3_rl_training_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "accuracy_alone_must_not_be_sufficient_for_final_acceptance" in invariants["violations"]
    assert "accuracy_and_cross_entropy_must_not_be_sufficient_for_final_acceptance" in invariants["violations"]
    assert "final_metric_bundle_must_not_be_marked_passed" in invariants["violations"]
    assert "final_strategy_quality_claim_must_remain_blocked_until_metric_bundle_passes" in invariants["violations"]
    assert "evaluation_metric_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "timed_out_full_pytest_must_not_be_delivery_approval" in invariants["violations"]
    assert "critical_validation_must_pass_for_delivery" in invariants["violations"]
    assert "delivery_verifier_must_pass_for_delivery" in invariants["violations"]
    assert "test_execution_gap_must_not_block_current_delivery" in invariants["violations"]
    assert "human_likeness_must_not_be_marked_fully_proven" in invariants["violations"]
    assert "final_human_likeness_claim_must_remain_blocked" in invariants["violations"]
    assert "human_likeness_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "human_likeness_dimension_final_proof_must_be_blocked:bet_sizing" in invariants["violations"]
    assert "human_likeness_dimension_must_be_required:position_based_behavior" in invariants["violations"]
    assert "human_likeness_claim_gate_decision_must_remain_blocked" in invariants["violations"]
    assert "human_likeness_claim_gate_must_not_allow_final_claim" in invariants["violations"]
    assert "human_likeness_claim_gate_must_not_mark_full_proof" in invariants["violations"]
    assert "human_likeness_claim_gate_must_reject_action_distribution_only_proof" in invariants["violations"]
    assert "human_likeness_claim_gate_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "human_likeness_claim_gate_dimension_must_not_be_currently_sufficient:timing" in invariants[
        "violations"
    ]
    assert "model_confidence_monitoring_must_be_required_for_real_traffic" in invariants["violations"]
    assert "real_production_traffic_must_not_be_approved_before_observability" in invariants["violations"]
    assert "real_production_traffic_status_must_require_observability" in invariants["violations"]
    assert "final_strategy_quality_claim_must_remain_blocked_until_challenger_passes" in invariants["violations"]
    assert "normalized_action_contract_must_be_implemented" in invariants["violations"]
    assert "raw_ocr_action_must_not_be_training_label" in invariants["violations"]
    assert "normalized_action_must_be_required_before_training" in invariants["violations"]
    assert "normalized_action_contract_must_not_remain_model_quality_risk" in invariants["violations"]
    assert "normalized_action_training_labels_must_pass" in invariants["violations"]
    assert "normalized_action_training_labels_must_not_contain_raw_ocr" in invariants["violations"]
    assert "normalized_action_example_must_pass:ra1se" in invariants["violations"]


def test_final_delivery_acceptance_endpoint_returns_contract() -> None:
    from poker_agent.service import final_delivery_acceptance_json

    payload = final_delivery_acceptance_json()

    assert payload["overall_status"] == "PASS"
    assert payload["final_status"] == "READY_WITH_TRACKED_COMPONENT_RISKS"
    assert payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_claim_allowed"] is False
