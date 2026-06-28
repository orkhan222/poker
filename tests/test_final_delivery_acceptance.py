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
                    "fine_tuning_completed": False,
                    "production_approved": False,
                    "current_delivery_blocker": False,
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
                    "real_traffic_claim_allowed_without_observability": False,
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
                },
                "calibration_boundary": {
                    "requires_more_real_player_behavior_labels": True,
                    "final_high_realism_claim_allowed": False,
                    "production_blocker_for_current_delivery": False,
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
    assert payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["production_approved"] is False
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["monitoring_required_for_real_traffic"] is True
    assert payload["tracked_component_risks"]["production_runtime_monitoring"]["real_traffic_blocker_if_disabled"] is True
    assert payload["tracked_component_risks"]["challenger_strategy_quality"]["challenger_required_before_final_claim"] is True
    assert payload["tracked_component_risks"]["challenger_strategy_quality"]["final_production_strategy_quality_claim_allowed"] is False
    assert payload["tracked_component_risks"]["raw_supervised_model"]["standalone_status"] == "NOT_STANDALONE_APPROVED"


def test_final_delivery_acceptance_blocks_false_claims(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_final_delivery_acceptance(tmp_path)
    payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_present"] = True
    payload["tracked_component_risks"]["raw_supervised_model"]["standalone_status"] = "STANDALONE_APPROVED"
    payload["tracked_component_risks"]["qlora_larger_llm_fine_tuning"]["production_approved"] = True
    payload["tracked_component_risks"]["production_runtime_monitoring"]["monitoring_required_for_real_traffic"] = False
    payload["tracked_component_risks"]["challenger_strategy_quality"]["final_production_strategy_quality_claim_allowed"] = True
    payload.pop("overall_status", None)

    invariants = validate_final_delivery_acceptance(payload)

    assert invariants["status"] == "FAIL"
    assert "fully_autonomous_llm_agent_must_not_be_present" in invariants["violations"]
    assert "raw_model_must_not_be_standalone_approved" in invariants["violations"]
    assert "qlora_must_not_be_marked_production_approved" in invariants["violations"]
    assert "monitoring_must_be_required_for_real_traffic" in invariants["violations"]
    assert "final_strategy_quality_claim_must_remain_blocked_until_challenger_passes" in invariants["violations"]


def test_final_delivery_acceptance_endpoint_returns_contract() -> None:
    from poker_agent.service import final_delivery_acceptance_json

    payload = final_delivery_acceptance_json()

    assert payload["overall_status"] == "PASS"
    assert payload["final_status"] == "READY_WITH_TRACKED_COMPONENT_RISKS"
    assert payload["tracked_component_risks"]["llm_work"]["fully_autonomous_llm_agent_claim_allowed"] is False
