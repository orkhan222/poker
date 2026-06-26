from __future__ import annotations

from pathlib import Path

from poker_agent.today_training import build_today_training_plan, build_today_training_report


def test_today_training_selects_routed_bundle_for_current_delivery() -> None:
    plan = build_today_training_plan(Path("."), max_examples=1000)

    assert plan["profile"] == "today_acceptance_training"
    assert plan["selected_architecture"] == "routed_policy_bundle"
    assert plan["delivery_decision"] == "RUN_NOW_FOR_CURRENT_DELIVERY"
    assert plan["not_selected_today"]["full_multi_agent_training"]["status"] == "DEFERRED_TO_HARDENING"


def test_today_training_report_keeps_full_training_boundary() -> None:
    plan = build_today_training_plan(Path("."), max_examples=1000)
    report = build_today_training_report(
        plan,
        training_result={"returncode": 0},
        model_metadata={"valid_metrics": {"macro_f1": 0.5}},
        gate_result={"status": "FAIL", "decision": "Not approved for production decision-policy deployment."},
    )

    assert report["delivery_status"] == "READY_FOR_CURRENT_DELIVERY"
    assert report["training_status"] == "PASS"
    assert report["production_gate_status"] == "FAIL"
    assert report["approval_boundary"]["full_multi_agent_training"] == "DEFERRED_TO_PRODUCTION_HARDENING"
    assert report["approval_boundary"]["raw_supervised_model_standalone"] == "NOT_STANDALONE_APPROVED"