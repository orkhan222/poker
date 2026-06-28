from __future__ import annotations

import json
from pathlib import Path

from poker_agent.production_runtime_monitoring import (
    RuntimeMonitoringState,
    build_production_runtime_monitoring,
    validate_production_runtime_monitoring,
)


def _write_reports(reports: Path) -> None:
    reports.mkdir()
    (reports / "final_delivery_acceptance.json").write_text(
        json.dumps(
            {
                "acceptance_summary": {
                    "service_delivery": "READY",
                    "deployed_strategy_stack": "APPROVED",
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_approval.json").write_text(
        json.dumps({"overall_status": "APPROVED_WITH_COMPONENT_RISK"}),
        encoding="utf-8",
    )
    (reports / "strategy_stack_maturity.json").write_text(
        json.dumps({"current_strategy_stack": {"deployment_mode": "monitored_rollout"}}),
        encoding="utf-8",
    )
    (reports / "deployed_strategy_gate.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")


def test_production_runtime_monitoring_requires_observability_for_real_traffic(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_production_runtime_monitoring(tmp_path)

    boundary = payload["runtime_observability_boundary"]
    assert payload["overall_status"] == "PASS"
    assert boundary["status"] == "REQUIRES_MONITORING_ROLLBACK_AND_LIVE_DRIFT_TRACKING"
    assert boundary["monitoring_required_for_real_traffic"] is True
    assert boundary["rollback_rules_required_for_real_traffic"] is True
    assert boundary["live_drift_tracking_required_for_real_traffic"] is True
    assert boundary["real_traffic_claim_allowed_without_observability"] is False
    assert boundary["real_traffic_blocker_if_disabled"] is True
    assert boundary["current_delivery_blocker"] is False


def test_production_runtime_monitoring_blocks_unmonitored_real_traffic_claim(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_production_runtime_monitoring(tmp_path)
    boundary = payload["runtime_observability_boundary"]
    boundary["monitoring_required_for_real_traffic"] = False
    boundary["rollback_rules_required_for_real_traffic"] = False
    boundary["live_drift_tracking_required_for_real_traffic"] = False
    boundary["real_traffic_claim_allowed_without_observability"] = True
    boundary["real_traffic_blocker_if_disabled"] = False
    payload.pop("overall_status", None)

    invariants = validate_production_runtime_monitoring(payload)

    assert invariants["status"] == "FAIL"
    assert "monitoring_must_be_required_for_real_traffic" in invariants["violations"]
    assert "rollback_rules_must_be_required_for_real_traffic" in invariants["violations"]
    assert "live_drift_tracking_must_be_required_for_real_traffic" in invariants["violations"]
    assert "unmonitored_real_traffic_claim_must_be_blocked" in invariants["violations"]


def test_runtime_monitoring_state_flags_action_distribution_drift() -> None:
    state = RuntimeMonitoringState(max_events=100)
    for _ in range(40):
        state.observe_prediction(
            {"action": "raise", "probabilities": {"raise": 0.91}, "model_status": "routed_policy_bundle"},
            latency_ms=20.0,
            request_payload={"street": "preflop", "position": "BTN", "hole_cards": ["AS", "KD"]},
        )

    snapshot = state.snapshot()

    assert snapshot["prediction_count"] == 40
    assert snapshot["rollback_evaluation"]["rollback_required"] is True
    assert any(trigger["metric"] == "action_distribution_js" for trigger in snapshot["rollback_evaluation"]["triggers"])


def test_production_runtime_monitoring_endpoint_returns_contract() -> None:
    from poker_agent.service import production_runtime_monitoring_json

    payload = production_runtime_monitoring_json()

    assert payload["overall_status"] == "PASS"
    assert payload["runtime_observability_boundary"]["monitoring_required_for_real_traffic"] is True
    assert payload["runtime_observability_boundary"]["current_delivery_blocker"] is False
