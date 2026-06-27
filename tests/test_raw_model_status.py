from __future__ import annotations

import json
from pathlib import Path

from poker_agent.raw_model_status import (
    assert_raw_model_status,
    build_raw_model_status,
    validate_raw_model_status,
)


def test_raw_model_status_preserves_component_risk_boundary(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    reports.mkdir()
    models.mkdir()
    (models / "poker_policy.joblib").write_bytes(b"model-bytes")
    (reports / "production_gate.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "valid_metrics": {
                    "accuracy": 0.68,
                    "macro_f1": 0.39,
                    "balanced_accuracy": 0.40,
                    "majority_baseline_accuracy": 0.70,
                    "lift_vs_majority": -0.02,
                },
                "gates": [
                    {"name": "macro_f1", "passed": False},
                    {"name": "calibration", "passed": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (reports / "model_risk_register.json").write_text(
        json.dumps(
            {
                "raw_artifact_runtime_status": {"status": "LOADABLE"},
                "raw_production_gate_status": "FAIL",
                "raw_supervised_model_status": "NOT_STANDALONE_APPROVED",
                "deployed_strategy_stack_status": "APPROVED",
                "risk_summary": {"component_risks": 1, "deployment_blockers": 0},
            }
        ),
        encoding="utf-8",
    )

    payload = build_raw_model_status(tmp_path)

    assert payload["raw_supervised_model"]["runtime_status"] == "LOADABLE"
    assert payload["raw_supervised_model"]["service_loadable"] is True
    assert payload["raw_supervised_model"]["quality_gate_status"] == "FAIL"
    assert payload["raw_supervised_model"]["standalone_status"] == "NOT_STANDALONE_APPROVED"
    assert payload["raw_supervised_model"]["approved_as_standalone_policy"] is False
    assert payload["release_boundary"]["component_risk"] is True
    assert payload["release_boundary"]["production_blocker"] is False
    assert "macro_f1" in payload["quality_evidence"]["failed_gates"]
    assert_raw_model_status(payload)


def test_raw_model_status_rejects_false_standalone_approval() -> None:
    payload = {
        "raw_supervised_model": {
            "runtime_status": "LOADABLE",
            "service_loadable": True,
            "quality_gate_status": "FAIL",
            "standalone_status": "STANDALONE_APPROVED",
            "approved_as_standalone_policy": True,
        },
        "release_boundary": {
            "component_risk": False,
            "production_blocker": False,
            "service_delivery_allowed": True,
        },
    }

    violations = validate_raw_model_status(payload)

    assert "raw_model_cannot_be_standalone_approved_when_quality_gate_fails" in violations
    assert "standalone_status_cannot_be_approved_when_quality_gate_fails" in violations


def test_raw_model_status_endpoint_returns_current_boundary() -> None:
    from poker_agent.service import raw_model_status_json

    payload = raw_model_status_json()

    assert payload["raw_supervised_model"]["runtime_status"] == "LOADABLE"
    assert payload["raw_supervised_model"]["quality_gate_status"] == "FAIL"
    assert payload["raw_supervised_model"]["standalone_status"] == "NOT_STANDALONE_APPROVED"
    assert payload["release_boundary"]["component_risk"] is True


def test_raw_model_challenger_endpoint_never_false_approves_missing_report() -> None:
    from poker_agent.service import raw_model_challenger_json

    payload = raw_model_challenger_json()

    if payload.get("status") == "MISSING":
        assert payload["standalone_status"] == "NOT_STANDALONE_APPROVED"
        assert payload["approved_as_standalone_policy"] is False
    else:
        gate = (payload.get("best_candidate") or {}).get("gate") or {}
        if gate.get("status") != "PASS":
            assert payload["standalone_status"] == "NOT_STANDALONE_APPROVED"
            assert payload["approved_as_standalone_policy"] is False
