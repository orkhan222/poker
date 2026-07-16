from __future__ import annotations

import tempfile
from pathlib import Path

from poker_agent.agents import RuleBasedAgent
from poker_agent.monitoring import (
    AUDIT_TRAIL_SCHEMA_VERSION,
    MONITORING_CONTRACT_VERSION,
    PREDICTION_LOG_SCHEMA_VERSION,
    append_jsonl,
    audit_trail_event,
    describe_monitoring_contract,
    drift_report,
    invalid_state_findings,
    monitoring_summary,
    prediction_log_event,
    read_jsonl,
    validate_monitoring_contract,
)
from poker_agent.schemas import PredictionRequest

ROOT = Path(__file__).resolve().parents[1]


def valid_payload(pot: float = 2.5, to_call: float = 1.0) -> dict:
    return {
        "position": "BTN",
        "street": "preflop",
        "hole_cards": ["Ah", "Kd"],
        "pot": pot,
        "to_call": to_call,
        "amount_to_call": to_call,
        "stack": 100.0,
        "effective_stack": 100.0,
        "min_raise": 2.0,
        "max_raise": 100.0,
    }


def event_from_payload(payload: dict, *, request_id: str, latency_ms: float = 20.0) -> dict:
    request = PredictionRequest.from_dict(payload)
    response = RuleBasedAgent().predict(request).to_dict()
    return prediction_log_event(
        request_id=request_id,
        raw_payload=payload,
        request=request,
        response=response,
        latency_ms=latency_ms,
        status="ok",
    )


def test_invalid_state_findings_catch_bad_payloads() -> None:
    findings = invalid_state_findings(
        {
            "street": "showdown",
            "pot": -1,
            "hole_cards": ["Ah", "Kd", "Qs"],
        }
    )
    codes = {finding["code"] for finding in findings}

    assert {"missing_position", "invalid_street", "negative_state_value", "too_many_hole_cards"}.issubset(codes)


def test_prediction_log_event_contains_latency_confidence_features_and_state() -> None:
    event = event_from_payload(valid_payload(), request_id="request-1")

    assert event["schema_version"] == PREDICTION_LOG_SCHEMA_VERSION
    assert event["latency_ms"] == 20.0
    assert 0.0 <= event["confidence"] <= 1.0
    assert event["feature_fingerprint"]
    assert event["feature_values"]
    assert event["invalid_state"] is False


def test_monitoring_summary_and_drift_report_cover_confidence_and_feature_drift() -> None:
    baseline = [
        event_from_payload(valid_payload(pot=2.5), request_id="baseline-1", latency_ms=18.0),
        event_from_payload(valid_payload(pot=4.0, to_call=0.0), request_id="baseline-2", latency_ms=22.0),
    ]
    current = [
        event_from_payload(valid_payload(pot=2.7), request_id="current-1", latency_ms=19.0),
        event_from_payload(valid_payload(pot=4.2, to_call=0.0), request_id="current-2", latency_ms=24.0),
    ]

    summary = monitoring_summary(current)
    report = drift_report(baseline, current)

    assert summary["latency"]["p95_ms"] > 0.0
    assert summary["invalid_states"]["rate"] == 0.0
    assert "confidence_mean_delta" in report["drift"]
    assert report["status"] == "PASS"


def test_prediction_logs_and_audit_trail_are_jsonl_contracts() -> None:
    with tempfile.TemporaryDirectory() as raw_temp:
        temp = Path(raw_temp)
        prediction_log = temp / "prediction_logs.jsonl"
        audit_trail = temp / "audit_trail.jsonl"
        event = event_from_payload(valid_payload(), request_id="request-2")
        audit = audit_trail_event(
            request_id="request-2",
            event_type="prediction_recorded",
            payload={"prediction_log_hash": event["feature_fingerprint"]},
        )
        append_jsonl(prediction_log, event)
        append_jsonl(audit_trail, audit)

        assert read_jsonl(prediction_log)[0]["schema_version"] == PREDICTION_LOG_SCHEMA_VERSION
        assert read_jsonl(audit_trail)[0]["schema_version"] == AUDIT_TRAIL_SCHEMA_VERSION
        assert read_jsonl(audit_trail)[0]["event_hash"]


def test_repo_monitoring_contract_files_are_present() -> None:
    contract = describe_monitoring_contract()
    checks = validate_monitoring_contract(ROOT)

    assert contract["schema_version"] == MONITORING_CONTRACT_VERSION
    assert {"latency", "invalid_states", "confidence_drift", "feature_drift", "prediction_logs", "audit_trail"}.issubset(
        set(contract["signals"])
    )
    assert not [check for check in checks if not check["passed"]]
