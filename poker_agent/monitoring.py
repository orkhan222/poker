from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poker_agent.api_contract import PREDICT_RESPONSE_SCHEMA_VERSION
from poker_agent.features import request_to_features
from poker_agent.mlops import stable_digest, utc_now
from poker_agent.schemas import PredictionRequest

MONITORING_CONTRACT_VERSION = "monitoring_contract.v1"
PREDICTION_LOG_SCHEMA_VERSION = "prediction_log.v1"
AUDIT_TRAIL_SCHEMA_VERSION = "audit_trail.v1"

DEFAULT_MONITORING_PATHS = {
    "prediction_log": "reports/prediction_logs.jsonl",
    "audit_trail": "reports/audit_trail.jsonl",
    "monitoring_report": "reports/monitoring_report.json",
}


@dataclass(frozen=True)
class MonitoringThresholds:
    latency_p95_ms_max: float = 150.0
    invalid_state_rate_max: float = 0.0
    confidence_mean_delta_max: float = 0.20
    feature_mean_delta_max: float = 3.0
    min_events: int = 1


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def invalid_state_findings(raw_payload: dict[str, Any], request: PredictionRequest | None = None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    street = str(raw_payload.get("street") or getattr(request, "street", "") or "").lower()
    if street and street not in {"preflop", "flop", "turn", "river"}:
        findings.append({"code": "invalid_street", "field": "street", "value": street})
    if not (raw_payload.get("position") or raw_payload.get("player_position") or getattr(request, "position", "")):
        findings.append({"code": "missing_position", "field": "position"})

    for field in ("pot", "pot_size", "current_bet", "to_call", "amount_to_call", "stack", "effective_stack"):
        if field not in raw_payload or raw_payload.get(field) in (None, ""):
            continue
        try:
            value = float(raw_payload[field])
        except (TypeError, ValueError):
            findings.append({"code": "non_numeric_state", "field": field, "value": str(raw_payload[field])})
            continue
        if value < 0:
            findings.append({"code": "negative_state_value", "field": field, "value": value})

    hole_cards = raw_payload.get("hole_cards", getattr(request, "hole_cards", []))
    board_cards = raw_payload.get("board_cards", getattr(request, "board_cards", []))
    if isinstance(hole_cards, list) and len(hole_cards) > 2:
        findings.append({"code": "too_many_hole_cards", "field": "hole_cards", "value": len(hole_cards)})
    if isinstance(board_cards, list) and len(board_cards) > 5:
        findings.append({"code": "too_many_board_cards", "field": "board_cards", "value": len(board_cards)})

    if request is not None:
        if request.amount_to_call > request.effective_stack > 0:
            findings.append(
                {
                    "code": "call_exceeds_effective_stack",
                    "field": "amount_to_call",
                    "value": request.amount_to_call,
                }
            )
        if not request.legal_actions:
            findings.append({"code": "empty_legal_actions", "field": "legal_actions"})
    return findings


def feature_snapshot(request: PredictionRequest) -> dict[str, Any]:
    features = request_to_features(request)
    numeric = {
        key: float(value)
        for key, value in features.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    return {
        "feature_count": len(numeric),
        "feature_fingerprint": stable_digest(numeric),
        "features": numeric,
    }


def prediction_log_event(
    *,
    request_id: str,
    raw_payload: dict[str, Any],
    request: PredictionRequest | None,
    response: dict[str, Any] | None,
    latency_ms: float,
    status: str,
    error_code: str | None = None,
) -> dict[str, Any]:
    state_findings = invalid_state_findings(raw_payload, request)
    features = feature_snapshot(request) if request is not None else {"feature_count": 0, "feature_fingerprint": "", "features": {}}
    probabilities = dict((response or {}).get("probabilities") or {})
    confidence = float((response or {}).get("confidence", 0.0) or 0.0)
    action = (response or {}).get("action")
    legal_actions = list((response or {}).get("legal_actions") or getattr(request, "legal_actions", []) or [])
    return {
        "schema_version": PREDICTION_LOG_SCHEMA_VERSION,
        "contract_version": MONITORING_CONTRACT_VERSION,
        "request_id": request_id,
        "created_at": utc_now(),
        "status": status,
        "error_code": error_code,
        "latency_ms": round(float(latency_ms), 4),
        "model_version": (response or {}).get("model_version", "unknown"),
        "response_schema_version": (response or {}).get("schema_version", PREDICT_RESPONSE_SCHEMA_VERSION),
        "action": action,
        "legal_actions": legal_actions,
        "invalid_state": bool(state_findings),
        "invalid_state_findings": state_findings,
        "confidence": confidence,
        "probabilities": probabilities,
        "feature_fingerprint": features["feature_fingerprint"],
        "feature_count": features["feature_count"],
        "feature_values": features["features"],
        "state_context": (response or {}).get("state_context", request.state_context() if request is not None else {}),
    }


def audit_trail_event(
    *,
    request_id: str,
    event_type: str,
    actor: str = "poker-decision-agent",
    payload: dict[str, Any] | None = None,
    previous_hash: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": AUDIT_TRAIL_SCHEMA_VERSION,
        "request_id": request_id,
        "created_at": utc_now(),
        "actor": actor,
        "event_type": event_type,
        "payload": payload or {},
        "previous_hash": previous_hash,
    }
    body["event_hash"] = stable_digest(body)
    return body


def append_prediction_monitoring(
    *,
    prediction_log_path: Path,
    audit_trail_path: Path,
    request_id: str,
    raw_payload: dict[str, Any],
    request: PredictionRequest | None,
    response: dict[str, Any] | None,
    latency_ms: float,
    status: str,
    error_code: str | None = None,
) -> dict[str, Any]:
    event = prediction_log_event(
        request_id=request_id,
        raw_payload=raw_payload,
        request=request,
        response=response,
        latency_ms=latency_ms,
        status=status,
        error_code=error_code,
    )
    append_jsonl(prediction_log_path, event)
    audit_event = audit_trail_event(
        request_id=request_id,
        event_type="prediction_recorded",
        payload={
            "status": status,
            "error_code": error_code,
            "prediction_log_hash": stable_digest(event),
        },
    )
    append_jsonl(audit_trail_path, audit_event)
    return event


def monitoring_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(event.get("latency_ms", 0.0) or 0.0) for event in events]
    confidences = [float(event.get("confidence", 0.0) or 0.0) for event in events if event.get("status") == "ok"]
    invalid = [event for event in events if event.get("invalid_state")]
    feature_sums: dict[str, float] = {}
    feature_counts: dict[str, int] = {}
    for event in events:
        values = event.get("feature_values") or {}
        if not isinstance(values, dict):
            continue
        for key, raw_value in values.items():
            if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
                feature_sums[key] = feature_sums.get(key, 0.0) + float(raw_value)
                feature_counts[key] = feature_counts.get(key, 0) + 1
    feature_means = {
        key: feature_sums[key] / feature_counts[key]
        for key in sorted(feature_sums)
        if feature_counts.get(key, 0) > 0
    }
    return {
        "count": len(events),
        "latency": {
            "p50_ms": percentile(latencies, 0.50),
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "max_ms": max(latencies) if latencies else 0.0,
        },
        "invalid_states": {
            "count": len(invalid),
            "rate": len(invalid) / len(events) if events else 0.0,
            "codes": sorted(
                {
                    finding.get("code")
                    for event in invalid
                    for finding in event.get("invalid_state_findings", [])
                    if finding.get("code")
                }
            ),
        },
        "confidence": {
            "mean": sum(confidences) / len(confidences) if confidences else 0.0,
            "min": min(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0,
        },
        "feature_means": feature_means,
    }


def drift_report(
    baseline_events: list[dict[str, Any]],
    current_events: list[dict[str, Any]],
    thresholds: MonitoringThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or MonitoringThresholds()
    baseline = monitoring_summary(baseline_events)
    current = monitoring_summary(current_events)
    baseline_confidence = float(baseline["confidence"]["mean"])
    current_confidence = float(current["confidence"]["mean"])
    confidence_delta = current_confidence - baseline_confidence

    baseline_features = baseline["feature_means"]
    current_features = current["feature_means"]
    shared = sorted(set(baseline_features) & set(current_features))
    feature_deltas = {
        key: float(current_features[key]) - float(baseline_features[key])
        for key in shared
    }
    max_feature_delta = max((abs(value) for value in feature_deltas.values()), default=0.0)
    checks = [
        {
            "name": "latency_p95_ms",
            "passed": float(current["latency"]["p95_ms"]) <= thresholds.latency_p95_ms_max,
            "observed": current["latency"]["p95_ms"],
            "threshold": thresholds.latency_p95_ms_max,
        },
        {
            "name": "invalid_state_rate",
            "passed": float(current["invalid_states"]["rate"]) <= thresholds.invalid_state_rate_max,
            "observed": current["invalid_states"]["rate"],
            "threshold": thresholds.invalid_state_rate_max,
        },
        {
            "name": "confidence_drift",
            "passed": abs(confidence_delta) <= thresholds.confidence_mean_delta_max,
            "observed": confidence_delta,
            "threshold": thresholds.confidence_mean_delta_max,
        },
        {
            "name": "feature_drift",
            "passed": max_feature_delta <= thresholds.feature_mean_delta_max,
            "observed": max_feature_delta,
            "threshold": thresholds.feature_mean_delta_max,
        },
    ]
    return {
        "schema_version": MONITORING_CONTRACT_VERSION,
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "baseline": baseline,
        "current": current,
        "drift": {
            "confidence_mean_delta": confidence_delta,
            "feature_mean_deltas": feature_deltas,
            "max_abs_feature_mean_delta": max_feature_delta,
        },
    }


def describe_monitoring_contract() -> dict[str, Any]:
    return {
        "schema_version": MONITORING_CONTRACT_VERSION,
        "signals": [
            "latency",
            "invalid_states",
            "confidence_drift",
            "feature_drift",
            "prediction_logs",
            "audit_trail",
        ],
        "paths": DEFAULT_MONITORING_PATHS,
        "prediction_log_schema": {
            "schema_version": PREDICTION_LOG_SCHEMA_VERSION,
            "required_fields": [
                "request_id",
                "created_at",
                "latency_ms",
                "invalid_state",
                "confidence",
                "probabilities",
                "feature_fingerprint",
                "feature_values",
            ],
        },
        "audit_trail_schema": {
            "schema_version": AUDIT_TRAIL_SCHEMA_VERSION,
            "required_fields": ["request_id", "event_type", "event_hash", "previous_hash", "payload"],
        },
        "thresholds": MonitoringThresholds().__dict__,
    }


def validate_monitoring_contract(root: Path) -> list[dict[str, Any]]:
    expected = [
        "poker_agent/monitoring.py",
        "scripts/check_monitoring_contract.py",
        "configs/monitoring/local.yaml",
        "configs/experiments/monitoring_smoke.yaml",
        "tests/test_monitoring_contract.py",
    ]
    checks = [
        {"name": f"file:{relative}", "passed": (root / relative).exists(), "detail": relative}
        for relative in expected
    ]
    return checks
