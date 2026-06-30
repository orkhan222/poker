from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

PRODUCTION_RUNTIME_MONITORING_VERSION = "2026-06-28"
REAL_TRAFFIC_REQUIREMENT = "REQUIRES_MONITORING_ROLLBACK_AND_LIVE_DRIFT_TRACKING"
READY_TO_ENABLE = "CONFIGURED_FOR_REAL_TRAFFIC_ENABLEMENT"
REAL_TRAFFIC_NOT_APPROVED_UNTIL_OBSERVABILITY = "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED"
BASELINE_ACTION_DISTRIBUTION = {
    "fold": 0.26,
    "check": 0.18,
    "call": 0.30,
    "bet": 0.14,
    "raise": 0.12,
}
ROLLBACK_THRESHOLDS = {
    "min_sample_count": 25,
    "max_error_rate": 0.02,
    "max_fallback_rate": 0.20,
    "max_p95_latency_ms": 750.0,
    "max_action_distribution_js": 0.08,
    "max_low_confidence_rate": 0.30,
}


class RuntimeMonitoringState:
    def __init__(self, max_events: int = 1000) -> None:
        self.max_events = max_events
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = Lock()

    def observe_prediction(
        self,
        result: Mapping[str, Any],
        latency_ms: float,
        request_payload: Mapping[str, Any] | None = None,
    ) -> None:
        probabilities = result.get("probabilities") if isinstance(result.get("probabilities"), Mapping) else {}
        clean_probabilities = _clean_probabilities(probabilities)
        confidence = _confidence(clean_probabilities)
        action = str(result.get("action") or result.get("recommended_action") or "unknown")
        model_status = str(result.get("model_status") or result.get("model_version") or "unknown")
        event = {
            "ts": time.time(),
            "kind": "prediction",
            "action": action,
            "confidence": confidence,
            "probabilities": clean_probabilities,
            "latency_ms": float(latency_ms),
            "model_status": model_status,
            "fallback": "fallback" in model_status.lower(),
            "street": _payload_value(request_payload, "street"),
            "position": _payload_value(request_payload, "position"),
            "missing_hole_cards": not bool(_payload_value(request_payload, "hole_cards")),
            "error": False,
        }
        with self._lock:
            self._events.append(event)

    def observe_error(self, latency_ms: float, error_type: str) -> None:
        with self._lock:
            self._events.append(
                {
                    "ts": time.time(),
                    "kind": "error",
                    "action": "error",
                    "confidence": 0.0,
                    "probabilities": {},
                    "latency_ms": float(latency_ms),
                    "model_status": error_type,
                    "fallback": False,
                    "street": None,
                    "position": None,
                    "missing_hole_cards": None,
                    "error": True,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        return build_runtime_snapshot(events)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


runtime_monitoring_state = RuntimeMonitoringState()


def build_runtime_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(events)
    predictions = [event for event in events if not event.get("error")]
    errors = [event for event in events if event.get("error")]
    action_counts = Counter(str(event.get("action")) for event in predictions)
    distribution = _normalize_counts(action_counts, BASELINE_ACTION_DISTRIBUTION.keys())
    latencies = sorted(float(event.get("latency_ms", 0.0)) for event in events)
    confidences = sorted(float(event.get("confidence", 0.0)) for event in predictions)
    fallback_count = sum(1 for event in predictions if event.get("fallback"))
    low_confidence_count = sum(1 for event in predictions if float(event.get("confidence", 0.0)) < 0.40)
    missing_card_count = sum(1 for event in predictions if event.get("missing_hole_cards") is True)
    snapshot = {
        "event_count": total,
        "prediction_count": len(predictions),
        "error_count": len(errors),
        "action_distribution": distribution,
        "prediction_distribution_tracking": {
            "status": "ACTIVE" if predictions else "CONFIGURED_NO_LIVE_TRAFFIC",
            "sample_count": len(predictions),
            "action_counts": {label: int(action_counts.get(label, 0)) for label in BASELINE_ACTION_DISTRIBUTION},
            "action_distribution": distribution,
            "probability_mean_by_action": _mean_probabilities(predictions),
        },
        "model_confidence_monitoring": {
            "status": "ACTIVE" if predictions else "CONFIGURED_NO_LIVE_TRAFFIC",
            "sample_count": len(predictions),
            "avg_confidence": _mean(confidences),
            "p05_confidence": _percentile(confidences, 0.05),
            "p50_confidence": _percentile(confidences, 0.50),
            "p95_confidence": _percentile(confidences, 0.95),
            "low_confidence_rate": _ratio(low_confidence_count, len(predictions)),
        },
        "error_rate": _ratio(len(errors), total),
        "fallback_rate": _ratio(fallback_count, len(predictions)),
        "low_confidence_rate": _ratio(low_confidence_count, len(predictions)),
        "missing_hole_card_rate": _ratio(missing_card_count, len(predictions)),
        "avg_latency_ms": _mean(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "action_distribution_js": _js_divergence(distribution, BASELINE_ACTION_DISTRIBUTION),
    }
    snapshot["rollback_evaluation"] = evaluate_rollback(snapshot)
    return snapshot


def evaluate_rollback(snapshot: Mapping[str, Any], thresholds: Mapping[str, float] | None = None) -> dict[str, Any]:
    thresholds = thresholds or ROLLBACK_THRESHOLDS
    triggers: list[dict[str, Any]] = []
    sample_count = int(snapshot.get("prediction_count") or 0)
    if sample_count < int(thresholds["min_sample_count"]):
        return {
            "status": "INSUFFICIENT_LIVE_TRAFFIC",
            "rollback_required": False,
            "sample_count": sample_count,
            "min_sample_count": int(thresholds["min_sample_count"]),
            "triggers": [],
        }
    _add_trigger(triggers, "error_rate", snapshot.get("error_rate"), thresholds["max_error_rate"])
    _add_trigger(triggers, "fallback_rate", snapshot.get("fallback_rate"), thresholds["max_fallback_rate"])
    _add_trigger(triggers, "p95_latency_ms", snapshot.get("p95_latency_ms"), thresholds["max_p95_latency_ms"])
    _add_trigger(triggers, "action_distribution_js", snapshot.get("action_distribution_js"), thresholds["max_action_distribution_js"])
    _add_trigger(triggers, "low_confidence_rate", snapshot.get("low_confidence_rate"), thresholds["max_low_confidence_rate"])
    return {
        "status": "ROLLBACK_REQUIRED" if triggers else "PASS",
        "rollback_required": bool(triggers),
        "sample_count": sample_count,
        "triggers": triggers,
    }


def build_production_runtime_monitoring(
    project_root: Path,
    runtime_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reports = project_root / "reports"
    final_acceptance = _read_optional_json(reports / "final_delivery_acceptance.json")
    production_approval = _read_optional_json(reports / "production_approval.json")
    strategy_maturity = _read_optional_json(reports / "strategy_stack_maturity.json")
    deployed_gate = _read_optional_json(reports / "deployed_strategy_gate.json")

    final_summary = final_acceptance.get("acceptance_summary") or {}
    maturity = strategy_maturity.get("current_strategy_stack") or {}
    runtime_snapshot = dict(runtime_snapshot or runtime_monitoring_state.snapshot())

    payload: dict[str, Any] = {
        "version": PRODUCTION_RUNTIME_MONITORING_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Production monitoring, rollback, and live drift tracking contract",
        "client_statement": (
            "Production monitoring, rollback rules, and live drift tracking must be enabled when the service "
            "is deployed against real traffic. The current delivery package is ready, but real-traffic rollout "
            "requires active operational telemetry and rollback enforcement."
        ),
        "runtime_observability_boundary": {
            "status": REAL_TRAFFIC_REQUIREMENT,
            "implementation_status": READY_TO_ENABLE,
            "monitoring_required_for_real_traffic": True,
            "rollback_rules_required_for_real_traffic": True,
            "live_drift_tracking_required_for_real_traffic": True,
            "prediction_distribution_tracking_required_for_real_traffic": True,
            "model_confidence_monitoring_required_for_real_traffic": True,
            "real_traffic_claim_allowed_without_observability": False,
            "real_production_traffic_approved": False,
            "real_production_traffic_approval_status": REAL_TRAFFIC_NOT_APPROVED_UNTIL_OBSERVABILITY,
            "real_traffic_blocker_if_disabled": True,
            "current_delivery_blocker": False,
            "service_delivery": final_summary.get("service_delivery"),
            "deployed_strategy_stack": final_summary.get("deployed_strategy_stack"),
            "production_approval_status": production_approval.get("overall_status"),
            "strategy_deployment_mode": maturity.get("deployment_mode"),
            "deployed_gate_status": deployed_gate.get("status"),
        },
        "monitoring_plan": {
            "required_streams": [
                "request_count",
                "prediction_latency_ms",
                "action_distribution",
                "probability_distribution",
                "confidence_distribution",
                "model_status_and_fallback_rate",
                "validation_error_rate",
                "missing_hole_card_rate",
                "bet_size_distribution",
                "wait_time_distribution",
                "realized_outcome_when_available",
            ],
            "required_labels": [
                "model_version",
                "policy_type",
                "street",
                "position",
                "action",
                "model_status",
                "deployment_environment",
            ],
            "windows": ["15m", "1h", "24h", "7d"],
            "minimum_live_samples_before_drift_decision": ROLLBACK_THRESHOLDS["min_sample_count"],
        },
        "rollback_rules": {
            "thresholds": dict(ROLLBACK_THRESHOLDS),
            "rules": [
                {"name": "model_load_failure", "action": "rollback_to_last_known_good_artifact"},
                {"name": "api_error_rate", "metric": "error_rate", "max": ROLLBACK_THRESHOLDS["max_error_rate"]},
                {"name": "latency_regression", "metric": "p95_latency_ms", "max": ROLLBACK_THRESHOLDS["max_p95_latency_ms"]},
                {"name": "fallback_rate_spike", "metric": "fallback_rate", "max": ROLLBACK_THRESHOLDS["max_fallback_rate"]},
                {"name": "action_distribution_drift", "metric": "action_distribution_js", "max": ROLLBACK_THRESHOLDS["max_action_distribution_js"]},
                {"name": "confidence_shift", "metric": "low_confidence_rate", "max": ROLLBACK_THRESHOLDS["max_low_confidence_rate"]},
            ],
            "rollback_target": "previous approved deployed strategy stack or rule-based safe fallback",
            "manual_review_required_after_trigger": True,
        },
        "live_drift_tracking": {
            "baseline_action_distribution": dict(BASELINE_ACTION_DISTRIBUTION),
            "metrics": [
                "action_distribution_js",
                "confidence_distribution",
                "latency_ms",
                "fallback_rate",
                "missing_hole_card_rate",
                "bet_size_distribution",
                "wait_time_distribution",
                "street_position_slice_drift",
            ],
            "slice_keys": ["street", "position", "model_status", "missing_hole_cards"],
            "storage_requirement": "Persist request-level telemetry outside process memory in production.",
        },
        "runtime_snapshot": runtime_snapshot,
        "allowed_claims": [
            "The delivery package includes a production monitoring and rollback contract.",
            "Real-traffic rollout is allowed only with active monitoring, rollback, and drift tracking.",
            "Real-traffic approval additionally requires prediction-distribution tracking and model-confidence monitoring.",
            "The in-process snapshot is for service-local visibility; production must persist telemetry externally.",
        ],
        "blocked_claims": [
            "The service is approved for real traffic without monitoring.",
            "The service is approved for real production traffic before observability is enabled.",
            "Rollback is optional for production rollout.",
            "Offline validation alone replaces live drift tracking.",
            "Prediction distribution and model confidence monitoring are optional for real traffic.",
            "In-process telemetry is sufficient as the only production monitoring store.",
        ],
        "evidence": {
            "final_delivery_acceptance": "reports/final_delivery_acceptance.json",
            "production_approval": "reports/production_approval.json",
            "strategy_stack_maturity": "reports/strategy_stack_maturity.json",
            "deployed_strategy_gate": "reports/deployed_strategy_gate.json",
        },
    }
    payload["invariants"] = validate_production_runtime_monitoring(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_production_runtime_monitoring(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    boundary = payload.get("runtime_observability_boundary") or {}
    plan = payload.get("monitoring_plan") or {}
    rollback = payload.get("rollback_rules") or {}
    drift = payload.get("live_drift_tracking") or {}
    rollback_names = {rule.get("name") for rule in rollback.get("rules", []) if isinstance(rule, Mapping)}
    drift_metrics = set(drift.get("metrics") or [])

    if payload.get("overall_status") == "PASS":
        violations.append("overall_status_must_be_assigned_after_invariant_validation")
    if boundary.get("status") != REAL_TRAFFIC_REQUIREMENT:
        violations.append("real_traffic_boundary_must_require_monitoring_rollback_and_drift")
    if boundary.get("monitoring_required_for_real_traffic") is not True:
        violations.append("monitoring_must_be_required_for_real_traffic")
    if boundary.get("rollback_rules_required_for_real_traffic") is not True:
        violations.append("rollback_rules_must_be_required_for_real_traffic")
    if boundary.get("live_drift_tracking_required_for_real_traffic") is not True:
        violations.append("live_drift_tracking_must_be_required_for_real_traffic")
    if boundary.get("prediction_distribution_tracking_required_for_real_traffic") is not True:
        violations.append("prediction_distribution_tracking_must_be_required_for_real_traffic")
    if boundary.get("model_confidence_monitoring_required_for_real_traffic") is not True:
        violations.append("model_confidence_monitoring_must_be_required_for_real_traffic")
    if boundary.get("real_traffic_claim_allowed_without_observability") is not False:
        violations.append("unmonitored_real_traffic_claim_must_be_blocked")
    if boundary.get("real_production_traffic_approved") is not False:
        violations.append("real_production_traffic_must_not_be_approved_without_enabled_observability")
    if boundary.get("real_production_traffic_approval_status") != REAL_TRAFFIC_NOT_APPROVED_UNTIL_OBSERVABILITY:
        violations.append("real_production_traffic_status_must_require_enabled_observability")
    if boundary.get("real_traffic_blocker_if_disabled") is not True:
        violations.append("disabled_observability_must_block_real_traffic_rollout")
    if boundary.get("current_delivery_blocker") is not False:
        violations.append("monitoring_contract_must_not_block_current_delivery_package")

    required_streams = {
        "prediction_latency_ms",
        "action_distribution",
        "probability_distribution",
        "confidence_distribution",
        "model_status_and_fallback_rate",
        "validation_error_rate",
    }
    if not required_streams.issubset(set(plan.get("required_streams") or [])):
        violations.append("monitoring_plan_missing_required_streams")

    required_rules = {"model_load_failure", "api_error_rate", "latency_regression", "fallback_rate_spike", "action_distribution_drift", "confidence_shift"}
    if not required_rules.issubset(rollback_names):
        violations.append("rollback_rules_missing_required_triggers")

    required_drift = {"action_distribution_js", "confidence_distribution", "latency_ms", "fallback_rate", "missing_hole_card_rate"}
    if not required_drift.issubset(drift_metrics):
        violations.append("live_drift_tracking_missing_required_metrics")
    if "external" not in str(drift.get("storage_requirement", "")).lower() and "persist" not in str(drift.get("storage_requirement", "")).lower():
        violations.append("production_telemetry_must_require_external_persistence")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_production_runtime_monitoring(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    runtime_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_production_runtime_monitoring(project_root, runtime_snapshot=runtime_snapshot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_production_runtime_monitoring_markdown(payload), encoding="utf-8")
    return payload


def render_production_runtime_monitoring_markdown(payload: Mapping[str, Any]) -> str:
    boundary = payload["runtime_observability_boundary"]
    rollback = payload["rollback_rules"]
    drift = payload["live_drift_tracking"]
    snapshot = payload["runtime_snapshot"]
    lines = [
        "# Production Runtime Monitoring Contract",
        "",
        payload["client_statement"],
        "",
        "## Runtime Boundary",
        "",
        f"- Status: `{boundary['status']}`",
        f"- Monitoring required for real traffic: `{boundary['monitoring_required_for_real_traffic']}`",
        f"- Rollback rules required for real traffic: `{boundary['rollback_rules_required_for_real_traffic']}`",
        f"- Live drift tracking required for real traffic: `{boundary['live_drift_tracking_required_for_real_traffic']}`",
        f"- Prediction distribution tracking required for real traffic: `{boundary['prediction_distribution_tracking_required_for_real_traffic']}`",
        f"- Model confidence monitoring required for real traffic: `{boundary['model_confidence_monitoring_required_for_real_traffic']}`",
        f"- Real production traffic approved: `{boundary['real_production_traffic_approved']}`",
        f"- Real production traffic approval status: `{boundary['real_production_traffic_approval_status']}`",
        f"- Real traffic blocker if disabled: `{boundary['real_traffic_blocker_if_disabled']}`",
        f"- Current delivery blocker: `{boundary['current_delivery_blocker']}`",
        "",
        "## Rollback Rules",
        "",
    ]
    lines.extend(f"- `{rule['name']}`" for rule in rollback["rules"])
    lines.extend([
        "",
        "## Live Drift Metrics",
        "",
    ])
    lines.extend(f"- `{metric}`" for metric in drift["metrics"])
    lines.extend([
        "",
        "## Runtime Snapshot",
        "",
        f"- Prediction count: `{snapshot['prediction_count']}`",
        f"- Error rate: `{snapshot['error_rate']}`",
        f"- Fallback rate: `{snapshot['fallback_rate']}`",
        f"- Action distribution JS: `{snapshot['action_distribution_js']}`",
        f"- Prediction distribution tracking: `{snapshot['prediction_distribution_tracking']['status']}`",
        f"- Confidence monitoring: `{snapshot['model_confidence_monitoring']['status']}`",
        f"- Low confidence rate: `{snapshot['model_confidence_monitoring']['low_confidence_rate']}`",
        f"- Rollback status: `{snapshot['rollback_evaluation']['status']}`",
        "",
        f"Invariant status: `{payload['invariants']['status']}`",
        "",
    ])
    return "\n".join(lines)


def _confidence(probabilities: Mapping[str, Any]) -> float:
    values: list[float] = []
    for value in probabilities.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _clean_probabilities(probabilities: Mapping[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for label in BASELINE_ACTION_DISTRIBUTION:
        try:
            clean[label] = max(float(probabilities.get(label, 0.0)), 0.0)
        except (TypeError, ValueError):
            clean[label] = 0.0
    total = sum(clean.values())
    if total <= 0.0:
        return {label: 0.0 for label in BASELINE_ACTION_DISTRIBUTION}
    return {label: value / total for label, value in clean.items()}


def _payload_value(payload: Mapping[str, Any] | None, key: str) -> Any:
    if payload is None:
        return None
    if key in payload:
        return payload[key]
    game_state = payload.get("game_state") if isinstance(payload, Mapping) else None
    if isinstance(game_state, Mapping):
        return game_state.get(key)
    return None


def _normalize_counts(counts: Counter[str], labels: Any) -> dict[str, float]:
    labels = list(labels)
    total = sum(counts.get(label, 0) for label in labels)
    if total <= 0:
        return {label: 0.0 for label in labels}
    return {label: counts.get(label, 0) / total for label in labels}


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_probabilities(events: list[dict[str, Any]]) -> dict[str, float]:
    if not events:
        return {label: 0.0 for label in BASELINE_ACTION_DISTRIBUTION}
    totals = {label: 0.0 for label in BASELINE_ACTION_DISTRIBUTION}
    for event in events:
        probabilities = event.get("probabilities") if isinstance(event.get("probabilities"), Mapping) else {}
        for label in totals:
            try:
                totals[label] += float(probabilities.get(label, 0.0))
            except (TypeError, ValueError):
                continue
    return {label: value / len(events) for label, value in totals.items()}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return float(values[index])


def _js_divergence(observed: Mapping[str, float], baseline: Mapping[str, float]) -> float:
    keys = set(observed) | set(baseline)
    p = {key: max(float(observed.get(key, 0.0)), 0.0) for key in keys}
    q = {key: max(float(baseline.get(key, 0.0)), 0.0) for key in keys}
    p = _renormalize(p)
    q = _renormalize(q)
    m = {key: 0.5 * (p[key] + q[key]) for key in keys}
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _renormalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        return {key: 0.0 for key in values}
    return {key: value / total for key, value in values.items()}


def _kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    total = 0.0
    for key, value in p.items():
        if value <= 0.0:
            continue
        denom = max(float(q.get(key, 0.0)), 1e-12)
        total += value * math.log(value / denom, 2)
    return float(total)


def _add_trigger(triggers: list[dict[str, Any]], metric: str, observed: Any, threshold: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return
    if value > float(threshold):
        triggers.append({"metric": metric, "observed": value, "threshold": float(threshold)})


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
