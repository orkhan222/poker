from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.agents import RuleBasedAgent
from poker_agent.monitoring import (
    MonitoringThresholds,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prediction monitoring, drift, logs, and audit trail contract")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--prediction-log-out", default=Path("reports/prediction_logs.jsonl"), type=Path)
    parser.add_argument("--audit-trail-out", default=Path("reports/audit_trail.jsonl"), type=Path)
    parser.add_argument("--out", default=Path("reports/monitoring_report.json"), type=Path)
    parser.add_argument("--latency-p95-ms-max", type=float, default=150.0)
    parser.add_argument("--invalid-state-rate-max", type=float, default=0.0)
    parser.add_argument("--confidence-mean-delta-max", type=float, default=0.20)
    parser.add_argument("--feature-mean-delta-max", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def sample_payload(*, pot: float, to_call: float, stack: float = 100.0) -> dict[str, Any]:
    return {
        "position": "BTN",
        "street": "preflop",
        "hole_cards": ["Ah", "Kd"],
        "board_cards": [],
        "pot": pot,
        "current_bet": to_call,
        "to_call": to_call,
        "amount_to_call": to_call,
        "stack": stack,
        "effective_stack": stack,
        "small_blind": 0.5,
        "big_blind": 1.0,
        "button_position": "BTN",
        "dealer_position": "BTN",
        "action_order": ["UTG", "MP", "CO", "BTN", "SB", "BB"],
        "min_raise": 2.0,
        "max_raise": stack,
    }


def prediction_event_from_payload(payload: dict[str, Any], *, request_id: str, latency_ms: float) -> dict[str, Any]:
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


def smoke_events() -> dict[str, Any]:
    baseline = [
        prediction_event_from_payload(sample_payload(pot=2.5, to_call=1.0), request_id="baseline-1", latency_ms=18.0),
        prediction_event_from_payload(sample_payload(pot=4.0, to_call=0.0), request_id="baseline-2", latency_ms=22.0),
    ]
    current = [
        prediction_event_from_payload(sample_payload(pot=2.7, to_call=1.0), request_id="current-1", latency_ms=19.0),
        prediction_event_from_payload(sample_payload(pot=4.2, to_call=0.0), request_id="current-2", latency_ms=24.0),
    ]
    invalid_payload = sample_payload(pot=-1.0, to_call=250.0, stack=100.0)
    invalid_payload["street"] = "showdown"
    invalid_request = PredictionRequest.from_dict(invalid_payload)
    invalid = prediction_log_event(
        request_id="invalid-state-1",
        raw_payload=invalid_payload,
        request=invalid_request,
        response=None,
        latency_ms=5.0,
        status="error",
        error_code="INVALID_REQUEST",
    )
    return {"baseline": baseline, "current": current, "invalid_sample": invalid}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    prediction_log = resolve(root, args.prediction_log_out)
    audit_trail = resolve(root, args.audit_trail_out)
    out = resolve(root, args.out)
    thresholds = MonitoringThresholds(
        latency_p95_ms_max=args.latency_p95_ms_max,
        invalid_state_rate_max=args.invalid_state_rate_max,
        confidence_mean_delta_max=args.confidence_mean_delta_max,
        feature_mean_delta_max=args.feature_mean_delta_max,
    )

    checks = validate_monitoring_contract(root)
    artifacts: dict[str, Any] = {}
    if args.smoke:
        events = smoke_events()
        for event in [*events["current"], events["invalid_sample"]]:
            append_jsonl(prediction_log, event)
            append_jsonl(
                audit_trail,
                audit_trail_event(
                    request_id=str(event["request_id"]),
                    event_type="prediction_recorded",
                    payload={
                        "status": event["status"],
                        "invalid_state": event["invalid_state"],
                        "prediction_log_hash": event["feature_fingerprint"],
                    },
                ),
            )
        report = drift_report(events["baseline"], events["current"], thresholds)
        invalid_findings = invalid_state_findings(
            {
                "position": "",
                "street": "showdown",
                "pot": -10,
                "hole_cards": ["Ah", "Kd", "Qs"],
            }
        )
        checks.extend(
            [
                {"name": "latency:p95", "passed": report["checks"][0]["passed"], "detail": report["checks"][0]},
                {"name": "invalid_states:detected", "passed": bool(invalid_findings), "detail": invalid_findings},
                {"name": "confidence_drift", "passed": report["checks"][2]["passed"], "detail": report["checks"][2]},
                {"name": "feature_drift", "passed": report["checks"][3]["passed"], "detail": report["checks"][3]},
                {"name": "prediction_logs:written", "passed": len(read_jsonl(prediction_log)) >= 3, "detail": str(prediction_log)},
                {"name": "audit_trail:written", "passed": len(read_jsonl(audit_trail)) >= 3, "detail": str(audit_trail)},
            ]
        )
        artifacts = {
            "drift_report": report,
            "prediction_log": str(prediction_log),
            "audit_trail": str(audit_trail),
            "logged_summary": monitoring_summary(read_jsonl(prediction_log)),
            "invalid_sample": events["invalid_sample"],
        }

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    payload = {
        "status": status,
        "contract": describe_monitoring_contract(),
        "checks": checks,
        "artifacts": artifacts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
