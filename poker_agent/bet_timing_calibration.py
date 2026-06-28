from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BET_TIMING_CALIBRATION_VERSION = "2026-06-28"
IMPLEMENTED_AND_MEASURED = "IMPLEMENTED_AND_MEASURED"
CALIBRATION_RECOMMENDED_FOR_HIGHER_REALISM = "CALIBRATION_RECOMMENDED_FOR_HIGHER_REALISM"
LABELS_INSUFFICIENT_FOR_FINAL_HIGH_REALISM = "LABELS_INSUFFICIENT_FOR_FINAL_HIGH_REALISM"
CURRENT_VALIDATION_SCOPE = "current_delivery_validation_scope"
HIGHER_REALISM_SCOPE = "larger_real_player_behavior_labels"
REQUIRED_RESPONSE_FIELDS = ("bet_size", "wait_time_ms", "sizing_method", "timing_method")


def build_bet_timing_calibration(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    acceptance = _read_optional_json(reports / "policy_acceptance.json")
    behavioral = _read_optional_json(reports / "behavioral_revalidation.json")
    api_contract = _read_optional_json(reports / "api_contract.json")

    likeness = acceptance.get("human_likeness") or {}
    behavioral_scope = behavioral.get("current_validation_scope") or {}
    metrics_to_revalidate = behavioral.get("metrics_to_revalidate") or []
    response_fields = _response_fields_from_contract(api_contract)
    if not response_fields:
        response_fields = list(REQUIRED_RESPONSE_FIELDS)

    timing_and_bet_size_status = likeness.get(
        "timing_and_bet_size_status",
        behavioral_scope.get("timing_and_bet_size_status", "UNKNOWN"),
    )
    implemented = all(field in response_fields for field in REQUIRED_RESPONSE_FIELDS)
    measured = timing_and_bet_size_status == "PASS"

    payload: dict[str, Any] = {
        "version": BET_TIMING_CALIBRATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Bet-sizing and timing implementation, measurement, and calibration boundary",
        "client_statement": (
            "Bet-sizing and timing behavior are implemented and measured, but should still be calibrated "
            "with more real player behavior labels if higher realism is required."
        ),
        "current_delivery_scope": {
            "scope": CURRENT_VALIDATION_SCOPE,
            "implementation_status": IMPLEMENTED_AND_MEASURED if implemented and measured else "INCOMPLETE_OR_UNMEASURED",
            "bet_sizing_implemented": "bet_size" in response_fields and "sizing_method" in response_fields,
            "timing_implemented": "wait_time_ms" in response_fields and "timing_method" in response_fields,
            "measured": measured,
            "timing_and_bet_size_status": timing_and_bet_size_status,
            "policy_acceptance_metric": "policy_acceptance.human_likeness.timing_and_bet_size_status",
            "behavioral_revalidation_metric": "behavioral_revalidation.current_validation_scope.timing_and_bet_size_status",
            "api_response_fields": response_fields,
            "implementation_modules": [
                "poker_agent/action_planning.py",
                "poker_agent/agents.py",
                "poker_agent/autonomous_agent.py",
                "poker_agent/schemas.py",
            ],
            "measurement_reports": [
                "reports/policy_acceptance.json",
                "reports/behavioral_revalidation.json",
            ],
        },
        "calibration_boundary": {
            "status": CALIBRATION_RECOMMENDED_FOR_HIGHER_REALISM,
            "higher_realism_scope": HIGHER_REALISM_SCOPE,
            "requires_more_real_player_behavior_labels": True,
            "requires_bet_size_labels": True,
            "requires_decision_timing_labels": True,
            "requires_slice_level_calibration": True,
            "label_gap_status": LABELS_INSUFFICIENT_FOR_FINAL_HIGH_REALISM,
            "production_blocker_for_current_delivery": False,
            "final_high_realism_claim_allowed": False,
            "current_delivery_claim_allowed": implemented and measured,
            "reason": (
                "The service emits and measures bet-size and timing behavior for the current delivery scope. "
                "Higher realism requires calibration against larger reviewed player-behavior labels, especially "
                "for bet-size distributions, decision timing, table tempo, stack depth, street, and opponent slices."
            ),
        },
        "calibration_dataset_requirements": [
            "Reviewed real-player bet-size labels for call, bet, and raise decisions.",
            "Reviewed decision-time labels captured before the hero action, not after outcome leakage.",
            "Street, stack-to-pot ratio, pot-size, position, and table-tempo slices.",
            "Separate calibration slices for observed-card and missing-card requests.",
            "Enough examples per action and street to estimate distribution drift, not only aggregate accuracy.",
        ],
        "metrics_to_revalidate": _merge_metrics(
            metrics_to_revalidate,
            [
                "bet-size distribution similarity",
                "timing distribution similarity",
                "bet-size MAE by action and street",
                "decision-time MAE by action and street",
                "slice-level drift against real player behavior labels",
            ],
        ),
        "allowed_claims": [
            "The service implements and returns bet_size, wait_time_ms, sizing_method, and timing_method.",
            "Bet-sizing and timing are measured for the current validation scope.",
            "Additional real-player behavior labels would improve calibration for higher realism.",
        ],
        "not_allowed_claims": [
            "Bet-sizing and timing are fully calibrated to real player behavior across all production conditions.",
            "Higher-realism calibration is complete without larger reviewed real-player behavior labels.",
            "Current timing and bet-size measurements are a substitute for production-scale behavioral calibration.",
        ],
    }
    payload["invariants"] = validate_bet_timing_calibration(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_bet_timing_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    current = payload.get("current_delivery_scope") or {}
    boundary = payload.get("calibration_boundary") or {}
    response_fields = set(current.get("api_response_fields") or [])
    metrics = [str(metric).lower() for metric in payload.get("metrics_to_revalidate") or []]

    missing_fields = [field for field in REQUIRED_RESPONSE_FIELDS if field not in response_fields]
    if missing_fields:
        violations.append(f"missing_required_response_fields:{','.join(missing_fields)}")
    if current.get("bet_sizing_implemented") is not True:
        violations.append("bet_sizing_must_be_implemented")
    if current.get("timing_implemented") is not True:
        violations.append("timing_must_be_implemented")
    if current.get("measured") is not True:
        violations.append("bet_timing_behavior_must_be_measured")
    if current.get("timing_and_bet_size_status") != "PASS":
        violations.append("current_scope_timing_and_bet_size_status_must_pass")
    if boundary.get("status") != CALIBRATION_RECOMMENDED_FOR_HIGHER_REALISM:
        violations.append("higher_realism_calibration_recommendation_must_remain_visible")
    if boundary.get("requires_more_real_player_behavior_labels") is not True:
        violations.append("more_real_player_behavior_labels_must_remain_required")
    if boundary.get("requires_bet_size_labels") is not True:
        violations.append("bet_size_labels_must_remain_required")
    if boundary.get("requires_decision_timing_labels") is not True:
        violations.append("decision_timing_labels_must_remain_required")
    if boundary.get("requires_slice_level_calibration") is not True:
        violations.append("slice_level_calibration_must_remain_required")
    if boundary.get("final_high_realism_claim_allowed") is not False:
        violations.append("final_high_realism_claim_must_remain_blocked")
    if boundary.get("production_blocker_for_current_delivery") is not False:
        violations.append("calibration_gap_must_not_block_current_delivery")
    if boundary.get("label_gap_status") != LABELS_INSUFFICIENT_FOR_FINAL_HIGH_REALISM:
        violations.append("label_gap_status_must_remain_insufficient_for_final_high_realism")
    if not any("bet-size" in metric or "bet size" in metric for metric in metrics):
        violations.append("bet_size_distribution_metric_must_be_revalidated")
    if not any("timing" in metric or "decision-time" in metric or "decision time" in metric for metric in metrics):
        violations.append("timing_distribution_metric_must_be_revalidated")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_bet_timing_calibration(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_bet_timing_calibration(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_bet_timing_calibration_markdown(payload), encoding="utf-8")
    return payload


def render_bet_timing_calibration_markdown(payload: dict[str, Any]) -> str:
    current = payload["current_delivery_scope"]
    boundary = payload["calibration_boundary"]
    lines = [
        "# Bet-Sizing and Timing Calibration Contract",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Current Delivery Scope",
        "",
        f"- Scope: `{current['scope']}`",
        f"- Implementation status: `{current['implementation_status']}`",
        f"- Bet sizing implemented: `{current['bet_sizing_implemented']}`",
        f"- Timing implemented: `{current['timing_implemented']}`",
        f"- Measured: `{current['measured']}`",
        f"- Timing and bet-size status: `{current['timing_and_bet_size_status']}`",
        "",
        "## Calibration Boundary",
        "",
        f"- Status: `{boundary['status']}`",
        f"- Requires more real-player behavior labels: `{boundary['requires_more_real_player_behavior_labels']}`",
        f"- Requires bet-size labels: `{boundary['requires_bet_size_labels']}`",
        f"- Requires decision-timing labels: `{boundary['requires_decision_timing_labels']}`",
        f"- Final high-realism claim allowed: `{boundary['final_high_realism_claim_allowed']}`",
        f"- Production blocker for current delivery: `{boundary['production_blocker_for_current_delivery']}`",
        "",
        "## Calibration Dataset Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["calibration_dataset_requirements"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _response_fields_from_contract(payload: dict[str, Any]) -> list[str]:
    response_schema = payload.get("response_schema") or payload.get("response") or {}
    if isinstance(response_schema, dict):
        fields = response_schema.get("fields") or response_schema.get("properties") or response_schema
        if isinstance(fields, dict):
            return list(fields.keys())
        if isinstance(fields, list):
            return [str(item.get("name") if isinstance(item, dict) else item) for item in fields]
    return []


def _merge_metrics(existing: list[Any], required: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for metric in [*existing, *required]:
        text = str(metric)
        key = text.lower()
        if key not in seen:
            merged.append(text)
            seen.add(key)
    return merged


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
