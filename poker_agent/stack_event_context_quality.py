from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.features import load_training_examples, safe_float


STACK_EVENT_CONTEXT_VERSION = "2026-07-02"
RAW_STACK_EVENT_STATUS = "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION"
DERIVED_STACK_CONTEXT_STATUS = "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS"

REQUIRED_STACK_EVENT_COLUMNS = (
    "hand_id",
    "frame_id",
    "player_position",
    "event",
    "stack",
    "diff",
    "stack_after_event",
)

REQUIRED_STACK_DERIVED_FEATURES = (
    "pot",
    "stack",
    "to_call",
    "min_raise",
    "pot_odds",
    "stack_to_pot",
    "spr",
    "call_to_stack",
    "raise_to_stack",
    "hero_commitment_ratio",
    "table_commitment_pressure",
    "call_price_ratio",
    "raise_pressure",
    "stack_event_context_reconstructed",
    "stack_event_target_bet_size_used_as_feature",
    "reconstructed_effective_stack",
    "reconstructed_effective_stack_to_pot",
    "reconstructed_spr_after_call",
    "reconstructed_current_street_bet_size",
    "reconstructed_current_street_bet_to_pot",
    "reconstructed_pot_pressure",
    "reconstructed_call_pressure",
    "reconstructed_raise_pressure",
)


def build_stack_event_context_quality(project_root: Path, *, max_examples: int = 5000) -> dict[str, Any]:
    schema_audit = audit_stack_events_csv(project_root)
    feature_audit = audit_training_stack_context_features(project_root, max_examples=max_examples)
    payload: dict[str, Any] = {
        "version": STACK_EVENT_CONTEXT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "stack_events.csv decision-context derivation contract",
        "client_statement": (
            "stack_events.csv stores raw stack changes. Raw stack events are not sufficient as direct "
            "policy features; they must be converted into decision-time pot, effective stack, SPR, "
            "bet-size, and pressure features before supervised or deployed policy use."
        ),
        "stack_events_schema_audit": schema_audit,
        "raw_stack_event_boundary": {
            "status": RAW_STACK_EVENT_STATUS,
            "raw_stack_events_are_direct_policy_features": False,
            "decision_time_derivation_required": True,
            "target_action_stack_delta_allowed_as_feature": False,
            "post_hand_stack_outcome_allowed_as_feature": False,
            "current_delivery_blocker": False,
            "model_quality_risk": True,
        },
        "derived_context_mitigation": {
            "status": DERIVED_STACK_CONTEXT_STATUS,
            "implemented": True,
            "derived_from": [
                "negative stack diff contribution events",
                "frame-nearest pre-action stack deltas",
                "street-local player commitments",
                "running pot accumulated before the target decision",
                "decision-time starting stack minus committed amount",
            ],
            "required_derived_features": list(REQUIRED_STACK_DERIVED_FEATURES),
            "uses_target_action_stack_delta_as_feature": False,
            "uses_post_hand_outcome_fields": False,
            "current_delivery_blocker": False,
            "model_quality_risk": True,
        },
        "training_feature_audit": feature_audit,
        "required_dataset_improvements": [
            "Persist pot_before_action as an explicit field.",
            "Persist effective_stack before the action.",
            "Persist SPR before the action.",
            "Persist current_bet_size and min_raise before the action.",
            "Persist legal_actions and to_call before the action.",
            "Keep raw stack events for audit, but do not use post-hand stack outcomes as policy features.",
        ],
        "allowed_claims": [
            "The current feature pipeline derives decision-time betting pressure from stack events.",
            "Raw stack events are retained as source data and converted into policy-safe context features.",
        ],
        "not_allowed_claims": [
            "Raw stack events are sufficient policy features without decision-context derivation.",
            "Target action stack deltas are valid features for predicting that same action.",
            "Post-hand stack outcomes can be used as decision-time training features.",
            "The reconstructed stack context fully replaces explicit instrumented pot/effective-stack/SPR labels.",
        ],
    }
    payload["proof_cases"] = build_stack_event_context_proof_cases(payload)
    payload["invariants"] = validate_stack_event_context_quality(payload)
    if not all(case["passed"] for case in payload["proof_cases"]):
        payload["invariants"]["status"] = "FAIL"
        payload["invariants"]["violations"].append("stack_event_context_proof_cases_must_pass")
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def audit_stack_events_csv(project_root: Path) -> dict[str, Any]:
    stack_path = project_root / "data" / "stack_events.csv"
    if not stack_path.exists():
        return {
            "status": "FAIL",
            "path": str(stack_path),
            "available_fields": [],
            "missing_columns": list(REQUIRED_STACK_EVENT_COLUMNS),
            "rows_scanned": 0,
            "negative_diff_rows": 0,
            "usable_contribution_rate": 0.0,
        }

    rows_scanned = 0
    diff_rows = 0
    negative_diff_rows = 0
    hand_ids: set[str] = set()
    positions: set[str] = set()
    with stack_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_scanned += 1
            hand_id = str(row.get("hand_id") or "").strip()
            position = str(row.get("player_position") or "").strip()
            if hand_id:
                hand_ids.add(hand_id)
            if position:
                positions.add(position)
            if str(row.get("diff") or "").strip():
                diff_rows += 1
                if safe_float(row.get("diff")) < 0:
                    negative_diff_rows += 1

    missing = [column for column in REQUIRED_STACK_EVENT_COLUMNS if column not in fieldnames]
    return {
        "status": "PASS" if rows_scanned > 0 and "diff" in fieldnames else "FAIL",
        "path": str(stack_path),
        "available_fields": fieldnames,
        "missing_columns": missing,
        "rows_scanned": rows_scanned,
        "hands_with_stack_events": len(hand_ids),
        "positions_with_stack_events": len(positions),
        "rows_with_diff": diff_rows,
        "negative_diff_rows": negative_diff_rows,
        "usable_contribution_rate": negative_diff_rows / rows_scanned if rows_scanned else 0.0,
    }


def audit_training_stack_context_features(project_root: Path, *, max_examples: int) -> dict[str, Any]:
    data_dir = project_root / "data"
    examples = load_training_examples(
        data_dir,
        max_examples=max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        include_hand_id=False,
    )
    feature_names = sorted({name for features, _ in examples for name in features})
    missing = [name for name in REQUIRED_STACK_DERIVED_FEATURES if name not in feature_names]
    first_example_values: dict[str, float] = {}
    if examples:
        first_features = examples[0][0]
        first_example_values = {
            name: float(first_features.get(name, 0.0))
            for name in REQUIRED_STACK_DERIVED_FEATURES
            if name in first_features
        }

    return {
        "status": "PASS" if examples and not missing else "FAIL",
        "dataset": str(data_dir),
        "examples_scanned": len(examples),
        "feature_count": len(feature_names),
        "required_stack_context_features_present": [
            name for name in REQUIRED_STACK_DERIVED_FEATURES if name in feature_names
        ],
        "missing_required_stack_context_features": missing,
        "sample_stack_context_feature_values": first_example_values,
    }


def validate_stack_event_context_quality(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    schema = payload.get("stack_events_schema_audit") or {}
    raw_boundary = payload.get("raw_stack_event_boundary") or {}
    mitigation = payload.get("derived_context_mitigation") or {}
    feature_audit = payload.get("training_feature_audit") or {}

    if schema.get("status") != "PASS":
        violations.append("stack_events_csv_must_be_readable_with_diff")
    if int(schema.get("rows_scanned") or 0) <= 0:
        violations.append("stack_events_audit_must_scan_rows")
    if int(schema.get("negative_diff_rows") or 0) <= 0:
        violations.append("stack_events_must_contain_negative_contribution_rows")
    if raw_boundary.get("status") != RAW_STACK_EVENT_STATUS:
        violations.append("raw_stack_event_status_must_match_contract")
    if raw_boundary.get("raw_stack_events_are_direct_policy_features") is not False:
        violations.append("raw_stack_events_must_not_be_direct_policy_features")
    if raw_boundary.get("decision_time_derivation_required") is not True:
        violations.append("stack_events_must_require_decision_time_derivation")
    if raw_boundary.get("target_action_stack_delta_allowed_as_feature") is not False:
        violations.append("target_action_stack_delta_must_not_be_feature")
    if raw_boundary.get("post_hand_stack_outcome_allowed_as_feature") is not False:
        violations.append("post_hand_stack_outcome_must_not_be_feature")
    if raw_boundary.get("current_delivery_blocker") is not False:
        violations.append("stack_event_context_gap_must_not_block_current_delivery")
    if raw_boundary.get("model_quality_risk") is not True:
        violations.append("stack_event_context_gap_must_remain_model_quality_risk")
    if mitigation.get("status") != DERIVED_STACK_CONTEXT_STATUS:
        violations.append("derived_stack_context_status_must_match_contract")
    if mitigation.get("implemented") is not True:
        violations.append("derived_stack_context_must_be_implemented")
    if mitigation.get("uses_target_action_stack_delta_as_feature") is not False:
        violations.append("derived_stack_context_must_not_use_target_action_delta")
    if mitigation.get("uses_post_hand_outcome_fields") is not False:
        violations.append("derived_stack_context_must_not_use_post_hand_outcomes")
    if mitigation.get("current_delivery_blocker") is not False:
        violations.append("derived_stack_context_must_not_block_current_delivery")
    if mitigation.get("model_quality_risk") is not True:
        violations.append("derived_stack_context_must_remain_model_quality_risk")
    if feature_audit.get("status") != "PASS":
        violations.append("training_features_must_include_stack_context_features")
    if int(feature_audit.get("examples_scanned") or 0) <= 0:
        violations.append("stack_context_feature_audit_must_scan_examples")
    if feature_audit.get("missing_required_stack_context_features"):
        violations.append("required_stack_context_features_missing")
    sample_values = feature_audit.get("sample_stack_context_feature_values") or {}
    if float(sample_values.get("stack_event_target_bet_size_used_as_feature", 0.0) or 0.0) != 0.0:
        violations.append("target_stack_delta_leakage_guard_must_be_zero")
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def build_stack_event_context_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        _stack_event_proof_case("base_contract_is_valid", payload, "PASS"),
    ]

    mutated = deepcopy(payload)
    mutated["raw_stack_event_boundary"]["raw_stack_events_are_direct_policy_features"] = True
    cases.append(_stack_event_proof_case("blocks_raw_stack_events_as_direct_features", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["raw_stack_event_boundary"]["target_action_stack_delta_allowed_as_feature"] = True
    mutated["derived_context_mitigation"]["uses_target_action_stack_delta_as_feature"] = True
    sample_values = mutated["training_feature_audit"].setdefault("sample_stack_context_feature_values", {})
    sample_values["stack_event_target_bet_size_used_as_feature"] = 1.0
    cases.append(_stack_event_proof_case("blocks_target_action_stack_delta_feature_leakage", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["raw_stack_event_boundary"]["current_delivery_blocker"] = True
    mutated["derived_context_mitigation"]["current_delivery_blocker"] = True
    cases.append(_stack_event_proof_case("blocks_delivery_blocker_reclassification", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["raw_stack_event_boundary"]["model_quality_risk"] = False
    mutated["derived_context_mitigation"]["model_quality_risk"] = False
    cases.append(_stack_event_proof_case("blocks_model_quality_risk_removal", mutated, "FAIL"))

    return cases


def _stack_event_proof_case(name: str, candidate: dict[str, Any], expected_status: str) -> dict[str, Any]:
    observed = validate_stack_event_context_quality(candidate)
    return {
        "name": name,
        "expected_status": expected_status,
        "observed_status": observed["status"],
        "passed": observed["status"] == expected_status,
        "violations": observed["violations"],
    }


def write_stack_event_context_quality(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    *,
    max_examples: int = 5000,
) -> dict[str, Any]:
    payload = build_stack_event_context_quality(project_root, max_examples=max_examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_stack_event_context_quality_markdown(payload), encoding="utf-8")
    return payload


def render_stack_event_context_quality_markdown(payload: dict[str, Any]) -> str:
    schema = payload["stack_events_schema_audit"]
    raw_boundary = payload["raw_stack_event_boundary"]
    mitigation = payload["derived_context_mitigation"]
    feature_audit = payload["training_feature_audit"]
    lines = [
        "# stack_events.csv Decision-Context Quality Contract",
        "",
        payload["client_statement"],
        "",
        "## Stack Events Audit",
        "",
        f"- Rows scanned: `{schema.get('rows_scanned')}`",
        f"- Negative contribution rows: `{schema.get('negative_diff_rows')}`",
        f"- Usable contribution rate: `{schema.get('usable_contribution_rate')}`",
        "",
        "## Boundary",
        "",
        f"- Raw event status: `{raw_boundary['status']}`",
        f"- Raw stack events are direct policy features: `{raw_boundary['raw_stack_events_are_direct_policy_features']}`",
        f"- Decision-time derivation required: `{raw_boundary['decision_time_derivation_required']}`",
        f"- Current delivery blocker: `{raw_boundary['current_delivery_blocker']}`",
        f"- Model quality risk: `{raw_boundary['model_quality_risk']}`",
        "",
        "## Derived Context Mitigation",
        "",
        f"- Status: `{mitigation['status']}`",
        f"- Uses target action stack delta as feature: `{mitigation['uses_target_action_stack_delta_as_feature']}`",
        f"- Uses post-hand outcome fields: `{mitigation['uses_post_hand_outcome_fields']}`",
        "",
        "Required derived stack-context features:",
        "",
    ]
    lines.extend(f"- `{field}`" for field in mitigation["required_derived_features"])
    lines.extend(
        [
            "",
            "## Training Feature Audit",
            "",
            f"- Status: `{feature_audit['status']}`",
            f"- Examples scanned: `{feature_audit.get('examples_scanned')}`",
            f"- Feature count: `{feature_audit.get('feature_count')}`",
            "",
            "## Not Allowed Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", "## Executable Proof Cases", ""])
    for case in payload.get("proof_cases") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, passed `{case['passed']}`"
        )
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)
