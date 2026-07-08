from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.actions_context_quality import (
    ACTION_CONTEXT_SOURCE_TABLE,
    DECISION_TIME_CONTEXT_POLICY,
    REQUIRED_EXPLICIT_ACTION_FIELDS,
    build_actions_context_quality,
)


ACTIONS_DATASET_EXPORT_CONTRACT_VERSION = "2026-07-08"
ACTIONS_DATASET_EXPORT_STATUS = "EXPLICIT_BETTING_CONTEXT_REQUIRED_FOR_NEXT_DATASET_EXPORT"


def build_actions_dataset_export_contract(project_root: Path) -> dict[str, Any]:
    actions_context = build_actions_context_quality(project_root)
    embedded_contract = actions_context.get("dataset_export_contract") or {}
    schema_audit = actions_context.get("actions_csv_schema_audit") or {}

    payload: dict[str, Any] = {
        "version": ACTIONS_DATASET_EXPORT_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "actions.csv future dataset export contract",
        "source_report": "reports/actions_context_quality.json",
        "source_table": ACTION_CONTEXT_SOURCE_TABLE,
        "status": embedded_contract.get("status", ACTIONS_DATASET_EXPORT_STATUS),
        "required_explicit_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
        "current_actions_csv_fields": schema_audit.get("available_fields", []),
        "currently_missing_explicit_fields": schema_audit.get("missing_explicit_context_fields", []),
        "field_contract": DECISION_TIME_CONTEXT_POLICY,
        "current_delivery_boundary": {
            "current_delivery_blocker": False,
            "reconstructed_context_allowed": True,
            "implementation": "leakage-safe pre-action reconstruction",
            "acceptance_boundary": (
                "Current delivery may use reconstructed betting context because the pipeline derives "
                "decision-time values from pre-action events and blocks target-row leakage."
            ),
        },
        "future_export_boundary": {
            "explicit_export_required": True,
            "model_quality_risk_until_export_is_instrumented": True,
            "must_persist_decision_time_values": True,
            "must_not_use_target_row_values": True,
            "must_not_use_future_outcome_fields": True,
            "acceptance_boundary": (
                "The next dataset export must persist amount, to_call, pot_before_action, min_raise, "
                "legal_actions, action_order, last_aggressor, and facing_bet as explicit decision-time fields."
            ),
        },
        "allowed_claims": [
            "The current delivery can reconstruct missing betting context from pre-action event streams.",
            "The missing explicit fields are not a current delivery blocker.",
            "The missing explicit fields remain a model-quality risk until the next dataset export is instrumented.",
        ],
        "not_allowed_claims": [
            "actions.csv is a complete decision-time betting-context table.",
            "Reconstructed context fully replaces explicit action-context labels for final strategy-quality claims.",
            "Future training exports can omit amount, to_call, pot_before_action, min_raise, legal_actions, action_order, last_aggressor, or facing_bet.",
        ],
    }
    payload["invariants"] = validate_actions_dataset_export_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_actions_dataset_export_contract(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    current_boundary = payload.get("current_delivery_boundary") or {}
    future_boundary = payload.get("future_export_boundary") or {}

    if payload.get("source_table") != ACTION_CONTEXT_SOURCE_TABLE:
        violations.append("dataset_export_source_table_must_be_actions_csv")
    if payload.get("status") != ACTIONS_DATASET_EXPORT_STATUS:
        violations.append("dataset_export_status_must_require_explicit_betting_context")
    if set(payload.get("required_explicit_fields") or []) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("dataset_export_required_fields_must_match_contract")
    if set(payload.get("field_contract") or {}) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("dataset_export_field_contract_must_cover_required_fields")
    if current_boundary.get("current_delivery_blocker") is not False:
        violations.append("dataset_export_gap_must_not_block_current_delivery")
    if current_boundary.get("reconstructed_context_allowed") is not True:
        violations.append("dataset_export_must_allow_current_reconstruction")
    if future_boundary.get("explicit_export_required") is not True:
        violations.append("future_dataset_export_must_require_explicit_context")
    if future_boundary.get("model_quality_risk_until_export_is_instrumented") is not True:
        violations.append("future_dataset_export_gap_must_remain_model_quality_risk")
    if future_boundary.get("must_persist_decision_time_values") is not True:
        violations.append("future_dataset_export_must_persist_decision_time_values")
    if future_boundary.get("must_not_use_target_row_values") is not True:
        violations.append("future_dataset_export_must_forbid_target_row_values")
    if future_boundary.get("must_not_use_future_outcome_fields") is not True:
        violations.append("future_dataset_export_must_forbid_future_outcome_fields")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_actions_dataset_export_contract(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_actions_dataset_export_contract(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_actions_dataset_export_contract_markdown(payload), encoding="utf-8")
    return payload


def render_actions_dataset_export_contract_markdown(payload: dict[str, Any]) -> str:
    current_boundary = payload["current_delivery_boundary"]
    future_boundary = payload["future_export_boundary"]
    lines = [
        "# actions.csv Future Dataset Export Contract",
        "",
        "This contract separates the current delivery boundary from the required next dataset export format.",
        "",
        "## Current Delivery Boundary",
        "",
        f"- Current delivery blocker: `{current_boundary['current_delivery_blocker']}`",
        f"- Reconstructed context allowed: `{current_boundary['reconstructed_context_allowed']}`",
        f"- Implementation: `{current_boundary['implementation']}`",
        f"- Boundary: {current_boundary['acceptance_boundary']}",
        "",
        "## Future Export Boundary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Explicit export required: `{future_boundary['explicit_export_required']}`",
        f"- Model-quality risk until instrumented: `{future_boundary['model_quality_risk_until_export_is_instrumented']}`",
        f"- Must persist decision-time values: `{future_boundary['must_persist_decision_time_values']}`",
        f"- Must not use target-row values: `{future_boundary['must_not_use_target_row_values']}`",
        f"- Must not use future outcome fields: `{future_boundary['must_not_use_future_outcome_fields']}`",
        f"- Boundary: {future_boundary['acceptance_boundary']}",
        "",
        "Required explicit fields:",
        "",
    ]
    lines.extend(f"- `{field}`" for field in payload["required_explicit_fields"])
    lines.extend(
        [
            "",
            "## Allowed Claims",
            "",
            *[f"- {claim}" for claim in payload["allowed_claims"]],
            "",
            "## Not Allowed Claims",
            "",
            *[f"- {claim}" for claim in payload["not_allowed_claims"]],
            "",
            "## Invariants",
            "",
            f"- Status: `{payload['invariants']['status']}`",
            f"- Violations: `{payload['invariants']['violations']}`",
            f"- Overall status: `{payload['overall_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"
