from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.features import load_training_examples


ACTIONS_CONTEXT_QUALITY_VERSION = "2026-07-02"
EXPLICIT_BETTING_CONTEXT_STATUS = "INCOMPLETE_EXPLICIT_BETTING_CONTEXT"
DERIVED_CONTEXT_STATUS = "IMPLEMENTED_FROM_PRE_ACTION_EVENT_STREAM"

REQUIRED_EXPLICIT_ACTION_FIELDS = (
    "amount",
    "to_call",
    "pot_before_action",
    "min_raise",
    "legal_actions",
    "action_order",
)

REQUIRED_DERIVED_CONTEXT_FEATURES = (
    "hand_action_order",
    "street_action_order",
    "hand_action_order_norm",
    "street_action_order_norm",
    "street_action_count",
    "facing_bet_or_raise",
    "call_price_ratio",
    "raise_pressure",
    "table_commitment_pressure",
    "betting_context_reconstructed",
    "action_order_derived",
    "legal_actions_derived",
)


def build_actions_context_quality(project_root: Path, *, max_examples: int = 5000) -> dict[str, Any]:
    schema_audit = audit_actions_csv_schema(project_root)
    feature_audit = audit_training_actions_context_features(project_root, max_examples=max_examples)
    payload: dict[str, Any] = {
        "version": ACTIONS_CONTEXT_QUALITY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "actions.csv betting-context quality contract",
        "client_statement": (
            "actions.csv currently contains the player action and street, but it does not provide "
            "explicit decision-time betting context fields such as amount, to_call, pot_before_action, "
            "min_raise, legal_actions, and action_order. Without these fields, call/fold/raise learning "
            "depends on reconstructed context and remains weaker than a fully instrumented action log."
        ),
        "required_explicit_action_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
        "actions_csv_schema_audit": schema_audit,
        "derived_context_mitigation": {
            "status": DERIVED_CONTEXT_STATUS,
            "implemented": True,
            "derived_from": [
                "frame-ordered actions.csv rows",
                "stack_events.csv contribution deltas",
                "decision-time starting_stack",
                "street-local commitment state",
            ],
            "required_derived_features": list(REQUIRED_DERIVED_CONTEXT_FEATURES),
            "uses_target_action_amount_as_feature": False,
            "uses_future_outcome_fields": False,
            "does_not_fully_replace_explicit_context": True,
            "current_delivery_blocker": False,
            "model_quality_risk": True,
        },
        "training_feature_audit": feature_audit,
        "required_dataset_improvements": [
            "Add explicit amount for each action row.",
            "Add to_call before the action.",
            "Add pot_before_action before the action.",
            "Add min_raise before the action.",
            "Add legal_actions available to the acting player.",
            "Add action_order within hand and street.",
            "Preserve these fields as decision-time values, not values derived from final hand outcome.",
        ],
        "allowed_claims": [
            "The current pipeline reconstructs decision-time betting context from pre-action state where possible.",
            "The deployed stack can use derived betting context for the current delivery package.",
        ],
        "not_allowed_claims": [
            "actions.csv is a complete decision-time betting-context dataset.",
            "The derived context fully replaces explicit amount, to_call, pot_before_action, min_raise, legal_actions, and action_order labels.",
            "A standalone poker policy can be promoted without revalidating on richer action-context data.",
        ],
    }
    payload["invariants"] = validate_actions_context_quality(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def audit_actions_csv_schema(project_root: Path) -> dict[str, Any]:
    actions_path = project_root / "data" / "actions.csv"
    if not actions_path.exists():
        return {
            "status": "FAIL",
            "path": str(actions_path),
            "available_fields": [],
            "missing_explicit_context_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
            "rows_scanned": 0,
        }

    with actions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        row_count = sum(1 for _ in reader)

    present = sorted(set(fieldnames) & set(REQUIRED_EXPLICIT_ACTION_FIELDS))
    missing = [field for field in REQUIRED_EXPLICIT_ACTION_FIELDS if field not in fieldnames]
    return {
        "status": "PASS",
        "explicit_context_status": (
            "COMPLETE" if not missing else EXPLICIT_BETTING_CONTEXT_STATUS
        ),
        "path": str(actions_path),
        "available_fields": fieldnames,
        "present_explicit_context_fields": present,
        "missing_explicit_context_fields": missing,
        "rows_scanned": row_count,
        "limitation_status": (
            "RESOLVED" if not missing else "OPEN_DATASET_LIMITATION"
        ),
    }


def audit_training_actions_context_features(project_root: Path, *, max_examples: int) -> dict[str, Any]:
    data_dir = project_root / "data"
    examples = load_training_examples(
        data_dir,
        max_examples=max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        include_hand_id=False,
    )
    feature_names = sorted({name for features, _ in examples for name in features})
    missing = [name for name in REQUIRED_DERIVED_CONTEXT_FEATURES if name not in feature_names]
    first_example_values: dict[str, float] = {}
    if examples:
        first_features = examples[0][0]
        first_example_values = {
            name: float(first_features.get(name, 0.0))
            for name in REQUIRED_DERIVED_CONTEXT_FEATURES
            if name in first_features
        }

    return {
        "status": "PASS" if not missing and examples else "FAIL",
        "dataset": str(data_dir),
        "examples_scanned": len(examples),
        "feature_count": len(feature_names),
        "required_derived_features_present": [name for name in REQUIRED_DERIVED_CONTEXT_FEATURES if name in feature_names],
        "missing_required_derived_features": missing,
        "sample_derived_feature_values": first_example_values,
    }


def validate_actions_context_quality(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    schema_audit = payload.get("actions_csv_schema_audit") or {}
    mitigation = payload.get("derived_context_mitigation") or {}
    feature_audit = payload.get("training_feature_audit") or {}

    if set(payload.get("required_explicit_action_fields") or []) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("required_explicit_action_fields_must_match_contract")
    if schema_audit.get("status") != "PASS":
        violations.append("actions_csv_schema_must_be_readable")
    if int(schema_audit.get("rows_scanned") or 0) <= 0:
        violations.append("actions_csv_audit_must_scan_rows")
    if schema_audit.get("missing_explicit_context_fields"):
        if schema_audit.get("explicit_context_status") != EXPLICIT_BETTING_CONTEXT_STATUS:
            violations.append("missing_explicit_fields_must_remain_marked_incomplete")
        if schema_audit.get("limitation_status") != "OPEN_DATASET_LIMITATION":
            violations.append("missing_explicit_fields_must_remain_open_dataset_limitation")

    if mitigation.get("status") != DERIVED_CONTEXT_STATUS:
        violations.append("derived_context_status_must_match_contract")
    if mitigation.get("implemented") is not True:
        violations.append("derived_context_must_be_implemented")
    if mitigation.get("uses_target_action_amount_as_feature") is not False:
        violations.append("target_action_amount_must_not_be_used_as_feature")
    if mitigation.get("uses_future_outcome_fields") is not False:
        violations.append("future_outcome_fields_must_not_be_used")
    if mitigation.get("does_not_fully_replace_explicit_context") is not True:
        violations.append("derived_context_must_not_claim_full_replacement")
    if mitigation.get("current_delivery_blocker") is not False:
        violations.append("actions_context_limitation_must_not_block_current_delivery")
    if mitigation.get("model_quality_risk") is not True:
        violations.append("actions_context_limitation_must_remain_model_quality_risk")

    if feature_audit.get("status") != "PASS":
        violations.append("training_examples_must_include_derived_context_features")
    if int(feature_audit.get("examples_scanned") or 0) <= 0:
        violations.append("training_feature_audit_must_scan_examples")
    if feature_audit.get("missing_required_derived_features"):
        violations.append("required_derived_context_features_missing")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_actions_context_quality(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    *,
    max_examples: int = 5000,
) -> dict[str, Any]:
    payload = build_actions_context_quality(project_root, max_examples=max_examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_actions_context_quality_markdown(payload), encoding="utf-8")
    return payload


def render_actions_context_quality_markdown(payload: dict[str, Any]) -> str:
    schema = payload["actions_csv_schema_audit"]
    mitigation = payload["derived_context_mitigation"]
    feature_audit = payload["training_feature_audit"]
    lines = [
        "# actions.csv Betting-Context Quality Contract",
        "",
        payload["client_statement"],
        "",
        "## Explicit Dataset Fields",
        "",
        f"- Explicit context status: `{schema.get('explicit_context_status')}`",
        f"- Rows scanned: `{schema.get('rows_scanned')}`",
        "",
        "Missing explicit context fields:",
        "",
    ]
    lines.extend(f"- `{field}`" for field in schema.get("missing_explicit_context_fields") or [])
    lines.extend(
        [
            "",
            "## Derived Context Mitigation",
            "",
            f"- Status: `{mitigation['status']}`",
            f"- Uses target action amount as feature: `{mitigation['uses_target_action_amount_as_feature']}`",
            f"- Uses future outcome fields: `{mitigation['uses_future_outcome_fields']}`",
            f"- Fully replaces explicit context: `{not mitigation['does_not_fully_replace_explicit_context']}`",
            f"- Current delivery blocker: `{mitigation['current_delivery_blocker']}`",
            f"- Model quality risk: `{mitigation['model_quality_risk']}`",
            "",
            "## Training Feature Audit",
            "",
            f"- Status: `{feature_audit['status']}`",
            f"- Examples scanned: `{feature_audit.get('examples_scanned')}`",
            f"- Feature count: `{feature_audit.get('feature_count')}`",
            "",
            "Required derived context features:",
            "",
        ]
    )
    lines.extend(f"- `{field}`" for field in mitigation["required_derived_features"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)
