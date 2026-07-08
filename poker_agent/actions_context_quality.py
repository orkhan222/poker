from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.features import load_training_examples


ACTIONS_CONTEXT_QUALITY_VERSION = "2026-07-06"
ACTION_CONTEXT_RISK_ID = "actions_csv_betting_context_incomplete"
ACTION_CONTEXT_ROOT_CAUSE = "actions_csv_lacks_decision_time_betting_context_fields"
ACTION_CONTEXT_SOURCE_TABLE = "actions.csv"
EXPLICIT_BETTING_CONTEXT_STATUS = "INCOMPLETE_EXPLICIT_BETTING_CONTEXT"
DERIVED_CONTEXT_STATUS = "IMPLEMENTED_FROM_PRE_ACTION_EVENT_STREAM"

REQUIRED_EXPLICIT_ACTION_FIELDS = (
    "amount",
    "to_call",
    "pot_before_action",
    "min_raise",
    "legal_actions",
    "action_order",
    "last_aggressor",
    "facing_bet",
)

DECISION_TIME_CONTEXT_POLICY: dict[str, dict[str, Any]] = {
    "amount": {
        "required_semantics": "Amount committed by an already observed prior action.",
        "reconstruction_source": "previous frame-ordered action rows and stack_events contribution deltas",
        "target_row_value_allowed_as_feature": False,
        "reason": "The current row action amount is part of the label/outcome for that decision and would leak the target action.",
    },
    "to_call": {
        "required_semantics": "Amount the acting player must call before choosing the target action.",
        "reconstruction_source": "street-local player commitments before the target frame",
        "target_row_value_allowed_as_feature": False,
        "reason": "The value must describe the pre-action state, not the contribution created by the target action.",
    },
    "pot_before_action": {
        "required_semantics": "Pot size available before the target action is selected.",
        "reconstruction_source": "running pot reconstructed from previous stack deltas and street commitments",
        "target_row_value_allowed_as_feature": False,
        "reason": "Post-action pot size would encode the target action size.",
    },
    "min_raise": {
        "required_semantics": "Minimum legal raise available before the target action.",
        "reconstruction_source": "current street highest commitment and previous raise increment",
        "target_row_value_allowed_as_feature": False,
        "reason": "The legal threshold must be derived from prior betting state only.",
    },
    "legal_actions": {
        "required_semantics": "Actions legally available to the acting player at decision time.",
        "reconstruction_source": "street, to_call, stack, min_raise, and table commitment state",
        "target_row_value_allowed_as_feature": False,
        "reason": "Legal actions must constrain prediction before the target action is known.",
    },
    "action_order": {
        "required_semantics": "Decision order within the hand and within the current street.",
        "reconstruction_source": "frame_id ordering grouped by hand_id and street",
        "target_row_value_allowed_as_feature": True,
        "reason": "Order is observable from prior event sequence and does not reveal which action is selected.",
    },
    "last_aggressor": {
        "required_semantics": "Most recent player who bet or raised before the target action.",
        "reconstruction_source": "previous street-local action rows with canonical bet or raise labels",
        "target_row_value_allowed_as_feature": False,
        "reason": "The target row cannot define its own prior aggressor; this must be derived from events before the target action.",
    },
    "facing_bet": {
        "required_semantics": "Whether the acting player is facing a live bet or raise before selecting the target action.",
        "reconstruction_source": "street-local commitment state and pre-action to_call",
        "target_row_value_allowed_as_feature": False,
        "reason": "The value must be computed before the action, not inferred from whether the target action became call or fold.",
    },
}

REQUIRED_DERIVED_CONTEXT_FEATURES = (
    "hand_action_order",
    "street_action_order",
    "hand_action_order_norm",
    "street_action_order_norm",
    "street_action_count",
    "facing_bet_or_raise",
    "facing_bet_derived",
    "call_price_ratio",
    "raise_pressure",
    "table_commitment_pressure",
    "last_aggressor_known",
    "last_aggressor_is_hero",
    "last_aggressor_derived",
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
        "risk_contract": {
            "risk_id": ACTION_CONTEXT_RISK_ID,
            "root_cause": ACTION_CONTEXT_ROOT_CAUSE,
            "source_table": ACTION_CONTEXT_SOURCE_TABLE,
            "current_native_fields": ["action", "street"],
            "missing_or_reconstructed_decision_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
            "decision_time_context_policy": DECISION_TIME_CONTEXT_POLICY,
            "target_row_values_are_labels_not_features": True,
            "impact": [
                "Call/fold/raise learning has weaker pot-odds and pressure signal when explicit context is absent.",
                "Bet sizing quality depends on reconstructed stack and pot state rather than direct labels.",
                "Legal-action and action-order ambiguity can bias supervised labels if not derived consistently.",
            ],
            "mitigation_status": "LEAKAGE_SAFE_RECONSTRUCTION_REQUIRED",
            "current_delivery_blocker": False,
            "model_quality_risk": True,
            "final_strategy_quality_claim_blocker_without_richer_action_context": True,
        },
        "client_statement": (
            "actions.csv currently contains the player action and street, but it does not provide "
            "explicit decision-time betting context fields such as amount, to_call, pot_before_action, "
            "min_raise, legal_actions, action_order, last_aggressor, and facing_bet. Without these fields, call/fold/raise learning "
            "depends on reconstructed context and remains weaker than a fully instrumented action log."
        ),
        "required_explicit_action_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
        "dataset_export_contract": {
            "status": "EXPLICIT_BETTING_CONTEXT_REQUIRED_FOR_NEXT_DATASET_EXPORT",
            "source_table": ACTION_CONTEXT_SOURCE_TABLE,
            "required_explicit_fields": list(REQUIRED_EXPLICIT_ACTION_FIELDS),
            "explicit_export_required": True,
            "reconstructed_context_allowed_for_current_delivery": True,
            "current_delivery_blocker": False,
            "model_quality_risk": True,
            "acceptance_boundary": (
                "The current delivery may use leakage-safe reconstructed betting context, "
                "but future dataset exports must persist explicit decision-time betting fields."
            ),
            "must_not_use_target_row_values": True,
            "must_not_use_future_outcome_fields": True,
        },
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
            "target_action_context_leakage_guard": True,
            "uses_future_outcome_fields": False,
            "does_not_fully_replace_explicit_context": True,
            "current_delivery_blocker": False,
            "model_quality_risk": True,
            "final_strategy_quality_claim_blocker_without_richer_action_context": True,
        },
        "training_feature_audit": feature_audit,
        "required_dataset_improvements": [
            "Add explicit amount for each action row.",
            "Add to_call before the action.",
            "Add pot_before_action before the action.",
            "Add min_raise before the action.",
            "Add legal_actions available to the acting player.",
            "Add action_order within hand and street.",
            "Add last_aggressor before the action.",
            "Add facing_bet before the action.",
            "Preserve these fields as decision-time values, not values derived from final hand outcome.",
        ],
        "allowed_claims": [
            "The current pipeline reconstructs decision-time betting context from pre-action state where possible.",
            "The deployed stack can use derived betting context for the current delivery package.",
        ],
        "not_allowed_claims": [
            "actions.csv is a complete decision-time betting-context dataset.",
            "The derived context fully replaces explicit amount, to_call, pot_before_action, min_raise, legal_actions, action_order, last_aggressor, and facing_bet labels.",
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
    risk = payload.get("risk_contract") or {}
    schema_audit = payload.get("actions_csv_schema_audit") or {}
    mitigation = payload.get("derived_context_mitigation") or {}
    feature_audit = payload.get("training_feature_audit") or {}
    export_contract = payload.get("dataset_export_contract") or {}

    if risk.get("risk_id") != ACTION_CONTEXT_RISK_ID:
        violations.append("actions_context_risk_id_must_match_contract")
    if risk.get("root_cause") != ACTION_CONTEXT_ROOT_CAUSE:
        violations.append("actions_context_root_cause_must_match_contract")
    if risk.get("source_table") != ACTION_CONTEXT_SOURCE_TABLE:
        violations.append("actions_context_source_table_must_be_actions_csv")
    if set(risk.get("missing_or_reconstructed_decision_fields") or []) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("actions_context_missing_fields_must_match_contract")
    policy = risk.get("decision_time_context_policy") or {}
    if set(policy) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("actions_context_policy_must_cover_every_required_field")
    for field in REQUIRED_EXPLICIT_ACTION_FIELDS:
        field_policy = policy.get(field) or {}
        if not field_policy.get("required_semantics"):
            violations.append(f"actions_context_policy_missing_semantics_for_{field}")
        if not field_policy.get("reconstruction_source"):
            violations.append(f"actions_context_policy_missing_reconstruction_source_for_{field}")
        if field != "action_order" and field_policy.get("target_row_value_allowed_as_feature") is not False:
            violations.append(f"actions_context_policy_must_forbid_target_row_value_for_{field}")
    if risk.get("target_row_values_are_labels_not_features") is not True:
        violations.append("actions_context_target_row_values_must_be_labels_not_features")
    if risk.get("mitigation_status") != "LEAKAGE_SAFE_RECONSTRUCTION_REQUIRED":
        violations.append("actions_context_mitigation_status_must_require_reconstruction")
    if risk.get("current_delivery_blocker") is not False:
        violations.append("actions_context_risk_must_not_block_current_delivery")
    if risk.get("model_quality_risk") is not True:
        violations.append("actions_context_risk_must_remain_model_quality_risk")
    if risk.get("final_strategy_quality_claim_blocker_without_richer_action_context") is not True:
        violations.append("actions_context_must_block_final_strategy_claim_without_richer_data")

    if set(payload.get("required_explicit_action_fields") or []) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("required_explicit_action_fields_must_match_contract")
    if export_contract.get("status") != "EXPLICIT_BETTING_CONTEXT_REQUIRED_FOR_NEXT_DATASET_EXPORT":
        violations.append("dataset_export_contract_status_must_require_explicit_betting_context")
    if export_contract.get("source_table") != ACTION_CONTEXT_SOURCE_TABLE:
        violations.append("dataset_export_contract_source_table_must_be_actions_csv")
    if set(export_contract.get("required_explicit_fields") or []) != set(REQUIRED_EXPLICIT_ACTION_FIELDS):
        violations.append("dataset_export_required_fields_must_match_contract")
    if export_contract.get("explicit_export_required") is not True:
        violations.append("dataset_export_must_require_explicit_context")
    if export_contract.get("reconstructed_context_allowed_for_current_delivery") is not True:
        violations.append("current_delivery_must_allow_reconstructed_context")
    if export_contract.get("current_delivery_blocker") is not False:
        violations.append("dataset_export_gap_must_not_block_current_delivery")
    if export_contract.get("model_quality_risk") is not True:
        violations.append("dataset_export_gap_must_remain_model_quality_risk")
    if export_contract.get("must_not_use_target_row_values") is not True:
        violations.append("dataset_export_must_forbid_target_row_values_as_features")
    if export_contract.get("must_not_use_future_outcome_fields") is not True:
        violations.append("dataset_export_must_forbid_future_outcome_fields")
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
    if mitigation.get("target_action_context_leakage_guard") is not True:
        violations.append("target_action_context_leakage_guard_must_be_enabled")
    if mitigation.get("uses_future_outcome_fields") is not False:
        violations.append("future_outcome_fields_must_not_be_used")
    if mitigation.get("does_not_fully_replace_explicit_context") is not True:
        violations.append("derived_context_must_not_claim_full_replacement")
    if mitigation.get("current_delivery_blocker") is not False:
        violations.append("actions_context_limitation_must_not_block_current_delivery")
    if mitigation.get("model_quality_risk") is not True:
        violations.append("actions_context_limitation_must_remain_model_quality_risk")
    if mitigation.get("final_strategy_quality_claim_blocker_without_richer_action_context") is not True:
        violations.append("actions_context_mitigation_must_block_final_strategy_claim_without_richer_data")

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
    risk = payload["risk_contract"]
    schema = payload["actions_csv_schema_audit"]
    mitigation = payload["derived_context_mitigation"]
    feature_audit = payload["training_feature_audit"]
    export_contract = payload["dataset_export_contract"]
    lines = [
        "# actions.csv Betting-Context Quality Contract",
        "",
        payload["client_statement"],
        "",
        "## Risk Contract",
        "",
        f"- Risk id: `{risk['risk_id']}`",
        f"- Root cause: `{risk['root_cause']}`",
        f"- Source table: `{risk['source_table']}`",
        f"- Current delivery blocker: `{risk['current_delivery_blocker']}`",
        f"- Model-quality risk: `{risk['model_quality_risk']}`",
        f"- Blocks final strategy-quality claim without richer action context: `{risk['final_strategy_quality_claim_blocker_without_richer_action_context']}`",
        "",
        "Missing or reconstructed decision-time fields:",
        "",
    ]
    lines.extend(f"- `{field}`" for field in risk["missing_or_reconstructed_decision_fields"])
    lines.extend(["", "Decision-time reconstruction policy:", ""])
    for field, policy in risk["decision_time_context_policy"].items():
        lines.append(
            f"- `{field}`: {policy['required_semantics']} Source: {policy['reconstruction_source']}. "
            f"Target-row value allowed as feature: `{policy['target_row_value_allowed_as_feature']}`."
        )
    lines.extend(
        [
            "",
            "## Explicit Dataset Fields",
            "",
            f"- Explicit context status: `{schema.get('explicit_context_status')}`",
            f"- Rows scanned: `{schema.get('rows_scanned')}`",
            "",
            "Missing explicit context fields:",
            "",
        ]
    )
    lines.extend(f"- `{field}`" for field in schema.get("missing_explicit_context_fields") or [])
    lines.extend(
        [
            "",
            "## Future Dataset Export Contract",
            "",
            f"- Status: `{export_contract['status']}`",
            f"- Explicit export required: `{export_contract['explicit_export_required']}`",
            f"- Reconstructed context allowed for current delivery: `{export_contract['reconstructed_context_allowed_for_current_delivery']}`",
            f"- Current delivery blocker: `{export_contract['current_delivery_blocker']}`",
            f"- Model-quality risk: `{export_contract['model_quality_risk']}`",
            f"- Acceptance boundary: {export_contract['acceptance_boundary']}",
            "",
            "Fields required in the next dataset export:",
            "",
        ]
    )
    lines.extend(f"- `{field}`" for field in export_contract["required_explicit_fields"])
    lines.extend(
        [
            "",
            "## Derived Context Mitigation",
            "",
            f"- Status: `{mitigation['status']}`",
            f"- Uses target action amount as feature: `{mitigation['uses_target_action_amount_as_feature']}`",
            f"- Target action context leakage guard: `{mitigation['target_action_context_leakage_guard']}`",
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
