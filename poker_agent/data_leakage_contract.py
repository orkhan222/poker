from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.features import load_training_examples, request_to_features
from poker_agent.leakage_guard import (
    FINAL_BOARD_FIELD_DEFINITIONS,
    FORBIDDEN_OUTCOME_FIELDS,
    OUTCOME_FIELD_DEFINITIONS,
    RAW_FINAL_BOARD_SNAPSHOT_FIELDS,
    forbidden_outcome_feature_names,
)
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest


DATA_LEAKAGE_CONTRACT_VERSION = "2026-07-02"
LEAKAGE_GUARD_STATUS = "PASS_NO_OUTCOME_FEATURES"
OUTCOME_ONLY_FIELD_STATUS = "DATASET_ONLY_NOT_TRAINING_FEATURES"

LEAKAGE_RISK_CONTRACT: dict[str, Any] = {
    "risk_id": "post_outcome_feature_leakage",
    "root_cause": "post_hand_outcome_fields_available_in_raw_dataset_schema",
    "risk_statement": (
        "Outcome and settlement fields are present in the raw CSV schema but are not observable "
        "at decision time. Using them as model inputs would let the model learn from future information."
    ),
    "temporal_requirement": "features_must_be_observable_before_target_action",
    "forbidden_fields": list(FORBIDDEN_OUTCOME_FIELDS),
    "field_definitions": OUTCOME_FIELD_DEFINITIONS,
    "feature_policy": {
        "raw_dataset_schema_presence": "allowed_for_audit_and_reporting_only",
        "training_feature_use": "forbidden",
        "prediction_request_use": "forbidden",
        "model_artifact_feature_use": "forbidden",
        "detected_violation": "production_blocker",
    },
    "impact_if_violated": [
        "optimistic offline metrics",
        "invalid holdout evaluation",
        "model learns hand outcome instead of decision policy",
        "unsafe production strategy-quality claim",
    ],
}

FINAL_BOARD_SNAPSHOT_CONTRACT: dict[str, Any] = {
    "risk_id": "final_board_snapshot_leakage",
    "root_cause": "hands_csv_board_cards_is_final_hand_snapshot",
    "risk_statement": (
        "hands.csv::board_cards stores the final board for the hand. For target actions before the river, "
        "using this value directly would expose future community cards to the model."
    ),
    "temporal_requirement": "board_features_must_be_truncated_to_cards_visible_at_target_street",
    "raw_final_board_snapshot_fields": list(RAW_FINAL_BOARD_SNAPSHOT_FIELDS),
    "field_definitions": FINAL_BOARD_FIELD_DEFINITIONS,
    "feature_policy": {
        "raw_dataset_schema_presence": "allowed_for_audit_and_street_truncation_only",
        "direct_training_feature_use": "forbidden",
        "prediction_request_board_cards": "allowed_only_as_decision_time_visible_board",
        "model_artifact_direct_final_board_feature_use": "forbidden",
        "detected_violation": "production_blocker",
    },
    "required_mitigation": {
        "truncate_final_board_by_street": True,
        "preflop_visible_board_count": 0,
        "flop_visible_board_count": 3,
        "turn_visible_board_count": 4,
        "river_visible_board_count": 5,
    },
}

GUARDED_SOURCE_FILES = (
    "poker_agent/features.py",
    "scripts/build_decision_context_holdout.py",
    "scripts/train_policy.py",
    "scripts/train_policy_bundle.py",
    "poker_agent/raw_model_challenger.py",
)

MODEL_ARTIFACTS = (
    "models/poker_policy.joblib",
    "models/poker_policy_bundle.joblib",
)


def build_data_leakage_contract(project_root: Path, *, max_examples: int = 5000) -> dict[str, Any]:
    feature_audit = audit_training_feature_names(project_root, max_examples=max_examples)
    request_audit = audit_prediction_request_features()
    model_audit = audit_model_artifact_features(project_root)
    source_audit = audit_training_source_usage(project_root)
    payload: dict[str, Any] = {
        "version": DATA_LEAKAGE_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Outcome-field data-leakage prevention contract",
        "client_statement": (
            "winner_positions, stack_delta, ending_stack, dealer_winner, dealer_pot, "
            "pot_from_stacks, and the raw final hands.csv board_cards snapshot are not direct "
            "decision-time features. They must not be used as training or prediction features because "
            "they would let the model observe future information."
        ),
        "leakage_risk_contract": LEAKAGE_RISK_CONTRACT,
        "final_board_snapshot_contract": FINAL_BOARD_SNAPSHOT_CONTRACT,
        "forbidden_outcome_fields": list(FORBIDDEN_OUTCOME_FIELDS),
        "raw_final_board_snapshot_fields": list(RAW_FINAL_BOARD_SNAPSHOT_FIELDS),
        "leakage_boundary": {
            "status": OUTCOME_ONLY_FIELD_STATUS,
            "decision_time_observability_required": True,
            "training_feature_use_allowed": False,
            "prediction_request_use_allowed": False,
            "model_artifact_feature_use_allowed": False,
            "direct_final_board_snapshot_feature_use_allowed": False,
            "decision_time_visible_board_cards_allowed": True,
            "dataset_schema_presence_allowed": True,
            "reporting_and_audit_use_allowed": True,
            "current_deployment_blocker": False,
            "production_blocker_if_detected": True,
        },
        "feature_name_audit": feature_audit,
        "prediction_request_audit": request_audit,
        "model_artifact_audit": model_audit,
        "source_usage_audit": source_audit,
        "raw_dataset_schema_audit": audit_raw_dataset_schema(project_root),
        "implemented_fixes": [
            "Removed ending_stack fallback from supervised training feature extraction.",
            "Removed ending_stack fallback from decision-context holdout generation.",
            "Added a machine-readable leakage contract covering feature names, model artifacts, and guarded training sources.",
            "Added a temporal availability contract for outcome-only fields that may remain in raw CSVs only for audit/reporting.",
            "Added a final-board snapshot contract: hands.csv::board_cards can be read only to derive street-visible board cards, not as a direct training feature.",
        ],
        "allowed_claims": [
            "Outcome-only fields may remain in the raw CSV schema for audit, reporting, and downstream settlement analysis.",
            "Training and prediction features are restricted to information observable before the target action.",
            "Prediction request board_cards are valid only when they contain community cards visible at the submitted street.",
        ],
        "not_allowed_claims": [
            "winner_positions, stack_delta, ending_stack, dealer_winner, dealer_pot, or pot_from_stacks are safe training features.",
            "A model trained with post-hand outcome fields can be accepted as leakage-free.",
            "ending_stack can be used as a fallback for decision-time stack when starting_stack is missing.",
            "The final hands.csv board_cards snapshot can be used directly for preflop, flop, or turn action prediction.",
        ],
    }
    payload["invariants"] = validate_data_leakage_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def audit_raw_dataset_schema(project_root: Path) -> dict[str, Any]:
    schema_hits: list[dict[str, Any]] = []
    final_board_hits: list[dict[str, Any]] = []
    missing_tables: list[str] = []
    for table in sorted({item["source_table"] for item in OUTCOME_FIELD_DEFINITIONS.values()}):
        path = project_root / "data" / table
        if not path.exists():
            missing_tables.append(table)
            continue
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",") if path.stat().st_size else []
        for field, definition in OUTCOME_FIELD_DEFINITIONS.items():
            if definition["source_table"] == table and field in header:
                schema_hits.append(
                    {
                        "table": table,
                        "field": field,
                        "availability": definition["availability"],
                        "presence_allowed": True,
                        "allowed_use": "audit_reporting_settlement_only",
                    }
                )
        for source_field, definition in FINAL_BOARD_FIELD_DEFINITIONS.items():
            if definition["source_table"] == table and definition["field"] in header:
                final_board_hits.append(
                    {
                        "table": table,
                        "field": definition["field"],
                        "source_field": source_field,
                        "availability": definition["availability"],
                        "presence_allowed": True,
                        "allowed_use": "audit_and_street_truncation_only",
                        "direct_training_feature_use_allowed": False,
                    }
                )
    return {
        "status": "PASS",
        "missing_tables": missing_tables,
        "outcome_fields_present_in_raw_schema": schema_hits,
        "final_board_snapshot_fields_present_in_raw_schema": final_board_hits,
        "presence_is_not_feature_approval": True,
        "final_board_snapshot_presence_is_not_feature_approval": True,
    }


def audit_training_feature_names(project_root: Path, *, max_examples: int) -> dict[str, Any]:
    data_dir = project_root / "data"
    examples = load_training_examples(
        data_dir,
        max_examples=max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        include_hand_id=False,
    )
    feature_names = sorted({name for features, _ in examples for name in features})
    return {
        "status": "PASS" if not _forbidden_names(feature_names) else "FAIL",
        "dataset": str(data_dir),
        "examples_scanned": len(examples),
        "feature_count": len(feature_names),
        "forbidden_feature_names_detected": _forbidden_names(feature_names),
        "sample_feature_names": feature_names[:40],
    }


def audit_prediction_request_features() -> dict[str, Any]:
    request = PredictionRequest(
        position="BTN",
        street="turn",
        hole_cards=["AS", "KD"],
        board_cards=["2C", "7D", "QS", "TH"],
        pot=12.5,
        to_call=3.0,
        stack=96.0,
        min_raise=6.0,
        player_count=6,
        betting_history=[
            {"player_position": "UTG", "action": "raise", "amount": 3.0, "street": "preflop"},
            {"player_position": "BB", "action": "call", "amount": 3.0, "street": "preflop"},
        ],
    )
    feature_names = sorted(request_to_features(request))
    return {
        "status": "PASS" if not _forbidden_names(feature_names) else "FAIL",
        "feature_count": len(feature_names),
        "forbidden_feature_names_detected": _forbidden_names(feature_names),
    }


def audit_model_artifact_features(project_root: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    forbidden_detected: list[dict[str, Any]] = []
    for relative in MODEL_ARTIFACTS:
        path = project_root / relative
        if not path.exists():
            artifacts.append({"path": relative, "status": "MISSING", "feature_count": 0})
            continue
        try:
            model = load_policy(path)
            feature_names = sorted(_collect_model_feature_names(model))
            forbidden = _forbidden_names(feature_names)
            artifacts.append(
                {
                    "path": relative,
                    "status": "PASS" if not forbidden else "FAIL",
                    "feature_count": len(feature_names),
                    "forbidden_feature_names_detected": forbidden,
                }
            )
            forbidden_detected.extend({"path": relative, "feature": name} for name in forbidden)
        except Exception as exc:
            artifacts.append(
                {
                    "path": relative,
                    "status": "UNREADABLE",
                    "feature_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": "PASS" if not forbidden_detected else "FAIL",
        "artifacts": artifacts,
        "forbidden_model_features_detected": forbidden_detected,
    }


def audit_training_source_usage(project_root: Path) -> dict[str, Any]:
    usages: list[dict[str, Any]] = []
    for relative in GUARDED_SOURCE_FILES:
        path = project_root / relative
        if not path.exists():
            usages.append({"path": relative, "field": "__missing_file__", "line_number": 0})
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for field in FORBIDDEN_OUTCOME_FIELDS:
                if field in line:
                    usages.append(
                        {
                            "path": relative,
                            "field": field,
                            "line_number": line_number,
                            "line": line.strip(),
                        }
                    )
    return {
        "status": "PASS" if not usages else "FAIL",
        "guarded_sources": list(GUARDED_SOURCE_FILES),
        "forbidden_source_usages": usages,
    }


def validate_data_leakage_contract(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    risk = payload.get("leakage_risk_contract") or {}
    final_board_contract = payload.get("final_board_snapshot_contract") or {}
    final_board_policy = final_board_contract.get("feature_policy") or {}
    final_board_mitigation = final_board_contract.get("required_mitigation") or {}
    final_board_definitions = final_board_contract.get("field_definitions") or {}
    feature_policy = risk.get("feature_policy") or {}
    field_definitions = risk.get("field_definitions") or {}
    boundary = payload.get("leakage_boundary") or {}
    feature_audit = payload.get("feature_name_audit") or {}
    request_audit = payload.get("prediction_request_audit") or {}
    model_audit = payload.get("model_artifact_audit") or {}
    source_audit = payload.get("source_usage_audit") or {}
    raw_schema_audit = payload.get("raw_dataset_schema_audit") or {}

    if risk.get("risk_id") != "post_outcome_feature_leakage":
        violations.append("leakage_risk_id_must_be_explicit")
    if risk.get("root_cause") != "post_hand_outcome_fields_available_in_raw_dataset_schema":
        violations.append("leakage_root_cause_must_be_post_hand_outcome_fields")
    if risk.get("temporal_requirement") != "features_must_be_observable_before_target_action":
        violations.append("decision_time_temporal_requirement_must_be_explicit")
    if set(risk.get("forbidden_fields") or []) != set(FORBIDDEN_OUTCOME_FIELDS):
        violations.append("leakage_risk_forbidden_fields_must_match_contract")
    if set(field_definitions) != set(FORBIDDEN_OUTCOME_FIELDS):
        violations.append("outcome_field_definitions_must_cover_all_forbidden_fields")
    for field, definition in field_definitions.items():
        if definition.get("availability") not in {"post_hand", "post_hand_reconstruction"}:
            violations.append(f"outcome_field_temporal_availability_invalid:{field}")
    if feature_policy.get("raw_dataset_schema_presence") != "allowed_for_audit_and_reporting_only":
        violations.append("raw_outcome_schema_presence_must_be_audit_only")
    if feature_policy.get("training_feature_use") != "forbidden":
        violations.append("risk_contract_training_feature_use_must_be_forbidden")
    if feature_policy.get("prediction_request_use") != "forbidden":
        violations.append("risk_contract_prediction_request_use_must_be_forbidden")
    if feature_policy.get("model_artifact_feature_use") != "forbidden":
        violations.append("risk_contract_model_feature_use_must_be_forbidden")
    if feature_policy.get("detected_violation") != "production_blocker":
        violations.append("risk_contract_detected_violation_must_be_production_blocker")
    if final_board_contract.get("risk_id") != "final_board_snapshot_leakage":
        violations.append("final_board_leakage_risk_id_must_be_explicit")
    if final_board_contract.get("root_cause") != "hands_csv_board_cards_is_final_hand_snapshot":
        violations.append("final_board_root_cause_must_be_final_hand_snapshot")
    if (
        final_board_contract.get("temporal_requirement")
        != "board_features_must_be_truncated_to_cards_visible_at_target_street"
    ):
        violations.append("final_board_temporal_requirement_must_require_street_truncation")
    if set(final_board_contract.get("raw_final_board_snapshot_fields") or []) != set(RAW_FINAL_BOARD_SNAPSHOT_FIELDS):
        violations.append("raw_final_board_snapshot_fields_must_match_contract")
    if set(payload.get("raw_final_board_snapshot_fields") or []) != set(RAW_FINAL_BOARD_SNAPSHOT_FIELDS):
        violations.append("payload_raw_final_board_snapshot_fields_must_match_contract")
    if set(final_board_definitions) != set(RAW_FINAL_BOARD_SNAPSHOT_FIELDS):
        violations.append("final_board_field_definitions_must_cover_raw_snapshot_fields")
    for field, definition in final_board_definitions.items():
        if definition.get("availability") != "post_hand_final_snapshot":
            violations.append(f"final_board_temporal_availability_invalid:{field}")
    if final_board_policy.get("raw_dataset_schema_presence") != "allowed_for_audit_and_street_truncation_only":
        violations.append("raw_final_board_schema_presence_must_be_audit_and_truncation_only")
    if final_board_policy.get("direct_training_feature_use") != "forbidden":
        violations.append("direct_final_board_training_feature_use_must_be_forbidden")
    if final_board_policy.get("prediction_request_board_cards") != "allowed_only_as_decision_time_visible_board":
        violations.append("prediction_request_board_cards_must_be_visible_board_only")
    if final_board_policy.get("model_artifact_direct_final_board_feature_use") != "forbidden":
        violations.append("model_direct_final_board_feature_use_must_be_forbidden")
    if final_board_policy.get("detected_violation") != "production_blocker":
        violations.append("final_board_detected_violation_must_be_production_blocker")
    if final_board_mitigation.get("truncate_final_board_by_street") is not True:
        violations.append("final_board_mitigation_must_truncate_by_street")
    expected_visible_counts = {
        "preflop_visible_board_count": 0,
        "flop_visible_board_count": 3,
        "turn_visible_board_count": 4,
        "river_visible_board_count": 5,
    }
    for key, expected in expected_visible_counts.items():
        if final_board_mitigation.get(key) != expected:
            violations.append(f"final_board_visible_count_invalid:{key}")
    if set(payload.get("forbidden_outcome_fields") or []) != set(FORBIDDEN_OUTCOME_FIELDS):
        violations.append("forbidden_outcome_fields_must_match_contract")
    if boundary.get("status") != OUTCOME_ONLY_FIELD_STATUS:
        violations.append("outcome_fields_must_be_dataset_only")
    if boundary.get("decision_time_observability_required") is not True:
        violations.append("decision_time_observability_must_be_required")
    if boundary.get("training_feature_use_allowed") is not False:
        violations.append("outcome_fields_must_not_be_training_features")
    if boundary.get("prediction_request_use_allowed") is not False:
        violations.append("outcome_fields_must_not_be_prediction_request_fields")
    if boundary.get("model_artifact_feature_use_allowed") is not False:
        violations.append("outcome_fields_must_not_be_model_artifact_features")
    if boundary.get("direct_final_board_snapshot_feature_use_allowed") is not False:
        violations.append("direct_final_board_snapshot_must_not_be_feature")
    if boundary.get("decision_time_visible_board_cards_allowed") is not True:
        violations.append("decision_time_visible_board_cards_must_remain_allowed")
    if boundary.get("dataset_schema_presence_allowed") is not True:
        violations.append("raw_dataset_schema_may_retain_outcome_fields")
    if boundary.get("reporting_and_audit_use_allowed") is not True:
        violations.append("audit_reporting_use_must_remain_allowed")
    if boundary.get("production_blocker_if_detected") is not True:
        violations.append("leakage_detection_must_be_a_production_blocker")

    if feature_audit.get("status") != "PASS":
        violations.append("training_feature_names_must_be_leakage_free")
    if feature_audit.get("forbidden_feature_names_detected"):
        violations.append("forbidden_training_feature_names_detected")
    if int(feature_audit.get("examples_scanned") or 0) <= 0:
        violations.append("training_feature_audit_must_scan_examples")
    if request_audit.get("status") != "PASS":
        violations.append("prediction_request_features_must_be_leakage_free")
    if request_audit.get("forbidden_feature_names_detected"):
        violations.append("forbidden_prediction_request_features_detected")
    if model_audit.get("status") != "PASS":
        violations.append("model_artifact_features_must_be_leakage_free")
    if model_audit.get("forbidden_model_features_detected"):
        violations.append("forbidden_model_artifact_features_detected")
    if source_audit.get("status") != "PASS":
        violations.append("guarded_training_sources_must_not_reference_outcome_fields")
    if source_audit.get("forbidden_source_usages"):
        violations.append("forbidden_outcome_fields_used_in_guarded_source")
    if raw_schema_audit.get("status") != "PASS":
        violations.append("raw_dataset_schema_audit_must_pass")
    if raw_schema_audit.get("presence_is_not_feature_approval") is not True:
        violations.append("raw_schema_presence_must_not_equal_feature_approval")
    if raw_schema_audit.get("final_board_snapshot_presence_is_not_feature_approval") is not True:
        violations.append("raw_final_board_presence_must_not_equal_feature_approval")
    for item in raw_schema_audit.get("outcome_fields_present_in_raw_schema") or []:
        if item.get("presence_allowed") is not True:
            violations.append("raw_outcome_field_presence_must_remain_allowed_for_audit")
        if item.get("allowed_use") != "audit_reporting_settlement_only":
            violations.append("raw_outcome_field_allowed_use_must_be_audit_only")
    final_board_schema_hits = raw_schema_audit.get("final_board_snapshot_fields_present_in_raw_schema") or []
    if {item.get("source_field") for item in final_board_schema_hits} != set(RAW_FINAL_BOARD_SNAPSHOT_FIELDS):
        violations.append("raw_final_board_schema_audit_must_cover_snapshot_field")
    for item in final_board_schema_hits:
        if item.get("presence_allowed") is not True:
            violations.append("raw_final_board_presence_must_remain_allowed_for_truncation")
        if item.get("allowed_use") != "audit_and_street_truncation_only":
            violations.append("raw_final_board_allowed_use_must_be_truncation_only")
        if item.get("direct_training_feature_use_allowed") is not False:
            violations.append("raw_final_board_direct_training_feature_must_be_forbidden")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_data_leakage_contract(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    *,
    max_examples: int = 5000,
) -> dict[str, Any]:
    payload = build_data_leakage_contract(project_root, max_examples=max_examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_data_leakage_contract_markdown(payload), encoding="utf-8")
    return payload


def render_data_leakage_contract_markdown(payload: dict[str, Any]) -> str:
    risk = payload["leakage_risk_contract"]
    final_board = payload["final_board_snapshot_contract"]
    boundary = payload["leakage_boundary"]
    feature_audit = payload["feature_name_audit"]
    request_audit = payload["prediction_request_audit"]
    model_audit = payload["model_artifact_audit"]
    source_audit = payload["source_usage_audit"]
    raw_schema_audit = payload["raw_dataset_schema_audit"]
    lines = [
        "# Data Leakage Contract",
        "",
        payload["client_statement"],
        "",
        "## Risk Contract",
        "",
        f"- Risk ID: `{risk['risk_id']}`",
        f"- Root cause: `{risk['root_cause']}`",
        f"- Temporal requirement: `{risk['temporal_requirement']}`",
        f"- Training feature use: `{risk['feature_policy']['training_feature_use']}`",
        f"- Prediction request use: `{risk['feature_policy']['prediction_request_use']}`",
        f"- Model artifact feature use: `{risk['feature_policy']['model_artifact_feature_use']}`",
        f"- Detected violation: `{risk['feature_policy']['detected_violation']}`",
        "",
        "## Final Board Snapshot Boundary",
        "",
        f"- Risk ID: `{final_board['risk_id']}`",
        f"- Root cause: `{final_board['root_cause']}`",
        f"- Temporal requirement: `{final_board['temporal_requirement']}`",
        f"- Direct training feature use: `{final_board['feature_policy']['direct_training_feature_use']}`",
        f"- Prediction request board cards: `{final_board['feature_policy']['prediction_request_board_cards']}`",
        f"- Required mitigation: `truncate_final_board_by_street={final_board['required_mitigation']['truncate_final_board_by_street']}`",
        "",
        "## Outcome Field Definitions",
        "",
    ]
    for field in payload["forbidden_outcome_fields"]:
        definition = risk["field_definitions"][field]
        lines.append(
            f"- `{field}`: `{definition['source_table']}`, `{definition['availability']}` - {definition['reason']}"
        )
    lines.extend(
        [
            "",
        "## Forbidden Outcome Fields",
        "",
        ]
    )
    lines.extend(f"- `{field}`" for field in payload["forbidden_outcome_fields"])
    lines.extend(
        [
            "",
            "## Leakage Boundary",
            "",
            f"- Status: `{boundary['status']}`",
            f"- Training feature use allowed: `{boundary['training_feature_use_allowed']}`",
            f"- Prediction request use allowed: `{boundary['prediction_request_use_allowed']}`",
            f"- Model artifact feature use allowed: `{boundary['model_artifact_feature_use_allowed']}`",
            f"- Dataset schema presence allowed: `{boundary['dataset_schema_presence_allowed']}`",
            f"- Production blocker if detected: `{boundary['production_blocker_if_detected']}`",
            "",
            "## Audit Results",
            "",
            f"- Training feature audit: `{feature_audit['status']}`; examples scanned: `{feature_audit.get('examples_scanned')}`; feature count: `{feature_audit.get('feature_count')}`",
            f"- Prediction request audit: `{request_audit['status']}`; feature count: `{request_audit.get('feature_count')}`",
            f"- Model artifact audit: `{model_audit['status']}`",
            f"- Source usage audit: `{source_audit['status']}`",
            f"- Raw dataset schema audit: `{raw_schema_audit['status']}`; outcome fields present: `{len(raw_schema_audit.get('outcome_fields_present_in_raw_schema') or [])}`",
            "",
            "## Implemented Fixes",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["implemented_fixes"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _forbidden_names(feature_names: list[str]) -> list[str]:
    return forbidden_outcome_feature_names(feature_names)


def _collect_model_feature_names(model: Any) -> set[str]:
    names: set[str] = set()
    raw_names = getattr(model, "feature_names", None)
    if raw_names:
        names.update(str(name) for name in raw_names)
    for attr in ("observed_policy", "missing_policy", "base_policy", "policy"):
        child = getattr(model, attr, None)
        if child is not None and child is not model:
            names.update(_collect_model_feature_names(child))
    return names
