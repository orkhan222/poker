from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.features import load_training_examples, request_to_features
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest


DATA_LEAKAGE_CONTRACT_VERSION = "2026-07-02"
LEAKAGE_GUARD_STATUS = "PASS_NO_OUTCOME_FEATURES"
OUTCOME_ONLY_FIELD_STATUS = "DATASET_ONLY_NOT_TRAINING_FEATURES"

FORBIDDEN_OUTCOME_FIELDS = (
    "winner_positions",
    "stack_delta",
    "ending_stack",
    "dealer_winner",
    "dealer_pot",
    "pot_from_stacks",
)

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
            "winner_positions, stack_delta, ending_stack, dealer_winner, dealer_pot, and "
            "pot_from_stacks are outcome-only or post-hand fields. They must not be used as "
            "training or prediction features because they would let the model observe future information."
        ),
        "forbidden_outcome_fields": list(FORBIDDEN_OUTCOME_FIELDS),
        "leakage_boundary": {
            "status": OUTCOME_ONLY_FIELD_STATUS,
            "decision_time_observability_required": True,
            "training_feature_use_allowed": False,
            "prediction_request_use_allowed": False,
            "model_artifact_feature_use_allowed": False,
            "dataset_schema_presence_allowed": True,
            "reporting_and_audit_use_allowed": True,
            "current_deployment_blocker": False,
            "production_blocker_if_detected": True,
        },
        "feature_name_audit": feature_audit,
        "prediction_request_audit": request_audit,
        "model_artifact_audit": model_audit,
        "source_usage_audit": source_audit,
        "implemented_fixes": [
            "Removed ending_stack fallback from supervised training feature extraction.",
            "Removed ending_stack fallback from decision-context holdout generation.",
            "Added a machine-readable leakage contract covering feature names, model artifacts, and guarded training sources.",
        ],
        "allowed_claims": [
            "Outcome-only fields may remain in the raw CSV schema for audit, reporting, and downstream settlement analysis.",
            "Training and prediction features are restricted to information observable before the target action.",
        ],
        "not_allowed_claims": [
            "winner_positions, stack_delta, ending_stack, dealer_winner, dealer_pot, or pot_from_stacks are safe training features.",
            "A model trained with post-hand outcome fields can be accepted as leakage-free.",
            "ending_stack can be used as a fallback for decision-time stack when starting_stack is missing.",
        ],
    }
    payload["invariants"] = validate_data_leakage_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


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
    boundary = payload.get("leakage_boundary") or {}
    feature_audit = payload.get("feature_name_audit") or {}
    request_audit = payload.get("prediction_request_audit") or {}
    model_audit = payload.get("model_artifact_audit") or {}
    source_audit = payload.get("source_usage_audit") or {}

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
    boundary = payload["leakage_boundary"]
    feature_audit = payload["feature_name_audit"]
    request_audit = payload["prediction_request_audit"]
    model_audit = payload["model_artifact_audit"]
    source_audit = payload["source_usage_audit"]
    lines = [
        "# Data Leakage Contract",
        "",
        payload["client_statement"],
        "",
        "## Forbidden Outcome Fields",
        "",
    ]
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
    return sorted(
        {
            name
            for name in feature_names
            for forbidden in FORBIDDEN_OUTCOME_FIELDS
            if forbidden in name
        }
    )


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
