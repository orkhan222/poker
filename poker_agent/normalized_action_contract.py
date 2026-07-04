from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.action_normalization import CANONICAL_ACTIONS, normalize_action_result
from poker_agent.features import load_training_examples


NORMALIZED_ACTION_CONTRACT_VERSION = "2026-07-04"
NORMALIZED_ACTION_STATUS = "IMPLEMENTED"
RAW_ACTION_SOURCE_STATUS = "RAW_OCR_OR_DEALER_TEXT"

NOISY_ACTION_CONTRACT_EXAMPLES = (
    ("ra1se", "raise"),
    ("Plyr3 ra1se $4.50", "raise"),
    ("cail", "call"),
    ("bett", "bet"),
    ("all-in", "all_in"),
    ("all in", "all_in"),
    ("checks", "check"),
    ("f0ld", "fold"),
)


def build_normalized_action_contract(project_root: Path, *, max_rows: int = 5000) -> dict[str, Any]:
    schema_audit = audit_actions_csv_normalization(project_root, max_rows=max_rows)
    training_label_audit = audit_training_labels(project_root, max_examples=max_rows)
    examples = [
        {
            "raw_action": raw,
            "expected": expected,
            "observed": normalize_action_result(raw).canonical_action,
            "method": normalize_action_result(raw).method,
            "passed": normalize_action_result(raw).canonical_action == expected,
        }
        for raw, expected in NOISY_ACTION_CONTRACT_EXAMPLES
    ]
    payload: dict[str, Any] = {
        "version": NORMALIZED_ACTION_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Normalized action label contract",
        "raw_action_source_status": RAW_ACTION_SOURCE_STATUS,
        "normalized_action_status": NORMALIZED_ACTION_STATUS,
        "canonical_actions": list(CANONICAL_ACTIONS),
        "source_field": "actions.csv::action",
        "normalized_field": "canonical_action",
        "raw_ocr_action_must_not_be_training_label": True,
        "normalization_required_before_training": True,
        "normalization_required_before_evaluation": True,
        "normalization_required_before_policy_comparison": True,
        "current_delivery_blocker": False,
        "model_quality_risk": False,
        "noisy_action_examples": examples,
        "actions_csv_audit": schema_audit,
        "training_label_audit": training_label_audit,
        "allowed_claims": [
            "Raw OCR/dealer action text is normalized into the canonical poker action set before supervised training.",
            "The current canonical action set is fold/call/check/bet/raise/all_in.",
        ],
        "not_allowed_claims": [
            "Raw OCR action strings such as ra1se, cail, bett, or all-in are valid model labels without normalization.",
            "A model-quality comparison can mix raw OCR labels and canonical action labels.",
            "Unknown or non-decision action text can be silently promoted to a canonical training label.",
        ],
    }
    payload["invariants"] = validate_normalized_action_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def audit_actions_csv_normalization(project_root: Path, *, max_rows: int) -> dict[str, Any]:
    actions_path = project_root / "data" / "actions.csv"
    if not actions_path.exists():
        return {
            "status": "FAIL",
            "path": str(actions_path),
            "action_column_present": False,
            "rows_scanned": 0,
            "canonical_decision_rows": 0,
            "unknown_rows": 0,
            "canonical_action_counts": {},
            "sample_normalized_rows": [],
        }

    canonical_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    action_column_present = False
    rows_scanned = 0

    with actions_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        action_column_present = "action" in (reader.fieldnames or [])
        for row in reader:
            if rows_scanned >= max_rows:
                break
            rows_scanned += 1
            result = normalize_action_result(row.get("action"))
            status_counts[result.status] += 1
            if result.is_decision_action:
                canonical_counts[result.canonical_action] += 1
            if len(samples) < 12:
                samples.append(
                    {
                        "raw_action": result.raw_action,
                        "canonical_action": result.canonical_action,
                        "status": result.status,
                        "method": result.method,
                        "confidence": result.confidence,
                    }
                )

    return {
        "status": "PASS" if action_column_present and rows_scanned > 0 else "FAIL",
        "path": str(actions_path),
        "action_column_present": action_column_present,
        "rows_scanned": rows_scanned,
        "canonical_decision_rows": sum(canonical_counts.values()),
        "unknown_rows": status_counts.get("unknown", 0),
        "non_decision_rows": status_counts.get("non_decision", 0),
        "canonical_action_counts": dict(sorted(canonical_counts.items())),
        "normalization_status_counts": dict(sorted(status_counts.items())),
        "sample_normalized_rows": samples,
    }


def audit_training_labels(project_root: Path, *, max_examples: int) -> dict[str, Any]:
    examples = load_training_examples(
        project_root / "data",
        max_examples=max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        include_hand_id=False,
    )
    labels = [label for _, label in examples]
    invalid = sorted({label for label in labels if label not in CANONICAL_ACTIONS})
    return {
        "status": "PASS" if examples and not invalid else "FAIL",
        "examples_scanned": len(examples),
        "canonical_label_counts": dict(sorted(Counter(labels).items())),
        "invalid_labels": invalid,
    }


def validate_normalized_action_contract(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    actions_audit = payload.get("actions_csv_audit") or {}
    training_audit = payload.get("training_label_audit") or {}
    examples = payload.get("noisy_action_examples") or []

    if payload.get("normalized_action_status") != NORMALIZED_ACTION_STATUS:
        violations.append("normalized_action_status_must_be_implemented")
    if payload.get("raw_action_source_status") != RAW_ACTION_SOURCE_STATUS:
        violations.append("raw_action_source_status_must_be_declared")
    if set(payload.get("canonical_actions") or []) != set(CANONICAL_ACTIONS):
        violations.append("canonical_actions_must_match_contract")
    if payload.get("raw_ocr_action_must_not_be_training_label") is not True:
        violations.append("raw_ocr_action_must_not_be_training_label")
    if payload.get("normalization_required_before_training") is not True:
        violations.append("normalization_must_be_required_before_training")
    if payload.get("normalization_required_before_evaluation") is not True:
        violations.append("normalization_must_be_required_before_evaluation")
    if payload.get("normalization_required_before_policy_comparison") is not True:
        violations.append("normalization_must_be_required_before_policy_comparison")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("normalized_action_contract_must_not_block_current_delivery")
    if payload.get("model_quality_risk") is not False:
        violations.append("implemented_normalized_action_contract_must_not_remain_open_model_quality_risk")
    if actions_audit.get("status") != "PASS":
        violations.append("actions_csv_action_column_must_be_readable")
    if actions_audit.get("action_column_present") is not True:
        violations.append("actions_csv_action_column_must_exist")
    if int(actions_audit.get("rows_scanned") or 0) <= 0:
        violations.append("actions_csv_normalization_must_scan_rows")
    if int(actions_audit.get("canonical_decision_rows") or 0) <= 0:
        violations.append("actions_csv_normalization_must_find_decision_rows")
    if training_audit.get("status") != "PASS":
        violations.append("training_labels_must_be_canonical")
    if training_audit.get("invalid_labels"):
        violations.append("training_labels_must_not_contain_raw_ocr_actions")
    for example in examples:
        if example.get("passed") is not True:
            violations.append(f"noisy_action_example_must_normalize:{example.get('raw_action')}")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_normalized_action_contract(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    *,
    max_rows: int = 5000,
) -> dict[str, Any]:
    payload = build_normalized_action_contract(project_root, max_rows=max_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_normalized_action_contract_markdown(payload), encoding="utf-8")
    return payload


def render_normalized_action_contract_markdown(payload: dict[str, Any]) -> str:
    actions_audit = payload["actions_csv_audit"]
    training_audit = payload["training_label_audit"]
    lines = [
        "# Normalized Action Contract",
        "",
        "Raw OCR/dealer action text must be normalized before training, evaluation, and policy comparison.",
        "",
        f"- Status: `{payload['normalized_action_status']}`",
        f"- Source field: `{payload['source_field']}`",
        f"- Normalized field: `{payload['normalized_field']}`",
        f"- Canonical actions: `{', '.join(payload['canonical_actions'])}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## OCR Examples",
        "",
    ]
    for example in payload["noisy_action_examples"]:
        lines.append(
            f"- `{example['raw_action']}` -> `{example['observed']}` "
            f"(expected `{example['expected']}`, passed=`{example['passed']}`)"
        )
    lines.extend(
        [
            "",
            "## Dataset Audit",
            "",
            f"- Rows scanned: `{actions_audit['rows_scanned']}`",
            f"- Canonical decision rows: `{actions_audit['canonical_decision_rows']}`",
            f"- Unknown rows: `{actions_audit['unknown_rows']}`",
            "",
            "## Training Label Audit",
            "",
            f"- Status: `{training_audit['status']}`",
            f"- Examples scanned: `{training_audit['examples_scanned']}`",
            f"- Invalid labels: `{training_audit['invalid_labels']}`",
            "",
            f"Invariant status: `{payload['invariants']['status']}`",
            "",
        ]
    )
    return "\n".join(lines)
