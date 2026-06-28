from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BEHAVIORAL_REVALIDATION_VERSION = "2026-06-28"
CURRENT_SCOPE = "current_delivery_validation_scope"
REVALIDATION_SCOPE = "larger_clean_real_gameplay_data"


def build_behavioral_revalidation(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    acceptance = _read_optional_json(reports / "policy_acceptance.json")
    deployed_gate = _read_optional_json(reports / "deployed_strategy_gate.json")
    maturity = _read_optional_json(reports / "strategy_stack_maturity.json")

    likeness = acceptance.get("human_likeness") or {}
    js_divergence = likeness.get("js_divergence")
    human_likeness_status = likeness.get("status", "UNKNOWN")
    action_distribution_status = "PASS" if human_likeness_status == "PASS" else "UNKNOWN"
    current_scope_passed = human_likeness_status == "PASS" and action_distribution_status == "PASS"

    payload: dict[str, Any] = {
        "version": BEHAVIORAL_REVALIDATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Human-likeness and action-distribution revalidation boundary",
        "client_statement": (
            "Human-likeness and action-distribution checks pass for the current validation scope, "
            "but they should be revalidated on larger and cleaner real gameplay data."
        ),
        "current_validation_scope": {
            "scope": CURRENT_SCOPE,
            "overall_status": "PASS" if current_scope_passed else "NOT_PASS",
            "human_likeness_status": human_likeness_status,
            "action_distribution_status": action_distribution_status,
            "action_distribution_source_metric": "policy_acceptance.human_likeness.js_divergence",
            "js_divergence": js_divergence,
            "timing_and_bet_size_status": likeness.get("timing_and_bet_size_status", "UNKNOWN"),
            "strategy_policy_status": deployed_gate.get("strategy_policy_status", "UNKNOWN"),
            "deployment_maturity_status": (maturity.get("current_strategy_stack") or {}).get("status", "UNKNOWN"),
        },
        "revalidation_boundary": {
            "larger_clean_real_gameplay_revalidation_required": True,
            "revalidation_scope": REVALIDATION_SCOPE,
            "current_scope_claim_allowed": current_scope_passed,
            "generalized_human_likeness_claim_allowed": False,
            "generalized_action_distribution_claim_allowed": False,
            "production_blocker": False,
            "reason": (
                "The current gates validate delivery behavior on the available held-out/simulation scope. "
                "They do not replace a larger reviewed real-gameplay validation set with cleaner OCR, "
                "more diverse opponents, and longer action histories."
            ),
        },
        "minimum_revalidation_dataset_requirements": [
            "Larger reviewed real gameplay sample covering multiple sessions and table states.",
            "Cleaner reconstructed actions with explicit audit of OCR/dealer-message corruption.",
            "Balanced coverage across fold, check, call, bet, and raise actions.",
            "Observed-card and missing-card slices reported separately.",
            "Opponent profile, street, pot-size, stack-depth, and bet-size slices reported separately.",
        ],
        "metrics_to_revalidate": [
            "human action alignment accuracy and macro F1",
            "action distribution divergence",
            "bet-size distribution similarity",
            "timing distribution similarity",
            "slice-level drift against current deployment metrics",
        ],
        "allowed_claims": [
            "Human-likeness and action-distribution checks pass for the current validation scope.",
            "The deployed stack can be monitored in rollout while larger real-gameplay validation is collected.",
        ],
        "not_allowed_claims": [
            "Human-likeness is fully proven across all real gameplay conditions.",
            "Action distribution is final and no longer needs validation.",
            "Current validation scope is a substitute for a larger clean production gameplay set.",
        ],
    }
    payload["invariants"] = validate_behavioral_revalidation(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_behavioral_revalidation(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    current = payload.get("current_validation_scope") or {}
    boundary = payload.get("revalidation_boundary") or {}

    if current.get("human_likeness_status") != "PASS":
        violations.append("Human-likeness must pass before this current-scope claim is allowed.")
    if current.get("action_distribution_status") != "PASS":
        violations.append("Action-distribution check must pass for the current scope.")
    if boundary.get("larger_clean_real_gameplay_revalidation_required") is not True:
        violations.append("Larger clean real-gameplay revalidation must remain required.")
    if boundary.get("generalized_human_likeness_claim_allowed") is not False:
        violations.append("Generalized human-likeness claims must remain blocked.")
    if boundary.get("generalized_action_distribution_claim_allowed") is not False:
        violations.append("Generalized action-distribution claims must remain blocked.")
    if boundary.get("production_blocker") is not False:
        violations.append("Revalidation requirement should not block the current monitored deployment package.")
    if boundary.get("revalidation_scope") != REVALIDATION_SCOPE:
        violations.append("Revalidation scope must target larger clean real gameplay data.")

    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def write_behavioral_revalidation(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_behavioral_revalidation(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_behavioral_revalidation_markdown(payload), encoding="utf-8")
    return payload


def render_behavioral_revalidation_markdown(payload: dict[str, Any]) -> str:
    current = payload["current_validation_scope"]
    boundary = payload["revalidation_boundary"]
    lines = [
        "# Behavioral Revalidation Contract",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Current Validation Scope",
        "",
        f"- Scope: `{current['scope']}`",
        f"- Overall status: `{current['overall_status']}`",
        f"- Human-likeness status: `{current['human_likeness_status']}`",
        f"- Action-distribution status: `{current['action_distribution_status']}`",
        f"- JS divergence: `{current['js_divergence']}`",
        "",
        "## Revalidation Boundary",
        "",
        f"- Larger clean real gameplay revalidation required: `{boundary['larger_clean_real_gameplay_revalidation_required']}`",
        f"- Generalized human-likeness claim allowed: `{boundary['generalized_human_likeness_claim_allowed']}`",
        f"- Generalized action-distribution claim allowed: `{boundary['generalized_action_distribution_claim_allowed']}`",
        f"- Production blocker: `{boundary['production_blocker']}`",
        "",
        "## Dataset Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["minimum_revalidation_dataset_requirements"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
