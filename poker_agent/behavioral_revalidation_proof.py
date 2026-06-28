from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.behavioral_revalidation import build_behavioral_revalidation, validate_behavioral_revalidation


BEHAVIORAL_REVALIDATION_PROOF_VERSION = "2026-06-28"


def build_behavioral_revalidation_proof(project_root: Path) -> dict[str, Any]:
    base = build_behavioral_revalidation(project_root)
    proof_cases = [
        _case(
            "base_contract_is_valid",
            base,
            expected_status="PASS",
            description="Current validation scope may claim PASS for human-likeness and action distribution.",
        ),
        _tamper_case(
            "blocks_missing_larger_real_gameplay_revalidation",
            base,
            [("revalidation_boundary", "larger_clean_real_gameplay_revalidation_required", False)],
            description="The contract must fail if larger clean real-gameplay revalidation is disabled.",
        ),
        _tamper_case(
            "blocks_generalized_human_likeness_claim",
            base,
            [("revalidation_boundary", "generalized_human_likeness_claim_allowed", True)],
            description="The contract must fail if current-scope results are generalized to all real gameplay.",
        ),
        _tamper_case(
            "blocks_generalized_action_distribution_claim",
            base,
            [("revalidation_boundary", "generalized_action_distribution_claim_allowed", True)],
            description="The contract must fail if action distribution is presented as final global evidence.",
        ),
        _tamper_case(
            "blocks_wrong_revalidation_scope",
            base,
            [("revalidation_boundary", "revalidation_scope", "current_delivery_validation_scope")],
            description="The contract must fail if revalidation is not targeted at larger clean real gameplay data.",
        ),
    ]
    all_passed = all(case["passed"] for case in proof_cases)
    payload = {
        "version": BEHAVIORAL_REVALIDATION_PROOF_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Executable proof for behavioral revalidation boundary",
        "proof_status": "PASS" if all_passed else "FAIL",
        "client_statement_proven": (
            "Human-likeness and action-distribution checks pass for the current validation scope, "
            "but they cannot be presented as final proof for larger clean real gameplay without revalidation."
        ),
        "base_contract_summary": {
            "overall_status": base.get("overall_status"),
            "human_likeness_status": (base.get("current_validation_scope") or {}).get("human_likeness_status"),
            "action_distribution_status": (base.get("current_validation_scope") or {}).get("action_distribution_status"),
            "larger_clean_real_gameplay_revalidation_required": (
                base.get("revalidation_boundary") or {}
            ).get("larger_clean_real_gameplay_revalidation_required"),
            "generalized_human_likeness_claim_allowed": (
                base.get("revalidation_boundary") or {}
            ).get("generalized_human_likeness_claim_allowed"),
            "generalized_action_distribution_claim_allowed": (
                base.get("revalidation_boundary") or {}
            ).get("generalized_action_distribution_claim_allowed"),
        },
        "proof_cases": proof_cases,
        "verifier_enforcement": {
            "script": "scripts/verify_delivery.py",
            "requires_report": "reports/behavioral_revalidation.json",
            "requires_proof_report": "reports/behavioral_revalidation_proof.json",
            "blocks_false_generalization": True,
        },
    }
    payload["invariants"] = validate_behavioral_revalidation_proof(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_behavioral_revalidation_proof(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    if payload.get("proof_status") != "PASS":
        violations.append("proof_cases_did_not_all_pass")
    cases = {case.get("name"): case for case in payload.get("proof_cases") or []}
    required = {
        "base_contract_is_valid",
        "blocks_missing_larger_real_gameplay_revalidation",
        "blocks_generalized_human_likeness_claim",
        "blocks_generalized_action_distribution_claim",
        "blocks_wrong_revalidation_scope",
    }
    missing = sorted(required - set(cases))
    if missing:
        violations.append(f"missing_proof_cases={missing}")
    for name in required:
        case = cases.get(name) or {}
        if case.get("passed") is not True:
            violations.append(f"proof_case_failed={name}")
    summary = payload.get("base_contract_summary") or {}
    if summary.get("human_likeness_status") != "PASS":
        violations.append("base_human_likeness_not_pass")
    if summary.get("action_distribution_status") != "PASS":
        violations.append("base_action_distribution_not_pass")
    if summary.get("larger_clean_real_gameplay_revalidation_required") is not True:
        violations.append("base_revalidation_not_required")
    if summary.get("generalized_human_likeness_claim_allowed") is not False:
        violations.append("base_generalized_human_likeness_not_blocked")
    if summary.get("generalized_action_distribution_claim_allowed") is not False:
        violations.append("base_generalized_action_distribution_not_blocked")
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_behavioral_revalidation_proof(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_behavioral_revalidation_proof(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_behavioral_revalidation_proof_markdown(payload), encoding="utf-8")
    return payload


def render_behavioral_revalidation_proof_markdown(payload: dict[str, Any]) -> str:
    summary = payload["base_contract_summary"]
    lines = [
        "# Behavioral Revalidation Proof",
        "",
        payload["client_statement_proven"],
        "",
        f"- Proof status: `{payload['proof_status']}`",
        f"- Human-likeness status: `{summary['human_likeness_status']}`",
        f"- Action-distribution status: `{summary['action_distribution_status']}`",
        f"- Larger clean real gameplay revalidation required: `{summary['larger_clean_real_gameplay_revalidation_required']}`",
        f"- Generalized human-likeness claim allowed: `{summary['generalized_human_likeness_claim_allowed']}`",
        f"- Generalized action-distribution claim allowed: `{summary['generalized_action_distribution_claim_allowed']}`",
        "",
        "## Executed Proof Cases",
        "",
    ]
    for case in payload["proof_cases"]:
        lines.append(f"- `{case['name']}`: passed=`{case['passed']}`, observed=`{case['observed_status']}`")
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _case(name: str, payload: dict[str, Any], *, expected_status: str, description: str) -> dict[str, Any]:
    observed = validate_behavioral_revalidation(payload)
    observed_status = observed["status"]
    return {
        "name": name,
        "description": description,
        "expected_status": expected_status,
        "observed_status": observed_status,
        "passed": observed_status == expected_status,
        "violations": observed.get("violations", []),
    }


def _tamper_case(
    name: str,
    base: dict[str, Any],
    mutations: list[tuple[str, str, Any]],
    *,
    description: str,
) -> dict[str, Any]:
    mutated = copy.deepcopy(base)
    for section, key, value in mutations:
        mutated.setdefault(section, {})[key] = value
    return _case(name, mutated, expected_status="FAIL", description=description)
