from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.human_likeness_evidence import (
    HUMAN_LIKENESS_BOUNDARY,
    REQUIRED_BEHAVIOR_DIMENSIONS,
    build_human_likeness_evidence,
)
from poker_agent.human_likeness_policy_guard import (
    CLAIM_DECISION_BLOCKED,
    FINAL_CLAIM_BLOCKING_REASONS,
    FULL_HUMAN_LIKENESS_CLAIM,
    evaluate_full_human_likeness_claim,
)


HUMAN_LIKENESS_CLAIM_GATE_VERSION = "2026-07-03"
HUMAN_LIKENESS_CLAIM = FULL_HUMAN_LIKENESS_CLAIM
HUMAN_LIKENESS_CLAIM_DECISION = CLAIM_DECISION_BLOCKED


def build_human_likeness_claim_gate(project_root: Path) -> dict[str, Any]:
    evidence = _load_or_build_evidence(project_root)
    decision = evaluate_full_human_likeness_claim(evidence)

    payload: dict[str, Any] = {
        "version": HUMAN_LIKENESS_CLAIM_GATE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "human_likeness_claim_gate",
        **decision,
    }
    payload["proof_cases"] = build_human_likeness_claim_gate_proof_cases(payload)
    payload["invariants"] = validate_human_likeness_claim_gate(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def build_human_likeness_claim_gate_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_human_likeness_claim_gate(candidate)
        observed = candidate["invariants"]["status"]
        cases.append(
            {
                "name": name,
                "expected_status": expected_status,
                "observed_status": observed,
                "result": "PASS" if observed == expected_status else "FAIL",
                "violations": candidate["invariants"].get("violations", []),
            }
        )

    record("base_contract_valid", json.loads(json.dumps(payload)), "PASS")

    candidate = json.loads(json.dumps(payload))
    candidate["claim_allowed"] = True
    candidate["decision"] = "APPROVED"
    candidate["human_likeness_fully_proven"] = True
    record("blocks_full_human_likeness_approval", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["action_distribution_only_proof_rejected"] = False
    record("blocks_action_distribution_only_proof", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["evidence_requirements"].pop("timing", None)
    record("blocks_missing_timing_dimension", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["evidence_requirements"]["bet_sizing"]["currently_sufficient_for_final_claim"] = True
    record("blocks_unreviewed_bet_sizing_sufficiency", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["evidence_requirements"]["timing"]["currently_sufficient_for_final_claim"] = True
    record("blocks_unreviewed_timing_sufficiency", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["evidence_requirements"]["position_based_behavior"]["currently_sufficient_for_final_claim"] = True
    record("blocks_unreviewed_position_based_sufficiency", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["evidence_requirements"]["street_level_strategy"]["currently_sufficient_for_final_claim"] = True
    record("blocks_unreviewed_street_level_sufficiency", candidate, "FAIL")

    return cases


def validate_human_likeness_claim_gate(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    requirements = payload.get("evidence_requirements") or {}
    source_evidence = payload.get("source_evidence") or {}

    if payload.get("boundary") != HUMAN_LIKENESS_BOUNDARY:
        violations.append("human_likeness_claim_boundary_must_be_present")
    if payload.get("claim") != HUMAN_LIKENESS_CLAIM:
        violations.append("human_likeness_claim_name_must_be_full_human_likeness")
    if payload.get("decision") != HUMAN_LIKENESS_CLAIM_DECISION:
        violations.append("full_human_likeness_decision_must_remain_blocked")
    if payload.get("claim_allowed") is not False:
        violations.append("full_human_likeness_claim_must_not_be_allowed")
    if payload.get("human_likeness_fully_proven") is not False:
        violations.append("human_likeness_must_not_be_marked_fully_proven")
    if payload.get("action_distribution_only_proof_rejected") is not True:
        violations.append("action_distribution_only_proof_must_be_rejected")
    if payload.get("current_scope_action_distribution_passed") is not True:
        violations.append("current_scope_action_distribution_must_remain_the_limited_pass")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("human_likeness_claim_gap_must_not_block_current_delivery")
    if payload.get("model_quality_risk") is not True:
        violations.append("human_likeness_claim_gap_must_remain_model_quality_risk")
    if source_evidence.get("overall_status") != "PASS" or source_evidence.get("invariant_status") != "PASS":
        violations.append("source_human_likeness_evidence_must_pass")
    if source_evidence.get("final_human_likeness_claim_allowed") is not False:
        violations.append("source_final_human_likeness_claim_must_be_blocked")
    if set(payload.get("required_evidence_dimensions") or []) != set(REQUIRED_BEHAVIOR_DIMENSIONS):
        violations.append("human_likeness_claim_dimensions_must_be_complete")

    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        item = requirements.get(dimension) or {}
        if not item:
            violations.append(f"human_likeness_claim_dimension_missing:{dimension}")
            continue
        if item.get("required_for_final_claim") is not True:
            violations.append(f"human_likeness_claim_dimension_must_be_required:{dimension}")
        if item.get("currently_sufficient_for_final_claim") is not False:
            violations.append(f"human_likeness_claim_dimension_must_not_be_currently_sufficient:{dimension}")

    reasons = set(payload.get("blocking_reasons") or [])
    required_reasons = set(FINAL_CLAIM_BLOCKING_REASONS)
    missing_reasons = required_reasons - reasons
    if missing_reasons:
        violations.append(f"human_likeness_claim_missing_blocking_reasons:{','.join(sorted(missing_reasons))}")

    return {"status": "PASS" if not violations else "FAIL", "violations": violations}


def write_human_likeness_claim_gate(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_human_likeness_claim_gate(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_human_likeness_claim_gate_markdown(payload), encoding="utf-8")
    return payload


def render_human_likeness_claim_gate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Human-Likeness Claim Gate",
        "",
        "Full human-likeness is not approved from action distribution alone.",
        "",
        f"- Claim: `{payload['claim']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Claim allowed: `{payload['claim_allowed']}`",
        f"- Action-distribution-only proof rejected: `{payload['action_distribution_only_proof_rejected']}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## Required Evidence",
        "",
    ]
    for name, item in payload["evidence_requirements"].items():
        lines.append(
            f"- `{name}`: required=`{item['required_for_final_claim']}`, "
            f"currently_sufficient=`{item['currently_sufficient_for_final_claim']}`, "
            f"status=`{item['current_status']}`"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in payload["blocking_reasons"])
    lines.extend(["", "## Proof Cases", ""])
    for case in payload["proof_cases"]:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, result `{case['result']}`"
        )
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _load_or_build_evidence(project_root: Path) -> dict[str, Any]:
    evidence_path = project_root / "reports" / "human_likeness_evidence.json"
    if evidence_path.exists():
        return json.loads(evidence_path.read_text(encoding="utf-8"))
    return build_human_likeness_evidence(project_root)
