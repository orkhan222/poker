from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HUMAN_LIKENESS_EVIDENCE_VERSION = "2026-07-03"
HUMAN_LIKENESS_BOUNDARY = "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF"
FINAL_HUMAN_LIKENESS_STATUS = "NOT_FULLY_PROVEN"

REQUIRED_BEHAVIOR_DIMENSIONS = (
    "action_distribution",
    "bet_sizing",
    "timing",
    "position_based_behavior",
    "street_level_strategy",
)


def build_human_likeness_evidence(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    behavioral = _read_optional_json(reports / "behavioral_revalidation.json")
    bet_timing = _read_optional_json(reports / "bet_timing_calibration.json")
    policy_acceptance = _read_optional_json(reports / "policy_acceptance.json")

    behavioral_scope = behavioral.get("current_validation_scope") or {}
    behavioral_boundary = behavioral.get("revalidation_boundary") or {}
    bet_current = bet_timing.get("current_delivery_scope") or {}
    bet_boundary = bet_timing.get("calibration_boundary") or {}
    timing_boundary = bet_timing.get("timing_label_quality_boundary") or {}
    likeness = policy_acceptance.get("human_likeness") or {}

    dimensions = {
        "action_distribution": {
            "required": True,
            "current_status": behavioral_scope.get("action_distribution_status") or likeness.get("status"),
            "current_scope_metric": "js_divergence",
            "current_scope_value": behavioral_scope.get("js_divergence") or likeness.get("js_divergence"),
            "final_proof_allowed": False,
            "remaining_requirement": "Revalidate distribution similarity on larger clean real gameplay data.",
            "source_reports": ["reports/policy_acceptance.json", "reports/behavioral_revalidation.json"],
        },
        "bet_sizing": {
            "required": True,
            "current_status": bet_current.get("timing_and_bet_size_status"),
            "current_scope_metric": "bet_size_mae",
            "current_scope_value": bet_current.get("bet_size_mae"),
            "final_proof_allowed": False,
            "remaining_requirement": "Collect reviewed real-player bet-size labels by action, street, position, SPR, and pot size.",
            "source_report": "reports/bet_timing_calibration.json",
        },
        "timing": {
            "required": True,
            "current_status": bet_current.get("timing_and_bet_size_status"),
            "current_scope_metric": "decision_time_mae",
            "current_scope_value": bet_current.get("decision_time_mae"),
            "timing_policy_type": timing_boundary.get("timing_policy_type"),
            "real_human_timing_labels_available": timing_boundary.get("real_human_timing_labels_available"),
            "requires_real_human_timing_labels": timing_boundary.get("requires_real_human_timing_labels"),
            "heuristic_timing_counts_as_full_human_likeness_proof": timing_boundary.get(
                "heuristic_timing_counts_as_full_human_likeness_proof"
            ),
            "final_proof_allowed": False,
            "remaining_requirement": "Calibrate decision timing against reviewed real human timing labels.",
            "source_report": "reports/bet_timing_calibration.json",
        },
        "position_based_behavior": {
            "required": True,
            "current_status": "REQUIRES_SLICE_REVALIDATION",
            "current_scope_metric": "position_slice_similarity",
            "current_scope_value": None,
            "final_proof_allowed": False,
            "remaining_requirement": "Measure behavior by hero position, opponent position, blind state, and table size.",
            "source_reports": ["reports/behavioral_revalidation.json", "reports/final_strategy_quality_status.json"],
        },
        "street_level_strategy": {
            "required": True,
            "current_status": "REQUIRES_SLICE_REVALIDATION",
            "current_scope_metric": "street_slice_similarity",
            "current_scope_value": None,
            "final_proof_allowed": False,
            "remaining_requirement": "Measure strategy slices separately for preflop, flop, turn, and river.",
            "source_reports": ["reports/behavioral_revalidation.json", "reports/final_strategy_quality_status.json"],
        },
    }

    payload: dict[str, Any] = {
        "version": HUMAN_LIKENESS_EVIDENCE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": HUMAN_LIKENESS_BOUNDARY,
        "status": FINAL_HUMAN_LIKENESS_STATUS,
        "human_likeness_fully_proven": False,
        "final_human_likeness_claim_allowed": False,
        "current_scope_action_distribution_passed": behavioral_scope.get("action_distribution_status") == "PASS",
        "current_scope_delivery_claim_allowed": behavioral_boundary.get("current_scope_claim_allowed"),
        "current_delivery_blocker": False,
        "model_quality_risk": True,
        "required_behavior_dimensions": list(REQUIRED_BEHAVIOR_DIMENSIONS),
        "behavior_dimensions": dimensions,
        "upstream_boundaries": {
            "larger_clean_real_gameplay_revalidation_required": behavioral_boundary.get(
                "larger_clean_real_gameplay_revalidation_required"
            ),
            "generalized_human_likeness_claim_allowed": behavioral_boundary.get(
                "generalized_human_likeness_claim_allowed"
            ),
            "requires_more_real_player_behavior_labels": bet_boundary.get("requires_more_real_player_behavior_labels"),
            "timing_human_likeness_final_proof_allowed": timing_boundary.get(
                "final_production_human_likeness_proof_allowed"
            ),
            "timing_alone_final_claim_allowed": timing_boundary.get(
                "final_human_likeness_claim_allowed_from_timing_alone"
            ),
            "heuristic_timing_counts_as_full_human_likeness_proof": timing_boundary.get(
                "heuristic_timing_counts_as_full_human_likeness_proof"
            ),
            "real_human_timing_labels_available": timing_boundary.get("real_human_timing_labels_available"),
        },
        "blocked_claims": [
            "Human-likeness is fully proven by action distribution alone.",
            "Bet sizing is human-like without reviewed bet-size labels.",
            "Timing is human-like without reviewed real human timing labels.",
            "Position-based behavior is proven without slice-level validation.",
            "Street-level strategy is proven without street-specific validation.",
        ],
        "allowed_claim": (
            "Action distribution passes in the current validation scope, and the delivery stack can be "
            "monitored. Full human-likeness remains blocked until all behavior dimensions are validated."
        ),
    }
    payload["proof_cases"] = build_human_likeness_evidence_proof_cases(payload)
    payload["invariants"] = validate_human_likeness_evidence(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def build_human_likeness_evidence_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_human_likeness_evidence(candidate)
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

    record("base_contract_valid", dict(payload), "PASS")

    candidate = json.loads(json.dumps(payload))
    candidate["human_likeness_fully_proven"] = True
    candidate["final_human_likeness_claim_allowed"] = True
    record("blocks_full_human_likeness_claim", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["behavior_dimensions"]["bet_sizing"]["final_proof_allowed"] = True
    record("blocks_bet_sizing_without_labels", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["behavior_dimensions"]["timing"]["final_proof_allowed"] = True
    record("blocks_timing_without_labels", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["behavior_dimensions"]["position_based_behavior"]["required"] = False
    record("blocks_missing_position_slice_requirement", candidate, "FAIL")

    candidate = json.loads(json.dumps(payload))
    candidate["behavior_dimensions"]["street_level_strategy"]["required"] = False
    record("blocks_missing_street_slice_requirement", candidate, "FAIL")

    return cases


def validate_human_likeness_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    dimensions = payload.get("behavior_dimensions") or {}
    upstream = payload.get("upstream_boundaries") or {}

    if payload.get("boundary") != HUMAN_LIKENESS_BOUNDARY:
        violations.append("human_likeness_boundary_must_be_present")
    if payload.get("status") != FINAL_HUMAN_LIKENESS_STATUS:
        violations.append("human_likeness_status_must_remain_not_fully_proven")
    if payload.get("human_likeness_fully_proven") is not False:
        violations.append("human_likeness_must_not_be_marked_fully_proven")
    if payload.get("final_human_likeness_claim_allowed") is not False:
        violations.append("final_human_likeness_claim_must_remain_blocked")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("human_likeness_gap_must_not_block_current_delivery")
    if payload.get("model_quality_risk") is not True:
        violations.append("human_likeness_gap_must_remain_model_quality_risk")
    if set(payload.get("required_behavior_dimensions") or []) != set(REQUIRED_BEHAVIOR_DIMENSIONS):
        violations.append("required_behavior_dimensions_must_be_complete")

    for dimension in REQUIRED_BEHAVIOR_DIMENSIONS:
        item = dimensions.get(dimension) or {}
        if item.get("required") is not True:
            violations.append(f"behavior_dimension_must_be_required:{dimension}")
        if item.get("final_proof_allowed") is not False:
            violations.append(f"behavior_dimension_final_proof_must_be_blocked:{dimension}")

    if dimensions.get("action_distribution", {}).get("current_status") != "PASS":
        violations.append("action_distribution_must_pass_for_current_scope_claim")
    if upstream.get("larger_clean_real_gameplay_revalidation_required") is not True:
        violations.append("larger_clean_real_gameplay_revalidation_must_remain_required")
    if upstream.get("generalized_human_likeness_claim_allowed") is not False:
        violations.append("generalized_human_likeness_claim_must_remain_blocked")
    if upstream.get("requires_more_real_player_behavior_labels") is not True:
        violations.append("real_player_behavior_labels_must_remain_required")
    if upstream.get("timing_human_likeness_final_proof_allowed") is not False:
        violations.append("timing_final_human_likeness_proof_must_remain_blocked")
    if upstream.get("timing_alone_final_claim_allowed") is not False:
        violations.append("timing_alone_final_human_likeness_claim_must_remain_blocked")
    if upstream.get("heuristic_timing_counts_as_full_human_likeness_proof") is not False:
        violations.append("heuristic_timing_must_not_count_as_full_human_likeness_proof")
    timing_dimension = dimensions.get("timing") or {}
    if timing_dimension.get("requires_real_human_timing_labels") is not True:
        violations.append("timing_dimension_must_require_real_human_timing_labels")
    if timing_dimension.get("real_human_timing_labels_available") is not False:
        violations.append("timing_dimension_must_not_claim_real_human_timing_labels_available")
    if timing_dimension.get("heuristic_timing_counts_as_full_human_likeness_proof") is not False:
        violations.append("timing_dimension_must_not_accept_heuristic_timing_as_full_proof")

    blocked = set(payload.get("blocked_claims") or [])
    if "Human-likeness is fully proven by action distribution alone." not in blocked:
        violations.append("blocked_claims_must_reject_action_distribution_only_proof")

    return {"status": "PASS" if not violations else "FAIL", "violations": violations}


def write_human_likeness_evidence(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_human_likeness_evidence(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_human_likeness_evidence_markdown(payload), encoding="utf-8")
    return payload


def render_human_likeness_evidence_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Human-Likeness Evidence Contract",
        "",
        "Action distribution alone is not full human-likeness proof.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Boundary: `{payload['boundary']}`",
        f"- Fully proven: `{payload['human_likeness_fully_proven']}`",
        f"- Final human-likeness claim allowed: `{payload['final_human_likeness_claim_allowed']}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## Behavior Dimensions",
        "",
    ]
    for name, item in payload["behavior_dimensions"].items():
        lines.append(
            f"- `{name}`: required=`{item['required']}`, current_status=`{item['current_status']}`, "
            f"final_proof_allowed=`{item['final_proof_allowed']}`"
        )
    lines.extend(["", "## Blocked Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.extend(["", "## Proof Cases", ""])
    for case in payload["proof_cases"]:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, result `{case['result']}`"
        )
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
