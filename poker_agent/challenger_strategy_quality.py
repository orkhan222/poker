from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHALLENGER_STRATEGY_QUALITY_VERSION = "2026-06-28"
FINAL_STRATEGY_QUALITY_STATUS = "BLOCKED_PENDING_CHALLENGER_GATE"
APPROVED_STRATEGY_QUALITY_STATUS = "APPROVED_AFTER_CHALLENGER_GATE"
REQUIRED_CHALLENGER_GATE = "PASS"
REQUIRED_RAW_GATE = "PASS"


def build_challenger_strategy_quality(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    raw_model_status = _read_optional_json(reports / "raw_model_status.json")
    raw_model_challenger = _read_optional_json(reports / "raw_model_challenger.json")
    production_gate = _read_optional_json(reports / "production_gate.json")
    deployed_strategy_gate = _read_optional_json(reports / "deployed_strategy_gate.json")

    raw_contract = raw_model_status.get("raw_supervised_model") or {}
    raw_release_boundary = raw_model_status.get("release_boundary") or {}
    challenger_best = raw_model_challenger.get("best_candidate") or {}
    challenger_gate = challenger_best.get("gate") or {}
    challenger_metrics = challenger_best.get("valid_metrics") or {}
    challenger_slices = challenger_best.get("valid_slice_metrics") or challenger_best.get("slice_metrics") or {}
    challenger_failed_gates = challenger_gate.get("failed_gates") or []
    gate_failure_analysis = _gate_failure_analysis(challenger_gate)

    raw_gate_status = str(production_gate.get("status") or raw_contract.get("quality_gate_status") or "UNKNOWN")
    challenger_gate_status = str(challenger_gate.get("status") or "MISSING")
    challenger_trained = bool(challenger_best) and challenger_best.get("status") not in {"MISSING", None}
    challenger_compared = _has_grouped_holdout_comparison(raw_model_challenger, challenger_metrics)
    final_claim_allowed = raw_gate_status == REQUIRED_RAW_GATE and challenger_gate_status == REQUIRED_CHALLENGER_GATE

    boundary = {
        "status": APPROVED_STRATEGY_QUALITY_STATUS if final_claim_allowed else FINAL_STRATEGY_QUALITY_STATUS,
        "final_production_strategy_quality_claim_allowed": final_claim_allowed,
        "claim_blocked_until_challenger_passes": not final_claim_allowed,
        "challenger_required_before_final_claim": True,
        "challenger_trained": challenger_trained,
        "challenger_compared_to_raw_model": challenger_compared,
        "raw_production_gate_status": raw_gate_status,
        "challenger_gate_status": challenger_gate_status,
        "current_raw_model_standalone_approved": raw_contract.get("approved_as_standalone_policy") is True,
        "raw_model_component_risk": raw_release_boundary.get("component_risk", True),
        "current_delivery_blocker": False,
        "deployed_strategy_stack_affected": False,
    }

    payload: dict[str, Any] = {
        "version": CHALLENGER_STRATEGY_QUALITY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Challenger model requirement before final production-level strategy quality claims",
        "strategy_quality_boundary": boundary,
        "current_raw_supervised_model": {
            "runtime_status": raw_contract.get("runtime_status"),
            "standalone_status": raw_contract.get("standalone_status"),
            "quality_gate_status": raw_contract.get("quality_gate_status"),
            "approved_as_standalone_policy": raw_contract.get("approved_as_standalone_policy"),
            "component_risk": raw_release_boundary.get("component_risk", True),
            "production_blocker": raw_release_boundary.get("production_blocker", False),
        },
        "challenger_result": {
            "status": raw_model_challenger.get("status", "MISSING"),
            "best_candidate": challenger_best.get("name"),
            "artifact_path": challenger_best.get("artifact_path"),
            "candidate_status": challenger_best.get("status"),
            "gate_status": challenger_gate_status,
            "passed_gates": challenger_gate.get("passed_gates"),
            "total_gates": challenger_gate.get("total_gates"),
            "failed_gates": challenger_failed_gates,
            "accuracy": challenger_metrics.get("accuracy"),
            "macro_f1": challenger_metrics.get("macro_f1"),
            "balanced_accuracy": challenger_metrics.get("balanced_accuracy"),
            "calibration_ece_10": challenger_metrics.get("ece_10"),
            "accuracy_lift_vs_majority": challenger_metrics.get("lift_vs_majority"),
            "majority_baseline_accuracy": challenger_metrics.get("majority_baseline_accuracy"),
            "observed_hole_cards_macro_f1": (challenger_slices.get("observed_hole_cards") or {}).get("macro_f1"),
            "facing_bet_macro_f1": (challenger_slices.get("facing_bet") or {}).get("macro_f1"),
        },
        "gate_failure_analysis": gate_failure_analysis,
        "minimum_promotion_requirements": [
            "Train at least one stronger challenger artifact on the grouped holdout contract.",
            "Compare the challenger against the current raw supervised artifact and majority baseline.",
            "Pass macro F1, balanced accuracy, calibration, observed-card, facing-bet, and dataset-audit gates.",
            "Keep final production-level strategy quality blocked while either the challenger gate or raw production gate fails.",
            "Promote only after the raw production gate can be independently reproduced as PASS.",
        ],
        "allowed_claims": [
            "The deployed strategy stack remains approved for monitored delivery.",
            "A challenger workflow exists and trains/evaluates standalone supervised candidates.",
            "The current best challenger is reported with its failed gates instead of being promoted incorrectly.",
            "The raw supervised model can remain a loadable service component while its standalone risk is tracked.",
        ],
        "blocked_claims": [
            "Final production-level strategy quality is approved without a passing challenger.",
            "The raw supervised model is a standalone production-approved poker policy.",
            "A failing challenger can be promoted as the production policy.",
            "The deployed stack approval also proves final maximally optimized strategy quality.",
        ],
        "evidence": {
            "raw_model_status": "reports/raw_model_status.json",
            "raw_model_challenger": "reports/raw_model_challenger.json",
            "production_gate": "reports/production_gate.json",
            "deployed_strategy_gate": "reports/deployed_strategy_gate.json",
        },
        "next_actions": _next_actions(challenger_failed_gates),
    }
    payload["invariants"] = validate_challenger_strategy_quality(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_challenger_strategy_quality(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    boundary = payload.get("strategy_quality_boundary") or {}
    raw = payload.get("current_raw_supervised_model") or {}
    challenger = payload.get("challenger_result") or {}
    failure_analysis = payload.get("gate_failure_analysis") or []
    blocked_claims = payload.get("blocked_claims") or []

    final_claim_allowed = boundary.get("final_production_strategy_quality_claim_allowed") is True
    raw_gate_status = boundary.get("raw_production_gate_status")
    challenger_gate_status = boundary.get("challenger_gate_status")

    if boundary.get("challenger_required_before_final_claim") is not True:
        violations.append("challenger_must_be_required_before_final_strategy_quality_claim")
    if boundary.get("challenger_compared_to_raw_model") is not True:
        violations.append("challenger_must_be_compared_before_final_strategy_quality_claim")
    if boundary.get("current_delivery_blocker") is not False:
        violations.append("challenger_gap_must_not_block_current_delivery")
    if boundary.get("deployed_strategy_stack_affected") is not False:
        violations.append("challenger_gap_must_not_change_deployed_stack_approval")
    if raw.get("standalone_status") == "STANDALONE_APPROVED" and not final_claim_allowed:
        violations.append("raw_standalone_approval_requires_final_strategy_quality_gate")
    if raw.get("approved_as_standalone_policy") is True and not final_claim_allowed:
        violations.append("raw_policy_approval_requires_final_strategy_quality_gate")
    if final_claim_allowed and raw_gate_status != REQUIRED_RAW_GATE:
        violations.append("final_strategy_quality_requires_raw_gate_pass")
    if final_claim_allowed and challenger_gate_status != REQUIRED_CHALLENGER_GATE:
        violations.append("final_strategy_quality_requires_challenger_gate_pass")
    if not final_claim_allowed and boundary.get("claim_blocked_until_challenger_passes") is not True:
        violations.append("failing_strategy_quality_gate_must_block_claim")
    if not final_claim_allowed and boundary.get("status") != FINAL_STRATEGY_QUALITY_STATUS:
        violations.append("failing_strategy_quality_gate_must_use_blocked_status")
    if challenger_gate_status != REQUIRED_CHALLENGER_GATE and challenger.get("gate_status") == REQUIRED_CHALLENGER_GATE:
        violations.append("boundary_and_challenger_gate_status_mismatch")
    if challenger_gate_status != REQUIRED_CHALLENGER_GATE and "A failing challenger can be promoted as the production policy." not in blocked_claims:
        violations.append("blocked_claims_must_reject_failing_challenger_promotion")
    if "Final production-level strategy quality is approved without a passing challenger." not in blocked_claims:
        violations.append("blocked_claims_must_reject_final_quality_without_challenger")
    if challenger_gate_status != REQUIRED_CHALLENGER_GATE:
        analyzed = {item.get("name") for item in failure_analysis}
        missing = set(challenger.get("failed_gates") or []) - analyzed
        if missing:
            violations.append(f"failed_challenger_gates_missing_failure_analysis:{','.join(sorted(missing))}")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_challenger_strategy_quality(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_challenger_strategy_quality(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_challenger_strategy_quality_markdown(payload), encoding="utf-8")
    return payload


def render_challenger_strategy_quality_markdown(payload: dict[str, Any]) -> str:
    boundary = payload["strategy_quality_boundary"]
    challenger = payload["challenger_result"]
    raw = payload["current_raw_supervised_model"]
    lines = [
        "# Challenger Strategy Quality Boundary",
        "",
        "Final production-level strategy quality cannot be claimed until a stronger challenger model passes the challenger gate and the raw production gate is reproducibly PASS.",
        "",
        "## Current Status",
        "",
        f"- Boundary status: `{boundary['status']}`",
        f"- Final production strategy-quality claim allowed: `{boundary['final_production_strategy_quality_claim_allowed']}`",
        f"- Raw production gate: `{boundary['raw_production_gate_status']}`",
        f"- Challenger gate: `{boundary['challenger_gate_status']}`",
        f"- Current delivery blocker: `{boundary['current_delivery_blocker']}`",
        f"- Deployed strategy stack affected: `{boundary['deployed_strategy_stack_affected']}`",
        "",
        "## Raw Model",
        "",
        f"- Runtime status: `{raw.get('runtime_status')}`",
        f"- Standalone status: `{raw.get('standalone_status')}`",
        f"- Component risk: `{raw.get('component_risk')}`",
        "",
        "## Best Challenger",
        "",
        f"- Candidate: `{challenger.get('best_candidate')}`",
        f"- Artifact: `{challenger.get('artifact_path')}`",
        f"- Macro F1: `{challenger.get('macro_f1')}`",
        f"- Balanced accuracy: `{challenger.get('balanced_accuracy')}`",
        f"- Calibration ECE@10: `{challenger.get('calibration_ece_10')}`",
        f"- Failed gates: `{', '.join(challenger.get('failed_gates') or []) or 'none'}`",
        "",
        "## Gate Failure Analysis",
        "",
    ]
    for item in payload.get("gate_failure_analysis") or []:
        lines.append(
            f"- `{item['name']}`: observed=`{item.get('observed')}`, threshold=`{item.get('threshold')}`, "
            f"shortfall=`{item.get('shortfall')}`, remediation={item.get('remediation')}"
        )
    lines.extend(
        [
            "",
        "## Blocked Claims",
        "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in payload["next_actions"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _has_grouped_holdout_comparison(raw_model_challenger: dict[str, Any], metrics: dict[str, Any]) -> bool:
    split = raw_model_challenger.get("split") or {}
    if split.get("split_type") != "stratified_hand_group_holdout":
        return False
    if "majority_baseline_accuracy" not in metrics:
        return False
    if "lift_vs_majority" not in metrics:
        return False
    return True


def _next_actions(failed_gates: list[str]) -> list[str]:
    actions = [
        "Keep final production-level strategy quality blocked until challenger and raw production gates pass.",
        "Continue reporting the deployed strategy stack separately from standalone raw-model quality.",
    ]
    failed = set(failed_gates)
    if "macro_f1" in failed or "balanced_accuracy" in failed:
        actions.append("Improve supervised features and imbalance handling until minority-action performance clears the gate.")
    if "calibration" in failed:
        actions.append("Calibrate the challenger with held-out probability calibration before promotion.")
    if "observed_hole_cards_macro_f1" in failed:
        actions.append("Increase reviewed hole-card coverage and train card-visible specialist features.")
    if "facing_bet_macro_f1" in failed:
        actions.append("Strengthen pot-odds, pressure, and betting-history features for facing-bet decisions.")
    if "dataset_audit_blockers" in failed:
        actions.append("Close dataset audit blockers before any standalone production-policy claim.")
    return actions


def _gate_failure_analysis(challenger_gate: dict[str, Any]) -> list[dict[str, Any]]:
    analysis: list[dict[str, Any]] = []
    for gate in challenger_gate.get("gates") or []:
        if gate.get("passed") is not False:
            continue
        name = str(gate.get("name"))
        observed = gate.get("observed")
        threshold = gate.get("threshold")
        analysis.append(
            {
                "name": name,
                "observed": observed,
                "threshold": threshold,
                "shortfall": _gate_gap_to_pass(name, observed, threshold),
                "impact": gate.get("impact"),
                "remediation": _remediation_for_gate(name),
            }
        )
    if not analysis:
        for name in challenger_gate.get("failed_gates") or []:
            analysis.append(
                {
                    "name": str(name),
                    "observed": None,
                    "threshold": None,
                    "shortfall": None,
                    "impact": None,
                    "remediation": _remediation_for_gate(str(name)),
                }
            )
    return analysis


def _gate_gap_to_pass(name: str, observed: Any, threshold: Any) -> float | None:
    if not isinstance(observed, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    lower_is_better = {"calibration", "dataset_audit_blockers"}
    if name in lower_is_better:
        return max(0.0, float(observed) - float(threshold))
    return max(0.0, float(threshold) - float(observed))


def _remediation_for_gate(name: str) -> str:
    mapping = {
        "macro_f1": "Improve minority-action recall with class weighting, resampling, richer betting-history features, and candidate-specific error analysis.",
        "calibration": "Apply held-out probability calibration and reject promotion until ECE is below the production threshold.",
        "observed_hole_cards_macro_f1": "Increase reviewed hole-card coverage and train card-visible specialist features.",
        "facing_bet_macro_f1": "Strengthen pot-odds, pressure, stack-to-pot, and previous-action features for call/fold/raise decisions under pressure.",
        "dataset_audit_blockers": "Close dataset audit blockers before any standalone production-policy claim.",
        "balanced_accuracy": "Improve recall across all classes instead of optimizing headline accuracy.",
    }
    return mapping.get(name, "Investigate the failing slice and add a targeted remediation before promotion.")


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

