from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.strategy_metric_gate import (
    DIAGNOSTIC_METRICS_NOT_SUFFICIENT,
    REQUIRED_PRODUCTION_METRICS,
    evaluate_strategy_metric_gate,
)


EVALUATION_METRIC_CONTRACT_VERSION = "2026-07-07"
METRIC_CONTRACT_STATUS = "METRIC_BUNDLE_REQUIRED"
METRIC_CONTRACT_PASSED_STATUS = "METRIC_BUNDLE_PASSED"
METRIC_CONTRACT_BOUNDARY = "ACCURACY_AND_CROSS_ENTROPY_NOT_SUFFICIENT"
FINAL_CLAIM_BLOCKED_STATUS = "BLOCKED_UNTIL_FULL_METRIC_BUNDLE_PASSES"
FINAL_CLAIM_ALLOWED_STATUS = "ALLOWED"

REQUIRED_METRIC_FAMILIES = (
    "action_classification",
    "calibration",
    "action_distribution",
    "bet_sizing",
    "simulation_return",
    "seed_stability",
)


def build_evaluation_metric_contract(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    production_gate = _read_optional_json(reports / "today_acceptance_production_gate.json")
    challenger = _read_optional_json(reports / "challenger_strategy_quality.json")
    acceptance = _read_optional_json(reports / "policy_acceptance.json")
    behavioral = _read_optional_json(reports / "behavioral_revalidation.json")
    bet_timing = _read_optional_json(reports / "bet_timing_calibration.json")
    self_play = _read_optional_json(reports / "production_self_play.json")
    multi_agent = _read_optional_json(reports / "multi_agent_training_status.json")
    phase3_arena = _read_optional_json(reports / "phase3_open_spiel_arena.json")

    valid_metrics = production_gate.get("valid_metrics") or {}
    challenger_result = challenger.get("challenger_result") or {}
    acceptance_alignment = acceptance.get("human_action_alignment") or {}
    acceptance_likeness = acceptance.get("human_likeness") or {}
    behavioral_scope = behavioral.get("current_validation_scope") or {}
    bet_current = bet_timing.get("current_delivery_scope") or {}
    bet_boundary = bet_timing.get("calibration_boundary") or {}
    multi_boundary = multi_agent.get("training_boundary") or {}
    multi_plan = multi_agent.get("hardening_training_plan") or {}
    phase3_proof = phase3_arena.get("rl_training_proof_boundary") or {}

    action_family = {
        "required": True,
        "metrics": {
            "accuracy": _first_number(valid_metrics.get("accuracy"), challenger_result.get("accuracy")),
            "macro_f1": _first_number(valid_metrics.get("macro_f1"), challenger_result.get("macro_f1")),
            "balanced_accuracy": _first_number(
                valid_metrics.get("balanced_accuracy"),
                challenger_result.get("balanced_accuracy"),
            ),
            "confusion_matrix": valid_metrics.get("confusion_matrix"),
            "human_action_alignment_accuracy": acceptance_alignment.get("accuracy"),
            "human_action_alignment_macro_f1": acceptance_alignment.get("macro_f1"),
        },
        "accuracy_only_approval_allowed": False,
        "confusion_matrix_required_for_final_claim": True,
        "source_reports": [
            "reports/today_acceptance_production_gate.json",
            "reports/challenger_strategy_quality.json",
            "reports/policy_acceptance.json",
        ],
    }

    calibration_family = {
        "required": True,
        "metrics": {
            "ece_10": _first_number(valid_metrics.get("ece_10"), challenger_result.get("calibration_ece_10")),
            "brier_loss": valid_metrics.get("brier_loss"),
            "cross_entropy": valid_metrics.get("cross_entropy"),
        },
        "max_ece_10": 0.10,
        "calibration_required_for_final_claim": True,
        "cross_entropy_only_approval_allowed": False,
        "diagnostic_loss_only_approval_allowed": False,
        "source_reports": [
            "reports/today_acceptance_production_gate.json",
            "reports/challenger_strategy_quality.json",
        ],
    }

    action_distribution_family = {
        "required": True,
        "metrics": {
            "js_divergence": _first_number(
                acceptance_likeness.get("js_divergence"),
                behavioral_scope.get("js_divergence"),
            ),
            "status": behavioral_scope.get("action_distribution_status") or acceptance_likeness.get("status"),
        },
        "larger_clean_real_gameplay_revalidation_required": (
            behavioral.get("revalidation_boundary") or {}
        ).get("larger_clean_real_gameplay_revalidation_required"),
        "generalized_action_distribution_claim_allowed": (
            behavioral.get("revalidation_boundary") or {}
        ).get("generalized_action_distribution_claim_allowed"),
        "source_reports": [
            "reports/policy_acceptance.json",
            "reports/behavioral_revalidation.json",
        ],
    }

    bet_sizing_family = {
        "required": True,
        "metrics": {
            "timing_and_bet_size_status": bet_current.get("timing_and_bet_size_status"),
            "bet_size_mae": bet_current.get("bet_size_mae"),
            "decision_time_mae": bet_current.get("decision_time_mae"),
        },
        "bet_size_mae_required_for_final_high_realism": True,
        "current_delivery_has_required_response_fields": bet_current.get("api_response_fields") is not None,
        "requires_more_real_player_behavior_labels": bet_boundary.get("requires_more_real_player_behavior_labels"),
        "final_high_realism_claim_allowed": bet_boundary.get("final_high_realism_claim_allowed"),
        "source_report": "reports/bet_timing_calibration.json",
    }

    simulation_return_family = {
        "required": True,
        "metrics": {
            "win_rate": self_play.get("mean_policy_win_rate"),
            "min_win_rate": self_play.get("min_policy_win_rate"),
            "max_win_rate": self_play.get("max_policy_win_rate"),
            "paired_hands": self_play.get("paired_hands"),
            "expected_value_delta_vs_baseline": self_play.get("mean_ev_delta_vs_baseline"),
        },
        "production_scale_status": self_play.get("production_scale_status"),
        "source_report": "reports/production_self_play.json",
    }

    seed_stability_family = {
        "required": True,
        "metrics": {
            "production_self_play_run_count": self_play.get("run_count"),
            "full_training_seed_stability_required": multi_plan.get("seed_stability_required"),
            "minimum_independent_training_seeds": multi_plan.get("minimum_independent_training_seeds"),
            "phase3_seed_stability_required": phase3_proof.get("seed_stability_required"),
            "phase3_seed_stability_evaluated": phase3_proof.get("seed_stability_evaluated"),
        },
        "full_production_scale_training_status": multi_boundary.get(
            "full_production_scale_multi_agent_training_status"
        ),
        "phase3_training_proof_status": phase3_proof.get("status"),
        "source_reports": [
            "reports/multi_agent_training_status.json",
            "reports/phase3_open_spiel_arena.json",
            "reports/production_self_play.json",
        ],
    }

    metric_families = {
        "action_classification": action_family,
        "calibration": calibration_family,
        "action_distribution": action_distribution_family,
        "bet_sizing": bet_sizing_family,
        "simulation_return": simulation_return_family,
        "seed_stability": seed_stability_family,
    }
    strategy_metric_gate = evaluate_strategy_metric_gate(metric_families)
    final_metric_bundle_passed = bool(strategy_metric_gate["final_metric_bundle_passed"])

    payload: dict[str, Any] = {
        "version": EVALUATION_METRIC_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Production evaluation metric coverage boundary",
        "status": METRIC_CONTRACT_PASSED_STATUS if final_metric_bundle_passed else METRIC_CONTRACT_STATUS,
        "boundary": METRIC_CONTRACT_BOUNDARY,
        "accuracy_alone_sufficient": False,
        "accuracy_and_cross_entropy_sufficient": False,
        "required_metric_families": list(REQUIRED_METRIC_FAMILIES),
        "required_production_metrics": list(REQUIRED_PRODUCTION_METRICS),
        "diagnostic_metrics_not_sufficient_for_final_claim": list(DIAGNOSTIC_METRICS_NOT_SUFFICIENT),
        "metric_families": metric_families,
        "strategy_metric_gate": strategy_metric_gate,
        "final_metric_bundle_passed": final_metric_bundle_passed,
        "final_strategy_quality_claim_allowed": final_metric_bundle_passed,
        "final_strategy_quality_claim_status": (
            FINAL_CLAIM_ALLOWED_STATUS if final_metric_bundle_passed else FINAL_CLAIM_BLOCKED_STATUS
        ),
        "current_delivery_blocker": False,
        "model_quality_risk": not final_metric_bundle_passed,
        "allowed_current_claim": (
            "The delivered stack is evaluated with a multi-metric contract; current delivery remains "
            "valid while final strategy-quality claims stay gated by the full metric bundle."
        ),
        "blocked_claims": [
            "Accuracy alone is sufficient for production strategy approval.",
            "Accuracy and cross-entropy are sufficient for production strategy approval.",
            "Cross-entropy alone is sufficient for production strategy approval.",
            "Final strategy quality is approved without macro F1 and balanced accuracy.",
            "Final strategy quality is approved without calibration/ECE.",
            "Final strategy quality is approved without action-distribution checks.",
            "Final strategy quality is approved without bet-size MAE or reviewed bet-size labels.",
            "Final strategy quality is approved without win-rate, expected-value, and seed-stability evidence.",
        ],
    }
    payload["proof_cases"] = build_evaluation_metric_proof_cases(payload)
    payload["invariants"] = validate_evaluation_metric_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def build_evaluation_metric_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    def cloned() -> dict[str, Any]:
        return json.loads(json.dumps(payload))

    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_evaluation_metric_contract(candidate)
        observed = candidate["invariants"]["status"]
        cases.append(
            {
                "name": name,
                "expected_status": expected_status,
                "observed_status": observed,
                "result": "PASS" if observed == expected_status else "FAIL",
                "violations": candidate["invariants"]["violations"],
            }
        )

    record("base_contract_valid", cloned(), "PASS")

    candidate = cloned()
    candidate["accuracy_alone_sufficient"] = True
    candidate["final_strategy_quality_claim_allowed"] = True
    record("blocks_accuracy_only_approval", candidate, "FAIL")

    candidate = cloned()
    candidate["accuracy_and_cross_entropy_sufficient"] = True
    candidate["metric_families"]["calibration"]["cross_entropy_only_approval_allowed"] = True
    candidate["metric_families"]["calibration"]["diagnostic_loss_only_approval_allowed"] = True
    candidate["final_strategy_quality_claim_allowed"] = True
    record("blocks_accuracy_and_cross_entropy_only_approval", candidate, "FAIL")

    for family_name in REQUIRED_METRIC_FAMILIES:
        candidate = cloned()
        candidate["metric_families"][family_name]["required"] = False
        record(f"blocks_missing_required_family:{family_name}", candidate, "FAIL")

    candidate = cloned()
    candidate["metric_families"]["calibration"]["calibration_required_for_final_claim"] = False
    record("blocks_final_claim_without_calibration_requirement", candidate, "FAIL")

    candidate = cloned()
    candidate["metric_families"]["action_classification"]["confusion_matrix_required_for_final_claim"] = False
    record("blocks_final_claim_without_confusion_matrix_requirement", candidate, "FAIL")

    candidate = cloned()
    candidate["metric_families"]["action_classification"]["metrics"]["confusion_matrix"] = None
    record("blocks_final_claim_without_confusion_matrix_measurement", candidate, "FAIL")

    candidate = cloned()
    candidate["metric_families"]["bet_sizing"]["bet_size_mae_required_for_final_high_realism"] = False
    record("blocks_final_claim_without_bet_size_mae_requirement", candidate, "FAIL")

    candidate = cloned()
    candidate["metric_families"]["seed_stability"]["metrics"]["phase3_seed_stability_required"] = False
    record("blocks_final_claim_without_seed_stability_requirement", candidate, "FAIL")

    return cases


def validate_evaluation_metric_contract(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    families = payload.get("metric_families") or {}

    if payload.get("overall_status") == "PASS":
        violations.append("overall_status_must_be_assigned_after_invariant_validation")
    final_passed = _final_metric_bundle_passed(families)
    expected_status = METRIC_CONTRACT_PASSED_STATUS if final_passed else METRIC_CONTRACT_STATUS
    if payload.get("status") != expected_status:
        violations.append("metric_contract_status_must_match_metric_bundle_state")
    if payload.get("boundary") != METRIC_CONTRACT_BOUNDARY:
        violations.append("metric_contract_boundary_must_block_accuracy_and_cross_entropy_approval")
    if payload.get("accuracy_alone_sufficient") is not False:
        violations.append("accuracy_alone_must_not_be_sufficient")
    if payload.get("accuracy_and_cross_entropy_sufficient") is not False:
        violations.append("accuracy_and_cross_entropy_must_not_be_sufficient")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("metric_contract_gap_must_not_block_current_delivery")
    required = set(payload.get("required_metric_families") or [])
    if required != set(REQUIRED_METRIC_FAMILIES):
        violations.append("required_metric_families_must_be_complete")
    required_metrics = set(payload.get("required_production_metrics") or [])
    if required_metrics != set(REQUIRED_PRODUCTION_METRICS):
        violations.append("required_production_metrics_must_be_complete")
    diagnostic_metrics = set(payload.get("diagnostic_metrics_not_sufficient_for_final_claim") or [])
    if not set(DIAGNOSTIC_METRICS_NOT_SUFFICIENT).issubset(diagnostic_metrics):
        violations.append("diagnostic_metrics_must_include_accuracy_and_cross_entropy")
    for family_name in REQUIRED_METRIC_FAMILIES:
        family = families.get(family_name) or {}
        if family.get("required") is not True:
            violations.append(f"metric_family_must_be_required:{family_name}")

    action = families.get("action_classification") or {}
    action_metrics = action.get("metrics") or {}
    if action.get("accuracy_only_approval_allowed") is not False:
        violations.append("action_classification_must_block_accuracy_only_approval")
    for metric_name in ("accuracy", "macro_f1", "balanced_accuracy"):
        if _as_float(action_metrics.get(metric_name)) is None:
            violations.append(f"action_classification_metric_missing:{metric_name}")
    if action.get("confusion_matrix_required_for_final_claim") is not True:
        violations.append("confusion_matrix_must_be_required_for_final_claim")
    if not _has_confusion_matrix(action_metrics.get("confusion_matrix")):
        violations.append("action_classification_confusion_matrix_missing_or_invalid")

    calibration = families.get("calibration") or {}
    calibration_metrics = calibration.get("metrics") or {}
    if calibration.get("calibration_required_for_final_claim") is not True:
        violations.append("calibration_must_be_required_for_final_claim")
    if calibration.get("cross_entropy_only_approval_allowed") is not False:
        violations.append("cross_entropy_only_must_not_be_sufficient")
    if calibration.get("diagnostic_loss_only_approval_allowed") is not False:
        violations.append("diagnostic_loss_only_must_not_be_sufficient")
    if _as_float(calibration_metrics.get("ece_10")) is None:
        violations.append("calibration_ece_must_be_present")
    if _as_float(calibration_metrics.get("cross_entropy")) is None:
        violations.append("calibration_cross_entropy_diagnostic_must_be_present")

    distribution = families.get("action_distribution") or {}
    distribution_metrics = distribution.get("metrics") or {}
    if _as_float(distribution_metrics.get("js_divergence")) is None:
        violations.append("action_distribution_js_divergence_must_be_present")
    if final_passed:
        if distribution.get("larger_clean_real_gameplay_revalidation_required") is not False:
            violations.append("passed_metric_bundle_requires_larger_clean_revalidation_closed")
        if distribution.get("generalized_action_distribution_claim_allowed") is not True:
            violations.append("passed_metric_bundle_requires_generalized_distribution_claim")
    else:
        if distribution.get("larger_clean_real_gameplay_revalidation_required") is not True:
            violations.append("action_distribution_must_require_larger_clean_revalidation")
        if distribution.get("generalized_action_distribution_claim_allowed") is not False:
            violations.append("generalized_action_distribution_claim_must_be_blocked")

    bet_sizing = families.get("bet_sizing") or {}
    if bet_sizing.get("bet_size_mae_required_for_final_high_realism") is not True:
        violations.append("bet_size_mae_must_be_required_for_final_high_realism")
    bet_sizing_metrics = bet_sizing.get("metrics") or {}
    if final_passed:
        if _as_float(bet_sizing_metrics.get("bet_size_mae")) is None:
            violations.append("bet_size_mae_must_be_measured_for_strategy_claim")
        if bet_sizing.get("requires_more_real_player_behavior_labels") is not False:
            violations.append("passed_metric_bundle_requires_bet_sizing_labels_closed")
        if bet_sizing.get("final_high_realism_claim_allowed") is not True:
            violations.append("passed_metric_bundle_requires_bet_sizing_high_realism_claim")
    else:
        if bet_sizing.get("requires_more_real_player_behavior_labels") is not True:
            violations.append("bet_sizing_must_require_more_real_player_labels")
        if bet_sizing.get("final_high_realism_claim_allowed") is not False:
            violations.append("bet_sizing_final_high_realism_claim_must_be_blocked")

    simulation = families.get("simulation_return") or {}
    simulation_metrics = simulation.get("metrics") or {}
    for metric_name in ("win_rate", "expected_value_delta_vs_baseline"):
        if _as_float(simulation_metrics.get(metric_name)) is None:
            violations.append(f"simulation_return_metric_missing:{metric_name}")

    seed = families.get("seed_stability") or {}
    seed_metrics = seed.get("metrics") or {}
    if seed_metrics.get("full_training_seed_stability_required") is not True:
        violations.append("full_training_seed_stability_must_be_required")
    if seed_metrics.get("phase3_seed_stability_required") is not True:
        violations.append("phase3_seed_stability_must_be_required")
    expected_training_status = "COMPLETED" if final_passed else "NOT_COMPLETED"
    expected_phase3_status = "TRAINING_PROOF_COMPLETED" if final_passed else "TRAINING_PROOF_NOT_COMPLETED"
    if seed.get("full_production_scale_training_status") != expected_training_status:
        violations.append("full_production_scale_training_status_must_match_metric_bundle_state")
    if seed.get("phase3_training_proof_status") != expected_phase3_status:
        violations.append("phase3_training_proof_status_must_match_metric_bundle_state")

    strategy_gate = payload.get("strategy_metric_gate") or {}
    if strategy_gate.get("final_metric_bundle_passed") is not final_passed:
        violations.append("strategy_metric_gate_status_must_match_metric_families")
    if strategy_gate.get("final_strategy_quality_claim_allowed") is not final_passed:
        violations.append("strategy_metric_gate_claim_status_must_match_metric_families")
    if payload.get("final_metric_bundle_passed") is not final_passed:
        violations.append("final_metric_bundle_status_must_match_metric_families")
    if not final_passed:
        if payload.get("final_strategy_quality_claim_allowed") is not False:
            violations.append("final_strategy_quality_claim_must_be_blocked_until_full_metric_bundle")
        if payload.get("model_quality_risk") is not True:
            violations.append("incomplete_metric_bundle_must_remain_model_quality_risk")
        if payload.get("final_strategy_quality_claim_status") != FINAL_CLAIM_BLOCKED_STATUS:
            violations.append("final_strategy_quality_claim_status_must_be_blocked")
    else:
        if payload.get("final_strategy_quality_claim_allowed") is not True:
            violations.append("final_strategy_quality_claim_must_open_when_full_metric_bundle_passes")
        if payload.get("model_quality_risk") is not False:
            violations.append("complete_metric_bundle_must_clear_model_quality_risk")
        if payload.get("final_strategy_quality_claim_status") != FINAL_CLAIM_ALLOWED_STATUS:
            violations.append("final_strategy_quality_claim_status_must_be_allowed")

    blocked = set(payload.get("blocked_claims") or [])
    if "Accuracy alone is sufficient for production strategy approval." not in blocked:
        violations.append("blocked_claims_must_reject_accuracy_only_approval")
    if "Accuracy and cross-entropy are sufficient for production strategy approval." not in blocked:
        violations.append("blocked_claims_must_reject_accuracy_and_cross_entropy_approval")
    if "Cross-entropy alone is sufficient for production strategy approval." not in blocked:
        violations.append("blocked_claims_must_reject_cross_entropy_only_approval")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_evaluation_metric_contract(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_evaluation_metric_contract(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_evaluation_metric_contract_markdown(payload), encoding="utf-8")
    return payload


def render_evaluation_metric_contract_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Metric Contract",
        "",
        "Accuracy and cross-entropy are diagnostic metrics; they are not sufficient for strategy-quality approval.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Boundary: `{payload['boundary']}`",
        f"- Final metric bundle passed: `{payload['final_metric_bundle_passed']}`",
        f"- Final strategy quality claim allowed: `{payload['final_strategy_quality_claim_allowed']}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## Required Metric Families",
        "",
    ]
    for family_name, family in payload["metric_families"].items():
        lines.append(f"- `{family_name}`: required=`{family['required']}`")
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


def _final_metric_bundle_passed(families: dict[str, Any]) -> bool:
    return bool(evaluate_strategy_metric_gate(families)["final_metric_bundle_passed"])


def _first_number(*values: Any) -> Any:
    for value in values:
        if _as_float(value) is not None:
            return value
    return None


def _meets(value: Any, threshold: float) -> bool:
    parsed = _as_float(value)
    return parsed is not None and parsed >= threshold


def _meets_max(value: Any, threshold: float) -> bool:
    parsed = _as_float(value)
    return parsed is not None and parsed <= threshold


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_confusion_matrix(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    labels = value.get("labels")
    matrix = value.get("matrix")
    if not isinstance(labels, list) or not labels:
        return False
    if not isinstance(matrix, list) or len(matrix) != len(labels):
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(labels):
            return False
        for cell in row:
            if _as_float(cell) is None:
                return False
    return True


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
