from __future__ import annotations

from typing import Any


REQUIRED_PRODUCTION_METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "confusion_matrix",
    "calibration_ece",
    "action_distribution_js_divergence",
    "bet_size_mae",
    "expected_value_delta_vs_baseline",
    "win_rate",
    "seed_stability",
)

DIAGNOSTIC_METRICS_NOT_SUFFICIENT = (
    "accuracy",
    "cross_entropy",
)


def evaluate_strategy_metric_gate(metric_families: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether the strategy stack has enough evidence for final approval."""

    failures = _collect_gate_failures(metric_families)
    final_metric_bundle_passed = not failures
    return {
        "gate_status": "PASS" if final_metric_bundle_passed else "BLOCKED",
        "final_metric_bundle_passed": final_metric_bundle_passed,
        "final_strategy_quality_claim_allowed": final_metric_bundle_passed,
        "required_production_metrics": list(REQUIRED_PRODUCTION_METRICS),
        "diagnostic_metrics_not_sufficient_for_final_claim": list(DIAGNOSTIC_METRICS_NOT_SUFFICIENT),
        "blocked_approval_shortcuts": {
            "accuracy_only": True,
            "accuracy_plus_cross_entropy": True,
            "cross_entropy_only": True,
            "diagnostic_loss_only": True,
        },
        "missing_or_failed_requirements": failures,
    }


def _collect_gate_failures(metric_families: dict[str, Any]) -> list[str]:
    action = metric_families.get("action_classification") or {}
    action_metrics = action.get("metrics") or {}
    calibration = metric_families.get("calibration") or {}
    calibration_metrics = calibration.get("metrics") or {}
    distribution = metric_families.get("action_distribution") or {}
    bet_sizing = metric_families.get("bet_sizing") or {}
    simulation = metric_families.get("simulation_return") or {}
    simulation_metrics = simulation.get("metrics") or {}
    seed = metric_families.get("seed_stability") or {}
    seed_metrics = seed.get("metrics") or {}

    failures: list[str] = []
    for family_name in (
        "action_classification",
        "calibration",
        "action_distribution",
        "bet_sizing",
        "simulation_return",
        "seed_stability",
    ):
        if (metric_families.get(family_name) or {}).get("required") is not True:
            failures.append(f"required_family_missing_or_disabled:{family_name}")

    if action.get("accuracy_only_approval_allowed") is not False:
        failures.append("accuracy_only_not_allowed")
    if calibration.get("cross_entropy_only_approval_allowed") is not False:
        failures.append("cross_entropy_only_not_allowed")
    if calibration.get("diagnostic_loss_only_approval_allowed") is not False:
        failures.append("diagnostic_loss_only_not_allowed")
    if not _meets(action_metrics.get("macro_f1"), 0.50):
        failures.append("macro_f1_below_threshold")
    if not _meets(action_metrics.get("balanced_accuracy"), 0.50):
        failures.append("balanced_accuracy_below_threshold")
    if not _has_confusion_matrix(action_metrics.get("confusion_matrix")):
        failures.append("confusion_matrix_missing_or_invalid")
    if not _meets_max(calibration_metrics.get("ece_10"), 0.10):
        failures.append("calibration_ece_above_threshold")
    if distribution.get("larger_clean_real_gameplay_revalidation_required") is not False:
        failures.append("larger_clean_real_gameplay_revalidation_required")
    if bet_sizing.get("final_high_realism_claim_allowed") is not True:
        failures.append("bet_sizing_high_realism_not_approved")
    if _as_float((bet_sizing.get("metrics") or {}).get("bet_size_mae")) is None:
        failures.append("bet_size_mae_missing")
    if not _meets(simulation_metrics.get("win_rate"), 0.52):
        failures.append("win_rate_below_threshold")

    ev_delta = _as_float(simulation_metrics.get("expected_value_delta_vs_baseline"))
    if ev_delta is None or ev_delta <= 0.0:
        failures.append("expected_value_delta_not_positive")

    if seed_metrics.get("full_training_seed_stability_required") is not True:
        failures.append("full_training_seed_stability_not_required")
    if seed_metrics.get("phase3_seed_stability_evaluated") is not True:
        failures.append("phase3_seed_stability_not_evaluated")
    if seed.get("full_production_scale_training_status") != "COMPLETED":
        failures.append("full_production_scale_training_not_completed")
    if seed.get("phase3_training_proof_status") != "TRAINING_PROOF_COMPLETED":
        failures.append("phase3_training_proof_not_completed")

    return failures


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
