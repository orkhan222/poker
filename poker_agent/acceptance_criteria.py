from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class AcceptanceCriteria:
    latency_p95_ms_max: float = 150.0
    latency_p99_ms_max: float = 300.0
    invalid_action_rate_max: float = 0.0
    validation_pass_rate_min: float = 1.0
    reproducibility_pass_rate_min: float = 1.0
    latency_sample_min: int = 1
    invalid_action_sample_min: int = 1
    validation_check_min: int = 1
    reproducibility_check_min: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_ACCEPTANCE_CRITERIA = AcceptanceCriteria()


def latency_summary(latencies_ms: list[float]) -> dict[str, Any]:
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": values[-1],
    }


def build_acceptance_metrics(
    *,
    latencies_ms: list[float],
    prediction_payloads: list[dict[str, Any]],
    validation_checks: list[bool | dict[str, Any]],
    reproducibility_checks: list[bool | dict[str, Any]],
) -> dict[str, Any]:
    invalid_count = 0
    for payload in prediction_payloads:
        action = str(payload.get("action") or "")
        legal_actions = {str(item) for item in payload.get("legal_actions", [])}
        if action not in legal_actions:
            invalid_count += 1

    validation_passed = sum(1 for check in validation_checks if check_passed(check))
    reproducibility_passed = sum(1 for check in reproducibility_checks if check_passed(check))
    hash_mismatch_count = sum(
        1
        for check in reproducibility_checks
        if isinstance(check, dict) and bool(check.get("hash_mismatch"))
    )

    prediction_total = len(prediction_payloads)
    validation_total = len(validation_checks)
    reproducibility_total = len(reproducibility_checks)
    return {
        "latency": latency_summary(latencies_ms),
        "invalid_actions": {
            "total": prediction_total,
            "invalid": invalid_count,
            "rate": invalid_count / prediction_total if prediction_total else 1.0,
        },
        "validation": {
            "total": validation_total,
            "passed": validation_passed,
            "pass_rate": validation_passed / validation_total if validation_total else 0.0,
        },
        "reproducibility": {
            "total": reproducibility_total,
            "passed": reproducibility_passed,
            "pass_rate": reproducibility_passed / reproducibility_total if reproducibility_total else 0.0,
            "hash_mismatch_count": hash_mismatch_count,
        },
    }


def evaluate_acceptance_criteria(
    metrics: dict[str, Any],
    criteria: AcceptanceCriteria = DEFAULT_ACCEPTANCE_CRITERIA,
) -> dict[str, Any]:
    latency = dict(metrics.get("latency") or {})
    invalid_actions = dict(metrics.get("invalid_actions") or {})
    validation = dict(metrics.get("validation") or {})
    reproducibility = dict(metrics.get("reproducibility") or {})

    checks = [
        gate(
            "latency_sample_size",
            int(latency.get("count", 0)) >= criteria.latency_sample_min,
            latency.get("count", 0),
            f">= {criteria.latency_sample_min}",
            "Latency target must be measured on at least the configured sample count.",
        ),
        gate(
            "latency_p95_ms",
            float(latency.get("p95_ms", float("inf"))) <= criteria.latency_p95_ms_max,
            latency.get("p95_ms"),
            f"<= {criteria.latency_p95_ms_max}",
            "p95 prediction latency must fit the interactive service budget.",
        ),
        gate(
            "latency_p99_ms",
            float(latency.get("p99_ms", float("inf"))) <= criteria.latency_p99_ms_max,
            latency.get("p99_ms"),
            f"<= {criteria.latency_p99_ms_max}",
            "p99 prediction latency must fit the worst-case service budget.",
        ),
        gate(
            "invalid_action_sample_size",
            int(invalid_actions.get("total", 0)) >= criteria.invalid_action_sample_min,
            invalid_actions.get("total", 0),
            f">= {criteria.invalid_action_sample_min}",
            "Invalid-action rate must be measured on at least the configured sample count.",
        ),
        gate(
            "invalid_action_rate",
            float(invalid_actions.get("rate", 1.0)) <= criteria.invalid_action_rate_max,
            invalid_actions.get("rate"),
            f"<= {criteria.invalid_action_rate_max}",
            "Selected actions must always be legal after action masking and fallback.",
        ),
        gate(
            "validation_check_count",
            int(validation.get("total", 0)) >= criteria.validation_check_min,
            validation.get("total", 0),
            f">= {criteria.validation_check_min}",
            "Validation pass rate must be measured on at least one validation check.",
        ),
        gate(
            "validation_pass_rate",
            float(validation.get("pass_rate", 0.0)) >= criteria.validation_pass_rate_min,
            validation.get("pass_rate"),
            f">= {criteria.validation_pass_rate_min}",
            "Delivery validation checks must meet the configured pass-rate target.",
        ),
        gate(
            "reproducibility_check_count",
            int(reproducibility.get("total", 0)) >= criteria.reproducibility_check_min,
            reproducibility.get("total", 0),
            f">= {criteria.reproducibility_check_min}",
            "Reproducibility must be measured on at least one deterministic check.",
        ),
        gate(
            "reproducibility_pass_rate",
            float(reproducibility.get("pass_rate", 0.0)) >= criteria.reproducibility_pass_rate_min,
            reproducibility.get("pass_rate"),
            f">= {criteria.reproducibility_pass_rate_min}",
            "Seeded training/evaluation/runtime checks must be reproducible.",
        ),
        gate(
            "reproducibility_hash_mismatch_count",
            int(reproducibility.get("hash_mismatch_count", 0)) == 0,
            reproducibility.get("hash_mismatch_count", 0),
            "== 0",
            "Deterministic artifact/hash comparisons must not diverge.",
        ),
    ]
    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "criteria": criteria.to_dict(),
        "observed": {
            "latency": latency,
            "invalid_actions": invalid_actions,
            "validation": validation,
            "reproducibility": reproducibility,
        },
        "checks": checks,
    }


def criteria_from_mapping(raw: dict[str, Any] | None) -> AcceptanceCriteria:
    values = dict(raw or {})
    allowed = set(AcceptanceCriteria.__dataclass_fields__)
    return AcceptanceCriteria(**{key: values[key] for key in sorted(allowed & set(values))})


def gate(name: str, passed: bool, observed: Any, threshold: Any, impact: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "impact": impact,
    }


def check_passed(check: bool | dict[str, Any]) -> bool:
    if isinstance(check, bool):
        return check
    status = str(check.get("status") or "").upper()
    if status:
        return status == "PASS"
    return bool(check.get("passed"))


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
