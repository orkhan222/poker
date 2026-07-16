from __future__ import annotations

from poker_agent.acceptance_criteria import (
    DEFAULT_ACCEPTANCE_CRITERIA,
    AcceptanceCriteria,
    build_acceptance_metrics,
    evaluate_acceptance_criteria,
    latency_summary,
)


def test_default_acceptance_criteria_are_numeric_and_explicit() -> None:
    criteria = DEFAULT_ACCEPTANCE_CRITERIA

    assert criteria.latency_p95_ms_max == 150.0
    assert criteria.latency_p99_ms_max == 300.0
    assert criteria.invalid_action_rate_max == 0.0
    assert criteria.validation_pass_rate_min == 1.0
    assert criteria.reproducibility_pass_rate_min == 1.0


def test_acceptance_metrics_pass_when_all_targets_are_met() -> None:
    metrics = build_acceptance_metrics(
        latencies_ms=[20.0, 25.0, 30.0, 35.0],
        prediction_payloads=[
            {"action": "call", "legal_actions": ["fold", "call", "raise"]},
            {"action": "check", "legal_actions": ["check", "bet"]},
        ],
        validation_checks=[True, {"status": "PASS"}],
        reproducibility_checks=[True, {"passed": True, "hash_mismatch": False}],
    )

    report = evaluate_acceptance_criteria(metrics)

    assert report["status"] == "PASS"
    assert report["observed"]["invalid_actions"]["rate"] == 0.0
    assert report["observed"]["validation"]["pass_rate"] == 1.0
    assert report["observed"]["reproducibility"]["pass_rate"] == 1.0


def test_acceptance_metrics_fail_invalid_actions_and_latency_regression() -> None:
    metrics = build_acceptance_metrics(
        latencies_ms=[100.0, 200.0, 400.0],
        prediction_payloads=[
            {"action": "check", "legal_actions": ["fold", "call"]},
        ],
        validation_checks=[True],
        reproducibility_checks=[True],
    )
    criteria = AcceptanceCriteria(latency_p95_ms_max=150.0, latency_p99_ms_max=300.0)

    report = evaluate_acceptance_criteria(metrics, criteria)
    failed = {check["name"] for check in report["checks"] if not check["passed"]}

    assert report["status"] == "FAIL"
    assert "latency_p95_ms" in failed
    assert "latency_p99_ms" in failed
    assert "invalid_action_rate" in failed


def test_latency_summary_reports_percentiles() -> None:
    summary = latency_summary([10.0, 20.0, 30.0, 40.0, 50.0])

    assert summary["count"] == 5
    assert summary["p50_ms"] == 30.0
    assert summary["p95_ms"] > summary["p50_ms"]
