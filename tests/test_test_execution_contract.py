from __future__ import annotations

import json
from pathlib import Path

from poker_agent.test_execution_contract import (
    CRITICAL_TEST_TARGETS,
    build_test_execution_contract,
    validate_test_execution_contract,
    write_test_execution_contract,
)


def _write_reports(reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "delivery_verification.json").write_text(
        json.dumps({"status": "PASS", "checks": [{"name": "zip_contract", "passed": True}]}),
        encoding="utf-8",
    )
    (reports / "evaluation_metric_contract.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "accuracy_alone_sufficient": False,
                "final_metric_bundle_passed": False,
                "final_strategy_quality_claim_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (reports / "final_delivery_acceptance.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "acceptance_summary": {
                    "service_delivery": "READY",
                    "deployed_strategy_stack": "APPROVED",
                },
            }
        ),
        encoding="utf-8",
    )


def test_test_execution_contract_records_timeout_without_using_it_as_approval(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_test_execution_contract(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["full_pytest"]["status"] == "TIMEOUT"
    assert payload["full_pytest"]["used_as_delivery_approval"] is False
    assert payload["critical_validation"]["status"] == "PASS"
    assert payload["critical_validation"]["targets"] == list(CRITICAL_TEST_TARGETS)
    assert payload["delivery_verifier"]["status"] == "PASS"
    assert payload["metric_contract"]["accuracy_alone_sufficient"] is False
    assert payload["current_delivery_blocker"] is False


def test_test_execution_contract_blocks_false_validation_claims(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_test_execution_contract(tmp_path)
    payload["full_pytest"]["used_as_delivery_approval"] = True
    payload["critical_validation"]["status"] = "FAIL"
    payload["delivery_verifier"]["status"] = "FAIL"
    payload["metric_contract"]["accuracy_alone_sufficient"] = True

    invariants = validate_test_execution_contract(payload)

    assert invariants["status"] == "FAIL"
    assert "timed_out_full_pytest_must_not_be_used_as_delivery_approval" in invariants["violations"]
    assert "critical_validation_must_pass" in invariants["violations"]
    assert "delivery_verifier_must_pass" in invariants["violations"]
    assert "accuracy_alone_must_remain_insufficient" in invariants["violations"]


def test_write_test_execution_contract_outputs_json_and_markdown(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = write_test_execution_contract(
        tmp_path,
        tmp_path / "reports" / "test_execution_contract.json",
        tmp_path / "reports" / "test_execution_contract.md",
    )

    assert payload["overall_status"] == "PASS"
    assert (tmp_path / "reports" / "test_execution_contract.json").exists()
    assert "Full pytest timeout is recorded" in (tmp_path / "reports" / "test_execution_contract.md").read_text(
        encoding="utf-8"
    )
