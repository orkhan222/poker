from __future__ import annotations

from pathlib import Path

from poker_agent.behavioral_revalidation_proof import (
    build_behavioral_revalidation_proof,
    validate_behavioral_revalidation_proof,
)


def test_behavioral_revalidation_proof_blocks_false_generalization() -> None:
    payload = build_behavioral_revalidation_proof(Path("."))
    cases = {case["name"]: case for case in payload["proof_cases"]}

    assert payload["overall_status"] == "PASS"
    assert payload["proof_status"] == "PASS"
    assert cases["base_contract_is_valid"]["observed_status"] == "PASS"
    assert cases["blocks_generalized_human_likeness_claim"]["observed_status"] == "FAIL"
    assert cases["blocks_generalized_action_distribution_claim"]["observed_status"] == "FAIL"
    assert cases["blocks_missing_larger_real_gameplay_revalidation"]["observed_status"] == "FAIL"


def test_behavioral_revalidation_proof_validator_rejects_missing_case() -> None:
    payload = build_behavioral_revalidation_proof(Path("."))
    payload["proof_cases"] = payload["proof_cases"][:1]

    invariants = validate_behavioral_revalidation_proof(payload)

    assert invariants["status"] == "FAIL"
    assert any("missing_proof_cases" in item for item in invariants["violations"])


def test_behavioral_revalidation_proof_endpoint_returns_executable_proof() -> None:
    from poker_agent.service import behavioral_revalidation_proof_json

    payload = behavioral_revalidation_proof_json()

    assert payload["overall_status"] == "PASS"
    assert payload["proof_status"] == "PASS"
    assert payload["base_contract_summary"]["generalized_human_likeness_claim_allowed"] is False
    assert payload["base_contract_summary"]["generalized_action_distribution_claim_allowed"] is False
