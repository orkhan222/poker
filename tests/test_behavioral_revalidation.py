from __future__ import annotations

import json
from pathlib import Path

from poker_agent.behavioral_revalidation import build_behavioral_revalidation, validate_behavioral_revalidation


def test_behavioral_revalidation_preserves_current_scope_pass_and_future_revalidation(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "policy_acceptance.json").write_text(
        json.dumps(
            {
                "human_likeness": {
                    "status": "PASS",
                    "timing_and_bet_size_status": "PASS",
                    "js_divergence": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "deployed_strategy_gate.json").write_text(
        json.dumps({"strategy_policy_status": "APPROVED"}),
        encoding="utf-8",
    )
    (reports / "strategy_stack_maturity.json").write_text(
        json.dumps({"current_strategy_stack": {"status": "APPROVED_FOR_DEPLOYMENT_WITH_MONITORING"}}),
        encoding="utf-8",
    )

    payload = build_behavioral_revalidation(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["current_validation_scope"]["human_likeness_status"] == "PASS"
    assert payload["current_validation_scope"]["action_distribution_status"] == "PASS"
    assert payload["revalidation_boundary"]["larger_clean_real_gameplay_revalidation_required"] is True
    assert payload["revalidation_boundary"]["generalized_human_likeness_claim_allowed"] is False
    assert payload["revalidation_boundary"]["generalized_action_distribution_claim_allowed"] is False
    assert payload["revalidation_boundary"]["production_blocker"] is False


def test_behavioral_revalidation_blocks_generalized_claims() -> None:
    payload = {
        "current_validation_scope": {
            "human_likeness_status": "PASS",
            "action_distribution_status": "PASS",
        },
        "revalidation_boundary": {
            "larger_clean_real_gameplay_revalidation_required": False,
            "revalidation_scope": "current_delivery_validation_scope",
            "generalized_human_likeness_claim_allowed": True,
            "generalized_action_distribution_claim_allowed": True,
            "production_blocker": False,
        },
    }

    invariants = validate_behavioral_revalidation(payload)

    assert invariants["status"] == "FAIL"
    assert len(invariants["violations"]) >= 4


def test_behavioral_revalidation_endpoint_returns_contract() -> None:
    from poker_agent.service import behavioral_revalidation_json

    payload = behavioral_revalidation_json()

    assert payload["overall_status"] == "PASS"
    assert payload["current_validation_scope"]["human_likeness_status"] == "PASS"
    assert payload["current_validation_scope"]["action_distribution_status"] == "PASS"
    assert payload["revalidation_boundary"]["larger_clean_real_gameplay_revalidation_required"] is True
