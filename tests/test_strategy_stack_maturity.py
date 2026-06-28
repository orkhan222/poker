from __future__ import annotations

import json
from pathlib import Path

from poker_agent.strategy_stack_maturity import (
    DEPLOYMENT_APPROVED_WITH_MONITORING,
    NOT_FINAL_ENGINE,
    build_strategy_stack_maturity,
    validate_strategy_stack_maturity,
)


def test_strategy_stack_is_deployment_approved_but_not_final_engine(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "deployed_strategy_gate.json").write_text(
        json.dumps({"status": "PASS", "strategy_policy_status": "APPROVED"}),
        encoding="utf-8",
    )
    (reports / "production_approval.json").write_text(
        json.dumps({"overall_status": "APPROVED_WITH_COMPONENT_RISK"}),
        encoding="utf-8",
    )
    (reports / "policy_acceptance.json").write_text(
        json.dumps(
            {
                "human_action_alignment_status": "PASS",
                "human_likeness": {"status": "PASS"},
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_self_play.json").write_text(
        json.dumps({"status": "PASS", "production_scale_status": "PASS"}),
        encoding="utf-8",
    )

    payload = build_strategy_stack_maturity(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["current_strategy_stack"]["status"] == DEPLOYMENT_APPROVED_WITH_MONITORING
    assert payload["current_strategy_stack"]["monitoring_required"] is True
    assert payload["final_engine_boundary"]["status"] == NOT_FINAL_ENGINE
    assert payload["final_engine_boundary"]["final_engine_claim_allowed"] is False
    assert payload["final_engine_boundary"]["maximally_optimized_claim_allowed"] is False


def test_strategy_stack_maturity_blocks_final_engine_claim() -> None:
    payload = build_strategy_stack_maturity(Path("."))
    payload["final_engine_boundary"]["status"] = "FINAL_MAXIMALLY_OPTIMIZED_ENGINE"
    payload["final_engine_boundary"]["final_engine_claim_allowed"] = True
    payload["final_engine_boundary"]["maximally_optimized_claim_allowed"] = True

    invariants = validate_strategy_stack_maturity(payload)

    assert invariants["status"] == "FAIL"
    assert len(invariants["violations"]) >= 3


def test_strategy_stack_maturity_endpoint_returns_contract() -> None:
    from poker_agent.service import strategy_stack_maturity_json

    payload = strategy_stack_maturity_json()

    assert payload["overall_status"] == "PASS"
    assert payload["current_strategy_stack"]["status"] == DEPLOYMENT_APPROVED_WITH_MONITORING
    assert payload["final_engine_boundary"]["status"] == NOT_FINAL_ENGINE
    assert payload["final_engine_boundary"]["final_engine_claim_allowed"] is False
