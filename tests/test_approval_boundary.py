from __future__ import annotations

import pytest

from poker_agent.approval_boundary import (
    assert_approval_boundary,
    calculate_approval_boundary,
    validate_approval_boundary,
)


def test_raw_model_component_risk_is_not_deployment_blocker() -> None:
    boundary = calculate_approval_boundary(
        delivery_readiness={"overall_status": "READY_FOR_PRODUCTION_POLICY"},
        deployed_gate={"status": "PASS", "strategy_policy_status": "APPROVED"},
        production_gate={"status": "FAIL"},
        risk_register={
            "raw_supervised_model_status": "NOT_STANDALONE_APPROVED",
            "raw_artifact_runtime_status": {"status": "LOADABLE"},
            "risk_summary": {"deployment_blockers": 0, "component_risks": 1},
        },
        hygiene={"status": "PASS"},
    )

    assert boundary.service_delivery == "READY"
    assert boundary.deployed_strategy_stack == "APPROVED"
    assert boundary.raw_supervised_model_runtime == "LOADABLE"
    assert boundary.raw_supervised_model_standalone == "NOT_STANDALONE_APPROVED"
    assert boundary.production_blocker is False
    assert boundary.component_risk is True
    assert boundary.release_status == "READY_WITH_COMPONENT_RISK"
    assert validate_approval_boundary(boundary) == []
    assert_approval_boundary(boundary)


def test_raw_gate_failure_cannot_be_reported_as_standalone_approved() -> None:
    violations = validate_approval_boundary(
        {
            "service_delivery": "READY",
            "deployed_strategy_stack": "APPROVED",
            "raw_supervised_model_runtime": "LOADABLE",
            "raw_supervised_model_standalone": "STANDALONE_APPROVED",
            "raw_production_gate": "FAIL",
            "production_blocker": False,
            "component_risk": False,
            "deployment_blockers": 0,
            "component_risks": 0,
            "release_status": "READY",
        }
    )

    assert "raw_model_cannot_be_standalone_approved_when_raw_gate_is_not_pass" in violations
    with pytest.raises(AssertionError):
        assert_approval_boundary(
            {
                "service_delivery": "READY",
                "deployed_strategy_stack": "APPROVED",
                "raw_supervised_model_runtime": "LOADABLE",
                "raw_supervised_model_standalone": "STANDALONE_APPROVED",
                "raw_production_gate": "FAIL",
                "production_blocker": False,
                "component_risk": False,
                "deployment_blockers": 0,
                "component_risks": 0,
                "release_status": "READY",
            }
        )


def test_raw_model_component_risk_must_not_be_promoted_to_production_blocker() -> None:
    violations = validate_approval_boundary(
        {
            "service_delivery": "READY",
            "deployed_strategy_stack": "APPROVED",
            "raw_supervised_model_runtime": "LOADABLE",
            "raw_supervised_model_standalone": "NOT_STANDALONE_APPROVED",
            "raw_production_gate": "FAIL",
            "production_blocker": True,
            "component_risk": True,
            "deployment_blockers": 0,
            "component_risks": 1,
            "release_status": "NOT_READY",
        }
    )

    assert "raw_model_component_risk_must_not_be_promoted_to_production_blocker" in violations
