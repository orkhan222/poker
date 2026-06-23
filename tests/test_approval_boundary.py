from __future__ import annotations

from poker_agent.approval_boundary import calculate_approval_boundary


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
