from __future__ import annotations

import json
from pathlib import Path

from poker_agent.project_scope import (
    PROJECT_SCOPE_CONTRACT_VERSION,
    describe_project_scope_contract,
    validate_project_scope,
    write_project_scope_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def test_project_scope_contract_captures_goals_phases_and_dataset_model() -> None:
    contract = describe_project_scope_contract()

    assert contract["schema_version"] == PROJECT_SCOPE_CONTRACT_VERSION
    assert "deploy_as_authorized_microservice" in contract["goals"]
    assert {phase["key"] for phase in contract["phases"]} == {
        "phase_1_baselines",
        "phase_2_selection_optimization",
        "phase_3_evaluation",
        "phase_4_deployment",
    }
    assert {"hands.csv", "players.csv", "actions.csv", "stack_events.csv"}.issubset(contract["dataset_model"])
    assert "action_amount" in contract["dataset_model"]["actions.csv"]["fields"]
    assert "ocr_confidence" in contract["dataset_model"]["actions.csv"]["fields"]


def test_project_scope_declares_senior_gap_requirements_as_machine_readable_contract() -> None:
    contract = describe_project_scope_contract()
    keys = {item["key"] for item in contract["senior_requirements"]}

    assert {
        "game_scope",
        "deployment_api",
        "dataset_schema_extensions",
        "data_validation",
        "labeling_contract",
        "action_and_state_space",
        "baseline_and_architecture",
        "rl_self_play",
        "evaluation_acceptance",
        "mlops_monitoring_security",
        "final_deliverables",
    }.issubset(keys)


def test_project_scope_validation_and_reports_pass_for_delivery_repo() -> None:
    outputs = write_project_scope_reports(ROOT)
    validation = validate_project_scope(ROOT)

    assert validation["status"] == "PASS"
    assert validation["fingerprint"]
    assert outputs["json"].exists()
    assert outputs["docs"].exists()
    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert report["validation"]["status"] == "PASS"
