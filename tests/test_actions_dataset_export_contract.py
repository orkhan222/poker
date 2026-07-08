from __future__ import annotations

from pathlib import Path

from poker_agent.actions_context_quality import REQUIRED_EXPLICIT_ACTION_FIELDS
from poker_agent.actions_dataset_export_contract import (
    ACTIONS_DATASET_EXPORT_STATUS,
    build_actions_dataset_export_contract,
    validate_actions_dataset_export_contract,
)
from poker_agent.api_contract import api_contract


def test_actions_dataset_export_contract_passes_with_current_delivery_boundary() -> None:
    project_root = Path(__file__).resolve().parents[1]

    payload = build_actions_dataset_export_contract(project_root)
    current_boundary = payload["current_delivery_boundary"]
    future_boundary = payload["future_export_boundary"]

    assert payload["overall_status"] == "PASS"
    assert payload["status"] == ACTIONS_DATASET_EXPORT_STATUS
    assert payload["source_table"] == "actions.csv"
    assert set(payload["required_explicit_fields"]) == set(REQUIRED_EXPLICIT_ACTION_FIELDS)
    assert set(payload["field_contract"]) == set(REQUIRED_EXPLICIT_ACTION_FIELDS)
    assert current_boundary["current_delivery_blocker"] is False
    assert current_boundary["reconstructed_context_allowed"] is True
    assert future_boundary["explicit_export_required"] is True
    assert future_boundary["model_quality_risk_until_export_is_instrumented"] is True
    assert future_boundary["must_persist_decision_time_values"] is True
    assert future_boundary["must_not_use_target_row_values"] is True
    assert future_boundary["must_not_use_future_outcome_fields"] is True


def test_actions_dataset_export_contract_is_exposed_through_api_and_service() -> None:
    from poker_agent.service import actions_dataset_export_contract_json

    contract = api_contract()["actions_dataset_export_contract"]
    payload = actions_dataset_export_contract_json()

    assert contract["endpoint"] == "/actions-dataset-export-contract.json"
    assert contract["status"] == ACTIONS_DATASET_EXPORT_STATUS
    assert contract["current_delivery_blocker"] is False
    assert contract["reconstructed_context_allowed_for_current_delivery"] is True
    assert contract["model_quality_risk"] is True
    assert set(contract["required_explicit_fields"]) == set(REQUIRED_EXPLICIT_ACTION_FIELDS)
    assert payload["overall_status"] == "PASS"
    assert payload["future_export_boundary"]["explicit_export_required"] is True


def test_actions_dataset_export_contract_blocks_false_resolution_claim() -> None:
    payload = {
        "source_table": "actions.csv",
        "status": "COMPLETE",
        "required_explicit_fields": ["to_call"],
        "field_contract": {"to_call": {}},
        "current_delivery_boundary": {
            "current_delivery_blocker": True,
            "reconstructed_context_allowed": False,
        },
        "future_export_boundary": {
            "explicit_export_required": False,
            "model_quality_risk_until_export_is_instrumented": False,
            "must_persist_decision_time_values": False,
            "must_not_use_target_row_values": False,
            "must_not_use_future_outcome_fields": False,
        },
    }

    invariants = validate_actions_dataset_export_contract(payload)

    assert invariants["status"] == "FAIL"
    assert "dataset_export_status_must_require_explicit_betting_context" in invariants["violations"]
    assert "dataset_export_required_fields_must_match_contract" in invariants["violations"]
    assert "dataset_export_field_contract_must_cover_required_fields" in invariants["violations"]
    assert "dataset_export_gap_must_not_block_current_delivery" in invariants["violations"]
    assert "dataset_export_must_allow_current_reconstruction" in invariants["violations"]
    assert "future_dataset_export_must_require_explicit_context" in invariants["violations"]
    assert "future_dataset_export_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "future_dataset_export_must_persist_decision_time_values" in invariants["violations"]
    assert "future_dataset_export_must_forbid_target_row_values" in invariants["violations"]
    assert "future_dataset_export_must_forbid_future_outcome_fields" in invariants["violations"]
