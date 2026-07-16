from __future__ import annotations

from poker_agent.action_space import CANONICAL_ACTIONS
from poker_agent.agents import RuleBasedAgent
from poker_agent.api_contract import (
    API_VERSION,
    ERROR_CODES,
    PREDICT_RESPONSE_SCHEMA_VERSION,
    api_error,
    deployment_api_contract,
    predict_request_schema,
    predict_response_schema,
)
from poker_agent.schemas import PredictionRequest


def test_predict_request_schema_declares_required_state_fields() -> None:
    schema = predict_request_schema()

    assert schema["schema_version"] == "predict_request.v1"
    assert {"position", "street", "hole_cards", "pot", "stack"}.issubset(set(schema["required"]))
    properties = schema["properties"]
    for field in ("current_bet", "amount_to_call", "effective_stack", "legal_actions", "game_scope", "usage_boundary"):
        assert field in properties
    assert "usage_boundary" in schema["required"]


def test_predict_response_schema_declares_model_version_confidence_and_probabilities() -> None:
    schema = predict_response_schema()

    assert schema["schema_version"] == PREDICT_RESPONSE_SCHEMA_VERSION
    assert {"model_version", "confidence", "probabilities", "action"}.issubset(set(schema["required"]))
    assert set(schema["properties"]["probabilities"]["required"]) == set(CANONICAL_ACTIONS)
    assert schema["properties"]["action"]["enum"] == list(CANONICAL_ACTIONS)


def test_deployment_contract_exposes_endpoint_model_version_and_error_codes() -> None:
    contract = deployment_api_contract(model_version="test-policy:v1")

    assert contract["api_version"] == API_VERSION
    assert contract["endpoint"] == "/predict"
    assert contract["model_version"] == "test-policy:v1"
    assert {
        "INVALID_REQUEST",
        "UNSUPPORTED_ACTION_SPACE",
        "MODEL_UNAVAILABLE",
        "PREDICTION_FAILED",
        "UNAUTHORIZED",
        "RATE_LIMITED",
        "SECURITY_MISCONFIGURED",
        "USAGE_BOUNDARY_VIOLATION",
    }.issubset(set(ERROR_CODES))
    assert contract["error_response_schema"]["properties"]["error"]["properties"]["code"]["enum"] == sorted(ERROR_CODES)


def test_prediction_response_payload_matches_deployment_schema_surface() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Kd"],
            "pot": 2.5,
            "to_call": 1.0,
            "stack": 100.0,
            "min_raise": 2.0,
            "max_raise": 100.0,
        }
    )

    response = RuleBasedAgent().predict(request).to_dict()

    assert response["schema_version"] == PREDICT_RESPONSE_SCHEMA_VERSION
    assert response["model_version"] == "rule_based:v1"
    assert 0.0 <= response["confidence"] <= 1.0
    assert set(response["probabilities"]) == set(CANONICAL_ACTIONS)
    assert response["action"] in response["legal_actions"]


def test_api_error_payload_is_machine_readable() -> None:
    payload = api_error("INVALID_REQUEST", "Missing required field: position.", {"field": "position"})

    assert payload["schema_version"] == "error_response.v1"
    assert payload["error"]["code"] == "INVALID_REQUEST"
    assert payload["error"]["retryable"] is False
    assert payload["error"]["details"]["field"] == "position"
