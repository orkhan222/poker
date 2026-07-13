from __future__ import annotations

import pytest

from poker_agent.api_contract import api_contract
from poker_agent.autonomous_agent import (
    AgentLifecycleError,
    AutonomousDecision,
    AutonomousPokerAgent,
)
from poker_agent.schemas import PredictionRequest, PredictionResponse


class FixedPolicy:
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        return PredictionResponse(
            action="check",
            probabilities={
                "fold": 0.10,
                "call": 0.70,
                "check": 0.15,
                "bet": 0.03,
                "raise": 0.02,
            },
            model_status="fixed_test_policy",
        )


def observation(sequence_number: int, event_id: str) -> dict:
    return {
        "hand_id": "hand-001",
        "sequence_number": sequence_number,
        "event_id": event_id,
        "state": {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["AS", "KD"],
            "board_cards": [],
            "pot": 6.0,
            "to_call": 2.0,
            "stack": 98.0,
            "min_raise": 4.0,
            "player_count": 6,
        },
    }


def test_agent_constrains_policy_to_legal_actions() -> None:
    agent = AutonomousPokerAgent(FixedPolicy())

    decision, replayed = agent.decide(observation(0, "event-0"))

    assert replayed is False
    assert decision.action == "call"
    assert decision.legal_actions == ("fold", "call", "raise")
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)
    assert set(decision.probabilities) == {"fold", "call", "raise"}
    assert decision.bet_size == 2.0
    assert any("replaced by legal action" in warning for warning in decision.warnings)


def test_agent_replays_duplicate_event_idempotently() -> None:
    agent = AutonomousPokerAgent(FixedPolicy())
    first, first_replay = agent.decide(observation(0, "event-0"))
    second, second_replay = agent.decide(observation(0, "event-0"))

    assert first_replay is False
    assert second_replay is True
    assert first.decision_id == second.decision_id
    assert agent.session("hand-001")["decision_count"] == 1


def test_agent_rejects_event_id_reuse_for_different_payload() -> None:
    agent = AutonomousPokerAgent(FixedPolicy())
    agent.decide(observation(0, "event-0"))
    conflicting = observation(1, "event-0")

    with pytest.raises(AgentLifecycleError, match="already used"):
        agent.decide(conflicting)


def test_agent_rejects_out_of_order_observation() -> None:
    agent = AutonomousPokerAgent(FixedPolicy())
    agent.decide(observation(1, "event-1"))

    with pytest.raises(AgentLifecycleError, match="sequence_number must be greater"):
        agent.decide(observation(0, "event-0"))


class TwoStepEnvironment:
    def __init__(self) -> None:
        self.step = 0
        self.actions: list[str] = []

    def reset(self) -> dict:
        return observation(0, "event-0")

    def apply_action(self, decision: AutonomousDecision) -> dict | None:
        self.actions.append(decision.action)
        self.step += 1
        if self.step == 1:
            return observation(1, "event-1")
        return None

    def is_terminal(self) -> bool:
        return self.step >= 2

    def result(self) -> dict:
        return {"chip_delta": 3.5, "actions": list(self.actions)}


def test_agent_runs_complete_simulation_episode() -> None:
    agent = AutonomousPokerAgent(FixedPolicy())

    result = agent.run_episode(TwoStepEnvironment())

    assert result["status"] == "completed"
    assert result["decision_count"] == 2
    assert result["result"]["chip_delta"] == 3.5
    assert result["session"]["status"] == "settled"


def test_settled_session_rejects_new_decisions() -> None:
    agent = AutonomousPokerAgent(FixedPolicy())
    agent.decide(observation(0, "event-0"))
    agent.settle("hand-001", {"chip_delta": 0.0})

    with pytest.raises(AgentLifecycleError, match="is settled"):
        agent.decide(observation(1, "event-1"))


def test_api_contract_exposes_autonomous_lifecycle() -> None:
    contract = api_contract()["autonomous_agent"]

    assert contract["agent_type"] == "controlled_stateful_policy_agent"
    assert contract["decision_endpoint"] == "/agent/decide"
    assert "legal-action enforcement" in contract["lifecycle_controls"]


def test_controlled_session_routes_are_hidden_from_public_openapi() -> None:
    from poker_agent.service import app

    paths = set(app.openapi()["paths"])

    assert paths == {"/predict"}
    assert "/predict" in paths
    assert "/health.json" not in paths
    assert "/agent/capabilities.json" not in paths
    assert "/agent/decide" not in paths
    assert "/agent/sessions/{hand_id}" not in paths
    assert "/agent/sessions/{hand_id}/settle" not in paths


def test_public_openapi_hides_internal_reports_and_validation_schemas() -> None:
    from poker_agent.service import app

    schema = app.openapi()
    schemas = (schema.get("components") or {}).get("schemas") or {}

    assert "/final-delivery-acceptance.json" not in schema["paths"]
    assert "/health.json" not in schema["paths"]
    assert "/strategy-stack-maturity.json" not in schema["paths"]
    assert "/strategy-readiness.json" not in schema["paths"]
    assert "/llm-role-boundary.json" not in schema["paths"]
    assert [tag["name"] for tag in schema.get("tags", [])] == ["Prediction"]
    assert set(schemas) == {
        "ActionProbabilitiesBody",
        "BettingHistoryBody",
        "GameScopeBody",
        "PredictRequestBody",
        "PredictResponseBody",
        "TimingContextBody",
    }
    assert "HTTPValidationError" not in schemas
    assert "ValidationError" not in schemas
    assert "additionalProp1" not in str(schema)
    assert "fold" in str(schemas["ActionProbabilitiesBody"])
    assert "raise" in str(schemas["ActionProbabilitiesBody"])

    request_example = schemas["PredictRequestBody"]["example"]
    assert set(request_example) == {
        "position",
        "street",
        "hole_cards",
        "board_cards",
        "pot",
        "to_call",
        "stack",
        "min_raise",
        "player_count",
        "game_scope",
    }
    assert request_example["game_scope"] == {
        "game_variant": "nl_holdem",
        "game_type": "cash",
        "table_format": "6_max",
        "small_blind": 0.5,
        "big_blind": 1.0,
        "ante": 0.0,
        "rake_percentage": 0.0,
        "rake_cap": 0.0,
        "stack_unit": "chips",
    }
    assert "betting_history" not in request_example
    assert "timing_context" not in request_example

    operation = schema["paths"]["/predict"]["post"]
    assert "JSON request body" in operation["description"]
    assert "No query parameters" in operation["description"]


def test_swagger_ui_hides_models_panel() -> None:
    from poker_agent.service import app

    assert app.swagger_ui_parameters["defaultModelsExpandDepth"] == -1
    assert app.swagger_ui_parameters["docExpansion"] == "full"


def test_public_docs_page_uses_client_facing_swagger_shell() -> None:
    from poker_agent.service import CLIENT_SWAGGER_HTML, app, public_docs_page

    docs_routes = [route for route in app.routes if getattr(route, "path", None) == "/docs"]
    html = public_docs_page()

    assert len(docs_routes) == 1
    assert docs_routes[0].include_in_schema is False
    assert html == CLIENT_SWAGGER_HTML
    assert "client-facing-swagger" in html
    assert "compact-public-docs" in html
    assert "client-docs-helper" in html
    assert "SwaggerUIBundle" in html
    assert "parameters-container" in html
    assert "hideEmptyParameterSections" in html
    assert "expandPredictOperation" in html
    assert "keepPredictExpanded" in html
    assert "onComplete: keepPredictExpanded" in html
    assert "tryItOutEnabled: false" in html
    assert "tryItOutEnabled: true" not in html
    assert "supportedSubmitMethods: []" in html
    assert "try-out__btn" in html
    assert 'url: "/openapi.json"' in html
    assert app.docs_url is None
