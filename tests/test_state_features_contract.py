from __future__ import annotations

from poker_agent.agents import RuleBasedAgent
from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


def test_state_fields_are_parsed_from_top_level_payload() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "turn",
            "pot": 25.0,
            "current_bet": 6.0,
            "amount_to_call": 4.0,
            "stack": 90.0,
            "effective_stack": 80.0,
            "small_blind": 0.5,
            "big_blind": 1.0,
            "ante": 0.1,
            "dealer_position": "BTN",
            "action_order": ["UTG", "CO", "BTN", "SB", "BB"],
            "min_raise": 8.0,
        }
    )

    assert request.to_call == 4.0
    assert request.amount_to_call == 4.0
    assert request.current_bet == 6.0
    assert request.effective_stack == 80.0
    assert request.button_position == "BTN"
    assert request.dealer_position == "BTN"
    assert request.action_order_index() == 2
    assert request.action_space.to_call == 4.0
    assert request.action_space.stack == 80.0


def test_state_fields_can_be_parsed_from_game_scope() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BB",
            "to_call": 2.0,
            "stack": 70.0,
            "game_scope": {
                "small_blind": 1.0,
                "big_blind": 2.0,
                "ante": 0.25,
                "button_position": "BTN",
            },
        }
    )

    assert request.small_blind == 1.0
    assert request.big_blind == 2.0
    assert request.ante == 0.25
    assert request.button_position == "BTN"


def test_request_to_features_exposes_required_state_context() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "turn",
            "pot": 25.0,
            "current_bet": 6.0,
            "amount_to_call": 4.0,
            "stack": 90.0,
            "effective_stack": 80.0,
            "small_blind": 0.5,
            "big_blind": 1.0,
            "ante": 0.1,
            "button_position": "BTN",
            "dealer_position": "BTN",
            "action_order": ["UTG", "CO", "BTN", "SB", "BB"],
            "min_raise": 8.0,
        }
    )

    features = request_to_features(request)

    assert features["pot_size"] == 25.0
    assert features["current_bet"] == 6.0
    assert features["amount_to_call"] == 4.0
    assert features["effective_stack"] == 80.0
    assert features["spr"] == 80.0 / 29.0
    assert features["small_blind"] == 0.5
    assert features["big_blind"] == 1.0
    assert features["ante"] == 0.1
    assert features["pot_in_big_blinds"] == 25.0
    assert features["amount_to_call_in_big_blinds"] == 4.0
    assert features["is_button"] == 1.0
    assert features["is_dealer"] == 1.0
    assert features["street=turn"] == 1.0
    assert features["is_turn"] == 1.0
    assert features["action_order_known"] == 1.0
    assert features["action_order_index"] == 2.0
    assert features["players_before_hero"] == 2.0
    assert features["players_after_hero"] == 2.0


def test_prediction_response_exposes_state_context() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "flop",
            "pot": 10.0,
            "current_bet": 3.0,
            "amount_to_call": 3.0,
            "stack": 50.0,
            "effective_stack": 45.0,
            "big_blind": 1.0,
            "button_position": "BTN",
            "action_order": ["CO", "BTN", "SB", "BB"],
        }
    )

    response = RuleBasedAgent().predict(request).to_dict()

    assert response["state_context"]["pot_size"] == 10.0
    assert response["state_context"]["current_bet"] == 3.0
    assert response["state_context"]["amount_to_call"] == 3.0
    assert response["state_context"]["effective_stack"] == 45.0
    assert response["state_context"]["action_order_index"] == 1
