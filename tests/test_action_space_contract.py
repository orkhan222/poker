from __future__ import annotations

from poker_agent.action_space import CANONICAL_ACTIONS, ActionSpace, constrain_probabilities
from poker_agent.agents import RuleBasedAgent
from poker_agent.schemas import PredictionRequest


def test_facing_bet_derives_call_fold_raise_all_in_space() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "to_call": 1.0,
            "stack": 100.0,
            "min_raise": 2.0,
            "min_raise_to": 3.0,
            "max_raise_to": 100.0,
            "min_raise_by": 2.0,
            "max_raise_by": 99.0,
        }
    )

    assert request.legal_actions == ("fold", "call", "raise", "all_in")
    assert request.min_raise_to == 3.0
    assert request.max_raise_to == 100.0
    assert request.min_raise_by == 2.0
    assert request.max_raise_by == 99.0


def test_no_bet_derives_check_bet_all_in_space() -> None:
    request = PredictionRequest.from_dict({"position": "BB", "to_call": 0.0, "stack": 50.0, "min_raise": 2.0})

    assert request.legal_actions == ("check", "bet", "all_in")


def test_explicit_legal_action_mask_is_filtered_to_physical_actions() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "to_call": 4.0,
            "stack": 80.0,
            "min_raise": 8.0,
            "legal_actions": ["check", "call", "raise"],
        }
    )

    assert request.legal_actions == ("call", "raise")


def test_probability_mask_removes_illegal_actions_and_keeps_canonical_keys() -> None:
    action_space = ActionSpace.from_state({}, to_call=2.0, stack=40.0, min_raise=4.0)

    action, probabilities, warnings = constrain_probabilities({"check": 0.99, "call": 0.01}, action_space)

    assert action == "call"
    assert set(probabilities) == set(CANONICAL_ACTIONS)
    assert probabilities["check"] == 0.0
    assert probabilities["call"] == 1.0
    assert warnings


def test_rule_agent_response_exposes_action_space_and_sizing() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Ad"],
            "pot": 2.5,
            "to_call": 1.0,
            "stack": 100.0,
            "min_raise": 2.0,
            "max_raise": 100.0,
            "min_raise_to": 3.0,
            "max_raise_to": 100.0,
            "min_raise_by": 2.0,
            "max_raise_by": 99.0,
            "legal_actions": ["fold", "call", "raise", "all_in"],
        }
    )

    response = RuleBasedAgent().predict(request).to_dict()

    assert response["action"] in response["legal_actions"]
    assert response["probabilities"]["check"] == 0.0
    assert response["probabilities"]["bet"] == 0.0
    assert response["action_space"]["min_raise_to"] == 3.0
    assert response["action_space"]["max_raise_by"] == 99.0
    assert abs(sum(response["probabilities"].values()) - 1.0) < 1e-9
    if response["action"] == "raise":
        assert response["raise_to"] == 3.0
        assert response["raise_by"] == 2.0


def test_prediction_payload_contains_action_space_contract() -> None:
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

    assert response["action"] in response["legal_actions"]
    assert "raise_to" in response
    assert "raise_by" in response
    assert "action_space" in response
