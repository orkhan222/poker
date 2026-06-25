from __future__ import annotations

import json

from poker_agent.llm_decision_context import (
    build_decision_context_report,
    build_decision_prompt,
    legal_actions_for_request,
    parse_decision_output,
)
from poker_agent.schemas import PredictionRequest


def test_legal_actions_when_facing_bet() -> None:
    request = PredictionRequest(position="BTN", street="preflop", to_call=2.0, pot=5.0, stack=100.0)

    assert legal_actions_for_request(request) == ("fold", "call", "raise")


def test_legal_actions_when_not_facing_bet() -> None:
    request = PredictionRequest(position="BB", street="flop", to_call=0.0, pot=6.0, stack=80.0)

    assert legal_actions_for_request(request) == ("check", "bet")


def test_full_context_prompt_contains_rules_guidelines_and_constraints() -> None:
    request = PredictionRequest(
        position="BTN",
        street="preflop",
        hole_cards=["Ah", "Kd"],
        pot=2.5,
        to_call=1.0,
        stack=100.0,
        min_raise=4.5,
    )

    prompt = build_decision_prompt(request, "full_in_context")

    assert "Texas Hold'em rules" in prompt.system_context
    assert "Decision guidelines" in prompt.system_context
    assert "Output constraints" in prompt.system_context
    assert "fold, call, raise" in prompt.system_context


def test_parse_decision_output_enforces_legal_actions() -> None:
    request = PredictionRequest(position="BB", street="flop", pot=6.0, to_call=0.0, stack=90.0)
    raw = json.dumps(
        {
            "action": "call",
            "probabilities": {"fold": 0.1, "call": 0.8, "raise": 0.1},
            "confidence": 0.8,
            "bet_size": 3.0,
            "reason_code": "uncertain",
        }
    )

    response = parse_decision_output(raw, request)

    assert response.action == "check"
    assert response.probabilities["check"] == 1.0
    assert response.probabilities["call"] == 0.0
    assert response.warnings


def test_parse_decision_output_reports_schema_repairs() -> None:
    request = PredictionRequest(position="BTN", street="preflop", to_call=1.0, pot=3.0, stack=50.0)
    raw = json.dumps(
        {
            "action": "call",
            "probabilities": {"call": 2.0},
            "confidence": 2.0,
            "bet_size": -1.0,
        }
    )

    response = parse_decision_output(raw, request)

    assert response.action == "call"
    assert abs(sum(response.probabilities.values()) - 1.0) < 1e-9
    assert len(response.warnings) >= 4


def test_decision_context_report_has_required_modes() -> None:
    report = build_decision_context_report()

    assert report["default_context_mode"] == "full_in_context"
    assert {"minimal_zero_shot", "rules_grounded", "full_in_context"} == set(
        report["supported_context_modes"]
    )
    assert any(item["contains_rules"] for item in report["prompt_records"])
    assert any(item["contains_strategy_guidelines"] for item in report["prompt_records"])


def test_full_context_serializes_observed_opponent_timing() -> None:
    request = PredictionRequest(
        position="BTN",
        street="turn",
        pot=20.0,
        stack=80.0,
        opponent_wait_before_turn_ms=1600.0,
        opponent_wait_after_hero_action_ms=2300.0,
    )

    prompt = build_decision_prompt(request, "full_in_context")

    assert '"opponent_wait_before_turn_ms": 1600.0' in prompt.user_context
    assert '"opponent_wait_after_hero_action_ms": 2300.0' in prompt.user_context
