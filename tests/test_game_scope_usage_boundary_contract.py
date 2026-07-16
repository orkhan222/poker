from __future__ import annotations

from poker_agent.features import request_to_features
from poker_agent.game_scope import GameScope, describe_game_scope_contract
from poker_agent.schemas import PredictionRequest
from poker_agent.usage_boundary import (
    ALLOWED_USAGE,
    BLOCKED_USAGE,
    describe_usage_boundary_contract,
    evaluate_usage_boundary,
)


def test_game_scope_normalizes_supported_nl_holdem_contract() -> None:
    scope = GameScope.from_payload(
        {
            "game_scope": {
                "game_type": "no-limit-holdem",
                "format": "cash_game",
                "table_size": "9max",
                "small_blind": 1,
                "big_blind": 2,
                "ante": 0.25,
                "rake_percentage": 5,
                "rake_cap": 3,
                "stack_unit": "bb",
            }
        }
    )

    assert scope.validate() == []
    assert scope.game_type == "nl_holdem"
    assert scope.format == "cash"
    assert scope.table_size == "9_max"
    assert scope.table_size_players == 9
    assert scope.rake_percentage == 0.05
    assert scope.stack_unit == "big_blinds"


def test_prediction_request_exposes_game_scope_in_state_context_and_features() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Kd"],
            "pot": 5,
            "stack": 100,
            "game_scope": {
                "game_type": "nl_holdem",
                "format": "tournament",
                "table_size": "9_max",
                "small_blind": 1,
                "big_blind": 2,
                "ante": 0.25,
                "stack_unit": "chips",
            },
        }
    )

    features = request_to_features(request)
    context = request.state_context()

    assert request.player_count == 9
    assert context["game_scope"]["game_type"] == "nl_holdem"
    assert context["game_scope"]["table_size"] == "9_max"
    assert features["scope_game_type_nl_holdem"] == 1.0
    assert features["scope_format_tournament"] == 1.0
    assert features["scope_table_9_max"] == 1.0
    assert features["big_blind"] == 2.0


def test_usage_boundary_allows_only_explicit_safe_use_cases() -> None:
    for declared_use in ALLOWED_USAGE:
        decision = evaluate_usage_boundary({"usage_boundary": {"declared_use": declared_use}})
        assert decision.allowed is True
        assert decision.declared_use == declared_use


def test_usage_boundary_blocks_missing_and_prohibited_uses() -> None:
    missing = evaluate_usage_boundary({})
    assert missing.allowed is False
    assert missing.reason_code == "missing_usage_boundary"

    blocked = evaluate_usage_boundary(
        {
            "usage_boundary": {
                "declared_use": "offline_research",
                "prohibited_use": "real_money_platform",
                "real_money": True,
            }
        }
    )
    assert blocked.allowed is False
    assert blocked.reason_code == "blocked_usage"
    assert "real_money_platform" in blocked.blocked_signals

    unauthorized = evaluate_usage_boundary(
        {
            "usage_boundary": {
                "declared_use": "authorized_environment",
                "authorized": False,
            }
        }
    )
    assert unauthorized.allowed is False
    assert "unauthorized_platform" in unauthorized.blocked_signals


def test_scope_and_usage_boundary_contracts_are_machine_readable() -> None:
    scope_contract = describe_game_scope_contract()
    usage_contract = describe_usage_boundary_contract()

    assert scope_contract["supported_game_types"] == ["nl_holdem"]
    assert {"6_max", "9_max"} == set(scope_contract["supported_table_sizes"])
    assert usage_contract["allowed_usage"] == list(ALLOWED_USAGE)
    assert usage_contract["blocked_usage"] == list(BLOCKED_USAGE)
    assert usage_contract["blocked_http_status"] == 403
    assert usage_contract["blocked_error_code"] == "USAGE_BOUNDARY_VIOLATION"
