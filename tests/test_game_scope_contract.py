from __future__ import annotations

import pytest

from poker_agent.api_contract import api_contract
from poker_agent.features import request_to_features
from poker_agent.schemas import GameScope, PredictionRequest
from poker_agent.scope_contract import build_scope_contract


def test_prediction_request_parses_explicit_game_scope_aliases() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "player_count": 9,
            "game_scope": {
                "game_variant": "NLHE",
                "game_type": "tournament",
                "table_format": "9-max",
                "small_blind": 50,
                "big_blind": 100,
                "ante": 12.5,
                "rake_percentage": 0,
                "rake_cap": 0,
                "stack_unit": "bb",
            },
        }
    )

    assert request.game_scope == GameScope(
        game_variant="nl_holdem",
        game_type="tournament",
        table_format="9_max",
        small_blind=50.0,
        big_blind=100.0,
        ante=12.5,
        rake_percentage=0.0,
        rake_cap=0.0,
        stack_unit="big_blinds",
    )


def test_game_scope_defaults_are_backward_compatible() -> None:
    request = PredictionRequest.from_dict({"position": "BTN", "player_count": 6})

    assert request.game_scope.game_variant == "nl_holdem"
    assert request.game_scope.game_type == "cash"
    assert request.game_scope.table_format == "6_max"
    assert request.game_scope.small_blind == 0.5
    assert request.game_scope.big_blind == 1.0
    assert request.game_scope.stack_unit == "chips"


def test_game_scope_rejects_invalid_blind_structure() -> None:
    with pytest.raises(ValueError, match="big_blind"):
        PredictionRequest.from_dict(
            {
                "position": "BTN",
                "game_scope": {
                    "small_blind": 2.0,
                    "big_blind": 1.0,
                },
            }
        )


def test_api_request_body_normalizes_game_scope_aliases() -> None:
    from poker_agent.service import PredictRequestBody

    body = PredictRequestBody.model_validate(
        {
            "position": "BTN",
            "game_scope": {
                "game_variant": "NLHE",
                "game_type": "cash_game",
                "table_format": "6-max",
                "small_blind": 1,
                "big_blind": 2,
                "stack_unit": "bb",
            },
        }
    )

    assert body.game_scope.game_variant == "nl_holdem"
    assert body.game_scope.game_type == "cash"
    assert body.game_scope.table_format == "6_max"
    assert body.game_scope.stack_unit == "big_blinds"


def test_game_scope_is_exposed_as_model_features() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "player_count": 9,
            "game_scope": {
                "game_type": "cash",
                "table_format": "9_max",
                "small_blind": 1,
                "big_blind": 2,
                "ante": 0.25,
                "rake_percentage": 5,
                "rake_cap": 3,
                "stack_unit": "chips",
            },
        }
    )

    features = request_to_features(request)

    assert features["scope_small_blind"] == 1.0
    assert features["scope_big_blind"] == 2.0
    assert features["scope_ante_bb"] == 0.125
    assert features["scope_rake_percentage"] == 5.0
    assert features["scope_rake_cap_bb"] == 1.5
    assert features["scope_game_variant=nl_holdem"] == 1.0
    assert features["scope_game_type=cash"] == 1.0
    assert features["scope_table_format=9_max"] == 1.0


def test_api_contract_exposes_game_scope_contract() -> None:
    contract = api_contract()
    game_scope = contract["game_scope_contract"]["contract"]

    assert contract["contract_version"] == "2026-07-13"
    assert "game_scope" in contract["prediction_request"]["request_fields"]
    assert game_scope["game_variant"]["supported_values"] == ["nl_holdem"]
    assert game_scope["game_type"]["supported_values"] == ["cash", "tournament"]
    assert game_scope["table_format"]["supported_values"] == ["6_max", "9_max"]
    assert game_scope["blind_structure"]["fields"] == ["small_blind", "big_blind", "ante"]
    assert game_scope["rake_structure"]["fields"] == ["rake_percentage", "rake_cap"]
    assert game_scope["stack_unit"]["supported_values"] == ["chips", "big_blinds"]


def test_scope_contract_includes_game_scope_contract(tmp_path) -> None:
    payload = build_scope_contract(tmp_path)

    game_scope = payload["game_scope_contract"]
    assert payload["version"] == "2026-07-13"
    assert game_scope["game_variant"]["default"] == "nl_holdem"
    assert game_scope["game_type"]["supported_values"] == ["cash", "tournament"]
    assert game_scope["table_format"]["supported_values"] == ["6_max", "9_max"]
