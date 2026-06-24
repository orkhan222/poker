from __future__ import annotations

from poker_agent.action_planning import build_action_plan
from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


def test_explicit_opponent_timing_is_exposed_as_features() -> None:
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "turn",
            "pot": 20,
            "stack": 80,
            "timing_context": {
                "opponent_wait_before_turn_ms": 1800,
                "opponent_wait_after_hero_action_ms": 2400,
            },
        }
    )

    features = request_to_features(request)

    assert features["opponent_wait_before_turn_seconds"] == 1.8
    assert features["opponent_wait_after_hero_action_seconds"] == 2.4
    assert features["explicit_timing_context_missing"] == 0.0


def test_betting_history_timing_uses_opponents_only() -> None:
    request = PredictionRequest(
        position="BTN",
        betting_history=[
            {"player_position": "BTN", "action": "raise", "wait_time_ms": 900},
            {"player_position": "BB", "action": "call", "wait_time_ms": 2100},
        ],
    )

    features = request_to_features(request)

    assert features["hist_opponent_timing_count"] == 1.0
    assert features["hist_opponent_wait_mean_seconds"] == 2.1
    assert features["hist_wait_after_hero_action_seconds"] == 2.1


def test_action_plan_calibrates_output_to_observed_table_tempo() -> None:
    request = PredictionRequest(
        position="BTN",
        street="river",
        pot=30,
        to_call=10,
        stack=100,
        opponent_wait_before_turn_ms=2200,
        betting_history=[
            {"player_position": "BB", "action": "bet", "wait_time_ms": 1800},
        ],
    )

    plan = build_action_plan(request, action="call", confidence=0.65, processing_time_ms=50)

    assert plan.timing_method == "table_tempo_calibrated"
    assert plan.wait_time_ms >= 75
    assert plan.wait_time_ms <= 3200
