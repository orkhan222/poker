from __future__ import annotations

import pytest

from poker_agent.stack_context import assert_stack_decision_context_feature_contract
from poker_agent.stack_context import build_stack_decision_context
from poker_agent.stack_context import derive_stack_decision_context_from_events


def test_stack_decision_context_derives_pre_action_pressure() -> None:
    context = build_stack_decision_context(
        running_pot=4.0,
        highest_commit=4.0,
        hero_commit=0.0,
        decision_stack=100.0,
        to_call=4.0,
        min_raise=4.0,
    )

    assert context.pot_base == 4.0
    assert context.effective_stack == 100.0
    assert context.current_street_bet_size == 4.0
    assert context.current_street_bet_to_pot == 1.0
    assert context.call_pressure == 1.0
    assert context.raise_pressure == 0.04
    assert context.call_price_ratio == 0.04
    assert context.spr_after_call > 10.0


def test_stack_decision_context_feature_dict_blocks_target_delta_leakage() -> None:
    context = build_stack_decision_context(
        running_pot=12.0,
        highest_commit=6.0,
        hero_commit=2.0,
        decision_stack=48.0,
        to_call=4.0,
        min_raise=8.0,
    )

    features = context.as_feature_dict()

    assert features["stack_event_context_reconstructed"] == 1.0
    assert features["stack_event_target_bet_size_used_as_feature"] == 0.0
    assert features["reconstructed_pot"] == 12.0
    assert features["reconstructed_effective_stack"] == 48.0
    assert features["reconstructed_spr_after_call"] > 0.0
    assert "stack_after_event" not in features
    assert "stack_delta" not in features
    assert_stack_decision_context_feature_contract(features, context="unit test")


def test_stack_decision_context_contract_blocks_raw_stack_event_fields() -> None:
    context = build_stack_decision_context(
        running_pot=8.0,
        highest_commit=4.0,
        hero_commit=0.0,
        decision_stack=80.0,
        to_call=4.0,
        min_raise=8.0,
    )
    features = context.as_feature_dict()
    features["stack_after_event"] = 76.0

    with pytest.raises(ValueError, match="stack_after_event"):
        assert_stack_decision_context_feature_contract(features, context="negative test")


def test_stack_decision_context_contract_requires_reconstructed_features() -> None:
    features = build_stack_decision_context(
        running_pot=8.0,
        highest_commit=4.0,
        hero_commit=0.0,
        decision_stack=80.0,
        to_call=4.0,
        min_raise=8.0,
    ).as_feature_dict()
    del features["reconstructed_spr_after_call"]

    with pytest.raises(ValueError, match="reconstructed_spr_after_call"):
        assert_stack_decision_context_feature_contract(features, context="negative test")


def test_stack_decision_context_derives_from_raw_pre_action_events_only() -> None:
    context = derive_stack_decision_context_from_events(
        [
            {
                "frame_id": "10",
                "player_position": "UTG",
                "diff": "-4",
            },
            {
                "frame_id": "20",
                "player_position": "BTN",
                "diff": "-4",
            },
            {
                "frame_id": "30",
                "player_position": "BB",
                "diff": "-50",
            },
        ],
        target_frame_id=20,
        hero_position="BTN",
        starting_stack=100.0,
        to_call=4.0,
        min_raise=8.0,
    )

    features = context.as_feature_dict()

    assert context.pot_base == 4.0
    assert context.effective_stack == 100.0
    assert context.current_street_bet_size == 4.0
    assert features["reconstructed_pot"] == 4.0
    assert features["stack_event_target_bet_size_used_as_feature"] == 0.0
    assert_stack_decision_context_feature_contract(features, context="raw pre-action derivation")


def test_stack_decision_context_ignores_target_delta_magnitude() -> None:
    prior_event = {"frame_id": "10", "player_position": "UTG", "diff": "-4"}
    small_target_delta = {"frame_id": "20", "player_position": "BTN", "diff": "-4"}
    large_target_delta = {"frame_id": "20", "player_position": "BTN", "diff": "-80"}

    small_context = derive_stack_decision_context_from_events(
        [prior_event, small_target_delta],
        target_frame_id=20,
        hero_position="BTN",
        starting_stack=100.0,
        to_call=4.0,
        min_raise=8.0,
    )
    large_context = derive_stack_decision_context_from_events(
        [prior_event, large_target_delta],
        target_frame_id=20,
        hero_position="BTN",
        starting_stack=100.0,
        to_call=4.0,
        min_raise=8.0,
    )

    assert small_context.as_feature_dict() == large_context.as_feature_dict()
