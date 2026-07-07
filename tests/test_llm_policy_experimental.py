from __future__ import annotations

import json

from poker_agent.llm_policy_experimental import (
    EXPERIMENTAL_POLICY_STATUS,
    ExperimentalLLMPolicyAdapter,
    StaticLLMPolicyBackend,
    build_experimental_llm_policy_contract,
    validate_experimental_llm_policy_contract,
)
from poker_agent.schemas import PredictionRequest


def _request_facing_raise() -> PredictionRequest:
    return PredictionRequest(
        position="BTN",
        street="preflop",
        hole_cards=["Ah", "As"],
        board_cards=[],
        pot=6.5,
        to_call=3.0,
        stack=100.0,
        min_raise=6.0,
        player_count=6,
        betting_history=[
            {"player_position": "UTG", "action": "raise", "amount": 3.0, "street": "preflop"},
            {"player_position": "CO", "action": "call", "amount": 3.0, "street": "preflop"},
        ],
    )


def test_experimental_llm_policy_accepts_valid_guarded_output() -> None:
    backend = StaticLLMPolicyBackend(
        json.dumps(
            {
                "action": "raise",
                "probabilities": {
                    "fold": 0.02,
                    "check": 0.0,
                    "call": 0.18,
                    "bet": 0.0,
                    "raise": 0.8,
                },
                "confidence": 0.8,
                "bet_size": 9.0,
                "reason_code": "hand_strength",
            }
        )
    )
    decision = ExperimentalLLMPolicyAdapter(backend, min_confidence=0.35).decide(_request_facing_raise())
    payload = decision.to_dict()

    assert payload["llm_policy_status"] == EXPERIMENTAL_POLICY_STATUS
    assert payload["action"] == "raise"
    assert payload["fallback_used"] is False
    assert payload["production_policy_approved"] is False
    assert payload["autonomous_policy_claim_allowed"] is False
    assert set(payload["legal_actions"]) == {"fold", "call", "raise"}
    assert abs(sum(payload["probabilities"].values()) - 1.0) <= 1e-9
    assert payload["probabilities"]["check"] == 0.0
    assert payload["probabilities"]["bet"] == 0.0


def test_experimental_llm_policy_falls_back_on_illegal_low_confidence_output() -> None:
    backend = StaticLLMPolicyBackend(
        json.dumps(
            {
                "action": "check",
                "probabilities": {
                    "fold": 0.1,
                    "check": 0.6,
                    "call": 0.1,
                    "bet": 0.1,
                    "raise": 0.1,
                },
                "confidence": 0.2,
                "bet_size": 0,
                "reason_code": "uncertain",
            }
        )
    )
    decision = ExperimentalLLMPolicyAdapter(backend, min_confidence=0.35).decide(_request_facing_raise())
    payload = decision.to_dict()

    assert payload["fallback_used"] is True
    assert payload["action"] in {"fold", "call", "raise"}
    assert payload["production_policy_approved"] is False
    assert payload["autonomous_policy_claim_allowed"] is False
    assert payload["probabilities"]["check"] == 0.0
    assert payload["probabilities"]["bet"] == 0.0
    assert payload["validation_warnings"]


def test_experimental_llm_policy_contract_blocks_false_production_claims() -> None:
    payload = build_experimental_llm_policy_contract()

    assert payload["overall_status"] == "PASS"
    assert payload["production_policy_approved"] is False
    assert payload["autonomous_policy_claim_allowed"] is False
    assert payload["served_by_predict_endpoint"] is False
    assert payload["deployed_strategy_stack_affected"] is False

    mutated = dict(payload)
    mutated["production_policy_approved"] = True
    assert validate_experimental_llm_policy_contract(mutated)["status"] == "FAIL"

    mutated = dict(payload)
    mutated["served_by_predict_endpoint"] = True
    assert validate_experimental_llm_policy_contract(mutated)["status"] == "FAIL"
