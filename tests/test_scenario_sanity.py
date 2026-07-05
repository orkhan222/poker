from __future__ import annotations

from poker_agent.scenario_sanity import (
    apply_critical_spot_guardrail,
    build_scenario_definitions,
    evaluate_scenario_sanity,
    validate_scenario_sanity,
)
from poker_agent.schemas import PredictionRequest, PredictionResponse


class GuardedWeakPolicy:
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        base = {
            "fold": 0.45,
            "call": 0.25,
            "check": 0.10,
            "bet": 0.10,
            "raise": 0.10,
        }
        action, probabilities, guardrails = apply_critical_spot_guardrail(request, base)
        return PredictionResponse(
            action=action,
            probabilities=probabilities,
            confidence=max(probabilities.values()),
            model_status="guarded_test_policy",
            strategy_guardrails=guardrails,
        )


def _request(scenario_id: str) -> PredictionRequest:
    scenarios = {scenario.scenario_id: scenario for scenario in build_scenario_definitions()}
    return PredictionRequest.from_dict(scenarios[scenario_id].payload)


def test_premium_pair_guardrail_forces_aggressive_preflop_response() -> None:
    action, probabilities, guardrails = apply_critical_spot_guardrail(
        _request("pocket_aces_preflop_bb_facing_raise"),
        {"fold": 0.70, "call": 0.20, "check": 0.04, "bet": 0.02, "raise": 0.04},
    )

    assert action == "raise"
    assert probabilities["fold"] <= 0.05
    assert probabilities["raise"] + probabilities["bet"] >= 0.55
    assert guardrails == ["premium_pair_preflop_3bet"]


def test_trash_hand_guardrail_forces_preflop_fold() -> None:
    action, probabilities, guardrails = apply_critical_spot_guardrail(
        _request("trash_72o_preflop_sb_facing_raise"),
        {"fold": 0.20, "call": 0.30, "check": 0.10, "bet": 0.20, "raise": 0.20},
    )

    assert action == "fold"
    assert probabilities["fold"] >= 0.60
    assert probabilities["raise"] + probabilities["bet"] <= 0.25
    assert guardrails == ["trash_hand_preflop_fold"]


def test_nut_flush_draw_guardrail_continues_against_flop_bet() -> None:
    action, probabilities, guardrails = apply_critical_spot_guardrail(
        _request("nut_flush_draw_flop_facing_bet"),
        {"fold": 0.65, "call": 0.15, "check": 0.05, "bet": 0.08, "raise": 0.07},
    )

    assert action in {"call", "raise", "bet"}
    assert probabilities["fold"] <= 0.15
    assert probabilities["call"] + probabilities["bet"] + probabilities["raise"] >= 0.55
    assert guardrails == ["nut_flush_draw_continue"]


def test_missed_river_guardrail_folds_to_large_bet() -> None:
    action, probabilities, guardrails = apply_critical_spot_guardrail(
        _request("missed_river_facing_large_bet"),
        {"fold": 0.25, "call": 0.35, "check": 0.05, "bet": 0.20, "raise": 0.15},
    )

    assert action == "fold"
    assert probabilities["fold"] >= 0.60
    assert probabilities["call"] + probabilities["bet"] + probabilities["raise"] <= 0.30
    assert guardrails == ["missed_river_fold"]


def test_scenario_sanity_report_passes_without_granting_full_strategy_claim() -> None:
    payload = evaluate_scenario_sanity(GuardedWeakPolicy())

    assert payload["overall_status"] == "PASS"
    assert payload["passed_scenarios"] == 4
    assert payload["problem_statement"]["implemented_control"] == "critical_spot_guardrail"
    assert payload["problem_statement"]["control_scope"] == "narrow_high_confidence_holdem_invariants"
    assert payload["boundary"]["evidence_level"] == "targeted_regression_gate"
    assert payload["boundary"]["claim_allowed"] == "critical_spot_sanity_passed"
    assert payload["boundary"]["full_production_strategy_proof"] is False
    assert payload["boundary"]["final_strategy_quality_claim_allowed"] is False
    assert payload["boundary"]["current_delivery_blocker"] is False
    assert "full_production_strategy_quality" in payload["boundary"]["claim_not_allowed"]
    assert "EV and win-rate in self-play" in payload["boundary"]["required_for_final_strategy_claim"]
    assert validate_scenario_sanity(payload) == []


def test_scenario_sanity_endpoint_is_hidden_from_public_openapi() -> None:
    from poker_agent.service import app

    assert "/scenario-sanity.json" not in app.openapi()["paths"]
