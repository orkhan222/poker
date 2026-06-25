from __future__ import annotations

import json

from poker_agent.llm_decision_benchmark import (
    DecisionExample,
    Generation,
    benchmark_context_modes,
    inspect_decision_output,
    _candidate_probabilities,
)
from poker_agent.llm_decision_context import DecisionPrompt
from poker_agent.schemas import PredictionRequest


class FixedProvider:
    name = "fixed"
    quality_claim_allowed = True

    def generate(self, prompt: DecisionPrompt, request: PredictionRequest) -> Generation:
        del prompt
        action = "call" if request.to_call > 0 else "check"
        probabilities = {name: 0.0 for name in ("fold", "check", "call", "bet", "raise")}
        probabilities[action] = 1.0
        return Generation(
            text=json.dumps(
                {
                    "action": action,
                    "probabilities": probabilities,
                    "confidence": 1.0,
                    "bet_size": request.to_call if action == "call" else 0.0,
                    "reason_code": "pot_odds",
                }
            ),
            latency_ms=5.0,
            prompt_tokens=100,
            completion_tokens=30,
        )


def test_inspection_rejects_illegal_action() -> None:
    request = PredictionRequest(position="BTN", to_call=2.0, pot=5.0, stack=50.0)
    raw = json.dumps(
        {
            "action": "check",
            "probabilities": {"fold": 0.0, "check": 1.0, "call": 0.0, "bet": 0.0, "raise": 0.0},
            "confidence": 1.0,
            "bet_size": 0.0,
            "reason_code": "uncertain",
        }
    )

    inspection = inspect_decision_output(raw, request)

    assert inspection.schema_valid is True
    assert inspection.legal_action is False


def test_benchmark_reports_context_metrics_and_claim_boundary() -> None:
    examples = [
        DecisionExample(
            example_id="facing-bet",
            request=PredictionRequest(position="BTN", to_call=2.0, pot=5.0, stack=50.0),
            expected_action="call",
        ),
        DecisionExample(
            example_id="free-option",
            request=PredictionRequest(position="BB", to_call=0.0, pot=5.0, stack=50.0),
            expected_action="check",
        ),
    ]

    result = benchmark_context_modes(
        examples,
        FixedProvider(),
        context_modes=["minimal_zero_shot", "full_in_context"],
        dataset_kind="smoke",
    )

    assert result["quality_claim_allowed"] is False
    assert result["best_mode"] is None
    assert result["provisional_best_mode"] is None
    assert result["systems"]["full_in_context"]["accuracy"] == 1.0
    assert result["systems"]["full_in_context"]["schema_valid_rate"] == 1.0
    assert result["systems"]["full_in_context"]["legal_action_rate"] == 1.0


def test_reconstructed_human_holdout_allows_only_provisional_comparison() -> None:
    examples = [
        DecisionExample(
            example_id="facing-bet",
            request=PredictionRequest(position="BTN", to_call=2.0, pot=5.0, stack=50.0),
            expected_action="call",
        )
    ]

    result = benchmark_context_modes(
        examples,
        FixedProvider(),
        context_modes=["minimal_zero_shot", "full_in_context"],
        dataset_kind="reconstructed_human_holdout",
    )

    assert result["comparison_allowed"] is True
    assert result["quality_claim_allowed"] is False
    assert result["best_mode"] is None
    assert result["provisional_best_mode"] in {"minimal_zero_shot", "full_in_context"}


def test_candidate_probabilities_are_normalized_and_loss_ordered() -> None:
    probabilities = _candidate_probabilities(
        {"fold": 2.0, "call": 0.5, "raise": 1.0},
        temperature=1.0,
    )

    assert max(probabilities, key=probabilities.get) == "call"
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12
    assert probabilities["call"] > probabilities["raise"] > probabilities["fold"]
