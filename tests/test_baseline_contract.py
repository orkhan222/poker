from __future__ import annotations

from poker_agent.baselines import (
    LLMPromptBaselinePolicy,
    RuleBaselinePolicy,
    baseline_names,
    build_baseline_policy,
    transform_examples_for_baseline,
)
from poker_agent.evaluator import evaluate_policy


def test_baseline_registry_declares_four_required_baselines() -> None:
    assert baseline_names() == ("rule", "imitation_learning", "llm", "end_to_end_policy")


def test_rule_and_llm_baselines_return_normalized_action_probabilities() -> None:
    features = sample_examples()[0][0]
    for policy in (RuleBaselinePolicy(), LLMPromptBaselinePolicy()):
        action, probabilities = policy.predict_from_features(features)
        assert action in probabilities
        assert set(probabilities) == {"fold", "check", "call", "bet", "raise", "all_in"}
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9


def test_imitation_learning_baseline_uses_public_context_features() -> None:
    transformed = transform_examples_for_baseline("imitation_learning", sample_examples())
    features, _label = transformed[0]

    assert "hole_high_rank" not in features
    assert "strength_proxy" not in features
    assert "pot_odds" in features


def test_trainable_baselines_fit_and_evaluate() -> None:
    examples = sample_examples()
    for name in ("imitation_learning", "end_to_end_policy"):
        model = build_baseline_policy(name, examples, epochs=2, learning_rate=0.05)
        metrics = evaluate_policy(model, transform_examples_for_baseline(name, examples))
        assert metrics["examples"] == float(len(examples))
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_llm_prompt_baseline_exposes_prompt_contract() -> None:
    policy = LLMPromptBaselinePolicy()
    prompt = policy.build_prompt(sample_examples()[0][0])

    assert "Choose one poker action" in prompt
    assert "pot_odds=" in prompt
    assert "spr=" in prompt


def sample_examples() -> list[tuple[dict[str, float], str]]:
    return [
        (
            {
                "bias": 1.0,
                "strength_proxy": 0.80,
                "hole_high_rank": 1.0,
                "pot_odds": 0.12,
                "spr": 9.0,
                "to_call": 2.0,
                "amount_to_call": 2.0,
                "street_aggression_ratio": 0.20,
                "street=preflop": 1.0,
                "position_group=btn": 1.0,
            },
            "raise",
        ),
        (
            {
                "bias": 1.0,
                "strength_proxy": 0.18,
                "hole_high_rank": 0.25,
                "pot_odds": 0.42,
                "spr": 4.0,
                "to_call": 8.0,
                "amount_to_call": 8.0,
                "street_aggression_ratio": 0.75,
                "street=turn": 1.0,
                "position_group=bb": 1.0,
            },
            "fold",
        ),
        (
            {
                "bias": 1.0,
                "strength_proxy": 0.38,
                "hole_high_rank": 0.70,
                "pot_odds": 0.0,
                "spr": 12.0,
                "to_call": 0.0,
                "amount_to_call": 0.0,
                "street_aggression_ratio": 0.0,
                "street=flop": 1.0,
                "position_group=co": 1.0,
            },
            "check",
        ),
        (
            {
                "bias": 1.0,
                "strength_proxy": 0.55,
                "hole_high_rank": 0.85,
                "pot_odds": 0.18,
                "spr": 7.0,
                "to_call": 3.0,
                "amount_to_call": 3.0,
                "street_aggression_ratio": 0.33,
                "street=river": 1.0,
                "position_group=sb": 1.0,
            },
            "call",
        ),
    ]
