from __future__ import annotations

from poker_agent.llm_architecture_comparison import compare_llm_decision_architectures


def test_candidate_ranker_is_selected_for_research_not_production() -> None:
    generation = {
        "provider": "transformers:qwen",
        "provisional_best_mode": "minimal_zero_shot",
        "systems": {
            "minimal_zero_shot": {
                "accuracy": 0.40,
                "macro_f1": 0.293,
                "schema_valid_rate": 0.20,
                "legal_action_rate": 1.0,
                "fallback_rate": 0.80,
                "average_latency_ms": 34300.0,
            }
        },
    }
    ranker = {
        "provider": "transformers_candidate_ranker:qwen",
        "provisional_best_mode": "full_in_context",
        "systems": {
            "full_in_context": {
                "accuracy": 0.35,
                "macro_f1": 0.298,
                "schema_valid_rate": 1.0,
                "legal_action_rate": 1.0,
                "fallback_rate": 0.0,
                "average_latency_ms": 5700.0,
            }
        },
    }

    result = compare_llm_decision_architectures(
        generation,
        ranker,
        {"status": "BASELINE_NOT_APPROVED", "failed_checks": ["schema_valid_rate"]},
        {"status": "BASELINE_NOT_APPROVED", "failed_checks": ["macro_f1"]},
    )

    assert result["recommended_architecture"] == "candidate_ranker"
    assert result["production_approved"] is False
    assert result["candidate_ranker_deltas"]["schema_valid_rate"] == 0.8
    assert result["approval_boundary"]["deployed_strategy_stack_affected"] is False
