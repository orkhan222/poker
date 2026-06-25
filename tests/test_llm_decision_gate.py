from __future__ import annotations

from poker_agent.llm_decision_gate import evaluate_llm_decision_report


def test_weak_reconstructed_baseline_is_not_approved() -> None:
    benchmark = {
        "provider": "transformers:Qwen/Qwen2.5-1.5B-Instruct:4bit_nf4",
        "dataset_kind": "reconstructed_human_holdout",
        "quality_claim_allowed": False,
        "provisional_best_mode": "minimal_zero_shot",
        "systems": {
            "minimal_zero_shot": {
                "examples": 20,
                "macro_f1": 0.29,
                "schema_valid_rate": 0.20,
                "legal_action_rate": 1.0,
                "average_latency_ms": 34000.0,
            }
        },
    }
    holdout = {"status": "PASS", "class_distribution": {"fold": 4, "check": 4, "call": 4, "bet": 4, "raise": 4}}

    result = evaluate_llm_decision_report(
        benchmark,
        holdout,
        min_examples=20,
        min_macro_f1=0.40,
        min_schema_valid_rate=0.95,
        min_legal_action_rate=0.99,
        max_average_latency_ms=5000.0,
    )

    assert result["status"] == "BASELINE_NOT_APPROVED"
    assert result["production_boundary"]["deployed_strategy_stack_affected"] is False
    assert "macro_f1" in result["failed_checks"]
    assert "manual_reviewed_labels" in result["failed_checks"]
