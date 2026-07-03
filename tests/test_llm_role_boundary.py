from __future__ import annotations

import json
from pathlib import Path

from poker_agent.llm_role_boundary import build_llm_role_boundary, validate_llm_role_boundary


def test_llm_role_boundary_keeps_llm_as_controlled_layer(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "llm_decision_context.json").write_text(
        json.dumps(
            {
                "default_context_mode": "full_in_context",
                "supported_context_modes": {"full_in_context": "rules and strategy context"},
                "required_controls": ["legal action filtering", "strict JSON-only output"],
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_event_gold_eval.json").write_text(
        json.dumps(
            {
                "examples": 24,
                "systems": {
                    "strict_schema_rules": {
                        "event_type": {"accuracy": 1.0, "macro_f1": 1.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_decision_gate.json").write_text(
        json.dumps({"production_boundary": {"llm_agent_production_approved": False}}),
        encoding="utf-8",
    )
    (reports / "llm_decision_candidate_gate.json").write_text(
        json.dumps({"production_boundary": {"llm_agent_production_approved": False}}),
        encoding="utf-8",
    )
    (reports / "llm_architecture_comparison.json").write_text(
        json.dumps(
            {
                "recommended_architecture": "candidate_ranker",
                "production_approved": False,
                "approval_boundary": {"deployed_strategy_stack_affected": False},
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_decision_candidate_ranker_qwen25.json").write_text(
        json.dumps({"provider": "transformers_candidate_ranker:Qwen/Qwen2.5-1.5B-Instruct:4bit_nf4"}),
        encoding="utf-8",
    )

    payload = build_llm_role_boundary(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["term_boundary"]["status"] == "LLM_BASED_AGENT_IS_UMBRELLA_TERM"
    assert payload["term_boundary"]["must_not_imply_fully_autonomous_policy"] is True
    assert payload["current_llm_role"]["status"] == "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
    assert set(payload["role_taxonomy"]) == {
        "event_normalization",
        "decision_context",
        "candidate_ranking",
        "real_policy_agent",
    }
    assert payload["role_taxonomy"]["event_normalization"]["status"] == "CONTROLLED_COMPONENT"
    assert payload["role_taxonomy"]["event_normalization"]["can_emit_policy_action"] is False
    assert payload["role_taxonomy"]["decision_context"]["status"] == "CONTROLLED_COMPONENT"
    assert payload["role_taxonomy"]["candidate_ranking"]["status"] == "RESEARCH_BASELINE_COMPONENT"
    assert payload["role_taxonomy"]["candidate_ranking"]["production_policy_approved"] is False
    assert payload["role_taxonomy"]["real_policy_agent"]["status"] == "NOT_CURRENT_DELIVERY_SCOPE"
    assert payload["role_taxonomy"]["real_policy_agent"]["implemented"] is False
    assert payload["current_llm_role"]["event_normalization_layer"]["implemented"] is True
    assert payload["current_llm_role"]["decision_context_layer"]["implemented"] is True
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_poker_playing_llm_agent_present"] is False
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] is False
    proof_cases = {case["name"]: case for case in payload["proof_cases"]}
    assert proof_cases["base_contract_is_valid"]["observed_status"] == "PASS"
    assert proof_cases["blocks_llm_based_agent_as_autonomous_policy"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_candidate_ranking_as_deployed_policy"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_real_policy_agent_current_scope_claim"]["observed_status"] == "FAIL"


def test_llm_role_boundary_blocks_false_autonomous_llm_claim() -> None:
    payload = {
        "current_llm_role": {
            "status": "FULLY_AUTONOMOUS_LLM_AGENT",
            "event_normalization_layer": {"implemented": True},
            "decision_context_layer": {"implemented": True},
            "production_status": "PRODUCTION_POLICY",
            "llm_decision_path_production_approved": True,
        },
        "autonomous_llm_agent_boundary": {
            "status": "AUTONOMOUS_LLM_AGENT",
            "fully_autonomous_poker_playing_llm_agent_present": True,
            "fully_autonomous_llm_agent_claim_allowed": True,
            "deployed_autonomous_endpoint_is_llm": True,
            "llm_can_choose_unconstrained_actions": True,
            "llm_can_bypass_schema_validation": True,
            "production_blocker_for_current_delivery": False,
        },
        "evidence": {
            "architecture_production_approved": True,
            "deployed_strategy_stack_affected": True,
        },
    }

    invariants = validate_llm_role_boundary(payload)

    assert invariants["status"] == "FAIL"
    assert "fully_autonomous_llm_agent_claim_must_be_blocked" in invariants["violations"]
    assert "llm_decision_path_must_not_be_production_approved" in invariants["violations"]
    assert "llm_based_agent_term_must_remain_umbrella_term" in invariants["violations"]
    assert "real_policy_agent_must_remain_out_of_current_scope" in invariants["violations"]


def test_llm_role_boundary_endpoint_returns_contract() -> None:
    from poker_agent.service import llm_role_boundary_json
    from poker_agent.api_contract import api_contract

    contract = api_contract()["llm_role_boundary"]
    payload = llm_role_boundary_json()

    assert contract["endpoint"] == "/llm-role-boundary.json"
    assert contract["term_status"] == "LLM_BASED_AGENT_IS_UMBRELLA_TERM"
    assert contract["role_taxonomy"] == [
        "event_normalization",
        "decision_context",
        "candidate_ranking",
        "real_policy_agent",
    ]
    assert payload["overall_status"] == "PASS"
    assert payload["current_llm_role"]["status"] == "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
    assert payload["role_taxonomy"]["real_policy_agent"]["production_policy_approved"] is False
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] is False
