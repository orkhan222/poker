from __future__ import annotations

import json
from pathlib import Path

from poker_agent.llm_role_boundary import (
    build_role_permissions_matrix,
    build_llm_role_boundary,
    validate_llm_agent_claim,
    validate_llm_production_scope_claim,
    validate_llm_role_boundary,
)


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
    acceptance = payload["controlled_layer_acceptance"]
    assert acceptance["status"] == "CONTROLLED_EVENT_CONTEXT_LAYER_APPROVED"
    assert acceptance["approved_for_current_delivery"] is True
    assert set(acceptance["approved_delivery_scope"]) == {"event_normalization", "decision_context"}
    assert set(acceptance["research_only_scope"]) == {"candidate_ranking"}
    assert "real_policy_agent" in acceptance["excluded_delivery_scope"]
    assert "fully_autonomous_poker_playing_llm_policy" in acceptance["excluded_delivery_scope"]
    assert acceptance["fully_autonomous_poker_playing_llm_policy_status"] == "FULLY_AUTONOMOUS_LLM_POLICY_NOT_APPROVED"
    assert acceptance["fully_autonomous_poker_playing_llm_policy_approved"] is False
    assert acceptance["fully_autonomous_policy_claim_allowed"] is False
    assert acceptance["production_blocker_for_current_delivery"] is False
    assert acceptance["future_policy_agent_requires_separate_approval"] is True
    assert payload["term_boundary"]["status"] == "LLM_BASED_AGENT_IS_UMBRELLA_TERM"
    assert payload["term_boundary"]["must_not_imply_fully_autonomous_policy"] is True
    assert payload["term_boundary"]["ambiguous_unqualified_usage_allowed"] is False
    assert payload["recommended_production_architecture"]["status"] == "SCHEMA_ROUTED_HYBRID_CONTROLLED_LAYER"
    assert payload["recommended_production_architecture"]["priority"] == "CONTROLLED_CONTEXT_EVENT_LAYER_FIRST"
    assert payload["recommended_production_architecture"]["approved_for_current_delivery"] is True
    assert payload["recommended_production_architecture"]["production_policy_claim_allowed"] is False
    assert payload["recommended_production_architecture"]["fully_autonomous_llm_agent_claim_allowed"] is False
    assert payload["recommended_production_architecture"]["final_policy_owner"] == "deployed_routed_policy_stack"
    assert (
        payload["recommended_production_architecture"]["llm_position"]
        == "controlled_fallback_before_schema_validation_for_event_context_layer"
    )
    assert (
        "LLM fallback for ambiguous event/context cases"
        in payload["recommended_production_architecture"]["pipeline"]
    )
    assert payload["scope_disambiguation_contract"]["status"] == "EXPLICITLY_DISAMBIGUATED"
    assert payload["scope_disambiguation_contract"]["ambiguous_llm_agent_term_allowed"] is False
    assert payload["scope_disambiguation_contract"]["role_type_mapping"] == {
        "event_normalization": "EVENT_NORMALIZER",
        "decision_context": "DECISION_CONTEXT_AGENT",
        "candidate_ranking": "CANDIDATE_RANKER",
        "real_policy_agent": "POLICY_AGENT",
    }
    assert payload["role_permissions_matrix"] == build_role_permissions_matrix()
    assert payload["role_permissions_matrix"]["event_normalization"]["may_emit_deployed_policy_action"] is False
    assert payload["role_permissions_matrix"]["decision_context"]["may_build_decision_context"] is True
    assert payload["role_permissions_matrix"]["candidate_ranking"]["may_rank_candidates"] is True
    assert payload["role_permissions_matrix"]["candidate_ranking"]["may_emit_deployed_policy_action"] is False
    assert payload["role_permissions_matrix"]["real_policy_agent"]["current_delivery_scope"] is False
    assert payload["role_permissions_matrix"]["real_policy_agent"]["production_policy_approved"] is False
    assert payload["current_llm_role"]["status"] == "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
    assert set(payload["role_taxonomy"]) == {
        "event_normalization",
        "decision_context",
        "candidate_ranking",
        "real_policy_agent",
    }
    assert payload["role_taxonomy"]["event_normalization"]["status"] == "CONTROLLED_COMPONENT"
    assert payload["role_taxonomy"]["event_normalization"]["role_type"] == "EVENT_NORMALIZER"
    assert payload["role_taxonomy"]["event_normalization"]["can_emit_policy_action"] is False
    assert payload["role_taxonomy"]["event_normalization"]["may_select_final_poker_action"] is False
    assert payload["role_taxonomy"]["decision_context"]["status"] == "CONTROLLED_COMPONENT"
    assert payload["role_taxonomy"]["decision_context"]["role_type"] == "DECISION_CONTEXT_AGENT"
    assert payload["role_taxonomy"]["decision_context"]["may_select_final_poker_action"] is False
    assert payload["role_taxonomy"]["candidate_ranking"]["status"] == "RESEARCH_BASELINE_COMPONENT"
    assert payload["role_taxonomy"]["candidate_ranking"]["role_type"] == "CANDIDATE_RANKER"
    assert payload["role_taxonomy"]["candidate_ranking"]["production_policy_approved"] is False
    assert payload["role_taxonomy"]["candidate_ranking"]["may_select_final_poker_action"] is False
    assert payload["role_taxonomy"]["real_policy_agent"]["status"] == "NOT_CURRENT_DELIVERY_SCOPE"
    assert payload["role_taxonomy"]["real_policy_agent"]["role_type"] == "POLICY_AGENT"
    assert payload["role_taxonomy"]["real_policy_agent"]["implemented"] is False
    assert payload["role_taxonomy"]["real_policy_agent"]["may_select_final_poker_action"] is True
    assert payload["current_llm_role"]["event_normalization_layer"]["implemented"] is True
    assert payload["current_llm_role"]["decision_context_layer"]["implemented"] is True
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_poker_playing_llm_agent_present"] is False
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] is False
    proof_cases = {case["name"]: case for case in payload["proof_cases"]}
    assert proof_cases["base_contract_is_valid"]["observed_status"] == "PASS"
    assert proof_cases["blocks_llm_based_agent_as_autonomous_policy"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_ambiguous_llm_agent_scope"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_autonomous_policy_under_controlled_layer_acceptance"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_candidate_ranking_as_deployed_policy"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_real_policy_agent_current_scope_claim"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_autonomous_architecture_as_recommended_production_path"]["observed_status"] == "FAIL"
    assert proof_cases["blocks_candidate_ranker_permission_escalation"]["observed_status"] == "FAIL"
    claim_cases = {case["name"]: case for case in payload["claim_validation_examples"]}
    assert claim_cases["blocks_unqualified_llm_based_agent_production_claim"]["observed_status"] == "FAIL"
    assert claim_cases["blocks_event_normalizer_as_policy_agent"]["observed_status"] == "FAIL"
    assert claim_cases["allows_decision_context_research_claim"]["observed_status"] == "PASS"
    assert claim_cases["blocks_candidate_ranker_as_deployed_policy"]["observed_status"] == "FAIL"
    assert claim_cases["blocks_real_policy_agent_current_delivery_claim"]["observed_status"] == "FAIL"
    production_scope_cases = {case["name"]: case for case in payload["production_scope_claim_examples"]}
    assert (
        production_scope_cases["allows_controlled_event_context_layer_production_claim"]["observed_status"]
        == "PASS"
    )
    assert production_scope_cases["blocks_autonomous_llm_policy_production_claim"]["observed_status"] == "FAIL"
    assert production_scope_cases["blocks_unqualified_llm_policy_claim_text"]["observed_status"] == "FAIL"


def test_llm_agent_claim_validator_requires_explicit_scope(tmp_path: Path) -> None:
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
                "systems": {
                    "strict_schema_rules": {
                        "event_type": {"accuracy": 1.0, "macro_f1": 1.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_architecture_comparison.json").write_text(
        json.dumps({"recommended_architecture": "candidate_ranker"}),
        encoding="utf-8",
    )
    (reports / "llm_decision_candidate_ranker_qwen25.json").write_text(
        json.dumps({"provider": "qwen25"}),
        encoding="utf-8",
    )
    boundary = build_llm_role_boundary(tmp_path)

    ambiguous = validate_llm_agent_claim(
        {
            "uses_llm_based_agent_term": True,
            "production_policy_claim": True,
            "autonomous_policy_claim": False,
            "final_poker_action_claim": False,
        },
        boundary,
    )
    event_policy = validate_llm_agent_claim(
        {
            "uses_llm_based_agent_term": True,
            "role": "event_normalization",
            "production_policy_claim": True,
            "autonomous_policy_claim": False,
            "final_poker_action_claim": True,
        },
        boundary,
    )
    decision_context = validate_llm_agent_claim(
        {
            "uses_llm_based_agent_term": True,
            "role": "decision_context",
            "production_policy_claim": False,
            "autonomous_policy_claim": False,
            "final_poker_action_claim": False,
        },
        boundary,
    )

    assert ambiguous["status"] == "FAIL"
    assert "llm_based_agent_claim_requires_explicit_role" in ambiguous["violations"]
    assert event_policy["status"] == "FAIL"
    assert "role_not_production_policy_approved:event_normalization" in event_policy["violations"]
    assert "role_cannot_select_final_poker_action:event_normalization" in event_policy["violations"]
    assert decision_context["status"] == "PASS"


def test_llm_production_scope_claim_guard_blocks_autonomous_policy_wording(tmp_path: Path) -> None:
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
                "systems": {
                    "strict_schema_rules": {
                        "event_type": {"accuracy": 1.0, "macro_f1": 1.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    boundary = build_llm_role_boundary(tmp_path)

    controlled_claim = validate_llm_production_scope_claim(
        {
            "role": "controlled_event_context_layer",
            "production_claim": True,
            "controlled_event_context_layer_claim": True,
            "autonomous_poker_policy_claim": False,
            "policy_agent_claim": False,
            "final_action_policy_claim": False,
            "claim_text": "The LLM is production-approved as a controlled event/context layer.",
        },
        boundary,
    )
    autonomous_claim = validate_llm_production_scope_claim(
        {
            "role": "real_policy_agent",
            "production_claim": True,
            "controlled_event_context_layer_claim": False,
            "autonomous_poker_policy_claim": True,
            "policy_agent_claim": True,
            "final_action_policy_claim": True,
            "claim_text": "The LLM is a production autonomous poker policy agent.",
        },
        boundary,
    )

    assert controlled_claim["status"] == "PASS"
    assert controlled_claim["approved_production_scope"] == "controlled_event_context_layer"
    assert controlled_claim["autonomous_policy_claim_allowed"] is False
    assert autonomous_claim["status"] == "FAIL"
    assert "autonomous_poker_playing_llm_policy_claim_must_be_blocked" in autonomous_claim["violations"]
    assert "llm_policy_agent_claim_must_be_blocked_for_current_delivery" in autonomous_claim["violations"]
    assert "llm_final_action_policy_claim_must_be_blocked_for_current_delivery" in autonomous_claim["violations"]


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
    assert "controlled_llm_layer_must_be_explicitly_approved" in invariants["violations"]
    assert "llm_based_agent_term_must_remain_umbrella_term" in invariants["violations"]
    assert "llm_scope_must_be_explicitly_disambiguated" in invariants["violations"]
    assert "real_policy_agent_must_remain_out_of_current_scope" in invariants["violations"]
    assert "llm_recommended_architecture_must_remain_schema_routed_hybrid_controlled_layer" in invariants["violations"]


def test_llm_role_boundary_endpoint_returns_contract() -> None:
    from poker_agent.api_contract import api_contract
    from poker_agent.service import llm_production_scope_claim_json, llm_role_boundary_json

    contract = api_contract()["llm_role_boundary"]
    production_scope_contract = api_contract()["llm_production_scope_claim"]
    payload = llm_role_boundary_json()
    production_scope_payload = llm_production_scope_claim_json()

    assert contract["endpoint"] == "/llm-role-boundary.json"
    assert contract["term_status"] == "LLM_BASED_AGENT_IS_UMBRELLA_TERM"
    assert contract["controlled_layer_acceptance_status"] == "CONTROLLED_EVENT_CONTEXT_LAYER_APPROVED"
    assert set(contract["approved_delivery_scope"]) == {"event_normalization", "decision_context"}
    assert set(contract["research_only_scope"]) == {"candidate_ranking"}
    assert "real_policy_agent" in contract["excluded_delivery_scope"]
    assert "fully_autonomous_poker_playing_llm_policy" in contract["excluded_delivery_scope"]
    assert contract["fully_autonomous_poker_playing_llm_policy_status"] == "FULLY_AUTONOMOUS_LLM_POLICY_NOT_APPROVED"
    assert contract["fully_autonomous_poker_playing_llm_policy_approved"] is False
    assert contract["fully_autonomous_policy_claim_allowed"] is False
    assert contract["current_delivery_approval"] == "controlled_event_context_layer_only"
    assert contract["recommended_production_architecture"] == "SCHEMA_ROUTED_HYBRID_CONTROLLED_LAYER"
    assert contract["architecture_priority"] == "CONTROLLED_CONTEXT_EVENT_LAYER_FIRST"
    assert contract["llm_position"] == "controlled_fallback_before_schema_validation_for_event_context_layer"
    assert contract["final_policy_owner"] == "deployed_routed_policy_stack"
    assert contract["not_recommended_first"] == "fully_autonomous_poker_playing_llm_policy"
    assert contract["ambiguous_llm_agent_term_allowed"] is False
    assert contract["unqualified_production_claim_allowed"] is False
    assert contract["claim_validator"] == "poker_agent.llm_role_boundary.validate_llm_agent_claim"
    assert (
        contract["production_scope_claim_validator"]
        == "poker_agent.llm_role_boundary.validate_llm_production_scope_claim"
    )
    assert contract["role_taxonomy"] == [
        "event_normalization",
        "decision_context",
        "candidate_ranking",
        "real_policy_agent",
    ]
    assert contract["role_types"]["event_normalization"] == "EVENT_NORMALIZER"
    assert contract["role_types"]["decision_context"] == "DECISION_CONTEXT_AGENT"
    assert contract["role_types"]["candidate_ranking"] == "CANDIDATE_RANKER"
    assert contract["role_types"]["real_policy_agent"] == "POLICY_AGENT"
    assert contract["role_permissions_matrix"]["event_normalization"]["may_emit_deployed_policy_action"] is False
    assert contract["role_permissions_matrix"]["decision_context"]["may_build_decision_context"] is True
    assert contract["role_permissions_matrix"]["candidate_ranking"]["may_rank_candidates"] is True
    assert contract["role_permissions_matrix"]["candidate_ranking"]["production_policy_approved"] is False
    assert contract["role_permissions_matrix"]["real_policy_agent"]["may_select_final_poker_action"] is True
    assert contract["role_permissions_matrix"]["real_policy_agent"]["production_policy_approved"] is False
    assert production_scope_contract["endpoint"] == "/llm-production-scope-claim.json"
    assert production_scope_contract["approved_production_scope"] == "controlled_event_context_layer"
    assert production_scope_contract["autonomous_policy_claim_allowed"] is False
    assert production_scope_contract["policy_agent_claim_allowed"] is False
    assert production_scope_contract["final_action_policy_claim_allowed"] is False
    assert (
        production_scope_contract["validator"]
        == "poker_agent.llm_role_boundary.validate_llm_production_scope_claim"
    )
    assert payload["overall_status"] == "PASS"
    assert payload["controlled_layer_acceptance"]["status"] == "CONTROLLED_EVENT_CONTEXT_LAYER_APPROVED"
    assert payload["recommended_production_architecture"]["status"] == "SCHEMA_ROUTED_HYBRID_CONTROLLED_LAYER"
    assert payload["controlled_layer_acceptance"]["fully_autonomous_policy_claim_allowed"] is False
    assert payload["current_llm_role"]["status"] == "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
    assert payload["role_taxonomy"]["real_policy_agent"]["production_policy_approved"] is False
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] is False
    assert production_scope_payload["status"] == "PASS"
    assert production_scope_payload["approved_production_scope"] == "controlled_event_context_layer"
    assert production_scope_payload["autonomous_policy_claim_allowed"] is False
    assert (
        production_scope_payload["validator"]
        == "poker_agent.llm_role_boundary.validate_llm_production_scope_claim"
    )
