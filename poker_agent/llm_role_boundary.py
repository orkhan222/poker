from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LLM_ROLE_BOUNDARY_VERSION = "2026-06-28"
CONTROLLED_LAYER = "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
NOT_AUTONOMOUS_LLM_AGENT = "NOT_FULLY_AUTONOMOUS_POKER_PLAYING_LLM_AGENT"
RESEARCH_BASELINE = "RESEARCH_BASELINE_NOT_PRODUCTION_POLICY"
LLM_BASED_AGENT_TERM = "LLM_BASED_AGENT_IS_UMBRELLA_TERM"
CONTROLLED_LAYER_APPROVED = "CONTROLLED_EVENT_CONTEXT_LAYER_APPROVED"
AUTONOMOUS_LLM_POLICY_NOT_APPROVED = "FULLY_AUTONOMOUS_LLM_POLICY_NOT_APPROVED"
RECOMMENDED_PRODUCTION_ARCHITECTURE = "SCHEMA_ROUTED_HYBRID_CONTROLLED_LAYER"
CONTROLLED_CONTEXT_EVENT_LAYER_FIRST = "CONTROLLED_CONTEXT_EVENT_LAYER_FIRST"
CONTROLLED_COMPONENT = "CONTROLLED_COMPONENT"
RESEARCH_COMPONENT = "RESEARCH_BASELINE_COMPONENT"
NOT_CURRENT_SCOPE = "NOT_CURRENT_DELIVERY_SCOPE"
EVENT_NORMALIZER_ROLE = "EVENT_NORMALIZER"
DECISION_CONTEXT_ROLE = "DECISION_CONTEXT_AGENT"
CANDIDATE_RANKER_ROLE = "CANDIDATE_RANKER"
POLICY_AGENT_ROLE = "POLICY_AGENT"


def validate_llm_agent_claim(claim: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    """Validate whether an external LLM-agent claim is allowed by the delivery boundary."""

    violations: list[str] = []
    taxonomy = boundary.get("role_taxonomy") or {}
    role_name = claim.get("role")
    uses_umbrella_term = bool(claim.get("uses_llm_based_agent_term", True))
    production_policy_claim = bool(claim.get("production_policy_claim"))
    autonomous_policy_claim = bool(claim.get("autonomous_policy_claim"))
    final_action_claim = bool(claim.get("final_poker_action_claim"))

    if uses_umbrella_term and not role_name:
        violations.append("llm_based_agent_claim_requires_explicit_role")
    if role_name and role_name not in taxonomy:
        violations.append(f"unknown_llm_role:{role_name}")

    role_payload = (taxonomy.get(role_name) if role_name else {}) or {}
    role_type = role_payload.get("role_type")
    if role_payload:
        if production_policy_claim and role_payload.get("production_policy_approved") is not True:
            violations.append(f"role_not_production_policy_approved:{role_name}")
        if final_action_claim and role_payload.get("may_select_final_poker_action") is not True:
            violations.append(f"role_cannot_select_final_poker_action:{role_name}")
        if autonomous_policy_claim and role_name != "real_policy_agent":
            violations.append(f"role_is_not_autonomous_policy_agent:{role_name}")
        if role_name == "real_policy_agent":
            if role_payload.get("implemented") is not True:
                violations.append("real_policy_agent_not_implemented")
            if role_payload.get("status") != "CURRENT_DELIVERY_SCOPE":
                violations.append("real_policy_agent_not_current_delivery_scope")
            if role_payload.get("requires_separate_stakeholder_approval") is not True:
                violations.append("real_policy_agent_missing_stakeholder_approval_gate")

    return {
        "status": "FAIL" if violations else "PASS",
        "claim": claim,
        "resolved_role": role_name,
        "resolved_role_type": role_type,
        "violations": violations,
    }


def validate_llm_production_scope_claim(claim: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    """Validate production-facing wording for the delivered LLM component."""

    violations: list[str] = []
    acceptance = boundary.get("controlled_layer_acceptance") or {}
    approved_scope = set(acceptance.get("approved_delivery_scope") or [])
    role_name = claim.get("role")
    claim_text = str(claim.get("claim_text") or "")
    normalized_text = claim_text.lower()
    production_claim = bool(claim.get("production_claim"))
    controlled_event_context_claim = bool(claim.get("controlled_event_context_layer_claim"))
    autonomous_policy_claim = bool(claim.get("autonomous_poker_policy_claim"))
    policy_agent_claim = bool(claim.get("policy_agent_claim"))
    final_action_policy_claim = bool(claim.get("final_action_policy_claim"))
    approved_controlled_roles = {
        "controlled_event_context_layer",
        "event_normalization",
        "decision_context",
    }

    if production_claim and role_name not in approved_controlled_roles:
        violations.append("llm_production_claim_must_be_controlled_event_context_layer")
    if controlled_event_context_claim and not {"event_normalization", "decision_context"}.issubset(
        approved_scope
    ):
        violations.append("controlled_event_context_layer_must_be_delivery_approved")
    if production_claim and controlled_event_context_claim is not True:
        violations.append("llm_production_claim_must_explicitly_state_controlled_event_context_layer")
    if production_claim and "controlled" not in normalized_text:
        violations.append("production_claim_text_must_qualify_llm_as_controlled_layer")
    if autonomous_policy_claim:
        violations.append("autonomous_poker_playing_llm_policy_claim_must_be_blocked")
    if policy_agent_claim:
        violations.append("llm_policy_agent_claim_must_be_blocked_for_current_delivery")
    if final_action_policy_claim:
        violations.append("llm_final_action_policy_claim_must_be_blocked_for_current_delivery")
    if "autonomous" in normalized_text and "llm" in normalized_text and "policy" in normalized_text:
        violations.append("claim_text_must_not_present_llm_as_autonomous_policy")
    if "fully autonomous" in normalized_text and "poker" in normalized_text:
        violations.append("claim_text_must_not_present_llm_as_fully_autonomous_poker_agent")

    return {
        "status": "FAIL" if violations else "PASS",
        "claim": claim,
        "approved_production_scope": "controlled_event_context_layer",
        "autonomous_policy_claim_allowed": False,
        "violations": violations,
    }


def build_role_permissions_matrix() -> dict[str, dict[str, Any]]:
    return {
        "event_normalization": {
            "role_type": EVENT_NORMALIZER_ROLE,
            "current_delivery_scope": True,
            "may_normalize_events": True,
            "may_build_decision_context": False,
            "may_rank_candidates": False,
            "may_select_final_poker_action": False,
            "may_emit_deployed_policy_action": False,
            "production_policy_approved": False,
        },
        "decision_context": {
            "role_type": DECISION_CONTEXT_ROLE,
            "current_delivery_scope": True,
            "may_normalize_events": False,
            "may_build_decision_context": True,
            "may_rank_candidates": False,
            "may_select_final_poker_action": False,
            "may_emit_deployed_policy_action": False,
            "production_policy_approved": False,
        },
        "candidate_ranking": {
            "role_type": CANDIDATE_RANKER_ROLE,
            "current_delivery_scope": "research_baseline_only",
            "may_normalize_events": False,
            "may_build_decision_context": False,
            "may_rank_candidates": True,
            "may_select_final_poker_action": False,
            "may_emit_deployed_policy_action": False,
            "production_policy_approved": False,
        },
        "real_policy_agent": {
            "role_type": POLICY_AGENT_ROLE,
            "current_delivery_scope": False,
            "may_normalize_events": False,
            "may_build_decision_context": False,
            "may_rank_candidates": True,
            "may_select_final_poker_action": True,
            "may_emit_deployed_policy_action": False,
            "production_policy_approved": False,
            "requires_separate_stakeholder_approval": True,
            "requires_independent_simulation_gate": True,
        },
    }


def build_llm_role_boundary(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    decision_context = _read_optional_json(reports / "llm_decision_context.json")
    event_eval = _read_optional_json(reports / "llm_event_gold_eval.json")
    decision_gate = _read_optional_json(reports / "llm_decision_gate.json")
    candidate_gate = _read_optional_json(reports / "llm_decision_candidate_gate.json")
    architecture = _read_optional_json(reports / "llm_architecture_comparison.json")
    api_contract = _read_optional_json(reports / "api_contract.json")

    strict_event = (event_eval.get("systems") or {}).get("strict_schema_rules") or {}
    strict_event_type = strict_event.get("event_type") or {}
    context_modes = decision_context.get("supported_context_modes") or {}
    required_controls = decision_context.get("required_controls") or []
    autonomous_api = _autonomous_api_contract(api_contract)
    decision_gate_boundary = decision_gate.get("production_boundary") or {}
    candidate_gate_boundary = candidate_gate.get("production_boundary") or {}
    architecture_boundary = architecture.get("approval_boundary") or {}
    candidate_ranker = _read_optional_json(reports / "llm_decision_candidate_ranker_qwen25.json")

    event_layer_available = bool(strict_event) and _as_float(strict_event_type.get("macro_f1"), 0.0) > 0.0
    decision_context_available = "full_in_context" in context_modes and bool(required_controls)
    llm_decision_approved = bool(decision_gate_boundary.get("llm_agent_production_approved")) or bool(
        candidate_gate_boundary.get("llm_agent_production_approved")
    )

    payload: dict[str, Any] = {
        "version": LLM_ROLE_BOUNDARY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "LLM role boundary for decision-context and event-normalization work",
        "client_statement": (
            "The LLM work is currently strongest as a controlled decision/context and event-normalization layer. "
            "It should not be presented as a fully autonomous poker-playing LLM agent."
        ),
        "controlled_layer_acceptance": {
            "status": CONTROLLED_LAYER_APPROVED,
            "approved_for_current_delivery": True,
            "approved_delivery_scope": ["event_normalization", "decision_context"],
            "research_only_scope": ["candidate_ranking"],
            "excluded_delivery_scope": ["real_policy_agent", "fully_autonomous_poker_playing_llm_policy"],
            "fully_autonomous_poker_playing_llm_policy_status": AUTONOMOUS_LLM_POLICY_NOT_APPROVED,
            "fully_autonomous_poker_playing_llm_policy_approved": False,
            "fully_autonomous_policy_claim_allowed": False,
            "may_be_presented_as": (
                "Controlled LLM event/context layer with schema validation, legal-action controls, "
                "and candidate-ranking research support."
            ),
            "must_not_be_presented_as": (
                "A fully autonomous poker-playing LLM policy or final production strategy engine."
            ),
            "production_blocker_for_current_delivery": False,
            "future_policy_agent_requires_separate_approval": True,
        },
        "term_boundary": {
            "document_term": "LLM-based agent",
            "status": LLM_BASED_AGENT_TERM,
            "requires_role_specific_qualification": True,
            "must_not_imply_fully_autonomous_policy": True,
            "ambiguous_unqualified_usage_allowed": False,
            "reason": (
                "The project documents an LLM-based agent baseline, but the term is ambiguous unless the "
                "implementation role is specified. In this delivery it means controlled LLM components, not "
                "a standalone autonomous poker policy."
            ),
        },
        "recommended_production_architecture": {
            "status": RECOMMENDED_PRODUCTION_ARCHITECTURE,
            "priority": CONTROLLED_CONTEXT_EVENT_LAYER_FIRST,
            "approved_for_current_delivery": True,
            "production_policy_claim_allowed": False,
            "fully_autonomous_llm_agent_claim_allowed": False,
            "why": (
                "The current bottleneck is data quality and state reconstruction, not an unconstrained LLM "
                "strategy engine. A schema-routed hybrid keeps deterministic parsing first, uses the LLM only "
                "for controlled ambiguous context/event handling, and preserves explicit validation gates."
            ),
            "pipeline": [
                "OCR/dealer logs",
                "deterministic parser",
                "candidate generator",
                "LLM fallback for ambiguous event/context cases",
                "JSON schema validation",
                "event stream",
                "feature builder",
                "deployed routed policy stack",
            ],
            "llm_position": "controlled_fallback_before_schema_validation_for_event_context_layer",
            "final_policy_owner": "deployed_routed_policy_stack",
            "not_recommended_first": "fully_autonomous_poker_playing_llm_policy",
        },
        "scope_disambiguation_contract": {
            "status": "EXPLICITLY_DISAMBIGUATED",
            "llm_based_agent_requires_explicit_role": True,
            "ambiguous_llm_agent_term_allowed": False,
            "required_roles": [
                "event_normalization",
                "decision_context",
                "candidate_ranking",
                "real_policy_agent",
            ],
            "role_type_mapping": {
                "event_normalization": EVENT_NORMALIZER_ROLE,
                "decision_context": DECISION_CONTEXT_ROLE,
                "candidate_ranking": CANDIDATE_RANKER_ROLE,
                "real_policy_agent": POLICY_AGENT_ROLE,
            },
            "current_delivery_controlled_roles": ["event_normalization", "decision_context"],
            "current_delivery_research_roles": ["candidate_ranking"],
            "not_current_delivery_roles": ["real_policy_agent"],
            "policy_agent_claim_requires": [
                "separate stakeholder approval",
                "independent simulation gate",
                "legal-action sandbox",
                "bankroll/session controls",
                "production monitoring and rollback",
            ],
        },
        "role_permissions_matrix": build_role_permissions_matrix(),
        "role_taxonomy": {
            "event_normalization": {
                "role_type": EVENT_NORMALIZER_ROLE,
                "status": CONTROLLED_COMPONENT,
                "implemented": event_layer_available,
                "purpose": "Normalize noisy OCR/dealer-log records into validated event JSON.",
                "input_contract": "noisy OCR or dealer-log text",
                "output_contract": "validated event JSON; no poker-policy action is emitted",
                "may_normalize_events": True,
                "may_rank_candidates": False,
                "may_select_final_poker_action": False,
                "can_emit_policy_action": False,
                "production_policy_approved": False,
                "requires_schema_validation": True,
            },
            "decision_context": {
                "role_type": DECISION_CONTEXT_ROLE,
                "status": CONTROLLED_COMPONENT,
                "implemented": decision_context_available,
                "purpose": "Provide explicit poker rules, legal actions, constraints, and output schema for LLM decision experiments.",
                "input_contract": "structured game state plus formal poker rules, constraints, and legal actions",
                "output_contract": "research decision suggestion constrained by legal-action filtering; not a production policy",
                "may_normalize_events": False,
                "may_rank_candidates": False,
                "may_select_final_poker_action": False,
                "may_return_research_action_suggestion": True,
                "can_emit_policy_action": True,
                "production_policy_approved": False,
                "requires_legal_action_filtering": True,
            },
            "candidate_ranking": {
                "role_type": CANDIDATE_RANKER_ROLE,
                "status": RESEARCH_COMPONENT,
                "implemented": bool(candidate_ranker) or architecture.get("recommended_architecture") == "candidate_ranker",
                "purpose": "Rank constrained candidate actions/events instead of free-form generation.",
                "input_contract": "raw context plus a deterministic candidate set",
                "output_contract": "candidate_id and confidence; no unconstrained action generation",
                "may_normalize_events": False,
                "may_rank_candidates": True,
                "may_select_final_poker_action": False,
                "can_emit_policy_action": False,
                "recommended_architecture": architecture.get("recommended_architecture"),
                "provider": candidate_ranker.get("provider"),
                "production_policy_approved": False,
                "deployed_strategy_stack_affected": False,
            },
            "real_policy_agent": {
                "role_type": POLICY_AGENT_ROLE,
                "status": NOT_CURRENT_SCOPE,
                "implemented": False,
                "purpose": "A standalone LLM policy that controls poker play end to end.",
                "input_contract": "full game state and session context",
                "output_contract": "final poker action, bet size, and timing for live policy control",
                "may_normalize_events": False,
                "may_rank_candidates": True,
                "may_select_final_poker_action": True,
                "production_policy_approved": False,
                "requires_separate_stakeholder_approval": True,
                "requires_independent_simulation_gate": True,
            },
        },
        "current_llm_role": {
            "status": CONTROLLED_LAYER,
            "event_normalization_layer": {
                "implemented": event_layer_available,
                "evaluation_report": "reports/llm_event_gold_eval.json",
                "best_controlled_system": "strict_schema_rules" if strict_event else "UNKNOWN",
                "gold_examples": event_eval.get("examples"),
                "event_type_accuracy": strict_event_type.get("accuracy"),
                "event_type_macro_f1": strict_event_type.get("macro_f1"),
                "schema_style": "strict controlled JSON/event schema",
            },
            "decision_context_layer": {
                "implemented": decision_context_available,
                "contract_report": "reports/llm_decision_context.json",
                "default_context_mode": decision_context.get("default_context_mode"),
                "supported_context_modes": sorted(context_modes.keys()),
                "required_controls": required_controls,
            },
            "production_status": RESEARCH_BASELINE,
            "llm_decision_path_production_approved": llm_decision_approved,
        },
        "autonomous_llm_agent_boundary": {
            "status": NOT_AUTONOMOUS_LLM_AGENT,
            "fully_autonomous_poker_playing_llm_agent_present": False,
            "fully_autonomous_llm_agent_claim_allowed": False,
            "deployed_autonomous_endpoint_is_llm": False,
            "deployed_autonomous_endpoint_agent_type": autonomous_api.get("agent_type", "controlled_stateful_policy_agent"),
            "deployed_autonomous_endpoint": autonomous_api.get("decision_endpoint", "/agent/decide"),
            "llm_can_choose_unconstrained_actions": False,
            "llm_can_bypass_schema_validation": False,
            "production_blocker_for_current_delivery": False,
            "reason": (
                "The current LLM work is constrained by explicit context, schema controls, legal-action filtering, "
                "candidate ranking or extraction rules, and independent gates. It is not an unconstrained autonomous "
                "LLM that plays poker end to end."
            ),
        },
        "evidence": {
            "decision_context_contract": "reports/llm_decision_context.json",
            "event_normalization_eval": "reports/llm_event_gold_eval.json",
            "llm_decision_gate": "reports/llm_decision_gate.json",
            "candidate_ranker_gate": "reports/llm_decision_candidate_gate.json",
            "architecture_comparison": "reports/llm_architecture_comparison.json",
            "decision_gate_status": decision_gate.get("status"),
            "candidate_gate_status": candidate_gate.get("status"),
            "architecture_production_approved": architecture.get("production_approved"),
            "deployed_strategy_stack_affected": architecture_boundary.get("deployed_strategy_stack_affected"),
        },
        "allowed_claims": [
            "The LLM work provides a controlled event-normalization layer for noisy OCR/dealer-log data.",
            "The LLM decision work is structured as an in-context decision/research layer with explicit controls.",
            "The current LLM components are research/control layers and do not override the deployed strategy stack.",
        ],
        "not_allowed_claims": [
            "The project contains a fully autonomous poker-playing LLM agent.",
            "The phrase LLM-based agent means a production-approved autonomous poker policy.",
            "The event-normalization LLM component is a poker policy agent.",
            "Candidate ranking is production-approved as the deployed poker policy.",
            "The LLM decision path is production-approved as the deployed poker policy.",
            "The LLM can make unconstrained poker decisions without legal-action filtering or schema validation.",
            "The controlled stateful policy endpoint is an autonomous LLM poker player.",
        ],
        "next_milestone_if_autonomous_llm_is_requested": [
            "Define a separate autonomous-LLM-agent milestone with legal-action sandboxing, bankroll limits, and session controls.",
            "Run supervised decision-context benchmarks against reviewed labels before enabling any LLM policy path.",
            "Add simulation gates and adversarial prompt/security tests for the LLM decision layer.",
            "Require explicit stakeholder approval before changing the LLM role from controlled layer to policy agent.",
        ],
    }
    payload["claim_validation_examples"] = build_llm_claim_validation_examples(payload)
    payload["production_scope_claim_examples"] = build_llm_production_scope_claim_examples(payload)
    payload["proof_cases"] = build_llm_role_boundary_proof_cases(payload)
    payload["invariants"] = validate_llm_role_boundary(payload)
    if not all(case["passed"] for case in payload["proof_cases"]):
        payload["invariants"]["status"] = "FAIL"
        payload["invariants"]["violations"].append("llm_role_boundary_proof_cases_must_pass")
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_llm_role_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    role = payload.get("current_llm_role") or {}
    term = payload.get("term_boundary") or {}
    acceptance = payload.get("controlled_layer_acceptance") or {}
    recommended_architecture = payload.get("recommended_production_architecture") or {}
    scope = payload.get("scope_disambiguation_contract") or {}
    permissions = payload.get("role_permissions_matrix") or {}
    taxonomy = payload.get("role_taxonomy") or {}
    event_role = taxonomy.get("event_normalization") or {}
    context_role = taxonomy.get("decision_context") or {}
    candidate_role = taxonomy.get("candidate_ranking") or {}
    policy_role = taxonomy.get("real_policy_agent") or {}
    event_layer = role.get("event_normalization_layer") or {}
    decision_context = role.get("decision_context_layer") or {}
    boundary = payload.get("autonomous_llm_agent_boundary") or {}
    evidence = payload.get("evidence") or {}
    claim_examples = {case.get("name"): case for case in payload.get("claim_validation_examples") or []}
    production_scope_examples = {
        case.get("name"): case for case in payload.get("production_scope_claim_examples") or []
    }

    if role.get("status") != CONTROLLED_LAYER:
        violations.append("llm_role_must_remain_controlled_layer")
    if term.get("status") != LLM_BASED_AGENT_TERM:
        violations.append("llm_based_agent_term_must_remain_umbrella_term")
    if term.get("requires_role_specific_qualification") is not True:
        violations.append("llm_based_agent_term_must_require_role_qualification")
    if term.get("must_not_imply_fully_autonomous_policy") is not True:
        violations.append("llm_based_agent_term_must_not_imply_autonomous_policy")
    if term.get("ambiguous_unqualified_usage_allowed") is not False:
        violations.append("llm_based_agent_ambiguous_unqualified_usage_must_be_blocked")
    if acceptance.get("status") != CONTROLLED_LAYER_APPROVED:
        violations.append("controlled_llm_layer_must_be_explicitly_approved")
    if acceptance.get("approved_for_current_delivery") is not True:
        violations.append("controlled_llm_layer_must_be_delivery_approved")
    if set(acceptance.get("approved_delivery_scope") or []) != {"event_normalization", "decision_context"}:
        violations.append("controlled_llm_approved_scope_must_be_event_and_context_only")
    if set(acceptance.get("research_only_scope") or []) != {"candidate_ranking"}:
        violations.append("controlled_llm_research_scope_must_be_candidate_ranking_only")
    if "real_policy_agent" not in set(acceptance.get("excluded_delivery_scope") or []):
        violations.append("real_policy_agent_must_be_excluded_from_current_delivery_scope")
    if "fully_autonomous_poker_playing_llm_policy" not in set(acceptance.get("excluded_delivery_scope") or []):
        violations.append("autonomous_llm_policy_must_be_excluded_from_current_delivery_scope")
    if acceptance.get("fully_autonomous_poker_playing_llm_policy_status") != AUTONOMOUS_LLM_POLICY_NOT_APPROVED:
        violations.append("autonomous_llm_policy_status_must_remain_not_approved")
    if acceptance.get("fully_autonomous_poker_playing_llm_policy_approved") is not False:
        violations.append("autonomous_llm_policy_must_not_be_approved")
    if acceptance.get("fully_autonomous_policy_claim_allowed") is not False:
        violations.append("autonomous_llm_policy_claim_must_not_be_allowed")
    if acceptance.get("production_blocker_for_current_delivery") is not False:
        violations.append("controlled_llm_layer_boundary_must_not_block_delivery")
    if acceptance.get("future_policy_agent_requires_separate_approval") is not True:
        violations.append("future_llm_policy_agent_must_require_separate_approval")
    if recommended_architecture.get("status") != RECOMMENDED_PRODUCTION_ARCHITECTURE:
        violations.append("llm_recommended_architecture_must_remain_schema_routed_hybrid_controlled_layer")
    if recommended_architecture.get("priority") != CONTROLLED_CONTEXT_EVENT_LAYER_FIRST:
        violations.append("llm_architecture_priority_must_remain_controlled_context_event_layer_first")
    if recommended_architecture.get("approved_for_current_delivery") is not True:
        violations.append("llm_recommended_architecture_must_be_approved_for_current_delivery")
    if recommended_architecture.get("production_policy_claim_allowed") is not False:
        violations.append("llm_recommended_architecture_must_not_allow_production_policy_claim")
    if recommended_architecture.get("fully_autonomous_llm_agent_claim_allowed") is not False:
        violations.append("llm_recommended_architecture_must_not_allow_autonomous_claim")
    if recommended_architecture.get("final_policy_owner") != "deployed_routed_policy_stack":
        violations.append("llm_final_policy_owner_must_remain_deployed_routed_policy_stack")
    if recommended_architecture.get("not_recommended_first") != "fully_autonomous_poker_playing_llm_policy":
        violations.append("llm_autonomous_policy_must_remain_not_recommended_first")
    if "LLM fallback for ambiguous event/context cases" not in set(recommended_architecture.get("pipeline") or []):
        violations.append("llm_architecture_must_place_llm_as_controlled_ambiguous_case_fallback")
    required_role_types = {
        "event_normalization": EVENT_NORMALIZER_ROLE,
        "decision_context": DECISION_CONTEXT_ROLE,
        "candidate_ranking": CANDIDATE_RANKER_ROLE,
        "real_policy_agent": POLICY_AGENT_ROLE,
    }
    expected_permissions = build_role_permissions_matrix()
    if set(permissions) != set(expected_permissions):
        violations.append("llm_role_permissions_matrix_must_cover_all_roles")
    for role_name, expected in expected_permissions.items():
        actual = permissions.get(role_name) or {}
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                violations.append(f"llm_role_permission_mismatch:{role_name}.{key}")
    for role_name, actual in permissions.items():
        if role_name != "real_policy_agent" and actual.get("may_select_final_poker_action") is not False:
            violations.append(f"llm_non_policy_role_must_not_select_final_action:{role_name}")
        if actual.get("may_emit_deployed_policy_action") is not False:
            violations.append(f"llm_role_must_not_emit_deployed_policy_action:{role_name}")
        if actual.get("production_policy_approved") is not False:
            violations.append(f"llm_role_must_not_be_production_policy_approved:{role_name}")
    if scope.get("status") != "EXPLICITLY_DISAMBIGUATED":
        violations.append("llm_scope_must_be_explicitly_disambiguated")
    if scope.get("llm_based_agent_requires_explicit_role") is not True:
        violations.append("llm_based_agent_must_require_explicit_role")
    if scope.get("ambiguous_llm_agent_term_allowed") is not False:
        violations.append("ambiguous_llm_agent_term_must_not_be_allowed")
    if set(scope.get("required_roles") or []) != set(required_role_types):
        violations.append("llm_scope_required_roles_must_match_taxonomy")
    if (scope.get("role_type_mapping") or {}) != required_role_types:
        violations.append("llm_scope_role_type_mapping_must_be_complete")
    if set(scope.get("current_delivery_controlled_roles") or []) != {"event_normalization", "decision_context"}:
        violations.append("llm_current_delivery_controlled_roles_must_be_explicit")
    if set(scope.get("current_delivery_research_roles") or []) != {"candidate_ranking"}:
        violations.append("llm_current_delivery_research_roles_must_be_explicit")
    if set(scope.get("not_current_delivery_roles") or []) != {"real_policy_agent"}:
        violations.append("llm_not_current_delivery_roles_must_be_explicit")
    for required_role in required_role_types:
        if required_role not in taxonomy:
            violations.append(f"llm_role_taxonomy_missing:{required_role}")
    for required_role, expected_role_type in required_role_types.items():
        role_payload = taxonomy.get(required_role) or {}
        if role_payload.get("role_type") != expected_role_type:
            violations.append(f"llm_role_type_mismatch:{required_role}")
        if not role_payload.get("input_contract"):
            violations.append(f"llm_role_input_contract_missing:{required_role}")
        if not role_payload.get("output_contract"):
            violations.append(f"llm_role_output_contract_missing:{required_role}")
        if "may_select_final_poker_action" not in role_payload:
            violations.append(f"llm_role_policy_action_capability_missing:{required_role}")
    if event_role.get("status") != CONTROLLED_COMPONENT:
        violations.append("event_normalization_role_must_remain_controlled_component")
    if event_role.get("may_normalize_events") is not True:
        violations.append("event_normalization_must_be_normalization_only")
    if event_role.get("may_rank_candidates") is not False:
        violations.append("event_normalization_must_not_rank_candidates")
    if event_role.get("may_select_final_poker_action") is not False:
        violations.append("event_normalization_must_not_select_final_poker_action")
    if event_role.get("implemented") is not True:
        violations.append("event_normalization_role_must_be_implemented")
    if event_role.get("can_emit_policy_action") is not False:
        violations.append("event_normalization_must_not_emit_policy_action")
    if event_role.get("production_policy_approved") is not False:
        violations.append("event_normalization_must_not_be_policy_approved")
    if event_role.get("requires_schema_validation") is not True:
        violations.append("event_normalization_must_require_schema_validation")
    if context_role.get("status") != CONTROLLED_COMPONENT:
        violations.append("decision_context_role_must_remain_controlled_component")
    if context_role.get("may_select_final_poker_action") is not False:
        violations.append("decision_context_must_not_select_final_poker_action")
    if context_role.get("implemented") is not True:
        violations.append("decision_context_role_must_be_implemented")
    if context_role.get("production_policy_approved") is not False:
        violations.append("decision_context_must_not_be_policy_approved")
    if context_role.get("requires_legal_action_filtering") is not True:
        violations.append("decision_context_must_require_legal_action_filtering")
    if candidate_role.get("status") != RESEARCH_COMPONENT:
        violations.append("candidate_ranking_must_remain_research_component")
    if candidate_role.get("may_rank_candidates") is not True:
        violations.append("candidate_ranking_must_rank_candidates")
    if candidate_role.get("can_emit_policy_action") is not False:
        violations.append("candidate_ranking_must_not_emit_policy_action")
    if candidate_role.get("may_select_final_poker_action") is not False:
        violations.append("candidate_ranking_must_not_select_final_poker_action")
    if candidate_role.get("implemented") is not True:
        violations.append("candidate_ranking_role_must_be_implemented")
    if candidate_role.get("production_policy_approved") is not False:
        violations.append("candidate_ranking_must_not_be_policy_approved")
    if candidate_role.get("deployed_strategy_stack_affected") is not False:
        violations.append("candidate_ranking_must_not_affect_deployed_strategy_stack")
    if policy_role.get("status") != NOT_CURRENT_SCOPE:
        violations.append("real_policy_agent_must_remain_out_of_current_scope")
    if policy_role.get("role_type") != POLICY_AGENT_ROLE:
        violations.append("real_policy_agent_role_type_must_be_policy_agent")
    if policy_role.get("may_select_final_poker_action") is not True:
        violations.append("real_policy_agent_must_be_the_only_final_action_role")
    if policy_role.get("implemented") is not False:
        violations.append("real_policy_agent_must_not_be_marked_implemented")
    if policy_role.get("production_policy_approved") is not False:
        violations.append("real_policy_agent_must_not_be_policy_approved")
    if policy_role.get("requires_separate_stakeholder_approval") is not True:
        violations.append("real_policy_agent_must_require_separate_stakeholder_approval")
    if policy_role.get("requires_independent_simulation_gate") is not True:
        violations.append("real_policy_agent_must_require_independent_simulation_gate")
    if event_layer.get("implemented") is not True:
        violations.append("event_normalization_layer_must_be_implemented")
    if decision_context.get("implemented") is not True:
        violations.append("decision_context_layer_must_be_implemented")
    if role.get("production_status") != RESEARCH_BASELINE:
        violations.append("llm_production_status_must_remain_research_baseline")
    if role.get("llm_decision_path_production_approved") is not False:
        violations.append("llm_decision_path_must_not_be_production_approved")
    if boundary.get("status") != NOT_AUTONOMOUS_LLM_AGENT:
        violations.append("llm_agent_boundary_must_remain_not_autonomous")
    if boundary.get("fully_autonomous_poker_playing_llm_agent_present") is not False:
        violations.append("fully_autonomous_llm_agent_presence_must_be_false")
    if boundary.get("fully_autonomous_llm_agent_claim_allowed") is not False:
        violations.append("fully_autonomous_llm_agent_claim_must_be_blocked")
    if boundary.get("deployed_autonomous_endpoint_is_llm") is not False:
        violations.append("deployed_autonomous_endpoint_must_not_be_labeled_llm")
    if boundary.get("llm_can_choose_unconstrained_actions") is not False:
        violations.append("llm_unconstrained_actions_must_be_blocked")
    if boundary.get("llm_can_bypass_schema_validation") is not False:
        violations.append("llm_schema_bypass_must_be_blocked")
    if boundary.get("production_blocker_for_current_delivery") is not False:
        violations.append("llm_role_boundary_must_not_block_current_delivery")
    if evidence.get("architecture_production_approved") is not False:
        violations.append("llm_architecture_must_not_grant_production_approval")
    if evidence.get("deployed_strategy_stack_affected") is not False:
        violations.append("llm_role_boundary_must_not_affect_deployed_strategy_stack")
    for required_claim_case in (
        "blocks_unqualified_llm_based_agent_production_claim",
        "blocks_event_normalizer_as_policy_agent",
        "allows_decision_context_research_claim",
        "blocks_candidate_ranker_as_deployed_policy",
        "blocks_real_policy_agent_current_delivery_claim",
    ):
        claim_case = claim_examples.get(required_claim_case) or {}
        if not claim_case:
            violations.append(f"llm_claim_validation_case_missing:{required_claim_case}")
            continue
        observed = validate_llm_agent_claim(claim_case.get("claim") or {}, payload)
        if observed["status"] != claim_case.get("expected_status"):
            violations.append(f"llm_claim_validation_case_status_mismatch:{required_claim_case}")
        if claim_case.get("passed") is not True:
            violations.append(f"llm_claim_validation_case_failed:{required_claim_case}")

    for required_scope_case in (
        "allows_controlled_event_context_layer_production_claim",
        "blocks_autonomous_llm_policy_production_claim",
        "blocks_unqualified_llm_policy_claim_text",
    ):
        scope_case = production_scope_examples.get(required_scope_case) or {}
        if not scope_case:
            violations.append(f"llm_production_scope_claim_case_missing:{required_scope_case}")
            continue
        observed = validate_llm_production_scope_claim(scope_case.get("claim") or {}, payload)
        if observed["status"] != scope_case.get("expected_status"):
            violations.append(f"llm_production_scope_claim_case_status_mismatch:{required_scope_case}")
        if scope_case.get("passed") is not True:
            violations.append(f"llm_production_scope_claim_case_failed:{required_scope_case}")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def build_llm_claim_validation_examples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    examples = [
        {
            "name": "blocks_unqualified_llm_based_agent_production_claim",
            "expected_status": "FAIL",
            "claim": {
                "uses_llm_based_agent_term": True,
                "role": None,
                "production_policy_claim": True,
                "autonomous_policy_claim": False,
                "final_poker_action_claim": False,
            },
        },
        {
            "name": "blocks_event_normalizer_as_policy_agent",
            "expected_status": "FAIL",
            "claim": {
                "uses_llm_based_agent_term": True,
                "role": "event_normalization",
                "production_policy_claim": True,
                "autonomous_policy_claim": False,
                "final_poker_action_claim": True,
            },
        },
        {
            "name": "allows_decision_context_research_claim",
            "expected_status": "PASS",
            "claim": {
                "uses_llm_based_agent_term": True,
                "role": "decision_context",
                "production_policy_claim": False,
                "autonomous_policy_claim": False,
                "final_poker_action_claim": False,
            },
        },
        {
            "name": "blocks_candidate_ranker_as_deployed_policy",
            "expected_status": "FAIL",
            "claim": {
                "uses_llm_based_agent_term": True,
                "role": "candidate_ranking",
                "production_policy_claim": True,
                "autonomous_policy_claim": False,
                "final_poker_action_claim": True,
            },
        },
        {
            "name": "blocks_real_policy_agent_current_delivery_claim",
            "expected_status": "FAIL",
            "claim": {
                "uses_llm_based_agent_term": True,
                "role": "real_policy_agent",
                "production_policy_claim": True,
                "autonomous_policy_claim": True,
                "final_poker_action_claim": True,
            },
        },
    ]
    for example in examples:
        observed = validate_llm_agent_claim(example["claim"], payload)
        example["observed_status"] = observed["status"]
        example["passed"] = observed["status"] == example["expected_status"]
        example["violations"] = observed["violations"]
    return examples


def build_llm_production_scope_claim_examples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    examples = [
        {
            "name": "allows_controlled_event_context_layer_production_claim",
            "expected_status": "PASS",
            "claim": {
                "role": "controlled_event_context_layer",
                "production_claim": True,
                "controlled_event_context_layer_claim": True,
                "autonomous_poker_policy_claim": False,
                "policy_agent_claim": False,
                "final_action_policy_claim": False,
                "claim_text": (
                    "The LLM work is production-approved as a controlled event/context layer, "
                    "not as the final poker policy."
                ),
            },
        },
        {
            "name": "blocks_autonomous_llm_policy_production_claim",
            "expected_status": "FAIL",
            "claim": {
                "role": "real_policy_agent",
                "production_claim": True,
                "controlled_event_context_layer_claim": False,
                "autonomous_poker_policy_claim": True,
                "policy_agent_claim": True,
                "final_action_policy_claim": True,
                "claim_text": "The LLM is a production autonomous poker policy agent.",
            },
        },
        {
            "name": "blocks_unqualified_llm_policy_claim_text",
            "expected_status": "FAIL",
            "claim": {
                "role": "decision_context",
                "production_claim": True,
                "controlled_event_context_layer_claim": False,
                "autonomous_poker_policy_claim": False,
                "policy_agent_claim": True,
                "final_action_policy_claim": False,
                "claim_text": "The LLM policy is approved for production.",
            },
        },
    ]
    for example in examples:
        observed = validate_llm_production_scope_claim(example["claim"], payload)
        example["observed_status"] = observed["status"]
        example["passed"] = observed["status"] == example["expected_status"]
        example["violations"] = observed["violations"]
    return examples


def build_llm_role_boundary_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        _llm_role_proof_case("base_contract_is_valid", payload, "PASS"),
    ]

    mutated = deepcopy(payload)
    mutated["term_boundary"]["must_not_imply_fully_autonomous_policy"] = False
    mutated["role_taxonomy"]["real_policy_agent"]["implemented"] = True
    mutated["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] = True
    cases.append(_llm_role_proof_case("blocks_llm_based_agent_as_autonomous_policy", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["role_taxonomy"]["event_normalization"]["can_emit_policy_action"] = True
    mutated["role_taxonomy"]["event_normalization"]["production_policy_approved"] = True
    cases.append(_llm_role_proof_case("blocks_event_normalization_as_policy_agent", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["role_taxonomy"]["candidate_ranking"]["production_policy_approved"] = True
    mutated["role_taxonomy"]["candidate_ranking"]["deployed_strategy_stack_affected"] = True
    cases.append(_llm_role_proof_case("blocks_candidate_ranking_as_deployed_policy", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["role_taxonomy"]["real_policy_agent"]["status"] = "CURRENT_DELIVERY_SCOPE"
    mutated["role_taxonomy"]["real_policy_agent"]["implemented"] = True
    mutated["role_taxonomy"]["real_policy_agent"]["production_policy_approved"] = True
    cases.append(_llm_role_proof_case("blocks_real_policy_agent_current_scope_claim", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["scope_disambiguation_contract"]["ambiguous_llm_agent_term_allowed"] = True
    mutated["role_taxonomy"]["event_normalization"]["role_type"] = POLICY_AGENT_ROLE
    mutated["role_taxonomy"]["event_normalization"]["may_select_final_poker_action"] = True
    cases.append(_llm_role_proof_case("blocks_ambiguous_llm_agent_scope", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["controlled_layer_acceptance"]["approved_delivery_scope"].append("real_policy_agent")
    mutated["controlled_layer_acceptance"]["fully_autonomous_poker_playing_llm_policy_approved"] = True
    mutated["controlled_layer_acceptance"]["fully_autonomous_policy_claim_allowed"] = True
    cases.append(_llm_role_proof_case("blocks_autonomous_policy_under_controlled_layer_acceptance", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["recommended_production_architecture"]["status"] = "FULLY_AUTONOMOUS_POKER_PLAYING_LLM_POLICY"
    mutated["recommended_production_architecture"]["production_policy_claim_allowed"] = True
    mutated["recommended_production_architecture"]["fully_autonomous_llm_agent_claim_allowed"] = True
    mutated["recommended_production_architecture"]["final_policy_owner"] = "llm_policy_agent"
    cases.append(_llm_role_proof_case("blocks_autonomous_architecture_as_recommended_production_path", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["role_permissions_matrix"]["candidate_ranking"]["may_emit_deployed_policy_action"] = True
    mutated["role_permissions_matrix"]["candidate_ranking"]["production_policy_approved"] = True
    cases.append(_llm_role_proof_case("blocks_candidate_ranker_permission_escalation", mutated, "FAIL"))

    mutated = deepcopy(payload)
    del mutated["role_taxonomy"]["candidate_ranking"]
    cases.append(_llm_role_proof_case("blocks_missing_role_taxonomy", mutated, "FAIL"))

    return cases


def _llm_role_proof_case(name: str, candidate: dict[str, Any], expected_status: str) -> dict[str, Any]:
    observed = validate_llm_role_boundary(candidate)
    return {
        "name": name,
        "expected_status": expected_status,
        "observed_status": observed["status"],
        "passed": observed["status"] == expected_status,
        "violations": observed["violations"],
    }


def write_llm_role_boundary(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_llm_role_boundary(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_llm_role_boundary_markdown(payload), encoding="utf-8")
    return payload


def render_llm_role_boundary_markdown(payload: dict[str, Any]) -> str:
    role = payload["current_llm_role"]
    event_layer = role["event_normalization_layer"]
    context_layer = role["decision_context_layer"]
    boundary = payload["autonomous_llm_agent_boundary"]
    taxonomy = payload["role_taxonomy"]
    scope = payload["scope_disambiguation_contract"]
    acceptance = payload["controlled_layer_acceptance"]
    recommended = payload["recommended_production_architecture"]
    lines = [
        "# LLM Role Boundary Contract",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Current LLM Role",
        "",
        f"- Status: `{role['status']}`",
        f"- Production status: `{role['production_status']}`",
        f"- LLM decision path production-approved: `{role['llm_decision_path_production_approved']}`",
        f"- Document term: `{payload['term_boundary']['document_term']}`",
        f"- Document term status: `{payload['term_boundary']['status']}`",
        f"- Ambiguous unqualified LLM-agent usage allowed: `{payload['term_boundary']['ambiguous_unqualified_usage_allowed']}`",
        f"- Event-normalization implemented: `{event_layer['implemented']}`",
        f"- Event-normalization macro F1: `{event_layer['event_type_macro_f1']}`",
        f"- Decision-context implemented: `{context_layer['implemented']}`",
        f"- Default context mode: `{context_layer['default_context_mode']}`",
        "",
        "## Controlled Layer Acceptance",
        "",
        f"- Status: `{acceptance['status']}`",
        f"- Approved for current delivery: `{acceptance['approved_for_current_delivery']}`",
        f"- Approved delivery scope: `{', '.join(acceptance['approved_delivery_scope'])}`",
        f"- Research-only scope: `{', '.join(acceptance['research_only_scope'])}`",
        f"- Excluded delivery scope: `{', '.join(acceptance['excluded_delivery_scope'])}`",
        f"- Fully autonomous LLM policy status: `{acceptance['fully_autonomous_poker_playing_llm_policy_status']}`",
        f"- Fully autonomous LLM policy approved: `{acceptance['fully_autonomous_poker_playing_llm_policy_approved']}`",
        f"- Fully autonomous policy claim allowed: `{acceptance['fully_autonomous_policy_claim_allowed']}`",
        f"- Production blocker for current delivery: `{acceptance['production_blocker_for_current_delivery']}`",
        f"- Future policy-agent approval required: `{acceptance['future_policy_agent_requires_separate_approval']}`",
        "",
        "## Recommended Production Architecture",
        "",
        f"- Status: `{recommended['status']}`",
        f"- Priority: `{recommended['priority']}`",
        f"- Approved for current delivery: `{recommended['approved_for_current_delivery']}`",
        f"- Production policy claim allowed: `{recommended['production_policy_claim_allowed']}`",
        f"- Fully autonomous LLM claim allowed: `{recommended['fully_autonomous_llm_agent_claim_allowed']}`",
        f"- LLM position: `{recommended['llm_position']}`",
        f"- Final policy owner: `{recommended['final_policy_owner']}`",
        f"- Not recommended first: `{recommended['not_recommended_first']}`",
        "",
        "## Scope Disambiguation",
        "",
        f"- Status: `{scope['status']}`",
        f"- Explicit role required: `{scope['llm_based_agent_requires_explicit_role']}`",
        f"- Ambiguous LLM-agent term allowed: `{scope['ambiguous_llm_agent_term_allowed']}`",
        f"- Controlled delivery roles: `{', '.join(scope['current_delivery_controlled_roles'])}`",
        f"- Research delivery roles: `{', '.join(scope['current_delivery_research_roles'])}`",
        f"- Not current delivery roles: `{', '.join(scope['not_current_delivery_roles'])}`",
        "",
        "## Role Taxonomy",
        "",
    ]
    for role_name, role_payload in taxonomy.items():
        lines.extend(
            [
                f"### `{role_name}`",
                "",
                f"- Status: `{role_payload.get('status')}`",
                f"- Role type: `{role_payload.get('role_type')}`",
                f"- Implemented: `{role_payload.get('implemented')}`",
                f"- Production policy approved: `{role_payload.get('production_policy_approved')}`",
                f"- Input contract: {role_payload.get('input_contract')}",
                f"- Output contract: {role_payload.get('output_contract')}",
                f"- May select final poker action: `{role_payload.get('may_select_final_poker_action')}`",
                f"- Purpose: {role_payload.get('purpose')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Autonomous LLM Boundary",
            "",
            f"- Status: `{boundary['status']}`",
            f"- Fully autonomous poker-playing LLM present: `{boundary['fully_autonomous_poker_playing_llm_agent_present']}`",
            f"- Fully autonomous LLM claim allowed: `{boundary['fully_autonomous_llm_agent_claim_allowed']}`",
            f"- Deployed autonomous endpoint is LLM: `{boundary['deployed_autonomous_endpoint_is_llm']}`",
            f"- Production blocker for current delivery: `{boundary['production_blocker_for_current_delivery']}`",
            "",
            "## Not Allowed Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", "## Executable Proof Cases", ""])
    for case in payload.get("proof_cases") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, passed `{case['passed']}`"
        )
    lines.extend(["", "## Claim Validation Examples", ""])
    for case in payload.get("claim_validation_examples") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, passed `{case['passed']}`"
        )
    lines.extend(["", "## Production Scope Claim Guard", ""])
    lines.extend(
        [
            "- Validator: `poker_agent.llm_role_boundary.validate_llm_production_scope_claim`",
            "- Approved production wording: controlled event/context layer only.",
            "- Blocked wording: autonomous poker-playing LLM policy or final-action policy agent.",
            "",
        ]
    )
    for case in payload.get("production_scope_claim_examples") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, passed `{case['passed']}`"
        )
    lines.extend(["", "## Next Milestone If Autonomous LLM Is Requested", ""])
    lines.extend(f"- {item}" for item in payload["next_milestone_if_autonomous_llm_is_requested"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _autonomous_api_contract(api_contract: dict[str, Any]) -> dict[str, Any]:
    if isinstance(api_contract.get("autonomous_agent"), dict):
        return api_contract["autonomous_agent"]
    if isinstance(api_contract.get("capabilities"), dict):
        maybe = api_contract["capabilities"].get("autonomous_agent")
        if isinstance(maybe, dict):
            return maybe
    return {}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
