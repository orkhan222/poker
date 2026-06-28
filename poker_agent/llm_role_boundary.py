from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LLM_ROLE_BOUNDARY_VERSION = "2026-06-28"
CONTROLLED_LAYER = "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
NOT_AUTONOMOUS_LLM_AGENT = "NOT_FULLY_AUTONOMOUS_POKER_PLAYING_LLM_AGENT"
RESEARCH_BASELINE = "RESEARCH_BASELINE_NOT_PRODUCTION_POLICY"


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
    payload["invariants"] = validate_llm_role_boundary(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_llm_role_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    role = payload.get("current_llm_role") or {}
    event_layer = role.get("event_normalization_layer") or {}
    decision_context = role.get("decision_context_layer") or {}
    boundary = payload.get("autonomous_llm_agent_boundary") or {}
    evidence = payload.get("evidence") or {}

    if role.get("status") != CONTROLLED_LAYER:
        violations.append("llm_role_must_remain_controlled_layer")
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

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


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
        f"- Event-normalization implemented: `{event_layer['implemented']}`",
        f"- Event-normalization macro F1: `{event_layer['event_type_macro_f1']}`",
        f"- Decision-context implemented: `{context_layer['implemented']}`",
        f"- Default context mode: `{context_layer['default_context_mode']}`",
        "",
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
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
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
