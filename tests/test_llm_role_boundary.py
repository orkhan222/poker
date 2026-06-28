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
        json.dumps({"production_approved": False, "approval_boundary": {"deployed_strategy_stack_affected": False}}),
        encoding="utf-8",
    )

    payload = build_llm_role_boundary(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["current_llm_role"]["status"] == "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
    assert payload["current_llm_role"]["event_normalization_layer"]["implemented"] is True
    assert payload["current_llm_role"]["decision_context_layer"]["implemented"] is True
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_poker_playing_llm_agent_present"] is False
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] is False


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


def test_llm_role_boundary_endpoint_returns_contract() -> None:
    from poker_agent.service import llm_role_boundary_json

    payload = llm_role_boundary_json()

    assert payload["overall_status"] == "PASS"
    assert payload["current_llm_role"]["status"] == "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
    assert payload["autonomous_llm_agent_boundary"]["fully_autonomous_llm_agent_claim_allowed"] is False
