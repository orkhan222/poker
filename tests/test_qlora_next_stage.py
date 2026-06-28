from __future__ import annotations

import json
from pathlib import Path

from poker_agent.qlora_next_stage import build_qlora_next_stage, validate_qlora_next_stage


def _write_reports(reports: Path) -> None:
    reports.mkdir()
    (reports / "llm_role_boundary.json").write_text(
        json.dumps(
            {
                "current_llm_role": {"status": "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"},
                "autonomous_llm_agent_boundary": {"fully_autonomous_llm_agent_claim_allowed": False},
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_architecture_comparison.json").write_text(
        json.dumps(
            {
                "production_approved": False,
                "approval_boundary": {"deployed_strategy_stack_affected": False},
                "recommended_architecture": "candidate_ranker",
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_decision_candidate_gate.json").write_text(
        json.dumps(
            {
                "status": "BASELINE_NOT_APPROVED",
                "production_boundary": {"llm_agent_production_approved": False},
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_decision_candidate_ranker_qwen25.json").write_text(
        json.dumps({"provider": "transformers_candidate_ranker:Qwen/Qwen2.5-1.5B-Instruct:4bit_nf4"}),
        encoding="utf-8",
    )
    (reports / "llm_event_gold_eval.json").write_text(json.dumps({"examples": 24}), encoding="utf-8")


def test_qlora_next_stage_tracks_improvement_without_approval(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_qlora_next_stage(tmp_path)

    boundary = payload["stage_boundary"]
    targets = payload["target_use_cases"]
    assert payload["overall_status"] == "PASS"
    assert boundary["stage_status"] == "NEXT_STAGE_IMPROVEMENT"
    assert boundary["fine_tuning_completed"] is False
    assert boundary["production_approved"] is False
    assert boundary["current_delivery_blocker"] is False
    assert boundary["autonomous_llm_agent_claim_allowed"] is False
    assert targets["structured_extraction"]["recommended"] is True
    assert targets["candidate_ranking"]["recommended"] is True
    assert targets["noisy_ocr_dealer_log_handling"]["recommended"] is True
    assert targets["autonomous_poker_policy"]["recommended"] is False


def test_qlora_next_stage_blocks_false_completion_or_production_claims(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_qlora_next_stage(tmp_path)
    payload["stage_boundary"]["fine_tuning_completed"] = True
    payload["stage_boundary"]["production_approved"] = True
    payload["stage_boundary"]["current_delivery_blocker"] = True
    payload["stage_boundary"]["autonomous_llm_agent_claim_allowed"] = True
    payload.pop("overall_status", None)

    invariants = validate_qlora_next_stage(payload)

    assert invariants["status"] == "FAIL"
    assert "qlora_fine_tuning_must_not_be_marked_completed" in invariants["violations"]
    assert "qlora_must_not_be_marked_production_approved" in invariants["violations"]
    assert "qlora_next_stage_must_not_block_current_delivery" in invariants["violations"]
    assert "qlora_plan_must_block_autonomous_llm_claims" in invariants["violations"]


def test_qlora_next_stage_endpoint_returns_contract() -> None:
    from poker_agent.service import qlora_next_stage_json

    payload = qlora_next_stage_json()

    assert payload["overall_status"] == "PASS"
    assert payload["stage_boundary"]["stage_status"] == "NEXT_STAGE_IMPROVEMENT"
    assert payload["stage_boundary"]["fine_tuning_completed"] is False
    assert payload["stage_boundary"]["production_approved"] is False
