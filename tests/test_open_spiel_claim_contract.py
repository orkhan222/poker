from __future__ import annotations

import json
from pathlib import Path

from poker_agent.open_spiel_claim_contract import (
    build_open_spiel_claim_contract,
    validate_open_spiel_claim_contract,
    write_open_spiel_claim_contract,
)
from poker_agent.open_spiel_llm_arena import ArenaRunConfig, write_phase3_open_spiel_arena_report
from poker_agent.rl_training_evidence_gate import REQUIRED_RL_EVIDENCE


def _write_pending_phase3_report(root: Path) -> None:
    write_phase3_open_spiel_arena_report(
        root,
        root / "reports" / "phase3_open_spiel_arena.json",
        root / "reports" / "phase3_open_spiel_arena.md",
        config=ArenaRunConfig(
            episodes=256,
            independent_seed_count=1,
            phase1_adapters_ready=False,
            policy_update_training_completed=False,
        ),
        run_if_available=False,
    )


def test_open_spiel_claim_contract_blocks_win_rate_claim_until_training_proof_exists(tmp_path: Path) -> None:
    _write_pending_phase3_report(tmp_path)

    contract = build_open_spiel_claim_contract(tmp_path)

    assert contract["overall_status"] == "PASS"
    assert contract["code_status"] == "READY"
    assert contract["arena_code_ready"] is True
    assert contract["agent_only_table"] is True
    assert contract["self_play_win_rate_claim_allowed"] is False
    assert contract["current_delivery_blocker"] is False
    assert contract["model_quality_risk"] is True
    assert set(contract["required_evidence_before_self_play_claim"]) == set(REQUIRED_RL_EVIDENCE)
    assert contract["missing_requirements"] == [
        "real_open_spiel_runtime",
        "two_phase1_trained_policy_artifacts",
        "seed_stability",
        "long_run_training_volume",
        "policy_update_training",
    ]


def test_open_spiel_claim_contract_rejects_false_completed_claim(tmp_path: Path) -> None:
    _write_pending_phase3_report(tmp_path)
    contract = build_open_spiel_claim_contract(tmp_path)
    contract["self_play_win_rate_claim_allowed"] = True
    contract["training_proof_completed"] = True
    contract["model_quality_risk"] = False

    validation = validate_open_spiel_claim_contract(contract)

    assert validation["status"] == "FAIL"
    assert "self_play_win_rate_claim_requires_complete_training_proof" in validation["violations"]
    assert "training_proof_completed_requires_all_evidence" in validation["violations"]
    assert "incomplete_open_spiel_training_proof_must_remain_model_quality_risk" in validation["violations"]


def test_open_spiel_claim_contract_writes_json_and_markdown(tmp_path: Path) -> None:
    _write_pending_phase3_report(tmp_path)
    json_out = tmp_path / "reports" / "open_spiel_claim_contract.json"
    markdown_out = tmp_path / "reports" / "open_spiel_claim_contract.md"

    payload = write_open_spiel_claim_contract(tmp_path, json_out, markdown_out)

    assert payload["overall_status"] == "PASS"
    assert json.loads(json_out.read_text(encoding="utf-8"))["gate_name"] == "open_spiel_self_play_claim_contract"
    assert "Self-play win-rate claim allowed" in markdown_out.read_text(encoding="utf-8")
