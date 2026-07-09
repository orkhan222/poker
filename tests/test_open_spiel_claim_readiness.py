from __future__ import annotations

import json

from poker_agent.open_spiel_claim_readiness import (
    build_open_spiel_claim_readiness,
    validate_open_spiel_claim_readiness,
    write_open_spiel_claim_readiness,
)


def test_claim_readiness_blocks_missing_evidence_without_delivery_blocker(tmp_path):
    payload = build_open_spiel_claim_readiness(
        tmp_path,
        pyspiel_runtime_available=False,
        policy_update_training_completed=False,
    )

    assert payload["overall_status"] == "PASS"
    assert payload["claim_ready"] is False
    assert payload["current_delivery_blocker"] is False
    assert payload["model_quality_risk"] is True
    assert "pyspiel_runtime" in payload["missing_requirements"]
    assert "two_phase1_trained_policy_artifacts" in payload["missing_requirements"]
    assert "ppo_or_equivalent_policy_update_training" in payload["missing_requirements"]
    assert "--claim-mode" in payload["claim_command"]
    assert payload["invariants"]["status"] == "PASS"


def test_claim_readiness_allows_complete_evidence(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "phase1_llm_policy_a.joblib").write_text("artifact-a", encoding="utf-8")
    (models / "phase1_llm_policy_b.joblib").write_text("artifact-b", encoding="utf-8")

    payload = build_open_spiel_claim_readiness(
        tmp_path,
        pyspiel_runtime_available=True,
        policy_update_training_completed=True,
        policy_update_algorithm="PPO",
        episodes=5000,
        independent_seed_count=5,
    )

    assert payload["overall_status"] == "PASS"
    assert payload["claim_ready"] is True
    assert payload["model_quality_risk"] is False
    assert payload["missing_requirements"] == []
    assert payload["phase1_policy_artifacts"]["existing_count"] == 2


def test_claim_readiness_writes_reports(tmp_path):
    out = tmp_path / "reports" / "open_spiel_claim_readiness.json"
    markdown = tmp_path / "reports" / "open_spiel_claim_readiness.md"

    payload = write_open_spiel_claim_readiness(
        tmp_path,
        out,
        markdown,
        pyspiel_runtime_available=False,
    )

    assert out.exists()
    assert markdown.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["gate_name"] == "open_spiel_claim_readiness"
    assert "--claim-mode" in markdown.read_text(encoding="utf-8")
    assert payload["overall_status"] == "PASS"


def test_claim_readiness_rejects_false_ready_payload(tmp_path):
    payload = build_open_spiel_claim_readiness(tmp_path, pyspiel_runtime_available=False)
    payload["claim_ready"] = True
    payload["model_quality_risk"] = False
    payload["missing_requirements"] = []

    validation = validate_open_spiel_claim_readiness(payload)

    assert validation["status"] == "FAIL"
    assert "claim_ready_must_match_all_required_evidence" in validation["violations"]
