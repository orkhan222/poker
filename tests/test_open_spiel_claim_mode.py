from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_phase3_open_spiel_arena import validate_claim_mode_args


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "run_if_available": True,
        "phase1_adapters_ready": True,
        "agent_a_model_path": "models/phase1_llm_policy_a.joblib",
        "agent_b_model_path": "models/phase1_llm_policy_b.joblib",
        "episodes": 5000,
        "independent_seed_count": 5,
        "policy_update_training_completed": True,
        "policy_update_algorithm": "PPO",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_claim_mode_requires_real_run_profile(tmp_path: Path) -> None:
    violations = validate_claim_mode_args(
        _args(
            run_if_available=False,
            phase1_adapters_ready=False,
            episodes=256,
            independent_seed_count=1,
            policy_update_training_completed=False,
        ),
        tmp_path,
    )

    assert "claim_mode_requires_run_if_available" in violations
    assert "claim_mode_requires_phase1_adapters_ready" in violations
    assert "claim_mode_requires_5000_or_more_episodes" in violations
    assert "claim_mode_requires_5_or_more_independent_seeds" in violations
    assert "claim_mode_requires_policy_update_training_completed" in violations


def test_claim_mode_requires_two_existing_phase1_artifacts(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "phase1_llm_policy_a.joblib").write_bytes(b"adapter-a")

    violations = validate_claim_mode_args(_args(), tmp_path)

    assert "claim_mode_missing_agent_a_model_path" not in violations
    assert "claim_mode_missing_agent_b_model_path" in violations


def test_claim_mode_accepts_complete_claim_prerequisites(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "phase1_llm_policy_a.joblib").write_bytes(b"adapter-a")
    (models / "phase1_llm_policy_b.joblib").write_bytes(b"adapter-b")

    violations = validate_claim_mode_args(_args(), tmp_path)

    assert violations == []
