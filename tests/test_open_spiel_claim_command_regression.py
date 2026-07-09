from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_direct_open_spiel_claim_command_fails_without_phase1_artifacts(tmp_path: Path) -> None:
    missing_agent_a = tmp_path / "missing_phase1_llm_policy_a.joblib"
    missing_agent_b = tmp_path / "missing_phase1_llm_policy_b.joblib"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_phase3_open_spiel_arena.py"),
        "--project-root",
        str(ROOT),
        "--claim-mode",
        "--run-if-available",
        "--phase1-adapters-ready",
        "--agent-a-model-path",
        str(missing_agent_a),
        "--agent-b-model-path",
        str(missing_agent_b),
        "--episodes",
        "5000",
        "--independent-seed-count",
        "5",
        "--policy-update-training-completed",
        "--policy-update-algorithm",
        "PPO",
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode != 0
    assert "OpenSpiel/RL claim mode is not eligible" in output
    assert "claim_mode_missing_agent_a_model_path" in output
    assert "claim_mode_missing_agent_b_model_path" in output
