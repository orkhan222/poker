from __future__ import annotations

import json
import tempfile
from pathlib import Path

from poker_agent.final_model_selection import (
    describe_final_model_selection,
    final_model_selection_status,
    validate_final_model_selection,
    write_final_model_selection_reports,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qwenpoker_checkpoint_40960_is_selected_from_balanced_benchmark() -> None:
    selection = describe_final_model_selection()

    assert selection["selected_model"]["model_family"] == "QwenPoker"
    assert selection["selected_model"]["checkpoint"] == "checkpoint_40960"
    assert selection["benchmark"]["balance"]["total_hands"] == 5000
    assert selection["benchmark"]["balance"]["hands_per_seat"] == 2500
    assert selection["benchmark"]["balance"]["seed"] == 20260714
    assert selection["metrics"]["win_rate"] == 0.6448
    assert selection["metrics"]["bb_per_100"] == 365.29


def test_final_model_selection_gates_require_positive_ci_and_position_profitability() -> None:
    status = final_model_selection_status()

    assert status["status"] == "PASS"
    checks = {item["name"]: item for item in status["checks"]}
    assert checks["returns_ci_95_positive"]["passed"] is True
    assert checks["both_positions_profitable"]["passed"] is True
    assert checks["seat_balance"]["passed"] is True
    assert checks["opponent_suite_weights"]["passed"] is True


def test_final_model_selection_validation_fails_when_return_ci_is_not_positive() -> None:
    selection = describe_final_model_selection()
    selection["metrics"]["returns_ci_95"]["is_entirely_positive"] = False
    checks = {item["name"]: item for item in validate_final_model_selection(selection)}

    assert checks["returns_ci_95_positive"]["passed"] is False


def test_final_model_selection_reports_are_machine_readable() -> None:
    with tempfile.TemporaryDirectory() as raw_temp:
        root = Path(raw_temp)
        (root / "reports").mkdir()
        (root / "docs").mkdir()
        outputs = write_final_model_selection_reports(root)
        report = json.loads(outputs["json"].read_text(encoding="utf-8"))
        docs = outputs["docs"].read_text(encoding="utf-8")

    assert report["status"] == "PASS"
    assert report["selected_model"]["checkpoint"] == "checkpoint_40960"
    assert "QwenPoker checkpoint_40960" in docs
