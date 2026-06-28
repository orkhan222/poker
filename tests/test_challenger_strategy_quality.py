from __future__ import annotations

import json
from pathlib import Path

from poker_agent.challenger_strategy_quality import (
    build_challenger_strategy_quality,
    validate_challenger_strategy_quality,
)


def _write_reports(reports: Path, challenger_gate: str = "FAIL", raw_gate: str = "FAIL") -> None:
    reports.mkdir()
    (reports / "raw_model_status.json").write_text(
        json.dumps(
            {
                "raw_supervised_model": {
                    "runtime_status": "LOADABLE",
                    "standalone_status": "NOT_STANDALONE_APPROVED",
                    "quality_gate_status": raw_gate,
                    "approved_as_standalone_policy": False,
                },
                "release_boundary": {"component_risk": True, "production_blocker": False},
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_gate.json").write_text(json.dumps({"status": raw_gate}), encoding="utf-8")
    (reports / "deployed_strategy_gate.json").write_text(
        json.dumps({"status": "PASS", "strategy_policy_status": "APPROVED"}),
        encoding="utf-8",
    )
    (reports / "raw_model_challenger.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "standalone_status": "NOT_STANDALONE_APPROVED",
                "approved_as_standalone_policy": False,
                "split": {"split_type": "stratified_hand_group_holdout"},
                "approval_boundary": {"existing_service_delivery_affected": False},
                "best_candidate": {
                    "name": "extra_trees_sqrt_balanced_full",
                    "status": "TRAINED",
                    "artifact_path": "models/raw_challengers/extra_trees_sqrt_balanced_full.joblib",
                    "gate": {
                        "status": challenger_gate,
                        "passed_gates": 2,
                        "total_gates": 7,
                        "failed_gates": ["macro_f1", "calibration"],
                    },
                    "valid_metrics": {
                        "accuracy": 0.70,
                        "macro_f1": 0.48,
                        "balanced_accuracy": 0.50,
                        "ece_10": 0.16,
                        "lift_vs_majority": 0.02,
                        "majority_baseline_accuracy": 0.68,
                    },
                    "slice_metrics": {
                        "observed_hole_cards": {"macro_f1": 0.43},
                        "facing_bet": {"macro_f1": 0.44},
                    },
                },
                "invariants": {"status": "PASS", "violations": []},
            }
        ),
        encoding="utf-8",
    )


def test_challenger_strategy_quality_blocks_final_claim_until_gates_pass(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")

    payload = build_challenger_strategy_quality(tmp_path)

    assert payload["overall_status"] == "PASS"
    boundary = payload["strategy_quality_boundary"]
    assert boundary["challenger_required_before_final_claim"] is True
    assert boundary["challenger_trained"] is True
    assert boundary["challenger_compared_to_raw_model"] is True
    assert boundary["final_production_strategy_quality_claim_allowed"] is False
    assert boundary["current_delivery_blocker"] is False
    assert boundary["deployed_strategy_stack_affected"] is False
    assert "A failing challenger can be promoted as the production policy." in payload["blocked_claims"]


def test_challenger_strategy_quality_rejects_false_final_approval(tmp_path: Path) -> None:
    _write_reports(tmp_path / "reports")
    payload = build_challenger_strategy_quality(tmp_path)
    payload["strategy_quality_boundary"]["final_production_strategy_quality_claim_allowed"] = True
    payload["strategy_quality_boundary"]["claim_blocked_until_challenger_passes"] = False
    payload["strategy_quality_boundary"]["status"] = "APPROVED_AFTER_CHALLENGER_GATE"
    payload.pop("overall_status", None)

    invariants = validate_challenger_strategy_quality(payload)

    assert invariants["status"] == "FAIL"
    assert "final_strategy_quality_requires_raw_gate_pass" in invariants["violations"]
    assert "final_strategy_quality_requires_challenger_gate_pass" in invariants["violations"]


def test_challenger_strategy_quality_endpoint_returns_contract() -> None:
    from poker_agent.service import challenger_strategy_quality_json

    payload = challenger_strategy_quality_json()

    assert payload["overall_status"] == "PASS"
    boundary = payload["strategy_quality_boundary"]
    assert boundary["challenger_required_before_final_claim"] is True
    assert boundary["final_production_strategy_quality_claim_allowed"] is False
    assert boundary["current_delivery_blocker"] is False
