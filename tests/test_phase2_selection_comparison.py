from __future__ import annotations

import json
from pathlib import Path

from poker_agent.phase2_selection_comparison import (
    COMMON_HOLDOUT_ID,
    COMMON_SIMULATION_ID,
    REQUIRED_CANDIDATES,
    build_phase2_selection_comparison,
    validate_phase2_selection_comparison,
    write_phase2_selection_comparison,
)


def _write_minimal_reports(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir()
    (reports / "llm_decision_gate.json").write_text(
        json.dumps(
            {
                "selection_is_provisional": True,
                "benchmark_metrics": {"accuracy": 0.37, "macro_f1": 0.14},
            }
        ),
        encoding="utf-8",
    )
    (reports / "llm_architecture_comparison.json").write_text(
        json.dumps(
            {
                "recommended_architecture": "candidate_ranker",
                "production_approved": False,
                "best_accuracy": 0.375,
                "best_macro_f1": 0.1406,
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_gate.json").write_text(
        json.dumps({"accuracy": 0.64, "macro_f1": 0.47, "balanced_accuracy": 0.49, "ece": 0.16}),
        encoding="utf-8",
    )
    (reports / "raw_model_status.json").write_text(
        json.dumps({"raw_supervised_model": {"runtime_status": "LOADABLE"}}),
        encoding="utf-8",
    )
    (reports / "deployed_strategy_gate.json").write_text(
        json.dumps({"status": "PASS", "action_distribution_js": 0.0026}),
        encoding="utf-8",
    )
    (reports / "production_self_play.json").write_text(
        json.dumps({"status": "PASS", "mean_win_rate": 0.55, "expected_value": 0.12}),
        encoding="utf-8",
    )


def test_phase2_selection_requires_common_holdout_and_simulation(tmp_path: Path) -> None:
    _write_minimal_reports(tmp_path)

    payload = build_phase2_selection_comparison(tmp_path)
    gate = payload["comparison_gate"]

    assert payload["overall_status"] == "PASS"
    assert set(payload["required_candidates"]) == set(REQUIRED_CANDIDATES)
    assert payload["common_holdout_contract"]["id"] == COMMON_HOLDOUT_ID
    assert payload["common_simulation_contract"]["id"] == COMMON_SIMULATION_ID
    assert gate["selected_for_current_delivery"] == "routed_policy_bundle"
    assert gate["final_selected_architecture"] is None
    assert gate["final_selection_claim_allowed"] is False
    assert gate["best_approach_claim_allowed"] is False
    assert gate["best_approach_claim_state"] == "BLOCKED_PENDING_FULL_COMMON_CONDITION_EVALUATION"
    assert gate["current_delivery_blocker"] is False
    assert gate["model_quality_risk"] is True
    assert gate["all_candidates_compared_on_common_holdout"] is False
    assert gate["all_candidates_compared_in_common_simulation"] is False
    assert gate["all_candidate_metric_bundles_complete"] is False
    assert "future_rl_agent" in gate["missing_common_holdout_candidates"]
    assert "future_rl_agent" in gate["missing_common_simulation_candidates"]
    assert "future_rl_agent" in gate["missing_metric_bundle_candidates"]
    assert "future_rl_agent" in gate["selection_ineligible_candidates"]
    assert set(gate["candidate_evidence_matrix"]) == set(REQUIRED_CANDIDATES)
    assert gate["candidate_evidence_matrix"]["future_rl_agent"]["selection_eligible"] is False
    assert "missing_required_metric_bundle" in gate["candidate_evidence_matrix"]["future_rl_agent"]["blocking_reasons"]
    assert payload["candidates"]["future_rl_agent"]["implementation_status"] == "NOT_AVAILABLE_YET"
    assert payload["candidates"]["routed_policy_bundle"]["selected_for_current_delivery"] is True
    final_result = payload["final_contract_result"]
    assert final_result["phase2_status"] == "PASS"
    assert final_result["strict_comparison_mechanism_ready"] is True
    assert final_result["current_delivery_stack"] == "routed_policy_bundle"
    assert final_result["final_selection_claim_allowed"] is False
    assert final_result["best_approach_claim_allowed"] is False
    assert final_result["all_candidate_metric_bundles_complete"] is False
    assert final_result["final_winner_claim_state"] == "BLOCKED_PENDING_COMMON_HOLDOUT_AND_SIMULATION"


def test_phase2_selection_blocks_false_final_winner_claim(tmp_path: Path) -> None:
    _write_minimal_reports(tmp_path)
    payload = build_phase2_selection_comparison(tmp_path)
    gate = payload["comparison_gate"]
    gate["final_selection_claim_allowed"] = True
    gate["final_selected_architecture"] = "routed_policy_bundle"
    gate["all_candidates_compared_on_common_holdout"] = True
    gate["all_candidates_compared_in_common_simulation"] = True
    gate["all_candidate_metric_bundles_complete"] = True
    gate["missing_common_holdout_candidates"] = []
    gate["missing_common_simulation_candidates"] = []
    gate["missing_metric_bundle_candidates"] = []
    gate["selection_ineligible_candidates"] = []
    gate["model_quality_risk"] = False
    gate["best_approach_claim_allowed"] = True
    gate["best_approach_claim_state"] = "ALLOWED"
    for evidence in gate["candidate_evidence_matrix"].values():
        evidence["holdout_complete"] = True
        evidence["simulation_complete"] = True
        evidence["missing_required_metrics"] = []
        evidence["metric_bundle_complete"] = True
        evidence["selection_eligible"] = True
        evidence["blocking_reasons"] = []
    payload["candidates"]["future_rl_agent"]["implementation_status"] = "AVAILABLE"
    payload["candidates"]["future_rl_agent"]["compared_in_common_simulation"] = True

    invariants = validate_phase2_selection_comparison(payload)

    assert invariants["status"] == "FAIL"
    assert "phase2_selection_final_claim_must_be_blocked_until_common_conditions" in invariants["violations"]
    assert "phase2_selection_final_architecture_must_not_be_selected_yet" in invariants["violations"]
    assert "phase2_selection_common_holdout_must_not_be_marked_complete_yet" in invariants["violations"]
    assert "phase2_selection_common_simulation_must_not_be_marked_complete_yet" in invariants["violations"]
    assert "phase2_selection_metric_bundles_must_not_be_marked_complete_yet" in invariants["violations"]
    assert "phase2_selection_future_rl_must_not_be_claimed_available" in invariants["violations"]
    assert "phase2_selection_future_rl_must_not_have_common_simulation_result_yet" in invariants["violations"]
    assert "phase2_selection_gap_must_remain_model_quality_risk" in invariants["violations"]
    assert "phase2_selection_best_approach_claim_must_be_blocked_until_common_conditions" in invariants["violations"]
    assert "phase2_selection_best_approach_claim_state_must_be_blocked" in invariants["violations"]
    assert "phase2_selection_evidence_metric_consistency:future_rl_agent" in invariants["violations"]


def test_phase2_selection_writer_creates_json_and_markdown(tmp_path: Path) -> None:
    _write_minimal_reports(tmp_path)
    json_out = tmp_path / "reports" / "phase2_selection_comparison.json"
    markdown_out = tmp_path / "reports" / "phase2_selection_comparison.md"

    payload = write_phase2_selection_comparison(tmp_path, json_out, markdown_out)

    assert payload["overall_status"] == "PASS"
    assert json.loads(json_out.read_text(encoding="utf-8"))["overall_status"] == "PASS"
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "Phase 2 status: `PASS`" in markdown
    assert "Final selection claim allowed: `False`" in markdown
    assert "Best approach claim allowed: `False`" in markdown
    assert "Metric bundles complete: `False`" in markdown
    assert "Final winner claim state: `BLOCKED_PENDING_COMMON_HOLDOUT_AND_SIMULATION`" in markdown
