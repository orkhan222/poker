from __future__ import annotations

import json
from pathlib import Path

from poker_agent.human_likeness_claim_gate import (
    HUMAN_LIKENESS_CLAIM_DECISION,
    build_human_likeness_claim_gate,
    validate_human_likeness_claim_gate,
    write_human_likeness_claim_gate,
)


def _write_evidence(reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "human_likeness_evidence.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "invariants": {"status": "PASS", "violations": []},
                "boundary": "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF",
                "current_scope_action_distribution_passed": True,
                "final_human_likeness_claim_allowed": False,
                "behavior_dimensions": {
                    "action_distribution": {
                        "current_status": "PASS",
                        "current_scope_metric": "js_divergence",
                        "current_scope_value": 0.0026,
                        "remaining_requirement": "Revalidate on larger clean real gameplay data.",
                    },
                    "bet_sizing": {
                        "current_status": "PASS",
                        "current_scope_metric": "bet_size_mae",
                        "current_scope_value": None,
                        "remaining_requirement": "Collect reviewed real-player bet-size labels.",
                    },
                    "timing": {
                        "current_status": "PASS",
                        "current_scope_metric": "decision_time_mae",
                        "current_scope_value": None,
                        "remaining_requirement": "Collect reviewed real human timing labels.",
                    },
                    "position_based_behavior": {
                        "current_status": "REQUIRES_SLICE_REVALIDATION",
                        "current_scope_metric": "position_slice_similarity",
                        "current_scope_value": None,
                        "remaining_requirement": "Validate position behavior slices.",
                    },
                    "street_level_strategy": {
                        "current_status": "REQUIRES_SLICE_REVALIDATION",
                        "current_scope_metric": "street_slice_similarity",
                        "current_scope_value": None,
                        "remaining_requirement": "Validate street-level strategy slices.",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_human_likeness_claim_gate_blocks_distribution_only_full_claim(tmp_path: Path) -> None:
    _write_evidence(tmp_path / "reports")

    payload = build_human_likeness_claim_gate(tmp_path)

    assert payload["overall_status"] == "PASS"
    assert payload["decision"] == HUMAN_LIKENESS_CLAIM_DECISION
    assert payload["claim_allowed"] is False
    assert payload["human_likeness_fully_proven"] is False
    assert payload["action_distribution_only_proof_rejected"] is True
    assert payload["current_scope_action_distribution_passed"] is True
    assert payload["current_delivery_blocker"] is False
    assert payload["model_quality_risk"] is True
    assert payload["evidence_requirements"]["action_distribution"]["currently_sufficient_for_final_claim"] is False
    assert payload["evidence_requirements"]["bet_sizing"]["currently_sufficient_for_final_claim"] is False
    assert payload["evidence_requirements"]["timing"]["currently_sufficient_for_final_claim"] is False


def test_human_likeness_claim_gate_rejects_false_approval(tmp_path: Path) -> None:
    _write_evidence(tmp_path / "reports")
    payload = build_human_likeness_claim_gate(tmp_path)
    payload["decision"] = "APPROVED"
    payload["claim_allowed"] = True
    payload["human_likeness_fully_proven"] = True
    payload["action_distribution_only_proof_rejected"] = False
    payload["evidence_requirements"]["bet_sizing"]["currently_sufficient_for_final_claim"] = True

    invariants = validate_human_likeness_claim_gate(payload)

    assert invariants["status"] == "FAIL"
    assert "full_human_likeness_decision_must_remain_blocked" in invariants["violations"]
    assert "full_human_likeness_claim_must_not_be_allowed" in invariants["violations"]
    assert "human_likeness_must_not_be_marked_fully_proven" in invariants["violations"]
    assert "action_distribution_only_proof_must_be_rejected" in invariants["violations"]
    assert "human_likeness_claim_dimension_must_not_be_currently_sufficient:bet_sizing" in invariants["violations"]


def test_write_human_likeness_claim_gate_outputs_reports(tmp_path: Path) -> None:
    _write_evidence(tmp_path / "reports")

    payload = write_human_likeness_claim_gate(
        tmp_path,
        tmp_path / "reports" / "human_likeness_claim_gate.json",
        tmp_path / "reports" / "human_likeness_claim_gate.md",
    )

    assert payload["overall_status"] == "PASS"
    assert (tmp_path / "reports" / "human_likeness_claim_gate.json").exists()
    assert "Full human-likeness is not approved" in (
        tmp_path / "reports" / "human_likeness_claim_gate.md"
    ).read_text(encoding="utf-8")
