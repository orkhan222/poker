from __future__ import annotations

import json
from pathlib import Path

from poker_agent.multi_agent_training_status import (
    build_multi_agent_training_status,
    validate_multi_agent_training_status,
)


def test_multi_agent_training_status_preserves_delivery_boundary(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "today_acceptance_training.json").write_text(
        json.dumps(
            {
                "profile": "today_acceptance_training",
                "selected_architecture": "routed_policy_bundle",
                "training_status": "PASS",
                "delivery_status": "READY_FOR_CURRENT_DELIVERY",
                "valid_metrics": {
                    "accuracy": 0.61,
                    "macro_f1": 0.44,
                    "balanced_accuracy": 0.47,
                    "cross_entropy": 1.18,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "client_gpu_training_response.json").write_text(
        json.dumps(
            {
                "gpu_boundary": {
                    "full_multi_agent_training": "separate production-hardening phase",
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "training_cluster_requirements.json").write_text(
        json.dumps({"run_profile": "immediate_delivery"}),
        encoding="utf-8",
    )
    (reports / "production_self_play.json").write_text(
        json.dumps({"status": "PASS", "production_scale_status": "PASS", "paired_hands": 5000, "run_count": 20}),
        encoding="utf-8",
    )

    payload = build_multi_agent_training_status(tmp_path)
    boundary = payload["training_boundary"]

    assert payload["overall_status"] == "PASS"
    assert boundary["delivery_validation_status"] == "PASS"
    assert boundary["acceptance_training_sufficient_for_delivery"] is True
    assert boundary["full_production_scale_multi_agent_training_status"] == "NOT_COMPLETED"
    assert boundary["full_long_running_self_play_completed"] is False
    assert boundary["production_blocker"] is False
    validation_boundary = payload["validation_vs_training_boundary"]
    assert validation_boundary["acceptance_self_play_paired_hands"] == 5000
    assert validation_boundary["acceptance_self_play_is_delivery_validation"] is True
    assert validation_boundary["acceptance_self_play_counts_as_full_training"] is False
    hardening_plan = payload["hardening_training_plan"]
    assert hardening_plan["required_profile"] == "full_multi_agent_training"
    assert hardening_plan["required_gpu_class"] == "single dedicated NVIDIA A100 or H100"
    assert hardening_plan["minimum_independent_training_seeds"] >= 5
    assert hardening_plan["seed_stability_required"] is True
    assert hardening_plan["minimum_paired_hands"] > validation_boundary["acceptance_self_play_paired_hands"]
    assert hardening_plan["estimated_duration_days_single_a100_or_h100"] == 5
    assert payload["approval_boundary"]["full_training_claim_allowed"] is False


def test_multi_agent_training_status_blocks_false_full_completion(tmp_path: Path) -> None:
    payload = build_multi_agent_training_status(tmp_path)
    payload["training_boundary"]["full_production_scale_multi_agent_training_status"] = "COMPLETED"
    payload["training_boundary"]["full_long_running_self_play_completed"] = True
    payload["validation_vs_training_boundary"]["acceptance_self_play_counts_as_full_training"] = True
    payload["hardening_training_plan"]["completion_status"] = "COMPLETED"
    payload["approval_boundary"]["full_training_claim_allowed"] = True

    invariants = validate_multi_agent_training_status(payload)

    assert invariants["status"] == "FAIL"
    assert len(invariants["violations"]) >= 5


def test_multi_agent_training_status_blocks_weak_hardening_plan(tmp_path: Path) -> None:
    payload = build_multi_agent_training_status(tmp_path)
    payload["hardening_training_plan"]["required_gpu_class"] = "shared consumer GPU"
    payload["hardening_training_plan"]["minimum_independent_training_seeds"] = 1
    payload["hardening_training_plan"]["minimum_paired_hands"] = 5000
    payload["hardening_training_plan"]["estimated_duration_days_single_a100_or_h100"] = 1
    payload["hardening_training_plan"]["seed_stability_required"] = False

    invariants = validate_multi_agent_training_status(payload)

    assert invariants["status"] == "FAIL"
    assert "Full training must require a dedicated A100/H100-class GPU profile." in invariants["violations"]
    assert "Full training seed count is below the required minimum." in invariants["violations"]
    assert "Full training simulation volume is below the required minimum." in invariants["violations"]
    assert "Full training must require seed-stability analysis." in invariants["violations"]


def test_multi_agent_training_status_endpoint_returns_boundary() -> None:
    from poker_agent.service import multi_agent_training_status_json

    payload = multi_agent_training_status_json()
    boundary = payload["training_boundary"]

    assert payload["overall_status"] == "PASS"
    assert boundary["full_production_scale_multi_agent_training_status"] == "NOT_COMPLETED"
    assert payload["validation_vs_training_boundary"]["acceptance_self_play_counts_as_full_training"] is False
    assert payload["hardening_training_plan"]["required_gpu_class"] == "single dedicated NVIDIA A100 or H100"
    assert payload["approval_boundary"]["full_training_claim_allowed"] is False
