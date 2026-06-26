from __future__ import annotations

import json
from pathlib import Path

from poker_agent.client_gpu_training_response import build_client_gpu_training_response


def test_client_gpu_response_contains_a100_h100_reply(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "today_acceptance_training.json").write_text(
        json.dumps(
            {
                "selected_architecture": "routed_policy_bundle",
                "profile": "today_acceptance_training",
                "training_status": "PASS",
                "delivery_status": "READY_FOR_CURRENT_DELIVERY",
                "model_out": "models/poker_policy_bundle.joblib",
                "valid_metrics": {"accuracy": 0.59, "macro_f1": 0.42, "balanced_accuracy": 0.45},
            }
        ),
        encoding="utf-8",
    )
    (reports / "training_cluster_requirements.json").write_text(
        json.dumps(
            {
                "run_profile": "immediate_delivery",
                "estimate": {
                    "status": "READY_FOR_IMMEDIATE_DELIVERY_VALIDATION",
                    "estimated_hours": 2.0,
                    "estimated_days": 0.08,
                    "confidence": "HIGH",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_client_gpu_training_response(tmp_path)

    assert "dedicated A100 or H100" in payload["recommended_reply"]
    assert payload["current_delivery_training"]["selected_architecture"] == "routed_policy_bundle"
    assert payload["current_delivery_training"]["training_status"] == "PASS"
    assert payload["gpu_boundary"]["full_multi_agent_training"] == "separate production-hardening phase"
    assert "Do not represent" in payload["gpu_boundary"]["do_not_claim"]

def test_client_gpu_training_response_endpoint_returns_client_ready_payload() -> None:
    from poker_agent.service import client_gpu_training_response_json

    payload = client_gpu_training_response_json()

    assert "dedicated A100 or H100" in payload["recommended_reply"]
    assert payload["current_delivery_training"]["training_status"] == "PASS"
    assert payload["gpu_boundary"]["full_multi_agent_training"] == "separate production-hardening phase"
