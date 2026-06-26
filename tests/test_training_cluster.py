from __future__ import annotations

from pathlib import Path

from poker_agent.training_cluster import (
    ClusterSpec,
    build_training_cluster_requirements,
    estimate_multi_agent_training,
)


def recommended_a100() -> ClusterSpec:
    return ClusterSpec(
        gpu_type="A100",
        gpu_count=1,
        vram_gb_per_gpu=80,
        cpu_cores=32,
        system_ram_gb=256,
        storage_gb=1000,
        interconnect="PCIe",
        dedicated_or_shared="dedicated",
    )


def recommended_h100() -> ClusterSpec:
    return ClusterSpec(
        gpu_type="NVIDIA H100",
        gpu_count=1,
        vram_gb_per_gpu=80,
        cpu_cores=32,
        system_ram_gb=256,
        storage_gb=1000,
        interconnect="NVLink",
        dedicated_or_shared="dedicated",
    )


def test_unknown_cluster_requests_required_fields() -> None:
    payload = build_training_cluster_requirements(Path("."))

    assert payload["run_profile"] == "immediate_delivery"
    assert payload["estimate"]["status"] == "PENDING_CLUSTER_CONFIRMATION"
    assert payload["estimate"]["estimated_hours"] is None
    assert "gpu_type" in payload["requested_fields"]
    assert "dedicated_or_shared" in payload["requested_fields"]


def test_single_a100_immediate_delivery_finishes_same_day() -> None:
    estimate = estimate_multi_agent_training(recommended_a100(), run_profile="immediate_delivery")

    assert estimate["status"] == "READY_FOR_IMMEDIATE_DELIVERY_VALIDATION"
    assert estimate["estimated_hours"] == 3.0
    assert estimate["estimated_days"] < 1.0
    assert estimate["confidence"] == "HIGH"


def test_single_h100_immediate_delivery_finishes_same_day() -> None:
    estimate = estimate_multi_agent_training(recommended_h100(), run_profile="immediate_delivery")

    assert estimate["status"] == "READY_FOR_IMMEDIATE_DELIVERY_VALIDATION"
    assert estimate["estimated_hours"] == 2.0
    assert estimate["estimated_days"] < 1.0


def test_full_multi_agent_training_remains_separate_reference() -> None:
    estimate = estimate_multi_agent_training(recommended_h100(), run_profile="full_multi_agent_training")

    assert estimate["status"] == "RECOMMENDED"
    assert estimate["estimated_days"] == 5.0
    assert estimate["estimated_hours"] == 120.0


def test_shared_cluster_marks_scheduling_risk() -> None:
    estimate = estimate_multi_agent_training(
        ClusterSpec(
            gpu_type="A100",
            gpu_count=1,
            vram_gb_per_gpu=40,
            cpu_cores=32,
            system_ram_gb=128,
            storage_gb=1000,
            interconnect="PCIe",
            dedicated_or_shared="shared",
        ),
        run_profile="immediate_delivery",
    )

    assert estimate["estimated_hours"] > 3.0
    assert any("Shared cluster" in risk for risk in estimate["risks"])


def test_training_cluster_endpoint_accepts_a100_query() -> None:
    from poker_agent.service import training_cluster_requirements_json

    payload = training_cluster_requirements_json(
        run_profile="immediate_delivery",
        gpu_type="A100",
        gpu_count=1,
        vram_gb_per_gpu=80,
        cpu_cores=32,
        system_ram_gb=256,
        storage_gb=1000,
        interconnect="PCIe",
        dedicated_or_shared="dedicated",
    )

    assert payload["estimate"]["status"] == "READY_FOR_IMMEDIATE_DELIVERY_VALIDATION"
    assert payload["estimate"]["estimated_hours"] == 3.0
    assert payload["provided_cluster"]["gpu_type"] == "A100"