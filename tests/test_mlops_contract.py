from __future__ import annotations

import tempfile
from pathlib import Path

from poker_agent.mlops import (
    DATASET_VERSION_SCHEMA_VERSION,
    DOCKER_VERSION_SCHEMA_VERSION,
    EXPERIMENT_TRACKING_BACKEND,
    MODEL_REGISTRY_SCHEMA_VERSION,
    ci_smoke_contract,
    dataset_version_manifest,
    describe_mlops_contract,
    docker_image_metadata,
    experiment_run_manifest,
    model_registry_entry,
    update_model_registry,
    validate_mlops_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_version_manifest_is_hash_based_and_machine_readable() -> None:
    manifest = dataset_version_manifest(ROOT / "sample_poker_log.jsonl")

    assert manifest["schema_version"] == DATASET_VERSION_SCHEMA_VERSION
    assert manifest["version"].startswith("dataset-")
    assert manifest["fingerprint"]
    assert manifest["file_count"] == 1
    assert manifest["schema_contract"] == "poker_csv.v1"
    assert manifest["split_policy"] == "source_player_time_holdout"


def test_experiment_tracking_manifest_has_run_identity_and_artifacts() -> None:
    run = experiment_run_manifest(
        experiment_name="mlops_smoke",
        command=["python", "scripts/check_mlops_contract.py", "--smoke"],
        parameters={"seed": 20260713},
        metrics={"smoke_contract": 1.0},
        artifacts=[{"type": "report", "path": "reports/mlops_contract.json"}],
        dataset_version="dataset-test",
        model_version="model-test",
        seed=20260713,
    )

    assert run["tracking_backend"] == EXPERIMENT_TRACKING_BACKEND
    assert run["run_id"].startswith("run-")
    assert run["dataset_version"] == "dataset-test"
    assert run["model_version"] == "model-test"
    assert run["artifacts"][0]["type"] == "report"


def test_model_registry_records_version_stage_dataset_and_artifact_hash() -> None:
    with tempfile.TemporaryDirectory() as raw_temp:
        registry_path = Path(raw_temp) / "model_registry.json"
        entry = model_registry_entry(
            ROOT / "models" / "poker_policy.json",
            run_id="run-test",
            dataset_version="dataset-test",
            metrics={"macro_f1": 0.42},
            stage="candidate",
            docker_image="poker-decision-agent:0.1.0",
        )
        registry = update_model_registry(registry_path, entry)

    model = registry["models"]["poker_policy"]
    assert registry["schema_version"] == MODEL_REGISTRY_SCHEMA_VERSION
    assert model["latest_version"] == entry["model_version"]
    assert model["versions"][entry["model_version"]]["dataset_version"] == "dataset-test"
    assert model["versions"][entry["model_version"]]["artifact"]["sha256"]


def test_docker_and_ci_contracts_are_versioned() -> None:
    docker = docker_image_metadata(dockerfile=ROOT / "Dockerfile")
    ci = ci_smoke_contract()

    assert docker["schema_version"] == DOCKER_VERSION_SCHEMA_VERSION
    assert docker["tag"] == "poker-decision-agent:0.1.0"
    assert docker["labels"]["org.opencontainers.image.version"] == "0.1.0"
    assert ci["workflow"] == ".github/workflows/smoke.yml"
    assert "python scripts/check_mlops_contract.py --smoke" in ci["required_commands"]


def test_repo_mlops_contract_files_and_metadata_are_present() -> None:
    contract = describe_mlops_contract()
    checks = validate_mlops_contract(ROOT)
    failed = [item for item in checks if not item["passed"]]

    assert contract["experiment_tracking"]["backend"] == "local_jsonl"
    assert contract["model_registry"]["stages"] == ["candidate", "staging", "production", "archived"]
    assert not failed
