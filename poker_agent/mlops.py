from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MLOPS_CONTRACT_VERSION = "mlops_contract.v1"
EXPERIMENT_TRACKING_BACKEND = "local_jsonl"
MODEL_REGISTRY_SCHEMA_VERSION = "model_registry.v1"
DATASET_VERSION_SCHEMA_VERSION = "dataset_version.v1"
DOCKER_VERSION_SCHEMA_VERSION = "docker_image.v1"
CI_SMOKE_SCHEMA_VERSION = "ci_smoke.v1"

DEFAULT_MLOPS_PATHS = {
    "experiment_log": "reports/experiments.jsonl",
    "model_registry": "reports/model_registry.json",
    "dataset_log": "reports/dataset_versions.jsonl",
    "mlops_report": "reports/mlops_contract.json",
    "ci_workflow": ".github/workflows/smoke.yml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(encoded)


def file_records(path: Path, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or path.parent
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    else:
        return []

    records: list[dict[str, Any]] = []
    for file_path in files:
        if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo", ".pyd"}:
            continue
        records.append(
            {
                "path": file_path.relative_to(root).as_posix() if file_path.is_relative_to(root) else str(file_path),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return records


def dataset_version_manifest(
    dataset_path: Path,
    *,
    name: str = "poker_dataset",
    version: str | None = None,
    schema_contract: str = "poker_csv.v1",
    split_policy: str = "source_player_time_holdout",
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    records = file_records(dataset_path, dataset_path if dataset_path.is_dir() else dataset_path.parent)
    fingerprint_payload = {
        "name": name,
        "schema_contract": schema_contract,
        "split_policy": split_policy,
        "files": records,
    }
    fingerprint = stable_digest(fingerprint_payload)
    return {
        "schema_version": DATASET_VERSION_SCHEMA_VERSION,
        "name": name,
        "version": version or f"dataset-{fingerprint[:12]}",
        "path": str(dataset_path),
        "created_at": utc_now(),
        "schema_contract": schema_contract,
        "split_policy": split_policy,
        "fingerprint": fingerprint,
        "file_count": len(records),
        "total_bytes": sum(int(item["size_bytes"]) for item in records),
        "files": records,
    }


def experiment_run_manifest(
    *,
    experiment_name: str,
    command: list[str],
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    dataset_version: str | None = None,
    model_version: str | None = None,
    status: str = "PASS",
    seed: int | str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    parameters = parameters or {}
    metrics = metrics or {}
    artifacts = artifacts or []
    identity = {
        "experiment_name": experiment_name,
        "command": command,
        "parameters": parameters,
        "dataset_version": dataset_version,
        "model_version": model_version,
        "seed": seed,
    }
    return {
        "schema_version": MLOPS_CONTRACT_VERSION,
        "tracking_backend": EXPERIMENT_TRACKING_BACKEND,
        "run_id": run_id or f"run-{stable_digest(identity)[:16]}",
        "experiment_name": experiment_name,
        "created_at": utc_now(),
        "status": status,
        "seed": str(seed) if seed is not None else None,
        "command": command,
        "parameters": parameters,
        "metrics": metrics,
        "dataset_version": dataset_version,
        "model_version": model_version,
        "artifacts": artifacts,
    }


def model_registry_entry(
    model_path: Path,
    *,
    model_name: str = "poker_policy",
    model_version: str | None = None,
    run_id: str,
    dataset_version: str,
    metrics: dict[str, Any] | None = None,
    stage: str = "candidate",
    docker_image: str | None = None,
    api_contract_version: str = "poker-decision-agent-api-v1",
) -> dict[str, Any]:
    model_path = model_path.resolve()
    model_hash = sha256_file(model_path) if model_path.is_file() else stable_digest({"missing": str(model_path)})
    version = model_version or f"model-{model_hash[:12]}"
    return {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "model_name": model_name,
        "model_version": version,
        "stage": stage,
        "registered_at": utc_now(),
        "artifact": {
            "path": str(model_path),
            "exists": model_path.exists(),
            "size_bytes": model_path.stat().st_size if model_path.is_file() else 0,
            "sha256": model_hash,
        },
        "run_id": run_id,
        "dataset_version": dataset_version,
        "metrics": metrics or {},
        "api_contract_version": api_contract_version,
        "docker_image": docker_image,
    }


def empty_model_registry() -> dict[str, Any]:
    return {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "models": {},
    }


def update_model_registry(registry_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = empty_model_registry()

    models = registry.setdefault("models", {})
    model_name = str(entry["model_name"])
    version = str(entry["model_version"])
    model_record = models.setdefault(model_name, {"versions": {}, "latest_version": None})
    model_record.setdefault("versions", {})[version] = entry
    model_record["latest_version"] = version
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return registry


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def docker_image_metadata(
    *,
    image_name: str = "poker-decision-agent",
    version: str = "0.1.0",
    dockerfile: Path | str = "Dockerfile",
    registry: str = "local",
) -> dict[str, Any]:
    dockerfile_path = Path(dockerfile)
    dockerfile_hash = sha256_file(dockerfile_path) if dockerfile_path.exists() else ""
    tag = f"{image_name}:{version}"
    return {
        "schema_version": DOCKER_VERSION_SCHEMA_VERSION,
        "registry": registry,
        "image": image_name,
        "version": version,
        "tag": tag,
        "build_args": {
            "APP_VERSION": version,
            "VCS_REF": "local",
            "BUILD_DATE": "unknown",
        },
        "labels": {
            "org.opencontainers.image.title": image_name,
            "org.opencontainers.image.version": version,
            "org.opencontainers.image.revision": "local",
        },
        "dockerfile": {
            "path": str(dockerfile_path),
            "sha256": dockerfile_hash,
        },
    }


def ci_smoke_contract() -> dict[str, Any]:
    return {
        "schema_version": CI_SMOKE_SCHEMA_VERSION,
        "workflow": DEFAULT_MLOPS_PATHS["ci_workflow"],
        "required_commands": [
            "python scripts/check_mlops_contract.py --smoke",
            "python scripts/check_monitoring_contract.py --smoke",
            "python scripts/check_security_contract.py --smoke",
            "python -B tests/test_mlops_contract.py",
        ],
        "required_checks": [
            "mlops_contract",
            "deployment_api_contract",
            "dataset_schema_contract",
            "data_validation_contract",
            "acceptance_criteria_contract",
        ],
    }


def describe_mlops_contract() -> dict[str, Any]:
    return {
        "schema_version": MLOPS_CONTRACT_VERSION,
        "experiment_tracking": {
            "backend": EXPERIMENT_TRACKING_BACKEND,
            "log_path": DEFAULT_MLOPS_PATHS["experiment_log"],
            "required_fields": ["run_id", "experiment_name", "command", "parameters", "metrics", "artifacts"],
        },
        "model_registry": {
            "path": DEFAULT_MLOPS_PATHS["model_registry"],
            "required_fields": ["model_name", "model_version", "stage", "artifact", "run_id", "dataset_version"],
            "stages": ["candidate", "staging", "production", "archived"],
        },
        "dataset_versioning": {
            "log_path": DEFAULT_MLOPS_PATHS["dataset_log"],
            "required_fields": ["version", "fingerprint", "schema_contract", "split_policy", "files"],
        },
        "docker_versioning": docker_image_metadata(),
        "ci_smoke": ci_smoke_contract(),
    }


def validate_mlops_contract(root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_files = [
        "poker_agent/mlops.py",
        "scripts/check_mlops_contract.py",
        "configs/mlops/local.yaml",
        "configs/experiments/mlops_smoke.yaml",
        DEFAULT_MLOPS_PATHS["ci_workflow"],
        "Dockerfile",
        "docker-compose.yml",
    ]
    for relative in expected_files:
        checks.append(
            {
                "name": f"file:{relative}",
                "passed": (root / relative).exists(),
                "detail": relative,
            }
        )

    dockerfile_text = (root / "Dockerfile").read_text(encoding="utf-8") if (root / "Dockerfile").exists() else ""
    for token in ("ARG APP_VERSION", "org.opencontainers.image.version", "POKER_AGENT_VERSION"):
        checks.append({"name": f"docker:{token}", "passed": token in dockerfile_text, "detail": token})

    compose_text = (root / "docker-compose.yml").read_text(encoding="utf-8") if (root / "docker-compose.yml").exists() else ""
    checks.append(
        {
            "name": "docker_compose:pinned_version",
            "passed": "poker-decision-agent:${POKER_AGENT_VERSION:-" in compose_text,
            "detail": "compose image tag must be versioned instead of latest-only",
        }
    )

    workflow = root / DEFAULT_MLOPS_PATHS["ci_workflow"]
    workflow_text = workflow.read_text(encoding="utf-8") if workflow.exists() else ""
    for token in ("check_mlops_contract.py", "--smoke", "tests/test_mlops_contract.py"):
        checks.append({"name": f"ci:{token}", "passed": token in workflow_text, "detail": token})
    return checks
