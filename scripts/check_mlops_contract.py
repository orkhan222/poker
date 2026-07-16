from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.mlops import (
    append_jsonl,
    dataset_version_manifest,
    describe_mlops_contract,
    docker_image_metadata,
    experiment_run_manifest,
    model_registry_entry,
    update_model_registry,
    validate_mlops_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and optionally emit the local MLOps contract artifacts")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--dataset", default=Path("sample_poker_log.jsonl"), type=Path)
    parser.add_argument("--model", default=Path("models/poker_policy.json"), type=Path)
    parser.add_argument("--out", default=Path("reports/mlops_contract.json"), type=Path)
    parser.add_argument("--registry-out", default=Path("reports/model_registry.json"), type=Path)
    parser.add_argument("--experiment-log-out", default=Path("reports/experiments.jsonl"), type=Path)
    parser.add_argument("--dataset-log-out", default=Path("reports/dataset_versions.jsonl"), type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def build_smoke_artifacts(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    dataset_path = resolve(root, args.dataset)
    model_path = resolve(root, args.model)
    registry_path = resolve(root, args.registry_out)
    experiment_log_path = resolve(root, args.experiment_log_out)
    dataset_log_path = resolve(root, args.dataset_log_out)

    dataset_manifest = dataset_version_manifest(dataset_path)
    append_jsonl(dataset_log_path, dataset_manifest)

    docker_metadata = docker_image_metadata(dockerfile=root / "Dockerfile")
    run_manifest = experiment_run_manifest(
        experiment_name="mlops_smoke",
        command=[
            "python",
            "scripts/check_mlops_contract.py",
            "--smoke",
            "--out",
            str(args.out),
        ],
        parameters={
            "dataset": str(args.dataset),
            "model": str(args.model),
            "registry_out": str(args.registry_out),
        },
        metrics={"contract_checks": "local"},
        artifacts=[
            {"type": "dataset_version", "version": dataset_manifest["version"]},
            {"type": "docker_image", "tag": docker_metadata["tag"]},
        ],
        dataset_version=dataset_manifest["version"],
        status="PASS",
        seed=20260713,
    )
    append_jsonl(experiment_log_path, run_manifest)

    registry_entry = model_registry_entry(
        model_path,
        run_id=run_manifest["run_id"],
        dataset_version=dataset_manifest["version"],
        metrics={"smoke_contract": 1.0},
        stage="candidate",
        docker_image=docker_metadata["tag"],
    )
    registry = update_model_registry(registry_path, registry_entry)
    return {
        "dataset_version": dataset_manifest,
        "experiment_run": run_manifest,
        "model_registry_entry": registry_entry,
        "model_registry": registry,
        "docker_image": docker_metadata,
        "paths": {
            "registry": str(registry_path),
            "experiment_log": str(experiment_log_path),
            "dataset_log": str(dataset_log_path),
        },
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    artifacts = build_smoke_artifacts(args, root) if args.smoke else {}
    checks = validate_mlops_contract(root)

    if args.smoke:
        registry = artifacts["model_registry"]
        model_name = artifacts["model_registry_entry"]["model_name"]
        model_version = artifacts["model_registry_entry"]["model_version"]
        checks.extend(
            [
                check(
                    "dataset_version:fingerprint",
                    bool(artifacts["dataset_version"].get("fingerprint")),
                    artifacts["dataset_version"]["version"],
                ),
                check(
                    "experiment_tracking:run_id",
                    bool(artifacts["experiment_run"].get("run_id")),
                    artifacts["experiment_run"]["run_id"],
                ),
                check(
                    "model_registry:latest_version",
                    registry["models"][model_name]["latest_version"] == model_version,
                    model_version,
                ),
                check(
                    "docker_version:tag",
                    artifacts["docker_image"]["tag"] == "poker-decision-agent:0.1.0",
                    artifacts["docker_image"]["tag"],
                ),
            ]
        )

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    report = {
        "status": status,
        "contract": describe_mlops_contract(),
        "checks": checks,
        "artifacts": artifacts,
    }
    out = resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
