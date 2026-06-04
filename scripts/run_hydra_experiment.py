from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def project_root_from_config(cfg: DictConfig) -> Path:
    configured = cfg.get("project_root")
    original = Path(get_original_cwd()).resolve()
    if configured in (None, "", "."):
        return original
    path = Path(str(configured))
    return path.resolve() if path.is_absolute() else (original / path).resolve()


def option_name(raw: str) -> str:
    return "--" + raw.replace("_", "-")


def stringify_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_command(cfg: DictConfig, root: Path) -> list[str]:
    experiment = cfg.experiments
    command_cfg = experiment.command
    entrypoint = root / str(command_cfg.entrypoint)
    if not entrypoint.exists():
        raise FileNotFoundError(f"Experiment entrypoint not found: {entrypoint}")

    command = [str(cfg.python_executable), str(entrypoint)]
    args = OmegaConf.to_container(command_cfg.get("args", {}), resolve=True) or {}
    if not isinstance(args, dict):
        raise TypeError("command.args must be a mapping")

    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(option_name(key))
            continue
        command.extend([option_name(key), stringify_value(value)])
    return command


def command_args(cfg: DictConfig) -> dict[str, Any]:
    args = OmegaConf.to_container(cfg.experiments.command.get("args", {}), resolve=True) or {}
    if not isinstance(args, dict):
        raise TypeError("command.args must be a mapping")
    return args


def runtime_environment(cfg: DictConfig) -> dict[str, str]:
    env = os.environ.copy()
    runtime = cfg.get("runtime", {}) or {}
    seed = int(runtime.get("seed", cfg.get("training", {}).get("seed", 42)))
    env["PYTHONHASHSEED"] = str(runtime.get("pythonhashseed", seed))
    env["POKER_EXPERIMENT_SEED"] = str(seed)
    env["OMP_NUM_THREADS"] = str(runtime.get("num_threads", 1))
    env["MKL_NUM_THREADS"] = str(runtime.get("num_threads", 1))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if bool(runtime.get("deterministic", True)):
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def git_metadata(root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"available": False}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return metadata
    return {
        "available": True,
        "revision": revision,
        "dirty": bool(status),
        "changed_paths": status,
    }


def environment_manifest(cfg: DictConfig, root: Path, env: dict[str, str]) -> dict[str, Any]:
    runtime = cfg.get("runtime", {}) or {}
    packages: dict[str, str | None] = {}
    for name in runtime.get("capture_packages", []):
        package_name = str(name)
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "determinism": {
            "enabled": bool(runtime.get("deterministic", True)),
            "seed": env.get("POKER_EXPERIMENT_SEED"),
            "pythonhashseed": env.get("PYTHONHASHSEED"),
            "num_threads": runtime.get("num_threads", 1),
        },
        "packages": packages,
        "git": git_metadata(root) if bool(runtime.get("capture_git", True)) else {"available": False},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_argument(key: str) -> bool:
    return key == "out" or key.endswith("_out") or key.endswith("_dir")


def artifact_manifest(
    cfg: DictConfig,
    root: Path,
    output_dir: Path,
    args: dict[str, Any],
) -> dict[str, Any]:
    logging_cfg = cfg.logging
    artifacts_dir = output_dir / str(logging_cfg.artifacts_dir)
    copy_artifacts = bool(logging_cfg.copy_artifacts)
    copy_max_bytes = int(logging_cfg.artifact_copy_max_bytes)
    artifacts: list[dict[str, Any]] = []
    for key, raw_value in sorted(args.items()):
        if not output_argument(key) or raw_value in (None, "") or isinstance(raw_value, (list, dict)):
            continue
        source = Path(str(raw_value))
        source = source if source.is_absolute() else root / source
        record: dict[str, Any] = {
            "argument": key,
            "source": str(source.resolve()),
            "exists": source.exists(),
        }
        if source.is_file():
            size = source.stat().st_size
            record.update({"type": "file", "size_bytes": size, "sha256": sha256_file(source)})
            if copy_artifacts and size <= copy_max_bytes:
                target = artifacts_dir / key / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                record["run_copy"] = str(target.resolve())
        elif source.is_dir():
            record.update(
                {
                    "type": "directory",
                    "file_count": sum(1 for path in source.rglob("*") if path.is_file()),
                }
            )
        artifacts.append(record)
    return {"artifacts": artifacts}


def parse_key_value_metrics(stdout: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in stdout.splitlines():
        text = line.strip()
        if not text or "=" not in text or text.startswith("{"):
            continue
        key, value = text.split("=", 1)
        if key and " " not in key:
            metrics[key] = value
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    root = project_root_from_config(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(cfg, root)
    args = command_args(cfg)
    env = runtime_environment(cfg)
    resolved_config_path = output_dir / str(cfg.logging.resolved_config_file)
    command_path = output_dir / "command.txt"
    stdout_path = output_dir / str(cfg.logging.stdout_file)
    stderr_path = output_dir / str(cfg.logging.stderr_file)
    run_metadata_path = output_dir / str(cfg.logging.run_metadata_file)
    environment_path = output_dir / str(cfg.logging.environment_file)
    artifact_manifest_path = output_dir / str(cfg.logging.artifact_manifest_file)

    resolved_config_path.write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )
    command_path.write_text(" ".join(command), encoding="utf-8")
    environment_path.write_text(
        json.dumps(environment_manifest(cfg, root, env), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    artifacts = artifact_manifest(cfg, root, output_dir, args)
    artifact_manifest_path.write_text(
        json.dumps(artifacts, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    allow_failure = bool(cfg.experiments.get("allow_failure", False))
    status = "pass" if completed.returncode == 0 else "failed"
    if completed.returncode != 0 and allow_failure:
        status = "allowed_failure"

    payload = {
        "experiment": cfg.experiments.name,
        "description": cfg.experiments.get("description", ""),
        "status": status,
        "returncode": completed.returncode,
        "allow_failure": allow_failure,
        "project_root": str(root),
        "output_dir": str(output_dir),
        "command": command,
        "seed": env.get("POKER_EXPERIMENT_SEED"),
        "pythonhashseed": env.get("PYTHONHASHSEED"),
        "deterministic": bool(cfg.get("runtime", {}).get("deterministic", True)),
        "metrics": parse_key_value_metrics(completed.stdout),
        "environment_manifest": str(environment_path),
        "artifact_manifest": str(artifact_manifest_path),
    }
    run_metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    if completed.returncode != 0 and not allow_failure:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
