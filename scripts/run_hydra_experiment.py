from __future__ import annotations

import json
import os
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


def runtime_environment(cfg: DictConfig) -> dict[str, str]:
    env = os.environ.copy()
    runtime = cfg.get("runtime", {}) or {}
    seed = int(runtime.get("seed", cfg.get("training", {}).get("seed", 42)))
    env["PYTHONHASHSEED"] = str(runtime.get("pythonhashseed", seed))
    env["POKER_EXPERIMENT_SEED"] = str(seed)
    env["OMP_NUM_THREADS"] = str(runtime.get("num_threads", 1))
    env["MKL_NUM_THREADS"] = str(runtime.get("num_threads", 1))
    if bool(runtime.get("deterministic", True)):
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


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
    env = runtime_environment(cfg)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(command), encoding="utf-8")

    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

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
    }
    (output_dir / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    if completed.returncode != 0 and not allow_failure:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
