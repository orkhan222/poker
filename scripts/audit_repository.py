from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_repo_hygiene import BANNED_PHRASES, SKIP_DIRS, should_scan


EXPERIMENT_NAME_TOKENS = (
    "audit",
    "benchmark",
    "evaluate",
    "eval",
    "experiment",
    "gate",
    "train",
    "verify",
)

REQUIRED_DOCS = (
    "README.md",
)

WEAK_TEXT_PATTERNS = (
    "TODO",
    "FIXME",
    "HACK",
    "".join(("place", "holder")),
    "debug only",
    "temporary",
    "quick and dirty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit repository experiment coverage and delivery hygiene")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--out", default=Path("reports/repository_audit.json"), type=Path)
    return parser.parse_args()


def python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts)
    )


def experiment_scripts(root: Path) -> list[str]:
    scripts_dir = root / "scripts"
    excluded = {"run_hydra_experiment.py"}
    scripts = []
    for path in sorted(scripts_dir.glob("*.py")):
        if path.name in excluded:
            continue
        if any(token in path.stem for token in EXPERIMENT_NAME_TOKENS):
            scripts.append(path.relative_to(root).as_posix())
    return scripts


def literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else type(node).__name__


def argparse_defaults(path: Path, root: Path) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    defaults: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        option = None
        if node.args:
            option = literal_value(node.args[0])
        default_value = None
        has_default = False
        for keyword in node.keywords:
            if keyword.arg == "default":
                has_default = True
                default_value = literal_value(keyword.value)
        if has_default:
            defaults.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "option": option,
                    "default": default_value,
                    "line": node.lineno,
                }
            )
    return defaults


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def experiment_config_coverage(root: Path, scripts: list[str]) -> dict[str, Any]:
    config_dir = root / "configs" / "experiments"
    configs: dict[str, dict[str, Any]] = {}
    entrypoints: defaultdict[str, list[str]] = defaultdict(list)
    fingerprints: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(config_dir.glob("*.yaml")):
        payload = load_yaml(path)
        configs[path.name] = payload
        command = payload.get("command") or {}
        entrypoint = str(command.get("entrypoint") or "")
        if entrypoint:
            entrypoints[entrypoint].append(path.name)
        comparable = {
            "entrypoint": entrypoint,
            "args": command.get("args") or {},
        }
        digest = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode("utf-8")).hexdigest()
        fingerprints[digest].append(path.name)

    covered = sorted(entrypoints)
    missing = sorted(script for script in scripts if script not in entrypoints)
    duplicated = {
        digest: names
        for digest, names in fingerprints.items()
        if len(names) > 1
    }
    return {
        "configs": sorted(configs),
        "covered_entrypoints": covered,
        "missing_hydra_configs": missing,
        "duplicated_command_configs": duplicated,
    }


def text_findings(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or not should_scan(path, root):
            continue
        if path.name == "audit_repository.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for phrase in BANNED_PHRASES:
                if phrase in line:
                    findings.append(
                        {
                            "kind": "blocked_phrase",
                            "file": relative,
                            "line": line_number,
                            "phrase": phrase,
                        }
                    )
            for phrase in WEAK_TEXT_PATTERNS:
                if phrase.lower() in line.lower():
                    findings.append(
                        {
                            "kind": "weak_text",
                            "file": relative,
                            "line": line_number,
                            "phrase": phrase,
                        }
                    )
    return findings


def docs_status(root: Path) -> dict[str, Any]:
    present = [doc for doc in REQUIRED_DOCS if (root / doc).exists()]
    missing = [doc for doc in REQUIRED_DOCS if not (root / doc).exists()]
    return {
        "required": list(REQUIRED_DOCS),
        "present": present,
        "missing": missing,
    }


def config_groups(root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    configs_root = root / "configs"
    for path in sorted(configs_root.iterdir()):
        if path.is_dir():
            groups[path.name] = sorted(item.name for item in path.glob("*.yaml"))
    return groups


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    scripts = experiment_scripts(root)
    defaults = [
        item
        for path in python_files(root)
        for item in argparse_defaults(path, root)
    ]
    coverage = experiment_config_coverage(root, scripts)
    findings = text_findings(root)
    defaults_by_file = Counter(item["file"] for item in defaults)
    report = {
        "status": "PASS"
        if not coverage["missing_hydra_configs"] and not findings and not docs_status(root)["missing"]
        else "FAIL",
        "root": str(root),
        "experiment_scripts": scripts,
        "hydra": {
            "groups": config_groups(root),
            **coverage,
        },
        "hardcoded_argparse_defaults": defaults,
        "hardcoded_defaults_by_file": dict(defaults_by_file.most_common()),
        "text_findings": findings,
        "documentation": docs_status(root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
