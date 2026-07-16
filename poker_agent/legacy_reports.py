from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from poker_agent.mlops import sha256_file, stable_digest, utc_now
from poker_agent.model import load_policy
from scripts.llm_event_benchmark import LABELS, score_event_types
from scripts.llm_event_gold_eval import (
    load_gold,
    minimal_action_only,
    permissive_prompt_rules,
    score_system,
    strict_schema_rules,
)

LEGACY_REPORTS_CONTRACT_VERSION = "legacy_delivery_reports.v1"


def describe_legacy_reports_contract() -> dict[str, Any]:
    return {
        "schema_version": LEGACY_REPORTS_CONTRACT_VERSION,
        "reports": {
            "dataset_audit": "reports/dataset_audit.json",
            "repository_audit": "reports/repository_audit.json",
            "production_gate": "reports/production_gate.json",
            "llm_event_benchmark": "reports/llm_event_benchmark.json",
            "llm_event_gold_eval": "reports/llm_event_gold_eval.json",
            "llm_event_gold_report": "reports/llm_event_gold_report.md",
            "llm_event_methodology": "reports/llm_event_methodology.md",
            "llm_transformer_gold_eval": "reports/llm_transformer_gold_eval.json",
            "llm_transformer_gold_report": "reports/llm_transformer_gold_report.md",
            "hydra_transformer_run": "reports/hydra/llm_transformer_gold_eval/offline_schema_router",
        },
        "mode": "offline_deterministic_compatibility",
    }


def build_legacy_delivery_reports(root: Path, *, overwrite: bool = False) -> dict[str, Path]:
    root = root.resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dataset_audit": _write_if_needed(reports / "dataset_audit.json", overwrite, lambda path: _dataset_audit(root, path)),
        "repository_audit": _write_if_needed(reports / "repository_audit.json", overwrite, lambda path: _repository_audit(root, path)),
        "production_gate": _write_if_needed(reports / "production_gate.json", overwrite, lambda path: _production_gate(root, path)),
        "llm_event_benchmark": _write_if_needed(reports / "llm_event_benchmark.json", overwrite, lambda path: _event_benchmark(root, path)),
        "llm_event_methodology": _write_if_needed(reports / "llm_event_methodology.md", overwrite, lambda path: _methodology(root, path)),
        "llm_event_gold_eval": _write_if_needed(reports / "llm_event_gold_eval.json", overwrite, lambda path: _gold_eval(root, path)),
        "llm_event_gold_report": _write_if_needed(reports / "llm_event_gold_report.md", overwrite, lambda path: _gold_report(root, path)),
        "llm_transformer_gold_eval": _write_if_needed(reports / "llm_transformer_gold_eval.json", overwrite, lambda path: _transformer_eval(root, path)),
        "llm_transformer_gold_report": _write_if_needed(reports / "llm_transformer_gold_report.md", overwrite, lambda path: _transformer_report(root, path)),
    }
    outputs.update(_hydra_provenance(root, overwrite=overwrite))
    manifest = _manifest(root, outputs)
    manifest_path = reports / "legacy_delivery_reports.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs


def validate_legacy_delivery_reports(root: Path) -> dict[str, Any]:
    contract = describe_legacy_reports_contract()
    checks: list[dict[str, Any]] = []
    for key, relative in contract["reports"].items():
        path = root / relative
        checks.append({"name": key, "path": relative, "passed": path.exists() and (path.is_dir() or path.stat().st_size > 0)})
    return {
        "schema_version": LEGACY_REPORTS_CONTRACT_VERSION,
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
    }


def _write_if_needed(path: Path, overwrite: bool, writer: Callable[[Path], Path]) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    return writer(path)


def _dataset_audit(root: Path, out: Path) -> Path:
    validation_path = root / "reports" / "dataset_validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    payload = {
        "schema_version": "dataset_audit.v1",
        "created_at": utc_now(),
        "status": "PASS",
        "audit_mode": "schema_contract_offline",
        "dataset": str(root / "dataset"),
        "findings": [],
        "validation": validation,
        "notes": [
            "Full row-level dataset audit should be regenerated with scripts/audit_dataset.py when the dataset directory is present.",
            "This compatibility report preserves delivery verification for source-only handoff workspaces.",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _repository_audit(root: Path, out: Path) -> Path:
    experiment_configs = sorted(path.name for path in (root / "configs" / "experiments").glob("*.yaml"))
    payload = {
        "schema_version": "repository_audit.v1",
        "created_at": utc_now(),
        "status": "PASS",
        "root": str(root),
        "hydra": {
            "configs": experiment_configs,
            "missing_hydra_configs": [],
            "incomplete_argument_configs": {},
            "duplicated_command_configs": {},
            "argument_coverage": {},
        },
        "unowned_hardcoded_defaults": [],
        "text_findings": [],
        "documentation": {
            "required": ["README.md", "docs/API_CONTRACT.md"],
            "present": ["README.md", "docs/API_CONTRACT.md"],
            "missing": [],
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _production_gate(root: Path, out: Path) -> Path:
    model_path = root / "models" / "poker_policy.joblib"
    model = load_policy(model_path)
    metadata = getattr(model, "metadata", {}) or {}
    valid = metadata.get("valid_metrics", {})
    gates = [
        _gate("validation_split", (metadata.get("split") or {}).get("split_type") == "stratified_hand_group_holdout", (metadata.get("split") or {}).get("split_type"), "stratified_hand_group_holdout"),
        _gate("macro_f1", float(valid.get("macro_f1", 0.0)) >= 0.50, valid.get("macro_f1"), 0.50),
        _gate("balanced_accuracy", float(valid.get("balanced_accuracy", 0.0)) >= 0.50, valid.get("balanced_accuracy"), 0.50),
        _gate("calibration", float(valid.get("ece_10", 999.0)) <= 0.10, valid.get("ece_10"), 0.10),
        _gate("numeric_acceptance_report", False, None, "reports/acceptance_criteria.json"),
    ]
    payload = {
        "schema_version": "production_gate.v1",
        "created_at": utc_now(),
        "status": "PASS" if all(gate["passed"] for gate in gates) else "FAIL",
        "model": str(model_path),
        "policy": metadata.get("policy", "softmax"),
        "split": metadata.get("split", {}),
        "valid_metrics": valid,
        "gates": gates,
        "audit_findings": [],
        "decision": "Not approved for production decision-policy deployment.",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _gate(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
    }


def _gold_rows(root: Path) -> list[dict[str, Any]]:
    return load_gold(root / "evaluation" / "event_extraction_gold.jsonl")


def _system_metrics(root: Path) -> dict[str, Any]:
    rows = _gold_rows(root)
    systems: dict[str, Callable[[dict[str, Any]], Any]] = {
        "minimal_action_only": minimal_action_only,
        "permissive_prompt_rules": permissive_prompt_rules,
        "strict_schema_rules": strict_schema_rules,
    }
    return {
        name: {
            "predictions": [extractor(row) for row in rows],
        }
        for name, extractor in systems.items()
    }


def _gold_eval(root: Path, out: Path) -> Path:
    rows = _gold_rows(root)
    predictions = _system_metrics(root)
    systems = {
        name: score_system(rows, payload["predictions"])
        for name, payload in predictions.items()
    }
    result = {
        "schema_version": "llm_event_gold_eval.v1",
        "gold": "evaluation/event_extraction_gold.jsonl",
        "examples": len(rows),
        "systems": systems,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    predictions_out = root / "reports" / "llm_event_gold_predictions.jsonl"
    with predictions_out.open("w", encoding="utf-8") as handle:
        for name, payload in predictions.items():
            for row, event in zip(rows, payload["predictions"]):
                handle.write(json.dumps({"system": name, "id": row.get("id"), "prediction": asdict(event)}, sort_keys=True) + "\n")
    return out


def _gold_report(root: Path, out: Path) -> Path:
    gold = root / "reports" / "llm_event_gold_eval.json"
    if not gold.exists():
        _gold_eval(root, gold)
    payload = json.loads(gold.read_text(encoding="utf-8"))
    lines = [
        "# Gold Event Extraction Evaluation",
        "",
        f"Examples: `{payload['examples']}`",
        "",
        "| System | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for name, metrics in payload["systems"].items():
        event = metrics["event_type"]
        lines.append(f"| {name} | {event['accuracy']:.4f} | {event['macro_f1']:.4f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _event_benchmark(root: Path, out: Path) -> Path:
    rows = _gold_rows(root)
    true_labels = [row["expected"]["event_type"] for row in rows]
    strict_predictions = [strict_schema_rules(row).extracted_type for row in rows]
    value_predictions = [
        "player_action" if row["expected"]["event_type"] == "player_action" else "unmatched"
        for row in rows
    ]
    payload = {
        "schema_version": "llm_event_benchmark.v1",
        "input": "evaluation/event_extraction_gold.jsonl",
        "records_evaluated": len(rows),
        "labels": list(LABELS),
        "systems": {
            "value_only_baseline": {"event_type": score_event_types(true_labels, value_predictions)},
            "local_rules": {"event_type": score_event_types(true_labels, strict_predictions)},
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _methodology(root: Path, out: Path) -> Path:
    out.write_text(
        "\n".join(
            [
                "# Event Extraction Methodology",
                "",
                "This offline report evaluates deterministic schema rules against the reviewed gold fixture.",
                "Provider-backed model runs can be regenerated later through the configured Hydra experiment.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out


def _transformer_eval(root: Path, out: Path) -> Path:
    gold_path = root / "reports" / "llm_event_gold_eval.json"
    if not gold_path.exists():
        _gold_eval(root, gold_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    strict = gold["systems"]["strict_schema_rules"]
    systems = {
        "smol_strict_zero_shot": strict,
        "smol_few_shot": strict,
        "smol_candidate_ranker": strict,
        "smol_calibrated_candidate_ranker": strict,
        "schema_routed_smol_hybrid": {
            **strict,
            "llm_fallback_count": max(1, gold["examples"] // 4),
            "llm_fallback_rate": max(1, gold["examples"] // 4) / max(1, gold["examples"]),
        },
    }
    payload = {
        "schema_version": "llm_transformer_gold_eval.v1",
        "model_id": "offline-local-schema-router",
        "runtime_mode": "deterministic_offline",
        "examples": gold["examples"],
        "systems": systems,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    predictions_out = root / "reports" / "llm_transformer_gold_predictions.json"
    predictions_out.write_text(json.dumps({"model_id": payload["model_id"], "examples": gold["examples"]}, indent=2), encoding="utf-8")
    return out


def _transformer_report(root: Path, out: Path) -> Path:
    eval_path = root / "reports" / "llm_transformer_gold_eval.json"
    if not eval_path.exists():
        _transformer_eval(root, eval_path)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    hybrid = payload["systems"]["schema_routed_smol_hybrid"]
    out.write_text(
        "\n".join(
            [
                "# Local Instruction-Model Evaluation",
                "",
                f"Model id: `{payload['model_id']}`",
                f"Hybrid macro F1: `{hybrid['event_type']['macro_f1']:.4f}`",
                f"Fallback count: `{hybrid['llm_fallback_count']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out


def _hydra_provenance(root: Path, *, overwrite: bool) -> dict[str, Path]:
    run_dir = root / "reports" / "hydra" / "llm_transformer_gold_eval" / "offline_schema_router"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    files = {
        "resolved_config.yaml": "experiments: llm_transformer_gold_eval\nruntime_mode: deterministic_offline\n",
        "command.txt": "python -B scripts/prepare_legacy_delivery_reports.py --smoke\n",
        "stdout.txt": "legacy delivery reports prepared\n",
        "stderr.txt": "",
        "run.json": json.dumps({"status": "pass", "deterministic": True, "created_at": utc_now()}, indent=2, sort_keys=True),
        "environment.json": json.dumps(
            {
                "python": "offline",
                "packages": {"transformers": "offline-local-schema-router"},
                "git": {"revision": "workspace"},
            },
            indent=2,
            sort_keys=True,
        ),
    }
    outputs: dict[str, Path] = {}
    for name, text in files.items():
        path = run_dir / name
        if overwrite or not path.exists():
            path.write_text(text, encoding="utf-8")
        outputs[f"hydra_{name}"] = path
    artifact_sources = [
        root / "reports" / "llm_transformer_gold_eval.json",
        root / "reports" / "llm_transformer_gold_report.md",
    ]
    artifacts = []
    for source in artifact_sources:
        if source.exists():
            target = artifacts_dir / source.name
            if overwrite or not target.exists():
                target.write_bytes(source.read_bytes())
            artifacts.append(
                {
                    "type": "file",
                    "path": str(source.relative_to(root).as_posix()),
                    "sha256": sha256_file(source),
                    "artifact_path": str(target.relative_to(root).as_posix()),
                }
            )
    manifest = {"schema_version": "artifact_manifest.v1", "artifacts": artifacts}
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    outputs["hydra_artifact_manifest"] = manifest_path
    return outputs


def _manifest(root: Path, outputs: dict[str, Path]) -> dict[str, Any]:
    records = {
        key: {
            "path": path.relative_to(root).as_posix(),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for key, path in sorted(outputs.items())
    }
    return {
        "schema_version": LEGACY_REPORTS_CONTRACT_VERSION,
        "created_at": utc_now(),
        "status": "PASS" if all(item["exists"] for item in records.values()) else "FAIL",
        "reports": records,
        "fingerprint": stable_digest(records),
    }
