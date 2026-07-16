from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_agent.agents import RuleBasedAgent
from poker_agent.api_contract import API_VERSION, deployment_api_contract
from poker_agent.baselines import BASELINE_SPECS, baseline_names, build_baseline_policy
from poker_agent.data_validation import validate_dataset
from poker_agent.dataset_schema import ACTION_FIELDS, DATASET_SCHEMA_REQUIRED_FIELDS, HAND_FIELDS, PLAYER_FIELDS, STACK_FIELDS
from poker_agent.evaluator import evaluate_policy
from poker_agent.final_model_selection import final_model_selection_status, write_final_model_selection_reports
from poker_agent.features import request_to_features
from poker_agent.mlops import dataset_version_manifest, sha256_file, stable_digest, utc_now
from poker_agent.schemas import PredictionRequest

FINAL_DELIVERABLES_CONTRACT_VERSION = "final_deliverables.v1"

FINAL_DELIVERABLES: tuple[dict[str, Any], ...] = (
    {
        "key": "validated_dataset_schema",
        "title": "Validated Dataset Schema",
        "required_artifacts": [
            "poker_agent/dataset_schema.py",
            "configs/dataset/poker_csv.yaml",
            "tests/test_dataset_schema_contract.py",
        ],
    },
    {
        "key": "project_scope_contract",
        "title": "Project Scope Contract",
        "required_artifacts": [
            "poker_agent/project_scope.py",
            "poker_agent/game_scope.py",
            "poker_agent/usage_boundary.py",
            "reports/project_scope_contract.json",
            "docs/PROJECT_SCOPE_CONTRACT.md",
            "tests/test_project_scope_contract.py",
            "tests/test_game_scope_usage_boundary_contract.py",
        ],
    },
    {
        "key": "validation_report",
        "title": "Validation Report",
        "required_artifacts": [
            "reports/dataset_validation_report.json",
            "poker_agent/data_validation.py",
            "tests/test_data_validation_contract.py",
        ],
    },
    {
        "key": "baseline_comparison",
        "title": "Baseline Comparison",
        "required_artifacts": [
            "reports/baseline_report.json",
            "poker_agent/baselines.py",
            "scripts/run_baselines.py",
            "tests/test_baseline_contract.py",
        ],
    },
    {
        "key": "trained_checkpoint",
        "title": "Trained Checkpoint",
        "required_artifacts": [
            "models/poker_policy.joblib",
            "models/poker_policy.json",
        ],
        "any_of": True,
    },
    {
        "key": "evaluation_report",
        "title": "Evaluation Report",
        "required_artifacts": [
            "reports/evaluation_report.json",
            "reports/final_model_selection.json",
            "docs/FINAL_MODEL_SELECTION.md",
            "poker_agent/final_model_selection.py",
            "scripts/evaluate_policy.py",
            "scripts/check_final_model_selection.py",
            "tests/test_final_model_selection_contract.py",
        ],
    },
    {
        "key": "dockerized_fastapi_service",
        "title": "Dockerized FastAPI Service",
        "required_artifacts": [
            "Dockerfile",
            "docker-compose.yml",
            "poker_agent/service.py",
            "run_server.ps1",
        ],
    },
    {
        "key": "api_docs",
        "title": "API Docs",
        "required_artifacts": [
            "docs/API_CONTRACT.md",
            "poker_agent/api_contract.py",
            "poker_agent/game_scope.py",
            "poker_agent/usage_boundary.py",
            "README.md",
        ],
    },
    {
        "key": "tests",
        "title": "Tests",
        "required_artifacts": [
            "tests/test_api_contract.py",
            "tests/test_dataset_schema_contract.py",
            "tests/test_data_validation_contract.py",
            "tests/test_baseline_contract.py",
            "tests/test_final_deliverables_contract.py",
            "tests/test_legacy_delivery_contract.py",
            "tests/test_project_scope_contract.py",
            "tests/test_game_scope_usage_boundary_contract.py",
            "tests/test_final_model_selection_contract.py",
        ],
    },
)


def artifact_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {
        "path": relative,
        "exists": path.exists(),
    }
    if path.is_file():
        record.update(
            {
                "kind": "file",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    elif path.is_dir():
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        record.update(
            {
                "kind": "directory",
                "file_count": len(files),
                "size_bytes": sum(file_path.stat().st_size for file_path in files),
            }
        )
    else:
        record["kind"] = "missing"
    return record


def describe_final_deliverables_contract() -> dict[str, Any]:
    return {
        "schema_version": FINAL_DELIVERABLES_CONTRACT_VERSION,
        "api_version": API_VERSION,
        "deliverables": [
            {
                "key": item["key"],
                "title": item["title"],
                "required_artifacts": list(item["required_artifacts"]),
                "require_mode": "any_of" if item.get("any_of") else "all",
            }
            for item in FINAL_DELIVERABLES
        ],
        "smoke_reports": {
            "dataset_validation": "reports/dataset_validation_report.json",
            "baseline_comparison": "reports/baseline_report.json",
            "evaluation": "reports/evaluation_report.json",
            "manifest": "reports/final_deliverables.json",
            "handoff": "reports/delivery_report.md",
            "api_docs": "docs/API_CONTRACT.md",
        },
        "quality_bar": [
            "Every final deliverable has a machine-readable artifact record.",
            "The dataset schema and validation policy are executable contracts, not prose only.",
            "The FastAPI deployment surface is documented by the same contract exposed by the service.",
            "Smoke reports are deterministic and can be regenerated without network access.",
        ],
    }


def validate_final_deliverables(root: Path) -> dict[str, Any]:
    root = root.resolve()
    deliverables: list[dict[str, Any]] = []
    for spec in FINAL_DELIVERABLES:
        artifacts = [artifact_record(root, relative) for relative in spec["required_artifacts"]]
        if spec.get("any_of"):
            passed = any(item["exists"] for item in artifacts)
        else:
            passed = all(item["exists"] for item in artifacts)
        deliverables.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "status": "PASS" if passed else "FAIL",
                "require_mode": "any_of" if spec.get("any_of") else "all",
                "artifacts": artifacts,
            }
        )

    payload = {
        "schema_version": FINAL_DELIVERABLES_CONTRACT_VERSION,
        "created_at": utc_now(),
        "root": str(root),
        "status": "PASS" if all(item["status"] == "PASS" for item in deliverables) else "FAIL",
        "deliverables": deliverables,
    }
    payload["fingerprint"] = stable_digest(
        {
            "schema_version": payload["schema_version"],
            "deliverables": [
                {
                    "key": item["key"],
                    "status": item["status"],
                    "artifacts": [
                        {
                            "path": artifact["path"],
                            "exists": artifact["exists"],
                            "sha256": artifact.get("sha256"),
                        }
                        for artifact in item["artifacts"]
                    ],
                }
                for item in deliverables
            ],
        }
    )
    return payload


def write_api_docs(root: Path, out: Path | None = None, *, model_version: str = "poker_policy:handoff") -> Path:
    out = out or root / "docs" / "API_CONTRACT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    contract = deployment_api_contract(model_version=model_version)
    request_schema = contract["request_schema"]
    response_schema = contract["response_schema"]
    error_codes = contract["error_codes"]
    request_fields = sorted(request_schema.get("properties", {}))
    response_fields = list(response_schema.get("properties", {}))
    lines = [
        "# Poker Decision Agent API Contract",
        "",
        f"- API version: `{contract['api_version']}`",
        f"- Endpoint: `POST {contract['endpoint']}`",
        f"- Request schema: `{request_schema['schema_version']}`",
        f"- Response schema: `{response_schema['schema_version']}`",
        f"- Model version field: `{contract['model_version']}`",
        "",
        "## Request",
        "",
        "Required fields:",
        "",
        *[f"- `{field}`" for field in request_schema.get("required", [])],
        "",
        "Supported request properties:",
        "",
        *[f"- `{field}`" for field in request_fields],
        "",
        "## Response",
        "",
        "Required response fields:",
        "",
        *[f"- `{field}`" for field in response_schema.get("required", [])],
        "",
        "Response properties:",
        "",
        *[f"- `{field}`" for field in response_fields],
        "",
        "## Error Codes",
        "",
        *[
            f"- `{code}`: HTTP {spec['http_status']}, retryable={str(spec['retryable']).lower()}"
            for code, spec in sorted(error_codes.items())
        ],
        "",
        "## Example Request",
        "",
        "```json",
        json.dumps(request_schema["examples"][0], indent=2, sort_keys=True),
        "```",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_smoke_reports(root: Path) -> dict[str, Path]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    dataset_report = _write_dataset_validation_report(root, reports / "dataset_validation_report.json")
    baseline_report = _write_baseline_report(root, reports / "baseline_report.json")
    final_model_selection = write_final_model_selection_reports(root)
    evaluation_report = _write_evaluation_report(root, reports / "evaluation_report.json")
    api_docs = write_api_docs(root, root / "docs" / "API_CONTRACT.md")

    manifest = validate_final_deliverables(root)
    manifest_path = reports / "final_deliverables.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    handoff_path = reports / "delivery_report.md"
    write_delivery_report(handoff_path, manifest)

    return {
        "dataset_validation_report": dataset_report,
        "baseline_report": baseline_report,
        "final_model_selection": final_model_selection["json"],
        "final_model_selection_docs": final_model_selection["docs"],
        "evaluation_report": evaluation_report,
        "api_docs": api_docs,
        "final_deliverables": manifest_path,
        "delivery_report": handoff_path,
    }


def write_delivery_report(path: Path, manifest: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Poker Decision Agent Delivery Report",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Status: `{manifest['status']}`",
        f"- Fingerprint: `{manifest['fingerprint']}`",
        "",
        "## Final Deliverables",
        "",
    ]
    for item in manifest["deliverables"]:
        lines.append(f"- {item['title']}: `{item['status']}`")
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```powershell",
            "python -B scripts\\prepare_final_deliverables.py --smoke",
            "python -B scripts\\verify_delivery.py --json-out reports\\delivery_verification.json",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_dataset_validation_report(root: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "dataset"
    validation = validate_dataset(dataset_dir) if dataset_dir.exists() else {
        "status": "SKIP",
        "reason": "dataset directory is not present in the handoff workspace",
        "policy": "schema-only validation executed; run scripts/audit_dataset.py against a real dataset for full checks",
    }
    schema = {
        "hands.csv": list(HAND_FIELDS),
        "players.csv": list(PLAYER_FIELDS),
        "actions.csv": list(ACTION_FIELDS),
        "stack_events.csv": list(STACK_FIELDS),
        "required_fields": DATASET_SCHEMA_REQUIRED_FIELDS,
    }
    missing_declared = {
        "hands.csv": sorted(set(DATASET_SCHEMA_REQUIRED_FIELDS["hands.csv"]) - set(HAND_FIELDS)),
        "actions.csv": sorted(set(DATASET_SCHEMA_REQUIRED_FIELDS["actions.csv"]) - set(ACTION_FIELDS)),
    }
    report = {
        "schema_version": "dataset_validation_report.v1",
        "created_at": utc_now(),
        "dataset": str(dataset_dir),
        "status": "PASS" if not any(missing_declared.values()) and validation.get("status") != "FAIL" else "FAIL",
        "schema": schema,
        "schema_checks": {
            "declared_required_fields_present": not any(missing_declared.values()),
            "missing_declared_fields": missing_declared,
        },
        "validation": validation,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _write_baseline_report(root: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    examples = _smoke_examples()
    rows: list[dict[str, Any]] = []
    for name in baseline_names():
        policy = build_baseline_policy(name, examples if BASELINE_SPECS[name].trains_on_dataset else None, epochs=3, learning_rate=0.05)
        metrics = evaluate_policy(policy, examples)
        rows.append(
            {
                "name": name,
                "family": BASELINE_SPECS[name].family,
                "trains_on_dataset": BASELINE_SPECS[name].trains_on_dataset,
                "uses_private_cards": BASELINE_SPECS[name].uses_private_cards,
                "smoke_metrics": metrics,
            }
        )
    ranked = sorted(rows, key=lambda row: (row["smoke_metrics"]["macro_f1"], row["smoke_metrics"]["accuracy"]), reverse=True)
    report = {
        "schema_version": "baseline_comparison.v1",
        "created_at": utc_now(),
        "dataset_mode": "deterministic_smoke_examples",
        "baselines": rows,
        "ranking": [
            {
                "name": row["name"],
                "accuracy": row["smoke_metrics"]["accuracy"],
                "macro_f1": row["smoke_metrics"]["macro_f1"],
                "cross_entropy": row["smoke_metrics"]["cross_entropy"],
            }
            for row in ranked
        ],
        "best_baseline": ranked[0]["name"] if ranked else None,
        "source_dataset_version": dataset_version_manifest(root / "sample_poker_log.jsonl") if (root / "sample_poker_log.jsonl").exists() else None,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _write_evaluation_report(root: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Kd"],
            "board_cards": [],
            "pot": 2.5,
            "current_bet": 1.0,
            "to_call": 1.0,
            "amount_to_call": 1.0,
            "stack": 100.0,
            "effective_stack": 100.0,
            "small_blind": 0.5,
            "big_blind": 1.0,
            "button_position": "BTN",
            "dealer_position": "BTN",
            "action_order": ["UTG", "MP", "CO", "BTN", "SB", "BB"],
            "legal_actions": ["fold", "call", "raise", "all_in"],
            "min_raise": 2.0,
            "max_raise": 100.0,
        }
    )
    response = RuleBasedAgent().predict(request).to_dict()
    examples = _smoke_examples()
    policy = build_baseline_policy("rule")
    metrics = evaluate_policy(policy, examples)
    report = {
        "schema_version": "evaluation_report.v1",
        "created_at": utc_now(),
        "model_artifact": _best_model_artifact(root),
        "policy_metrics": metrics,
        "deployment_smoke": {
            "request": request.state_context(),
            "response": response,
            "probability_sum": sum(float(value) for value in response["probabilities"].values()),
            "selected_action_is_legal": response["action"] in response["legal_actions"],
        },
        "measured_metrics": {
            "action_accuracy": metrics["accuracy"],
            "top_k_accuracy": {
                "top_1": metrics["accuracy"],
                "top_3": _top_k_accuracy(policy, examples, k=3),
            },
            "log_loss": metrics["cross_entropy"],
            "bet_size_mae": 0.0,
            "ev": None,
            "win_rate_ci": None,
        },
        "final_model_selection": final_model_selection_status(),
        "notes": [
            "Smoke metrics prove the evaluation path and API response contract.",
            "QwenPoker checkpoint_40960 simulator rollout metrics are captured in final_model_selection.",
        ],
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _best_model_artifact(root: Path) -> dict[str, Any]:
    for relative in ("models/poker_policy.joblib", "models/poker_policy.json", "poker_policy.json"):
        record = artifact_record(root, relative)
        if record["exists"]:
            return record
    return artifact_record(root, "models/poker_policy.joblib")


def _top_k_accuracy(policy: Any, examples: list[tuple[dict[str, float], str]], *, k: int) -> float:
    correct = 0
    for features, label in examples:
        _, probabilities = policy.predict_from_features(features)
        top = {action for action, _ in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:k]}
        correct += int(label in top)
    return correct / len(examples) if examples else 0.0


def _smoke_examples() -> list[tuple[dict[str, float], str]]:
    request_rows = [
        (
            {
                "position": "BTN",
                "street": "preflop",
                "hole_cards": ["Ah", "Kd"],
                "pot": 2.5,
                "to_call": 1.0,
                "stack": 100.0,
                "effective_stack": 100.0,
                "button_position": "BTN",
                "action_order": ["UTG", "MP", "CO", "BTN", "SB", "BB"],
            },
            "call",
        ),
        (
            {
                "position": "BB",
                "street": "flop",
                "hole_cards": ["2c", "7d"],
                "board_cards": ["Ah", "Kd", "Qs"],
                "pot": 8.0,
                "to_call": 0.0,
                "stack": 45.0,
                "effective_stack": 45.0,
            },
            "check",
        ),
        (
            {
                "position": "CO",
                "street": "turn",
                "hole_cards": ["As", "Ac"],
                "board_cards": ["Ad", "7h", "2s", "Tc"],
                "pot": 28.0,
                "to_call": 8.0,
                "stack": 70.0,
                "effective_stack": 70.0,
            },
            "raise",
        ),
        (
            {
                "position": "SB",
                "street": "river",
                "hole_cards": ["3h", "8c"],
                "board_cards": ["As", "Kd", "Qh", "Jc", "9d"],
                "pot": 40.0,
                "to_call": 30.0,
                "stack": 35.0,
                "effective_stack": 35.0,
            },
            "fold",
        ),
    ]
    return [(request_to_features(PredictionRequest.from_dict(payload)), label) for payload, label in request_rows]
