from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_agent.acceptance_criteria import DEFAULT_ACCEPTANCE_CRITERIA
from poker_agent.action_space import CANONICAL_ACTIONS
from poker_agent.baselines import baseline_names
from poker_agent.dataset_schema import ACTION_FIELDS, DATASET_SCHEMA_REQUIRED_FIELDS, HAND_FIELDS, PLAYER_FIELDS, STACK_FIELDS
from poker_agent.mlops import sha256_file, stable_digest, utc_now

PROJECT_SCOPE_CONTRACT_VERSION = "project_scope.v1"

PROJECT_GOALS = (
    "learn_from_historical_human_poker_hands",
    "improve_via_offline_self_play",
    "match_human_decision_patterns",
    "deploy_as_authorized_microservice",
)

PROJECT_PHASES: tuple[dict[str, Any], ...] = (
    {
        "key": "phase_1_baselines",
        "title": "Phase 1 - Two Baselines",
        "tracks": ["llm_based_agent", "end_to_end_policy_model"],
        "deliverables": [
            "working_llm_decision_agent",
            "trained_supervised_policy_checkpoint",
            "baseline_evaluation_metrics",
        ],
        "evidence": [
            "poker_agent/baselines.py",
            "scripts/run_baselines.py",
            "reports/baseline_report.json",
            "tests/test_baseline_contract.py",
        ],
    },
    {
        "key": "phase_2_selection_optimization",
        "title": "Phase 2 - Selection and Optimization",
        "tracks": ["baseline_comparison", "reproducible_experiments", "model_selection"],
        "deliverables": ["best_performing_model", "reproducible_pipeline"],
        "evidence": [
            "configs/experiments/run_baselines.yaml",
            "configs/experiments/evaluate_policy.yaml",
            "reports/evaluation_report.json",
            "reports/model_registry.json",
        ],
    },
    {
        "key": "phase_3_evaluation",
        "title": "Phase 3 - Evaluation",
        "tracks": ["held_out_human_alignment", "simulation_performance", "seed_stability"],
        "deliverables": ["final_model", "evaluation_report"],
        "evidence": [
            "poker_agent/evaluator.py",
            "poker_agent/acceptance_criteria.py",
            "reports/evaluation_report.json",
            "reports/final_model_selection.json",
            "docs/FINAL_MODEL_SELECTION.md",
            "tests/test_acceptance_criteria_contract.py",
            "tests/test_final_model_selection_contract.py",
        ],
    },
    {
        "key": "phase_4_deployment",
        "title": "Phase 4 - Deployment",
        "tracks": ["fastapi_service", "predict_endpoint", "dockerized_runtime", "api_docs"],
        "deliverables": ["deployable_agent_service"],
        "evidence": [
            "poker_agent/service.py",
            "poker_agent/api_contract.py",
            "Dockerfile",
            "docker-compose.yml",
            "docs/API_CONTRACT.md",
        ],
    },
)

SENIOR_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "key": "game_scope",
        "title": "Game Scope and Operating Boundary",
        "required_contract": {
            "game_type": "nl_holdem",
            "formats": ["cash", "tournament"],
            "table_sizes": ["6_max", "9_max"],
            "stack_unit": "chips_or_big_blinds",
            "allowed_usage": ["offline_research", "simulation", "authorized_environment"],
            "blocked_usage": ["real_money_platform", "unauthorized_platform", "stealth_automation", "tos_bypass"],
        },
        "evidence": [
            "poker_agent/game_scope.py",
            "poker_agent/usage_boundary.py",
            "poker_agent/schemas.py",
            "poker_agent/security.py",
            "tests/test_game_scope_usage_boundary_contract.py",
            "README.md",
        ],
    },
    {
        "key": "deployment_api",
        "title": "Deployment API Contract",
        "required_contract": {
            "service": "FastAPI",
            "endpoints": ["/health.json", "/contract.json", "/predict"],
            "response_fields": [
                "schema_version",
                "request_id",
                "model_version",
                "action",
                "probabilities",
                "confidence",
                "legal_actions",
                "action_space",
                "state_context",
            ],
        },
        "evidence": ["poker_agent/api_contract.py", "poker_agent/service.py", "docs/API_CONTRACT.md", "tests/test_api_contract.py"],
    },
    {
        "key": "dataset_schema_extensions",
        "title": "Dataset Schema Extensions",
        "required_contract": DATASET_SCHEMA_REQUIRED_FIELDS,
        "evidence": ["poker_agent/dataset_schema.py", "configs/dataset/poker_csv.yaml", "tests/test_dataset_schema_contract.py"],
    },
    {
        "key": "data_validation",
        "title": "Data Validation Rules",
        "required_contract": [
            "pot_conservation",
            "stack_delta_consistency",
            "duplicate_hand_detection",
            "missing_ocr_conflict_policy",
        ],
        "evidence": ["poker_agent/data_validation.py", "scripts/audit_dataset.py", "tests/test_data_validation_contract.py"],
    },
    {
        "key": "labeling_contract",
        "title": "Labeling Contract",
        "required_contract": {
            "human_action_label": list(CANONICAL_ACTIONS),
            "bet_size_label": ["raw_amount", "pot_ratio", "stack_ratio", "bb_ratio"],
            "timing_label": ["waiting_time", "clipped_outliers"],
            "confidence_weighting": True,
            "noisy_label_filtering": True,
        },
        "evidence": ["poker_agent/features.py", "poker_agent/action_space.py", "build_poker_dataset_optimized.py"],
    },
    {
        "key": "action_and_state_space",
        "title": "Action and State Space",
        "required_contract": {
            "actions": list(CANONICAL_ACTIONS),
            "sizing": ["raise_to", "raise_by", "min_raise_to", "max_raise_to", "min_raise_by", "max_raise_by", "all_in_amount"],
            "state": ["pot_size", "current_bet", "amount_to_call", "effective_stack", "spr", "action_order"],
        },
        "evidence": ["poker_agent/action_space.py", "poker_agent/features.py", "tests/test_state_features_contract.py"],
    },
    {
        "key": "baseline_and_architecture",
        "title": "Baseline and Model Architecture",
        "required_contract": {
            "baselines": list(baseline_names()),
            "policy_heads": ["policy_head", "bet_size_regression_head", "value_head"],
        },
        "evidence": ["poker_agent/baselines.py", "poker_agent/model.py", "poker_agent/sequence_models.py", "tests/test_baseline_contract.py"],
    },
    {
        "key": "rl_self_play",
        "title": "RL Environment and Self-Play",
        "required_contract": ["simulator", "self_play_league", "opponent_pool", "seed_policy", "reward_shaping"],
        "evidence": ["poker_agent/rl_environment.py", "scripts/inspect_rl_environment.py", "tests/test_rl_environment_contract.py"],
    },
    {
        "key": "evaluation_acceptance",
        "title": "Evaluation and Acceptance Criteria",
        "required_contract": {
            "metrics": ["action_accuracy", "top_k_accuracy", "log_loss", "bet_size_mae", "ev", "win_rate_ci"],
            "latency_p95_ms_max": DEFAULT_ACCEPTANCE_CRITERIA.latency_p95_ms_max,
            "latency_p99_ms_max": DEFAULT_ACCEPTANCE_CRITERIA.latency_p99_ms_max,
            "invalid_action_rate_max": DEFAULT_ACCEPTANCE_CRITERIA.invalid_action_rate_max,
            "validation_pass_rate_min": DEFAULT_ACCEPTANCE_CRITERIA.validation_pass_rate_min,
            "reproducibility_pass_rate_min": DEFAULT_ACCEPTANCE_CRITERIA.reproducibility_pass_rate_min,
        },
        "evidence": ["poker_agent/evaluator.py", "poker_agent/acceptance_criteria.py", "reports/evaluation_report.json"],
    },
    {
        "key": "mlops_monitoring_security",
        "title": "MLOps, Monitoring, and Security",
        "required_contract": ["experiment_tracking", "model_registry", "dataset_versioning", "monitoring", "api_auth", "rate_limiting", "log_retention"],
        "evidence": ["poker_agent/mlops.py", "poker_agent/monitoring.py", "poker_agent/security.py"],
    },
    {
        "key": "final_deliverables",
        "title": "Final Deliverables",
        "required_contract": [
            "validated_dataset_schema",
            "validation_report",
            "baseline_comparison",
            "trained_checkpoint",
            "evaluation_report",
            "dockerized_fastapi_service",
            "api_docs",
            "tests",
        ],
        "evidence": ["poker_agent/deliverables.py", "reports/final_deliverables.json", "tests/test_final_deliverables_contract.py"],
    },
)


def describe_project_scope_contract() -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCOPE_CONTRACT_VERSION,
        "goals": list(PROJECT_GOALS),
        "phases": [dict(item) for item in PROJECT_PHASES],
        "dataset_model": {
            "hands.csv": {"granularity": "one_hand", "primary_key": "hand_id", "fields": list(HAND_FIELDS)},
            "players.csv": {"granularity": "one_player_in_hand", "primary_key": "hand_id+position", "fields": list(PLAYER_FIELDS)},
            "actions.csv": {"granularity": "one_player_action", "primary_key": "hand_id+frame_id+player_position", "fields": list(ACTION_FIELDS)},
            "stack_events.csv": {"granularity": "one_stack_change_event", "primary_key": "hand_id+frame_id+player_position+event", "fields": list(STACK_FIELDS)},
        },
        "senior_requirements": [dict(item) for item in SENIOR_REQUIREMENTS],
    }


def validate_project_scope(root: Path) -> dict[str, Any]:
    root = root.resolve()
    contract = describe_project_scope_contract()
    phase_checks = [_evidence_check(root, phase["key"], phase["evidence"]) for phase in contract["phases"]]
    requirement_checks = [
        _evidence_check(root, requirement["key"], requirement["evidence"])
        for requirement in contract["senior_requirements"]
    ]
    dataset_checks = _dataset_model_checks(contract["dataset_model"])
    checks = phase_checks + requirement_checks + dataset_checks
    payload = {
        "schema_version": PROJECT_SCOPE_CONTRACT_VERSION,
        "created_at": utc_now(),
        "root": str(root),
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
    }
    payload["fingerprint"] = stable_digest(
        {
            "schema_version": payload["schema_version"],
            "checks": [
                {
                    "name": check["name"],
                    "passed": check["passed"],
                    "missing": check.get("missing", []),
                    "artifact_hashes": [artifact.get("sha256") for artifact in check.get("artifacts", []) if artifact.get("sha256")],
                }
                for check in checks
            ],
        }
    )
    return payload


def write_project_scope_reports(
    root: Path,
    *,
    out: Path | None = None,
    docs_out: Path | None = None,
) -> dict[str, Path]:
    root = root.resolve()
    out = out or root / "reports" / "project_scope_contract.json"
    docs_out = docs_out or root / "docs" / "PROJECT_SCOPE_CONTRACT.md"
    contract = describe_project_scope_contract()
    validation = validate_project_scope(root)
    payload = {
        "contract": contract,
        "validation": validation,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    docs_out.write_text(project_scope_markdown(contract, validation), encoding="utf-8")
    return {"json": out, "docs": docs_out}


def project_scope_markdown(contract: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        "# Poker ML Project Scope Contract",
        "",
        f"- Schema: `{contract['schema_version']}`",
        f"- Validation status: `{validation['status']}`",
        f"- Fingerprint: `{validation['fingerprint']}`",
        "",
        "## Goals",
        "",
        *[f"- `{goal}`" for goal in contract["goals"]],
        "",
        "## Phases",
        "",
    ]
    for phase in contract["phases"]:
        lines.extend(
            [
                f"### {phase['title']}",
                "",
                f"- Tracks: {', '.join(phase['tracks'])}",
                f"- Deliverables: {', '.join(phase['deliverables'])}",
                "",
            ]
        )
    lines.extend(["## Senior Requirements", ""])
    for requirement in contract["senior_requirements"]:
        lines.extend([f"### {requirement['title']}", "", f"- Key: `{requirement['key']}`", ""])
    lines.extend(["## Dataset Tables", ""])
    for table, spec in contract["dataset_model"].items():
        lines.append(f"- `{table}`: {spec['granularity']} ({len(spec['fields'])} fields)")
    lines.append("")
    return "\n".join(lines)


def _evidence_check(root: Path, name: str, evidence: list[str]) -> dict[str, Any]:
    artifacts = [_artifact_record(root, relative) for relative in evidence]
    missing = [item["path"] for item in artifacts if not item["exists"]]
    return {
        "name": name,
        "passed": not missing,
        "missing": missing,
        "artifacts": artifacts,
    }


def _artifact_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {"path": relative, "exists": path.exists()}
    if path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    elif path.is_dir():
        record.update({"kind": "directory"})
    return record


def _dataset_model_checks(dataset_model: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required_tables = {"hands.csv", "players.csv", "actions.csv", "stack_events.csv"}
    checks.append(
        {
            "name": "dataset_tables_declared",
            "passed": required_tables.issubset(set(dataset_model)),
            "missing": sorted(required_tables - set(dataset_model)),
        }
    )
    action_required = set(DATASET_SCHEMA_REQUIRED_FIELDS["actions.csv"])
    actions = set(dataset_model["actions.csv"]["fields"])
    checks.append(
        {
            "name": "actions_schema_extended_context",
            "passed": action_required.issubset(actions),
            "missing": sorted(action_required - actions),
        }
    )
    return checks
