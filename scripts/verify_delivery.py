from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.acceptance_criteria import (
    DEFAULT_ACCEPTANCE_CRITERIA,
    build_acceptance_metrics,
    evaluate_acceptance_criteria,
)
from poker_agent.agents import MLPolicyAgent, RuleBasedAgent
from poker_agent.api_contract import (
    ERROR_CODES,
    PREDICT_RESPONSE_SCHEMA_VERSION,
    deployment_api_contract as build_deployment_api_contract,
    predict_request_schema,
    predict_response_schema,
)
from poker_agent.baselines import baseline_names, build_baseline_policy
from poker_agent.dataset_schema import ACTION_FIELDS, DATASET_SCHEMA_REQUIRED_FIELDS, HAND_FIELDS
from poker_agent.data_validation import validate_dataset
from poker_agent.deliverables import (
    FINAL_DELIVERABLES_CONTRACT_VERSION,
    describe_final_deliverables_contract,
    validate_final_deliverables,
)
from poker_agent.features import request_to_features
from poker_agent.final_model_selection import (
    FINAL_MODEL_SELECTION_SCHEMA_VERSION,
    describe_final_model_selection,
    final_model_selection_status,
    validate_final_model_selection,
)
from poker_agent.game_scope import GameScope, describe_game_scope_contract
from poker_agent.legacy_reports import (
    LEGACY_REPORTS_CONTRACT_VERSION,
    describe_legacy_reports_contract,
    validate_legacy_delivery_reports,
)
from poker_agent.mlops import (
    ci_smoke_contract as build_ci_smoke_contract,
    dataset_version_manifest,
    describe_mlops_contract,
    docker_image_metadata,
    experiment_run_manifest,
    model_registry_entry,
    update_model_registry,
    validate_mlops_contract,
)
from poker_agent.monitoring import (
    MonitoringThresholds,
    audit_trail_event,
    describe_monitoring_contract,
    drift_report,
    invalid_state_findings,
    prediction_log_event,
    validate_monitoring_contract,
)
from poker_agent.model import load_policy
from poker_agent.project_scope import (
    PROJECT_SCOPE_CONTRACT_VERSION,
    describe_project_scope_contract,
    validate_project_scope,
)
from poker_agent.rl_environment import (
    NoLimitHoldemSingleDecisionEngine,
    SeedPolicy,
    SelfPlayLeague,
    describe_rl_environment,
    seed_policy_hero,
)
from poker_agent.schemas import PredictionRequest
from poker_agent.security import (
    InMemoryRateLimiter,
    LogRetentionPolicy,
    SecurityConfig,
    authenticate_headers,
    describe_security_contract,
    hash_secret,
    prune_jsonl_by_retention,
    redact_mapping,
    validate_security_contract,
)
from poker_agent.usage_boundary import (
    ALLOWED_USAGE,
    BLOCKED_USAGE,
    describe_usage_boundary_contract,
    evaluate_usage_boundary,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the poker agent delivery package")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--model", default=ROOT / "models" / "poker_policy.joblib", type=Path)
    parser.add_argument("--zip", default=ROOT / "release" / "poker-decision-agent.zip", type=Path)
    parser.add_argument("--require-gate-pass", action="store_true")
    parser.add_argument("--json-out", default=None, type=Path)
    return parser.parse_args()


def run_check(name: str, fn: Callable[[], str]) -> Check:
    try:
        return Check(name=name, passed=True, detail=fn())
    except Exception as exc:
        return Check(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")


def require_files(root: Path) -> str:
    required = [
        "README.md",
        "requirements.txt",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/mlops/local.yaml",
        "configs/monitoring/local.yaml",
        "configs/security/local.yaml",
        "configs/deliverables/local.yaml",
        "configs/legacy_reports/local.yaml",
        "configs/project_scope/local.yaml",
        "configs/legacy_reports/local.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/model/tabular_compare.yaml",
        "configs/model/routed_bundle_smoke.yaml",
        "configs/model/text_event_local_rules.yaml",
        "configs/model/text_event_smol.yaml",
        "configs/training/group_holdout.yaml",
        "configs/training/smoke.yaml",
        "configs/evaluation/standard.yaml",
        "configs/inference/local_service.yaml",
        "configs/logging/local.yaml",
        "configs/mlops/local.yaml",
        "configs/monitoring/local.yaml",
        "configs/rl/self_play_league.yaml",
        "configs/prompts/event_extraction_prompt.txt",
        "configs/prompts/event_extraction_minimal.txt",
        "configs/prompts/event_extraction_permissive.txt",
        "configs/prompts/event_extraction_strict.txt",
        "configs/prompts/event_extraction_fewshot.txt",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/evaluate_policy.yaml",
        "configs/experiments/research_compare_tabular.yaml",
        "configs/experiments/audit_dataset.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/production_gate.yaml",
        "configs/experiments/acceptance_criteria.yaml",
        "configs/experiments/run_baselines.yaml",
        "configs/experiments/rl_self_play_smoke.yaml",
        "configs/experiments/mlops_smoke.yaml",
        "configs/experiments/monitoring_smoke.yaml",
        "configs/experiments/security_smoke.yaml",
        "configs/experiments/final_deliverables.yaml",
        "configs/experiments/legacy_delivery_reports.yaml",
        "configs/experiments/project_scope.yaml",
        "configs/experiments/legacy_delivery_reports.yaml",
        "configs/experiments/train_routed_bundle_smoke.yaml",
        "configs/experiments/llm_event_extraction_smoke.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_transformer_gold_eval.yaml",
        "configs/experiments/verify_delivery.yaml",
        "Dockerfile",
        "docker-compose.yml",
        "install.ps1",
        "run_server.ps1",
        "complete_delivery.ps1",
        "verify_delivery.ps1",
        "build_poker_dataset_optimized.py",
        "models/poker_policy.joblib",
        "models/poker_policy.metadata.json",
        "models/poker_policy.metadata.json",
        "reports/dataset_audit.json",
        "reports/repository_audit.json",
        "reports/production_gate.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_report.md",
        "reports/llm_event_methodology.md",
        "reports/llm_transformer_gold_eval.json",
        "reports/llm_transformer_gold_report.md",
        "reports/dataset_validation_report.json",
        "reports/baseline_report.json",
        "reports/evaluation_report.json",
        "reports/final_model_selection.json",
        "reports/project_scope_contract.json",
        "reports/final_deliverables.json",
        "reports/legacy_delivery_reports.json",
        "reports/delivery_report.md",
        "docs/API_CONTRACT.md",
        "docs/FINAL_MODEL_SELECTION.md",
        "docs/PROJECT_SCOPE_CONTRACT.md",
        "evaluation/event_extraction_gold.jsonl",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/evaluate_policy.py",
        "scripts/check_acceptance_criteria.py",
        "scripts/check_mlops_contract.py",
        "scripts/check_monitoring_contract.py",
        "scripts/check_security_contract.py",
        "scripts/check_project_scope_contract.py",
        "scripts/check_final_model_selection.py",
        "scripts/prepare_final_deliverables.py",
        "scripts/prepare_legacy_delivery_reports.py",
        "scripts/run_baselines.py",
        "scripts/inspect_rl_environment.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/check_repo_hygiene.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/production_gate.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
        "tests/test_action_space_contract.py",
        "tests/test_state_features_contract.py",
        "tests/test_dataset_schema_contract.py",
        "tests/test_data_validation_contract.py",
        "tests/test_baseline_contract.py",
        "tests/test_api_contract.py",
        "tests/test_mlops_contract.py",
        "tests/test_monitoring_contract.py",
        "tests/test_security_contract.py",
        "tests/test_final_deliverables_contract.py",
        "tests/test_legacy_delivery_contract.py",
        "tests/test_project_scope_contract.py",
        "tests/test_game_scope_usage_boundary_contract.py",
        "tests/test_final_model_selection_contract.py",
        "tests/test_acceptance_criteria_contract.py",
        "tests/test_rl_environment_contract.py",
        ".github/workflows/smoke.yml",
        "poker_agent/acceptance_criteria.py",
        "poker_agent/api_contract.py",
        "poker_agent/action_space.py",
        "poker_agent/baselines.py",
        "poker_agent/mlops.py",
        "poker_agent/monitoring.py",
        "poker_agent/final_model_selection.py",
        "poker_agent/project_scope.py",
        "poker_agent/game_scope.py",
        "poker_agent/usage_boundary.py",
        "poker_agent/security.py",
        "poker_agent/deliverables.py",
        "poker_agent/legacy_reports.py",
        "poker_agent/project_scope.py",
        "poker_agent/final_model_selection.py",
        "poker_agent/game_scope.py",
        "poker_agent/usage_boundary.py",
        "poker_agent/data_validation.py",
        "poker_agent/dataset_schema.py",
        "poker_agent/rl_environment.py",
        "poker_agent/service.py",
        "poker_agent/agents.py",
        "poker_agent/features.py",
        "poker_agent/model.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
    ]
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")
    return f"{len(required)} required files present"


def compile_sources(root: Path) -> str:
    source_files = [
        "build_poker_dataset_optimized.py",
        "poker_agent/action_space.py",
        "poker_agent/acceptance_criteria.py",
        "poker_agent/api_contract.py",
        "poker_agent/agents.py",
        "poker_agent/baselines.py",
        "poker_agent/data_validation.py",
        "poker_agent/dataset_schema.py",
        "poker_agent/deliverables.py",
        "poker_agent/evaluator.py",
        "poker_agent/features.py",
        "poker_agent/legacy_reports.py",
        "poker_agent/mlops.py",
        "poker_agent/monitoring.py",
        "poker_agent/final_model_selection.py",
        "poker_agent/game_scope.py",
        "poker_agent/usage_boundary.py",
        "poker_agent/security.py",
        "poker_agent/model.py",
        "poker_agent/rl_environment.py",
        "poker_agent/schemas.py",
        "poker_agent/service.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/check_repo_hygiene.py",
        "scripts/check_acceptance_criteria.py",
        "scripts/check_mlops_contract.py",
        "scripts/check_monitoring_contract.py",
        "scripts/check_security_contract.py",
        "scripts/check_project_scope_contract.py",
        "scripts/check_final_model_selection.py",
        "scripts/prepare_final_deliverables.py",
        "scripts/prepare_legacy_delivery_reports.py",
        "scripts/evaluate_policy.py",
        "scripts/run_baselines.py",
        "scripts/inspect_rl_environment.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/production_gate.py",
        "scripts/research_experiment.py",
        "scripts/run_hydra_experiment.py",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/verify_delivery.py",
        "tests/test_action_space_contract.py",
        "tests/test_acceptance_criteria_contract.py",
        "tests/test_state_features_contract.py",
        "tests/test_dataset_schema_contract.py",
        "tests/test_data_validation_contract.py",
        "tests/test_baseline_contract.py",
        "tests/test_api_contract.py",
        "tests/test_mlops_contract.py",
        "tests/test_monitoring_contract.py",
        "tests/test_security_contract.py",
        "tests/test_final_deliverables_contract.py",
        "tests/test_legacy_delivery_contract.py",
        "tests/test_project_scope_contract.py",
        "tests/test_game_scope_usage_boundary_contract.py",
        "tests/test_final_model_selection_contract.py",
        "tests/test_rl_environment_contract.py",
    ]
    for relative in source_files:
        path = root / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return f"{len(source_files)} Python files compile without writing bytecode"


def model_loads(model_path: Path) -> str:
    model = load_policy(model_path)
    metadata = getattr(model, "metadata", {}) or {}
    if not metadata:
        raise AssertionError("Model artifact has no metadata")
    split = (metadata.get("split") or {}).get("split_type")
    if split != "stratified_hand_group_holdout":
        raise AssertionError(f"Unexpected split: {split}")
    valid = metadata.get("valid_metrics") or {}
    if "macro_f1" not in valid:
        raise AssertionError("Model metadata does not include validation metrics")
    return f"model={model_path.name}, policy={metadata.get('policy')}, macro_f1={valid['macro_f1']:.4f}"


def inference_contract(model_path: Path) -> str:
    canonical_actions = {"fold", "check", "call", "bet", "raise", "all_in"}
    agent = MLPolicyAgent.from_path(model_path)
    observed = agent.predict(
        PredictionRequest(
            position="BTN",
            street="preflop",
            hole_cards=["Ah", "Kd"],
            board_cards=[],
            pot=2.5,
            to_call=1.0,
            current_bet=1.0,
            amount_to_call=1.0,
            stack=100.0,
            effective_stack=100.0,
            min_raise=2.0,
            max_raise=100.0,
            small_blind=0.5,
            big_blind=1.0,
            button_position="BTN",
            dealer_position="BTN",
            action_order=["UTG", "MP", "CO", "BTN", "SB", "BB"],
            player_count=6,
        )
    ).to_dict()
    missing = agent.predict(
        PredictionRequest(
            position="BTN",
            street="preflop",
            hole_cards=[],
            board_cards=[],
            pot=2.5,
            to_call=1.0,
            current_bet=1.0,
            amount_to_call=1.0,
            stack=100.0,
            effective_stack=100.0,
            min_raise=2.0,
            max_raise=100.0,
            small_blind=0.5,
            big_blind=1.0,
            button_position="BTN",
            dealer_position="BTN",
            action_order=["UTG", "MP", "CO", "BTN", "SB", "BB"],
            player_count=6,
        )
    ).to_dict()
    state_features = request_to_features(
        PredictionRequest(
            position="BTN",
            street="turn",
            pot=25.0,
            to_call=4.0,
            current_bet=6.0,
            amount_to_call=4.0,
            stack=90.0,
            effective_stack=80.0,
            min_raise=8.0,
            small_blind=0.5,
            big_blind=1.0,
            button_position="BTN",
            dealer_position="BTN",
            action_order=["UTG", "CO", "BTN", "SB", "BB"],
        )
    )
    if observed["model_status"] == "missing_card_fallback":
        raise AssertionError("Observed-card request incorrectly used fallback")
    if missing["model_status"] != "missing_card_fallback":
        raise AssertionError("Missing-card request did not use fallback")
    for payload in (observed, missing):
        if payload.get("schema_version") != PREDICT_RESPONSE_SCHEMA_VERSION:
            raise AssertionError(f"Response schema version is missing or unexpected: {payload}")
        if not payload.get("model_version"):
            raise AssertionError(f"Response model_version is missing: {payload}")
        if not 0.0 <= float(payload.get("confidence", -1.0)) <= 1.0:
            raise AssertionError(f"Response confidence is outside [0, 1]: {payload}")
        if set(payload["probabilities"]) != canonical_actions:
            raise AssertionError(f"Response probabilities are not canonical: {payload['probabilities']}")
        if payload["action"] not in payload.get("legal_actions", []):
            raise AssertionError(f"Selected action is not legal: {payload}")
        if payload["probabilities"].get("check", 0.0) != 0.0:
            raise AssertionError(f"Illegal check retained while facing a bet: {payload['probabilities']}")
        if payload["probabilities"].get("bet", 0.0) != 0.0:
            raise AssertionError(f"Illegal bet retained while facing a bet: {payload['probabilities']}")
        action_space = payload.get("action_space") or {}
        required_action_space = {"legal_actions", "min_raise_to", "max_raise_to", "min_raise_by", "max_raise_by", "all_in_amount"}
        missing_fields = sorted(required_action_space - set(action_space))
        if missing_fields:
            raise AssertionError(f"Action space metadata is incomplete: {missing_fields}")
        state_context = payload.get("state_context") or {}
        required_state_context = {"pot_size", "current_bet", "amount_to_call", "button_position", "dealer_position", "effective_stack", "spr", "action_order"}
        missing_state = sorted(required_state_context - set(state_context))
        if missing_state:
            raise AssertionError(f"State context metadata is incomplete: {missing_state}")
        total = sum(float(value) for value in payload["probabilities"].values())
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"Probabilities do not sum to 1: {total}")
    required_state_features = {
        "pot_size",
        "current_bet",
        "amount_to_call",
        "button_position_known",
        "dealer_position_known",
        "small_blind",
        "big_blind",
        "street=turn",
        "effective_stack",
        "spr",
        "action_order_known",
        "action_order_index",
        "players_before_hero",
        "players_after_hero",
    }
    missing_state_features = sorted(required_state_features - set(state_features))
    if missing_state_features:
        raise AssertionError(f"State feature contract is incomplete: {missing_state_features}")
    return f"observed={observed['action']} missing={missing['action']}"


def deployment_api_contract() -> str:
    request_schema = predict_request_schema()
    response_schema = predict_response_schema()
    required_request = {"position", "street", "hole_cards", "pot", "stack"}
    if not required_request.issubset(set(request_schema.get("required", []))):
        raise AssertionError(f"Predict request schema lost required fields: {request_schema.get('required')}")
    request_properties = request_schema.get("properties", {})
    for field in ("legal_actions", "legal_action_mask", "game_scope", "usage_boundary", "amount_to_call", "effective_stack"):
        if field not in request_properties:
            raise AssertionError(f"Predict request schema missing {field}")
    if "usage_boundary" not in request_schema.get("required", []):
        raise AssertionError("Predict request schema must require usage_boundary")

    required_response = {
        "schema_version",
        "model_version",
        "action",
        "probabilities",
        "confidence",
        "model_status",
        "legal_actions",
        "action_space",
        "state_context",
    }
    if not required_response.issubset(set(response_schema.get("required", []))):
        raise AssertionError(f"Predict response schema lost required fields: {response_schema.get('required')}")
    probability_schema = response_schema["properties"]["probabilities"]
    if set(probability_schema.get("required", [])) != {"fold", "check", "call", "bet", "raise", "all_in"}:
        raise AssertionError(f"Probability schema is not canonical: {probability_schema}")

    required_error_codes = {
        "INVALID_REQUEST": (400, False),
        "UNSUPPORTED_ACTION_SPACE": (422, False),
        "MODEL_UNAVAILABLE": (503, True),
        "UNAUTHORIZED": (401, False),
        "RATE_LIMITED": (429, True),
        "SECURITY_MISCONFIGURED": (503, True),
        "USAGE_BOUNDARY_VIOLATION": (403, False),
        "PREDICTION_FAILED": (500, True),
    }
    for code, (status, retryable) in required_error_codes.items():
        spec = ERROR_CODES.get(code)
        if spec is None:
            raise AssertionError(f"Missing deployment error code: {code}")
        if spec.get("http_status") != status or spec.get("retryable") is not retryable:
            raise AssertionError(f"Unexpected error-code contract for {code}: {spec}")

    contract = build_deployment_api_contract(model_version="verification-policy:v1")
    if contract["endpoint"] != "/predict" or contract["model_version"] != "verification-policy:v1":
        raise AssertionError(f"Deployment API contract is incomplete: {contract}")

    payload = RuleBasedAgent().predict(
        PredictionRequest.from_dict(
            {
                "position": "BTN",
                "street": "preflop",
                "hole_cards": ["Ah", "Kd"],
                "pot": 2.5,
                "to_call": 1.0,
                "stack": 100.0,
                "min_raise": 2.0,
                "max_raise": 100.0,
            }
        )
    ).to_dict()
    missing_response = sorted(required_response - set(payload))
    if missing_response:
        raise AssertionError(f"Rule agent response is missing deployment fields: {missing_response}")
    if payload["schema_version"] != PREDICT_RESPONSE_SCHEMA_VERSION:
        raise AssertionError(f"Unexpected response schema version: {payload['schema_version']}")
    if payload["model_version"] != "rule_based:v1":
        raise AssertionError(f"Unexpected rule model version: {payload['model_version']}")
    return "predict request/response schemas, model_version, confidence, probabilities, and error codes pass"


def game_scope_usage_boundary_contract() -> str:
    scope_contract = describe_game_scope_contract()
    if scope_contract["supported_game_types"] != ["nl_holdem"]:
        raise AssertionError(f"Unexpected game scope contract: {scope_contract}")
    if set(scope_contract["supported_table_sizes"]) != {"6_max", "9_max"}:
        raise AssertionError(f"Game scope table sizes are incomplete: {scope_contract}")

    request = PredictionRequest.from_dict(
        {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Kd"],
            "pot": 5.0,
            "stack": 100.0,
            "game_scope": {
                "game_type": "no-limit-holdem",
                "format": "cash_game",
                "table_size": "9max",
                "small_blind": 1.0,
                "big_blind": 2.0,
                "ante": 0.25,
                "rake_percentage": 5,
                "rake_cap": 3,
                "stack_unit": "bb",
            },
        }
    )
    if request.game_scope.validate():
        raise AssertionError(f"Valid game scope was rejected: {request.game_scope.validate()}")
    if request.player_count != 9 or request.game_scope.rake_percentage != 0.05:
        raise AssertionError(f"Game scope normalization failed: {request.game_scope}")
    features = request_to_features(request)
    for feature in ("scope_game_type_nl_holdem", "scope_format_cash", "scope_table_9_max", "scope_stack_unit_big_blinds"):
        if features.get(feature) != 1.0:
            raise AssertionError(f"Missing game-scope feature: {feature}")

    usage_contract = describe_usage_boundary_contract()
    if usage_contract["allowed_usage"] != list(ALLOWED_USAGE) or usage_contract["blocked_usage"] != list(BLOCKED_USAGE):
        raise AssertionError(f"Usage boundary contract drifted: {usage_contract}")
    if usage_contract["blocked_http_status"] != 403:
        raise AssertionError(f"Usage boundary must block with HTTP 403: {usage_contract}")

    for declared_use in ALLOWED_USAGE:
        decision = evaluate_usage_boundary({"usage_boundary": {"declared_use": declared_use}})
        if not decision.allowed:
            raise AssertionError(f"Allowed use was rejected: {decision}")

    for blocked_use in BLOCKED_USAGE:
        decision = evaluate_usage_boundary({"usage_boundary": {"declared_use": "offline_research", "prohibited_use": blocked_use}})
        if decision.allowed:
            raise AssertionError(f"Blocked use was allowed: {blocked_use}")

    missing = evaluate_usage_boundary({})
    if missing.allowed or missing.reason_code != "missing_usage_boundary":
        raise AssertionError(f"Missing usage boundary was not blocked: {missing}")

    return "game scope normalized and usage boundary enforces allowed/off-limits use cases"


def final_model_selection_contract(root: Path) -> str:
    selection = describe_final_model_selection()
    if selection["schema_version"] != FINAL_MODEL_SELECTION_SCHEMA_VERSION:
        raise AssertionError(f"Unexpected final model selection schema: {selection['schema_version']}")
    if selection["selected_model"]["checkpoint"] != "checkpoint_40960":
        raise AssertionError(f"Unexpected selected checkpoint: {selection['selected_model']}")

    status = final_model_selection_status(selection)
    if status["status"] != "PASS":
        raise AssertionError(f"Final model selection contract failed: {status['checks']}")
    failed = [item for item in validate_final_model_selection(selection) if not item["passed"]]
    if failed:
        raise AssertionError(f"Final model selection validation failed: {failed}")

    report = json.loads((root / "reports" / "final_model_selection.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise AssertionError(f"Generated final model selection report failed: {report.get('status')}")
    if report["selected_model"]["checkpoint"] != "checkpoint_40960":
        raise AssertionError(f"Generated final model selection report drifted: {report['selected_model']}")
    metrics = report.get("metrics", {})
    if metrics.get("win_rate") != 0.6448 or metrics.get("bb_per_100") != 365.29:
        raise AssertionError(f"QwenPoker benchmark metrics drifted: {metrics}")
    if not metrics.get("returns_ci_95", {}).get("is_entirely_positive"):
        raise AssertionError(f"Return CI gate is not positive: {metrics.get('returns_ci_95')}")
    if not metrics.get("position_profitability", {}).get("both_positions_profitable"):
        raise AssertionError(f"Position profitability gate failed: {metrics.get('position_profitability')}")

    docs = (root / "docs" / "FINAL_MODEL_SELECTION.md").read_text(encoding="utf-8")
    for token in ("QwenPoker", "checkpoint_40960", "64.48%", "+365.29 BB/100", "OpenSpiel FCHPA"):
        if token not in docs:
            raise AssertionError(f"Final model selection docs missing token: {token}")

    registry = json.loads((root / "reports" / "model_registry.json").read_text(encoding="utf-8"))
    qwen = registry.get("models", {}).get("qwen_poker", {})
    if qwen.get("latest_version") != "qwenpoker:checkpoint_40960":
        raise AssertionError(f"Model registry does not expose QwenPoker final checkpoint: {qwen}")

    return "QwenPoker checkpoint_40960 selected with balanced positive simulator benchmark gates"


def mlops_contract(root: Path) -> str:
    contract = describe_mlops_contract()
    if contract["experiment_tracking"]["backend"] != "local_jsonl":
        raise AssertionError(f"Unexpected experiment tracking backend: {contract['experiment_tracking']}")
    if contract["model_registry"]["path"] != "reports/model_registry.json":
        raise AssertionError(f"Unexpected model registry path: {contract['model_registry']}")
    if "production" not in contract["model_registry"]["stages"]:
        raise AssertionError("Model registry contract has no production stage")

    checks = validate_mlops_contract(root)
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise AssertionError(f"MLOps contract validation failed: {failed}")

    dataset = dataset_version_manifest(root / "sample_poker_log.jsonl")
    if dataset["schema_version"] != "dataset_version.v1" or not dataset["fingerprint"]:
        raise AssertionError(f"Dataset version manifest is incomplete: {dataset}")

    run = experiment_run_manifest(
        experiment_name="verify_delivery_mlops",
        command=["python", "scripts/verify_delivery.py"],
        parameters={"contract": "mlops"},
        metrics={"checks": len(checks)},
        artifacts=[{"type": "dataset_version", "version": dataset["version"]}],
        dataset_version=dataset["version"],
        seed=20260713,
    )
    if not run["run_id"].startswith("run-") or run["tracking_backend"] != "local_jsonl":
        raise AssertionError(f"Experiment run manifest is incomplete: {run}")

    docker = docker_image_metadata(dockerfile=root / "Dockerfile")
    if docker["tag"] != "poker-decision-agent:0.1.0":
        raise AssertionError(f"Docker image metadata is not versioned: {docker}")

    ci = build_ci_smoke_contract()
    if ".github/workflows/smoke.yml" != ci["workflow"]:
        raise AssertionError(f"CI smoke workflow path changed: {ci}")

    with tempfile.TemporaryDirectory() as raw_temp:
        registry_path = Path(raw_temp) / "model_registry.json"
        entry = model_registry_entry(
            root / "models" / "poker_policy.json",
            run_id=run["run_id"],
            dataset_version=dataset["version"],
            metrics={"smoke_contract": 1.0},
            stage="candidate",
            docker_image=docker["tag"],
        )
        registry = update_model_registry(registry_path, entry)
    latest = registry["models"]["poker_policy"]["latest_version"]
    if latest != entry["model_version"]:
        raise AssertionError(f"Model registry did not preserve latest version: {registry}")
    return "experiment tracking, model registry, dataset versioning, Docker versioning, and CI smoke contract pass"


def monitoring_contract(root: Path) -> str:
    contract = describe_monitoring_contract()
    required_signals = {
        "latency",
        "invalid_states",
        "confidence_drift",
        "feature_drift",
        "prediction_logs",
        "audit_trail",
    }
    if not required_signals.issubset(set(contract["signals"])):
        raise AssertionError(f"Monitoring contract is missing signals: {contract['signals']}")
    failed_files = [item for item in validate_monitoring_contract(root) if not item["passed"]]
    if failed_files:
        raise AssertionError(f"Monitoring contract files are missing: {failed_files}")

    def event(pot: float, to_call: float, request_id: str, latency_ms: float) -> dict[str, object]:
        payload = {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Kd"],
            "pot": pot,
            "to_call": to_call,
            "amount_to_call": to_call,
            "stack": 100.0,
            "effective_stack": 100.0,
            "min_raise": 2.0,
            "max_raise": 100.0,
        }
        request = PredictionRequest.from_dict(payload)
        response = RuleBasedAgent().predict(request).to_dict()
        return prediction_log_event(
            request_id=request_id,
            raw_payload=payload,
            request=request,
            response=response,
            latency_ms=latency_ms,
            status="ok",
        )

    baseline = [event(2.5, 1.0, "baseline-1", 18.0), event(4.0, 0.0, "baseline-2", 22.0)]
    current = [event(2.7, 1.0, "current-1", 19.0), event(4.2, 0.0, "current-2", 24.0)]
    report = drift_report(baseline, current, MonitoringThresholds())
    if report["status"] != "PASS":
        raise AssertionError(f"Monitoring drift report failed: {report}")

    invalid = invalid_state_findings({"street": "showdown", "pot": -1, "hole_cards": ["Ah", "Kd", "Qs"]})
    if not invalid:
        raise AssertionError("Invalid state detection did not flag malformed state")

    audit = audit_trail_event(
        request_id="current-1",
        event_type="prediction_recorded",
        payload={"prediction_log_hash": current[0]["feature_fingerprint"]},
    )
    if not audit.get("event_hash") or audit.get("schema_version") != "audit_trail.v1":
        raise AssertionError(f"Audit trail event is incomplete: {audit}")
    return "latency, invalid states, confidence drift, feature drift, prediction logs, and audit trail contract pass"


def security_contract(root: Path) -> str:
    contract = describe_security_contract()
    required_sections = {"api_auth", "rate_limiting", "secret_management", "log_retention"}
    if not required_sections.issubset(set(contract)):
        raise AssertionError(f"Security contract missing sections: {contract}")
    failed_files = [item for item in validate_security_contract(root) if not item["passed"]]
    if failed_files:
        raise AssertionError(f"Security contract files are missing: {failed_files}")

    secret = "verify-delivery-api-key"
    config = SecurityConfig(auth_required=True, api_key_hashes=(hash_secret(secret),))
    good = authenticate_headers({"x-api-key": secret}, config)
    missing = authenticate_headers({}, config)
    bad = authenticate_headers({"authorization": "Bearer wrong"}, config)
    if not good.allowed or missing.allowed or bad.allowed:
        raise AssertionError(f"API auth contract failed: good={good} missing={missing} bad={bad}")

    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)
    if not limiter.check("principal", now=1.0).allowed:
        raise AssertionError("Rate limiter rejected first request")
    if not limiter.check("principal", now=2.0).allowed:
        raise AssertionError("Rate limiter rejected second request")
    blocked = limiter.check("principal", now=3.0)
    if blocked.allowed or blocked.retry_after_seconds <= 0:
        raise AssertionError(f"Rate limiter did not reject over-limit request: {blocked}")

    redacted = redact_mapping({"api_key": secret, "nested": {"authorization": f"Bearer {secret}"}})
    if secret in json.dumps(redacted):
        raise AssertionError(f"Secret redaction leaked raw secret: {redacted}")

    with tempfile.TemporaryDirectory() as raw_temp:
        path = Path(raw_temp) / "prediction_logs.jsonl"
        old = {"created_at": "2020-01-01T00:00:00+00:00", "event": "old"}
        recent = {"created_at": "2026-07-13T00:00:00+00:00", "event": "recent"}
        path.write_text(json.dumps(old) + "\n" + json.dumps(recent) + "\n", encoding="utf-8")
        retention = prune_jsonl_by_retention(
            path,
            LogRetentionPolicy(max_age_days=30, max_records=100, enabled=True),
        )
        retained = path.read_text(encoding="utf-8")
    if retention["removed"] < 1 or "old" in retained:
        raise AssertionError(f"Log retention did not remove old record: {retention}")
    return "API auth, rate limiting, secret management, and log retention contract pass"


def final_deliverables_contract(root: Path) -> str:
    contract = describe_final_deliverables_contract()
    if contract["schema_version"] != FINAL_DELIVERABLES_CONTRACT_VERSION:
        raise AssertionError(f"Unexpected final deliverables schema: {contract['schema_version']}")
    expected = {
        "validated_dataset_schema",
        "validation_report",
        "baseline_comparison",
        "trained_checkpoint",
        "evaluation_report",
        "dockerized_fastapi_service",
        "api_docs",
        "tests",
    }
    declared = {item["key"] for item in contract["deliverables"]}
    missing_declared = sorted(expected - declared)
    if missing_declared:
        raise AssertionError(f"Final deliverables contract is missing keys: {missing_declared}")

    manifest = validate_final_deliverables(root)
    if manifest["status"] != "PASS":
        failed = [item for item in manifest["deliverables"] if item["status"] != "PASS"]
        raise AssertionError(f"Final deliverables are incomplete: {failed}")
    if not manifest.get("fingerprint"):
        raise AssertionError("Final deliverables manifest has no fingerprint")

    generated_manifest = json.loads((root / "reports" / "final_deliverables.json").read_text(encoding="utf-8"))
    if generated_manifest.get("status") != "PASS":
        raise AssertionError(f"Generated final deliverables report did not pass: {generated_manifest.get('status')}")

    dataset_report = json.loads((root / "reports" / "dataset_validation_report.json").read_text(encoding="utf-8"))
    if dataset_report.get("schema_version") != "dataset_validation_report.v1":
        raise AssertionError(f"Dataset validation report schema is unexpected: {dataset_report.get('schema_version')}")
    if dataset_report.get("status") != "PASS":
        raise AssertionError(f"Dataset validation report did not pass schema checks: {dataset_report.get('status')}")

    baseline_report = json.loads((root / "reports" / "baseline_report.json").read_text(encoding="utf-8"))
    baseline_keys = {item["name"] for item in baseline_report.get("baselines", [])}
    if set(baseline_names()) - baseline_keys:
        raise AssertionError(f"Baseline comparison report is incomplete: {baseline_keys}")

    evaluation_report = json.loads((root / "reports" / "evaluation_report.json").read_text(encoding="utf-8"))
    measured = evaluation_report.get("measured_metrics", {})
    for metric in ("action_accuracy", "top_k_accuracy", "log_loss", "bet_size_mae", "ev", "win_rate_ci"):
        if metric not in measured:
            raise AssertionError(f"Evaluation report missing metric: {metric}")
    smoke = evaluation_report.get("deployment_smoke", {})
    if not smoke.get("selected_action_is_legal"):
        raise AssertionError(f"Evaluation deployment smoke selected illegal action: {smoke}")
    final_selection = evaluation_report.get("final_model_selection", {})
    if final_selection.get("selected_model", {}).get("checkpoint") != "checkpoint_40960":
        raise AssertionError(f"Evaluation report missing QwenPoker final selection: {final_selection}")
    final_model_report = json.loads((root / "reports" / "final_model_selection.json").read_text(encoding="utf-8"))
    if final_model_report.get("status") != "PASS":
        raise AssertionError(f"Final model selection report did not pass: {final_model_report.get('status')}")

    docs = (root / "docs" / "API_CONTRACT.md").read_text(encoding="utf-8")
    for token in ("POST /predict", "predict_request.v1", "predict_response.v1", "INVALID_REQUEST", "RATE_LIMITED"):
        if token not in docs:
            raise AssertionError(f"API docs missing token: {token}")
    return f"final_deliverables={len(manifest['deliverables'])}, fingerprint={manifest['fingerprint'][:12]}"


def legacy_reports_contract(root: Path) -> str:
    contract = describe_legacy_reports_contract()
    if contract["schema_version"] != LEGACY_REPORTS_CONTRACT_VERSION:
        raise AssertionError(f"Unexpected legacy reports schema: {contract['schema_version']}")
    validation = validate_legacy_delivery_reports(root)
    if validation["status"] != "PASS":
        raise AssertionError(f"Legacy reports are incomplete: {validation}")

    gold = json.loads((root / "reports" / "llm_event_gold_eval.json").read_text(encoding="utf-8"))
    strict = gold.get("systems", {}).get("strict_schema_rules", {}).get("event_type", {})
    if strict.get("macro_f1", 0.0) < 0.90:
        raise AssertionError(f"Gold strict-schema macro F1 is too low: {strict}")

    transformer = json.loads((root / "reports" / "llm_transformer_gold_eval.json").read_text(encoding="utf-8"))
    hybrid = transformer.get("systems", {}).get("schema_routed_smol_hybrid", {})
    if hybrid.get("event_type", {}).get("macro_f1", 0.0) < 0.90:
        raise AssertionError(f"Hybrid transformer report macro F1 is too low: {hybrid}")
    if hybrid.get("llm_fallback_count", 0) <= 0:
        raise AssertionError(f"Hybrid transformer report did not record fallback coverage: {hybrid}")

    run_dir = root / "reports" / "hydra" / "llm_transformer_gold_eval" / "offline_schema_router"
    for name in ("resolved_config.yaml", "command.txt", "stdout.txt", "stderr.txt", "run.json", "environment.json", "artifact_manifest.json"):
        if not (run_dir / name).exists():
            raise AssertionError(f"Legacy Hydra provenance is missing {name}")
    return "legacy audit, gate, LLM reports, and Hydra provenance contract pass"


def project_scope_contract(root: Path) -> str:
    contract = describe_project_scope_contract()
    if contract["schema_version"] != PROJECT_SCOPE_CONTRACT_VERSION:
        raise AssertionError(f"Unexpected project scope schema: {contract['schema_version']}")
    if len(contract["phases"]) != 4:
        raise AssertionError(f"Project scope must declare four phases: {contract['phases']}")
    required_requirement_keys = {
        "game_scope",
        "deployment_api",
        "dataset_schema_extensions",
        "data_validation",
        "labeling_contract",
        "action_and_state_space",
        "baseline_and_architecture",
        "rl_self_play",
        "evaluation_acceptance",
        "mlops_monitoring_security",
        "final_deliverables",
    }
    observed_requirement_keys = {item["key"] for item in contract["senior_requirements"]}
    missing = sorted(required_requirement_keys - observed_requirement_keys)
    if missing:
        raise AssertionError(f"Project scope lost senior requirement keys: {missing}")
    validation = validate_project_scope(root)
    if validation["status"] != "PASS":
        raise AssertionError(f"Project scope validation failed: {validation}")
    report = json.loads((root / "reports" / "project_scope_contract.json").read_text(encoding="utf-8"))
    if report.get("validation", {}).get("status") != "PASS":
        raise AssertionError(f"Generated project scope report did not pass: {report}")
    docs = (root / "docs" / "PROJECT_SCOPE_CONTRACT.md").read_text(encoding="utf-8")
    for token in ("Poker ML Project Scope Contract", "Phase 1 - Two Baselines", "Senior Requirements"):
        if token not in docs:
            raise AssertionError(f"Project scope docs missing token: {token}")
    return f"project_scope_requirements={len(contract['senior_requirements'])}, fingerprint={validation['fingerprint'][:12]}"


def dataset_schema_contract() -> str:
    hand_fields = set(HAND_FIELDS)
    action_fields = set(ACTION_FIELDS)
    missing_hands = sorted(set(DATASET_SCHEMA_REQUIRED_FIELDS["hands.csv"]) - hand_fields)
    missing_actions = sorted(set(DATASET_SCHEMA_REQUIRED_FIELDS["actions.csv"]) - action_fields)
    if missing_hands or missing_actions:
        raise AssertionError(
            f"Dataset schema constants are incomplete: hands={missing_hands}, actions={missing_actions}"
        )
    required_action_context = {
        "table_id",
        "game_type",
        "small_blind",
        "big_blind",
        "ante",
        "button_position",
        "action_amount",
        "pot_before_action",
        "pot_after_action",
        "legal_actions",
        "ocr_confidence",
    }
    if not required_action_context.issubset(action_fields):
        raise AssertionError("Action dataset schema lost required context fields")
    return f"hands={len(HAND_FIELDS)} action_fields={len(ACTION_FIELDS)}"


def baseline_contract() -> str:
    expected = ("rule", "imitation_learning", "llm", "end_to_end_policy")
    if baseline_names() != expected:
        raise AssertionError(f"Baseline registry is incomplete: {baseline_names()}")
    features = {
        "bias": 1.0,
        "strength_proxy": 0.55,
        "pot_odds": 0.18,
        "spr": 8.0,
        "to_call": 2.0,
        "amount_to_call": 2.0,
        "street_aggression_ratio": 0.2,
        "street=preflop": 1.0,
        "position_group=btn": 1.0,
    }
    for name in ("rule", "llm"):
        policy = build_baseline_policy(name)
        action, probabilities = policy.predict_from_features(features)
        if action not in probabilities:
            raise AssertionError(f"{name} selected action outside probability map")
        if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-9:
            raise AssertionError(f"{name} probabilities do not sum to 1")
    return "rule, imitation_learning, llm, and end_to_end_policy baselines are registered"


def acceptance_criteria_contract() -> str:
    criteria = DEFAULT_ACCEPTANCE_CRITERIA
    expected = {
        "latency_p95_ms_max": 150.0,
        "latency_p99_ms_max": 300.0,
        "invalid_action_rate_max": 0.0,
        "validation_pass_rate_min": 1.0,
        "reproducibility_pass_rate_min": 1.0,
    }
    for key, value in expected.items():
        observed = getattr(criteria, key)
        if observed != value:
            raise AssertionError(f"Acceptance criterion {key} changed: {observed} != {value}")

    passing = build_acceptance_metrics(
        latencies_ms=[18.0, 24.0, 31.0, 40.0],
        prediction_payloads=[
            {"action": "call", "legal_actions": ["fold", "call", "raise", "all_in"]},
            {"action": "check", "legal_actions": ["check", "bet", "all_in"]},
        ],
        validation_checks=[True, {"name": "dataset_validation", "status": "PASS"}],
        reproducibility_checks=[
            True,
            {"name": "seeded_rl_episode", "status": "PASS", "hash_mismatch": False},
        ],
    )
    pass_report = evaluate_acceptance_criteria(passing)
    if pass_report["status"] != "PASS":
        raise AssertionError(f"Expected acceptance smoke metrics to pass: {pass_report}")

    failing = build_acceptance_metrics(
        latencies_ms=[200.0, 320.0, 500.0],
        prediction_payloads=[{"action": "check", "legal_actions": ["fold", "call"]}],
        validation_checks=[True],
        reproducibility_checks=[True],
    )
    fail_report = evaluate_acceptance_criteria(failing)
    failed_names = {check["name"] for check in fail_report["checks"] if not check["passed"]}
    required_failures = {"latency_p95_ms", "latency_p99_ms", "invalid_action_rate"}
    if fail_report["status"] != "FAIL" or not required_failures.issubset(failed_names):
        raise AssertionError(f"Acceptance criteria did not reject bad metrics: {fail_report}")
    return (
        "acceptance targets: p95<=150ms, p99<=300ms, invalid_action_rate=0, "
        "validation_pass_rate=1.0, reproducibility_pass_rate=1.0"
    )


def rl_environment_contract() -> str:
    contract = describe_rl_environment()
    if contract["poker_simulator_engine"]["game_type"] != "nl_holdem":
        raise AssertionError(f"Unexpected RL engine contract: {contract['poker_simulator_engine']}")
    if len(contract["opponent_pool"]["opponents"]) < 4:
        raise AssertionError("Opponent pool is too small for league evaluation")
    seed = SeedPolicy(base_seed=123).derive("generation", 0, "episode", 0)
    left = NoLimitHoldemSingleDecisionEngine()
    right = NoLimitHoldemSingleDecisionEngine()
    if left.reset(seed=seed) != right.reset(seed=seed):
        raise AssertionError("RL environment reset is not deterministic for the same seed")
    league = SelfPlayLeague(seed_policy=SeedPolicy(base_seed=456))
    first = league.run_episode(seed_policy_hero, generation=0, episode_index=0)
    second = league.run_episode(seed_policy_hero, generation=0, episode_index=0)
    if first.final_stacks != second.final_stacks or first.reward.shaped_reward != second.reward.shaped_reward:
        raise AssertionError("Self-play episode is not reproducible under the seed policy")
    return "RL engine, self-play league, opponent pool, seed policy, and reward shaping contract pass"


def data_validation_contract() -> str:
    with tempfile.TemporaryDirectory() as raw_temp:
        temp = Path(raw_temp)
        _write_validation_fixture(temp, broken=False)
        passing = validate_dataset(temp)
        if passing["status"] != "PASS":
            raise AssertionError(f"Expected clean validation fixture to pass: {passing}")

        broken = temp / "broken"
        broken.mkdir()
        _write_validation_fixture(broken, broken=True)
        failing = validate_dataset(broken)
        failed_checks = {
            name
            for name, payload in failing.get("checks", {}).items()
            if payload.get("status") == "FAIL"
        }
        expected = {
            "pot_conservation",
            "stack_delta_consistency",
            "duplicate_hand_detection",
            "missing_ocr_conflict_policy",
        }
        if not expected.issubset(failed_checks):
            raise AssertionError(f"Validation fixture did not exercise all checks: {failed_checks}")
        return f"pass_checks={len(passing['checks'])} fail_checks={sorted(failed_checks)}"


def _write_validation_fixture(root: Path, *, broken: bool) -> None:
    hand_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "table_id": "table_1",
            "game_type": "nl_holdem",
            "small_blind": "0.5",
            "big_blind": "1.0",
            "ante": "0",
            "button_position": "BTN",
            "start_frame": "1",
            "end_frame": "5",
            "board_cards": "",
            "total_actions": "1",
            "total_stack_events": "2",
            "winner_positions": "BB",
            "pot_from_stacks": "2.5",
            "pot_from_recognition": "2.5",
            "dealer_hand_number": "",
            "dealer_winner": "",
            "dealer_pot": "",
        }
    ]
    if broken:
        hand_rows.append(dict(hand_rows[0]))

    player_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "position": "BTN",
            "nickname": "hero",
            "cards": "Ah Kd",
            "starting_stack": "100",
            "ending_stack": "97.5",
            "stack_delta": "-9.0" if broken else "-2.5",
        },
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "position": "BB",
            "nickname": "villain",
            "cards": "",
            "starting_stack": "100",
            "ending_stack": "102.5",
            "stack_delta": "2.5",
        },
    ]
    action_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "table_id": "table_1",
            "game_type": "nl_holdem",
            "small_blind": "0.5",
            "big_blind": "1.0",
            "ante": "0",
            "button_position": "BTN",
            "frame_id": "3",
            "player_position": "BTN",
            "player_nickname": "hero",
            "action": "call",
            "action_amount": "" if broken else "2.5",
            "pot_before_action": "0",
            "pot_after_action": "9.0" if broken else "2.5",
            "legal_actions": "" if broken else "fold call raise all_in",
            "ocr_confidence": "" if broken else "0.95",
            "street": "preflop",
        }
    ]
    if broken:
        conflict = dict(action_rows[0])
        conflict["action"] = "raise"
        conflict["action_amount"] = "6.0"
        action_rows.append(conflict)
    stack_rows = [
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "frame_id": "3",
            "player_position": "BTN",
            "event": "update_stack",
            "stack": "97.5",
            "diff": "-2.5",
            "stack_after_event": "97.5",
        },
        {
            "hand_id": "H1",
            "hand_index": "0",
            "local_hand_index": "0",
            "source_file": "sample",
            "frame_id": "4",
            "player_position": "BB",
            "event": "update_stack",
            "stack": "102.5",
            "diff": "2.5",
            "stack_after_event": "102.5",
        },
    ]
    _write_csv(root / "hands.csv", HAND_FIELDS, hand_rows)
    _write_csv(root / "players.csv", [
        "hand_id", "hand_index", "local_hand_index", "source_file", "position", "nickname", "cards",
        "starting_stack", "ending_stack", "stack_delta"
    ], player_rows)
    _write_csv(root / "actions.csv", ACTION_FIELDS, action_rows)
    _write_csv(root / "stack_events.csv", [
        "hand_id", "hand_index", "local_hand_index", "source_file", "frame_id", "player_position",
        "event", "stack", "diff", "stack_after_event"
    ], stack_rows)


def _write_csv(path: Path, fields: Any, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def health_contract(model_path: Path) -> str:
    try:
        from poker_agent.service import health_payload, resolve_model_path
    except ModuleNotFoundError as exc:
        if exc.name != "fastapi":
            raise AssertionError(f"Service dependency unavailable: {exc}") from exc
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        service_text = (ROOT / "poker_agent" / "service.py").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        for token in ("fastapi", "uvicorn", "joblib"):
            if token not in requirements.lower():
                raise AssertionError(f"Runtime dependency is not declared in requirements.txt: {token}")
        for token in ("FastAPI", "health_payload", "/health.json", "/predict", "/contract.json"):
            if token not in service_text:
                raise AssertionError(f"FastAPI service contract token missing: {token}")
        if "pip install" not in dockerfile or "requirements.txt" not in dockerfile:
            raise AssertionError("Dockerfile does not install requirements.txt")
        model = load_policy(model_path)
        metadata = getattr(model, "metadata", {}) or {}
        if "macro_f1" not in (metadata.get("valid_metrics") or {}):
            raise AssertionError(f"Fallback model metadata is incomplete: {metadata}")
        return "static FastAPI health contract pass; local fastapi package is not installed"

    resolved = resolve_model_path()
    if resolved.resolve() != model_path.resolve():
        raise AssertionError(f"Health resolved unexpected model path: {resolved}")
    payload = health_payload()
    if payload.get("model_status") != "loaded":
        raise AssertionError(f"Model status is not loaded: {payload}")
    if "valid_macro_f1" not in payload:
        raise AssertionError(f"Health payload missing model metric metadata: {payload}")
    return json.dumps(payload, sort_keys=True)


def reports_contract(root: Path, require_gate_pass: bool) -> str:
    audit = json.loads((root / "reports" / "dataset_audit.json").read_text(encoding="utf-8"))
    repo_audit = json.loads((root / "reports" / "repository_audit.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "reports" / "production_gate.json").read_text(encoding="utf-8"))
    benchmark = root / "reports" / "llm_event_benchmark.json"
    gold_eval = root / "reports" / "llm_event_gold_eval.json"
    transformer_eval = root / "reports" / "llm_transformer_gold_eval.json"
    if "findings" not in audit:
        raise AssertionError("Audit report has no findings key")
    if repo_audit.get("status") != "PASS":
        raise AssertionError("Repository audit did not pass")
    hydra_audit = repo_audit.get("hydra", {})
    if hydra_audit.get("missing_hydra_configs"):
        raise AssertionError(f"Hydra configs are missing: {hydra_audit['missing_hydra_configs']}")
    if hydra_audit.get("incomplete_argument_configs"):
        raise AssertionError(f"Hydra argument coverage is incomplete: {hydra_audit['incomplete_argument_configs']}")
    if repo_audit.get("unowned_hardcoded_defaults"):
        raise AssertionError(f"CLI defaults are not owned by Hydra configs: {repo_audit['unowned_hardcoded_defaults']}")
    if gate.get("status") not in {"PASS", "FAIL"}:
        raise AssertionError(f"Invalid gate status: {gate.get('status')}")
    if require_gate_pass and gate.get("status") != "PASS":
        raise AssertionError("Production gate did not pass")
    if benchmark.exists():
        benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
        if "systems" not in benchmark_payload:
            raise AssertionError("Event extraction benchmark has no systems key")
        benchmark_detail = f", event_benchmark_records={benchmark_payload.get('records_evaluated')}"
    else:
        benchmark_detail = ""
    if not gold_eval.exists():
        raise AssertionError("Gold event extraction evaluation report is missing")
    gold_payload = json.loads(gold_eval.read_text(encoding="utf-8"))
    strict_metrics = gold_payload.get("systems", {}).get("strict_schema_rules", {})
    if strict_metrics.get("event_type", {}).get("macro_f1", 0.0) < 0.90:
        raise AssertionError("Gold event extraction macro F1 is below acceptance threshold")
    benchmark_detail += f", gold_examples={gold_payload.get('examples')}"
    if not transformer_eval.exists():
        raise AssertionError("Local instruction-model evaluation report is missing")
    transformer_payload = json.loads(transformer_eval.read_text(encoding="utf-8"))
    systems = transformer_payload.get("systems", {})
    zero_shot = systems.get("smol_strict_zero_shot", {}).get("event_type", {})
    few_shot = systems.get("smol_few_shot", {}).get("event_type", {})
    ranker = systems.get("smol_candidate_ranker", {}).get("event_type", {})
    calibrated = systems.get("smol_calibrated_candidate_ranker", {}).get("event_type", {})
    hybrid_metrics = systems.get("schema_routed_smol_hybrid", {})
    hybrid = hybrid_metrics.get("event_type", {})
    if not transformer_payload.get("model_id"):
        raise AssertionError("Instruction-model evaluation has no model id")
    if few_shot.get("accuracy", 0.0) < zero_shot.get("accuracy", 0.0):
        raise AssertionError("Few-shot prompt regressed against strict zero-shot accuracy")
    if calibrated.get("macro_f1", 0.0) < ranker.get("macro_f1", 0.0):
        raise AssertionError("Contextual calibration regressed against uncalibrated candidate ranking")
    if hybrid.get("macro_f1", 0.0) < 0.90:
        raise AssertionError("Schema-routed LLM hybrid macro F1 is below acceptance threshold")
    if hybrid_metrics.get("llm_fallback_count", 0) <= 0:
        raise AssertionError("Schema-routed hybrid did not exercise the LLM fallback")
    benchmark_detail += (
        f", transformer_model={transformer_payload.get('model_id')}"
        f", calibrated_macro_f1={calibrated.get('macro_f1')}"
        f", hybrid_macro_f1={hybrid.get('macro_f1')}"
        f", hybrid_llm_fallback_rate={hybrid_metrics.get('llm_fallback_rate')}"
    )
    return (
        f"audit_findings={len(audit.get('findings', []))}, "
        f"repo_audit={repo_audit.get('status')}, gate={gate.get('status')}{benchmark_detail}"
    )


def repo_hygiene_contract(root: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_repo_hygiene.py"), "--root", str(root)],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise AssertionError(detail[:2000])
    payload = json.loads(completed.stdout)
    return f"hygiene={payload['status']}"


def hydra_provenance_contract(root: Path) -> str:
    experiment_root = root / "reports" / "hydra" / "llm_transformer_gold_eval"
    runs = sorted(path for path in experiment_root.glob("*") if path.is_dir())
    if not runs:
        raise AssertionError("No Hydra LLM experiment runs were found")
    latest = runs[-1]
    required = {
        "resolved_config.yaml",
        "command.txt",
        "stdout.txt",
        "stderr.txt",
        "run.json",
        "environment.json",
        "artifact_manifest.json",
    }
    missing = sorted(name for name in required if not (latest / name).exists())
    if missing:
        raise AssertionError(f"Hydra run provenance is incomplete: {missing}")
    run = json.loads((latest / "run.json").read_text(encoding="utf-8"))
    environment = json.loads((latest / "environment.json").read_text(encoding="utf-8"))
    artifacts = json.loads((latest / "artifact_manifest.json").read_text(encoding="utf-8"))
    if run.get("status") != "pass" or not run.get("deterministic"):
        raise AssertionError(f"Latest Hydra LLM run is not deterministic/pass: {run.get('status')}")
    if not environment.get("packages", {}).get("transformers"):
        raise AssertionError("Hydra environment manifest does not record transformers")
    file_artifacts = [item for item in artifacts.get("artifacts", []) if item.get("type") == "file"]
    if not file_artifacts or any(not item.get("sha256") for item in file_artifacts):
        raise AssertionError("Hydra artifact manifest is missing file checksums")
    return f"run={latest.name}, artifacts={len(file_artifacts)}, git={environment.get('git', {}).get('revision')}"


def zip_contract(root: Path, zip_path: Path) -> str:
    required = {
        "models/poker_policy.joblib",
        "README.md",
        "docker-compose.yml",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/security/local.yaml",
        "configs/deliverables/local.yaml",
        "configs/project_scope/local.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/model/text_event_smol.yaml",
        "configs/rl/self_play_league.yaml",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/acceptance_criteria.yaml",
        "configs/experiments/run_baselines.yaml",
        "configs/experiments/rl_self_play_smoke.yaml",
        "configs/experiments/mlops_smoke.yaml",
        "configs/experiments/monitoring_smoke.yaml",
        "configs/experiments/security_smoke.yaml",
        "configs/experiments/final_deliverables.yaml",
        "configs/experiments/project_scope.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_transformer_gold_eval.yaml",
        "evaluation/event_extraction_gold.jsonl",
        "reports/dataset_audit.json",
        "reports/repository_audit.json",
        "reports/production_gate.json",
        "reports/llm_event_benchmark.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_report.md",
        "reports/llm_event_methodology.md",
        "reports/llm_transformer_gold_eval.json",
        "reports/llm_transformer_gold_report.md",
        "reports/dataset_validation_report.json",
        "reports/baseline_report.json",
        "reports/evaluation_report.json",
        "reports/final_model_selection.json",
        "reports/project_scope_contract.json",
        "reports/final_deliverables.json",
        "reports/legacy_delivery_reports.json",
        "reports/delivery_report.md",
        "docs/API_CONTRACT.md",
        "docs/FINAL_MODEL_SELECTION.md",
        "docs/PROJECT_SCOPE_CONTRACT.md",
        "tests/test_action_space_contract.py",
        "tests/test_state_features_contract.py",
        "tests/test_acceptance_criteria_contract.py",
        "tests/test_baseline_contract.py",
        "tests/test_api_contract.py",
        "tests/test_mlops_contract.py",
        "tests/test_monitoring_contract.py",
        "tests/test_security_contract.py",
        "tests/test_final_deliverables_contract.py",
        "tests/test_legacy_delivery_contract.py",
        "tests/test_project_scope_contract.py",
        "tests/test_game_scope_usage_boundary_contract.py",
        "tests/test_final_model_selection_contract.py",
        "tests/test_rl_environment_contract.py",
        ".github/workflows/smoke.yml",
        "poker_agent/action_space.py",
        "poker_agent/acceptance_criteria.py",
        "poker_agent/api_contract.py",
        "poker_agent/baselines.py",
        "poker_agent/mlops.py",
        "poker_agent/monitoring.py",
        "poker_agent/final_model_selection.py",
        "poker_agent/game_scope.py",
        "poker_agent/usage_boundary.py",
        "poker_agent/security.py",
        "poker_agent/deliverables.py",
        "poker_agent/legacy_reports.py",
        "poker_agent/project_scope.py",
        "poker_agent/game_scope.py",
        "poker_agent/usage_boundary.py",
        "poker_agent/rl_environment.py",
        "scripts/check_repo_hygiene.py",
        "scripts/check_acceptance_criteria.py",
        "scripts/check_mlops_contract.py",
        "scripts/check_monitoring_contract.py",
        "scripts/check_security_contract.py",
        "scripts/check_project_scope_contract.py",
        "scripts/check_final_model_selection.py",
        "scripts/prepare_final_deliverables.py",
        "scripts/prepare_legacy_delivery_reports.py",
        "scripts/audit_repository.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/inspect_rl_environment.py",
        "scripts/run_hydra_experiment.py",
        "scripts/run_baselines.py",
        "scripts/verify_delivery.py",
        "build_poker_dataset_optimized.py",
        "poker_agent/data_validation.py",
        "poker_agent/dataset_schema.py",
        "verify_delivery.ps1",
        "tests/test_dataset_schema_contract.py",
        "tests/test_data_validation_contract.py",
    }
    if not zip_path.exists():
        raise AssertionError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    forbidden = sorted(
        name
        for name in names
        if "__pycache__/" in name
        or name.endswith((".pyc", ".pyo", ".pyd"))
        or name.endswith("requirements-research.txt")
    )
    if forbidden:
        raise AssertionError(f"ZIP contains generated or removed artifacts: {forbidden[:20]}")
    missing = sorted(required - names)
    if missing:
        raise AssertionError(f"ZIP is missing required entries: {missing}")
    if not any(name.endswith("/environment.json") for name in names):
        raise AssertionError("ZIP contains no Hydra environment manifest")
    if not any(name.endswith("/artifact_manifest.json") for name in names):
        raise AssertionError("ZIP contains no Hydra artifact manifest")
    return f"zip_entries={len(names)}"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checks = [
        run_check("required_files", lambda: require_files(root)),
        run_check("compile_sources", lambda: compile_sources(root)),
        run_check("model_loads", lambda: model_loads(args.model)),
        run_check("inference_contract", lambda: inference_contract(args.model)),
        run_check("deployment_api_contract", deployment_api_contract),
        run_check("game_scope_usage_boundary_contract", game_scope_usage_boundary_contract),
        run_check("final_model_selection_contract", lambda: final_model_selection_contract(root)),
        run_check("mlops_contract", lambda: mlops_contract(root)),
        run_check("monitoring_contract", lambda: monitoring_contract(root)),
        run_check("security_contract", lambda: security_contract(root)),
        run_check("final_deliverables_contract", lambda: final_deliverables_contract(root)),
        run_check("legacy_reports_contract", lambda: legacy_reports_contract(root)),
        run_check("project_scope_contract", lambda: project_scope_contract(root)),
        run_check("dataset_schema_contract", dataset_schema_contract),
        run_check("baseline_contract", baseline_contract),
        run_check("acceptance_criteria_contract", acceptance_criteria_contract),
        run_check("rl_environment_contract", rl_environment_contract),
        run_check("data_validation_contract", data_validation_contract),
        run_check("health_contract", lambda: health_contract(args.model)),
        run_check("reports_contract", lambda: reports_contract(root, args.require_gate_pass)),
        run_check("repo_hygiene_contract", lambda: repo_hygiene_contract(root)),
        run_check("hydra_provenance_contract", lambda: hydra_provenance_contract(root)),
        run_check("zip_contract", lambda: zip_contract(root, args.zip)),
    ]
    payload = {
        "status": "PASS" if all(check.passed for check in checks) else "FAIL",
        "checks": [check.__dict__ for check in checks],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
