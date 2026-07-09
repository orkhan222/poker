from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.agents import MLPolicyAgent
from poker_agent.api_contract import api_contract
from poker_agent.approval_boundary import assert_approval_boundary, build_approval_boundary
from poker_agent.model import load_policy
from poker_agent.scenario_sanity import validate_scenario_sanity
from poker_agent.schemas import PredictionRequest
from poker_agent.service import (
    CLIENT_SWAGGER_HTML,
    app,
    get_agent,
    get_autonomous_agent,
    health_payload,
    resolve_model_path,
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
        "requirements.txt",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
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
        "configs/prompts/event_extraction_prompt.txt",
        "configs/prompts/event_extraction_minimal.txt",
        "configs/prompts/event_extraction_permissive.txt",
        "configs/prompts/event_extraction_strict.txt",
        "configs/prompts/event_extraction_fewshot.txt",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/prompts/poker_decision_minimal_zero_shot.txt",
        "configs/prompts/poker_decision_rules_grounded.txt",
        "configs/prompts/poker_decision_full_context.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/evaluate_policy.yaml",
        "configs/experiments/research_compare_tabular.yaml",
        "configs/experiments/audit_dataset.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/production_gate.yaml",
        "configs/experiments/strategy_stack_maturity.yaml",
        "configs/experiments/behavioral_revalidation.yaml",
        "configs/experiments/behavioral_revalidation_proof.yaml",
        "configs/experiments/human_likeness_evidence.yaml",
        "configs/experiments/human_likeness_claim_gate.yaml",
        "configs/experiments/bet_timing_calibration.yaml",
        "configs/experiments/hole_card_data_quality.yaml",
        "configs/experiments/data_leakage_contract.yaml",
        "configs/experiments/normalized_action_contract.yaml",
        "configs/experiments/actions_context_quality.yaml",
        "configs/experiments/actions_dataset_export_contract.yaml",
        "configs/experiments/stack_event_context_quality.yaml",
        "configs/experiments/scenario_sanity.yaml",
        "configs/experiments/train_routed_bundle_smoke.yaml",
        "configs/experiments/llm_event_extraction_smoke.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_transformer_gold_eval.yaml",
        "configs/experiments/llm_decision_context.yaml",
        "configs/experiments/llm_decision_context_smoke.yaml",
        "configs/experiments/llm_decision_context_qwen25.yaml",
        "configs/experiments/build_decision_context_holdout.yaml",
        "configs/experiments/llm_decision_gate.yaml",
        "configs/experiments/llm_decision_candidate_ranker_qwen25.yaml",
        "configs/experiments/llm_decision_candidate_gate.yaml",
        "configs/experiments/llm_architecture_comparison.yaml",
        "configs/experiments/phase2_selection_comparison.yaml",
        "configs/experiments/llm_role_boundary.yaml",
        "configs/experiments/llm_policy_experimental.yaml",
        "configs/experiments/qlora_next_stage.yaml",
        "configs/experiments/production_runtime_monitoring.yaml",
        "configs/experiments/project_completion.yaml",
        "configs/experiments/final_delivery_acceptance.yaml",
        "configs/experiments/final_strategy_quality_status.yaml",
        "configs/experiments/final_delivery_acceptance.yaml",
        "configs/experiments/training_cluster_requirements.yaml",
        "configs/experiments/today_acceptance_training.yaml",
        "configs/experiments/client_gpu_training_response.yaml",
        "configs/experiments/multi_agent_training_status.yaml",
        "configs/experiments/phase3_open_spiel_arena.yaml",
        "configs/experiments/phase3_open_spiel_claim.yaml",
        "configs/experiments/open_spiel_claim_readiness.yaml",
        "configs/experiments/open_spiel_claim_contract.yaml",
        "configs/experiments/rl_delivery_boundary.yaml",
        "configs/experiments/evaluation_metric_contract.yaml",
        "configs/experiments/test_execution_contract.yaml",
        "configs/experiments/strategy_stack_maturity.yaml",
        "configs/experiments/behavioral_revalidation.yaml",
        "configs/experiments/behavioral_revalidation_proof.yaml",
        "configs/experiments/human_likeness_evidence.yaml",
        "configs/experiments/human_likeness_claim_gate.yaml",
        "configs/experiments/raw_model_status.yaml",
        "configs/experiments/raw_model_challenger.yaml",
        "configs/experiments/challenger_strategy_quality.yaml",
        "configs/experiments/verify_delivery.yaml",
        "Dockerfile",
        "docker-compose.yml",
        "install.ps1",
        "activate_env.cmd",
        "run_server.ps1",
        "complete_delivery.ps1",
        "verify_delivery.ps1",
        "models/poker_policy.joblib",
        "models/poker_policy_bundle.joblib",
        "reports/production_gate.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_eval.md",
        "reports/llm_decision_context.json",
        "reports/llm_decision_context.md",
        "reports/llm_decision_context_smoke.json",
        "reports/llm_decision_context_smoke_predictions.jsonl",
        "reports/llm_decision_context_smoke.md",
        "reports/decision_context_holdout.json",
        "reports/llm_decision_context_qwen25.json",
        "reports/llm_decision_context_qwen25_predictions.jsonl",
        "reports/llm_decision_context_qwen25.md",
        "reports/llm_decision_gate.json",
        "reports/llm_decision_gate.md",
        "reports/llm_decision_candidate_ranker_qwen25.json",
        "reports/llm_decision_candidate_ranker_qwen25_predictions.jsonl",
        "reports/llm_decision_candidate_ranker_qwen25.md",
        "reports/llm_decision_candidate_gate.json",
        "reports/llm_decision_candidate_gate.md",
        "reports/llm_architecture_comparison.json",
        "reports/llm_architecture_comparison.md",
        "reports/phase2_selection_comparison.json",
        "reports/phase2_selection_comparison.md",
        "reports/llm_role_boundary.json",
        "reports/llm_role_boundary.md",
        "reports/llm_policy_experimental.json",
        "reports/llm_policy_experimental.md",
        "reports/qlora_next_stage.json",
        "reports/qlora_next_stage.md",
        "reports/production_runtime_monitoring.json",
        "reports/production_runtime_monitoring.md",
        "reports/llm_role_boundary.json",
        "reports/llm_role_boundary.md",
        "reports/policy_acceptance.json",
        "reports/production_self_play.json",
        "reports/deployed_strategy_gate.json",
        "reports/delivery_readiness.json",
        "reports/scope_contract.json",
        "reports/scope_contract.md",
        "reports/project_completion.json",
        "reports/project_completion.md",
        "reports/final_delivery_acceptance.json",
        "reports/final_delivery_acceptance.md",
        "reports/final_strategy_quality_status.json",
        "reports/final_strategy_quality_status.md",
        "reports/final_delivery_acceptance.json",
        "reports/final_delivery_acceptance.md",
        "reports/model_risk_register.json",
        "reports/model_risk_register.md",
        "reports/production_approval.json",
        "reports/production_approval.md",
        "reports/strategy_stack_maturity.json",
        "reports/strategy_stack_maturity.md",
        "reports/strategy_stack_maturity.json",
        "reports/strategy_stack_maturity.md",
        "reports/behavioral_revalidation.json",
        "reports/behavioral_revalidation.md",
        "reports/behavioral_revalidation_proof.json",
        "reports/behavioral_revalidation_proof.md",
        "reports/human_likeness_evidence.json",
        "reports/human_likeness_evidence.md",
        "reports/human_likeness_claim_gate.json",
        "reports/human_likeness_claim_gate.md",
        "reports/bet_timing_calibration.json",
        "reports/bet_timing_calibration.md",
        "reports/bet_timing_calibration.json",
        "reports/bet_timing_calibration.md",
        "reports/hole_card_data_quality.json",
        "reports/hole_card_data_quality.md",
        "reports/hole_card_data_quality.json",
        "reports/hole_card_data_quality.md",
        "reports/data_leakage_contract.json",
        "reports/data_leakage_contract.md",
        "reports/normalized_action_contract.json",
        "reports/normalized_action_contract.md",
        "reports/actions_context_quality.json",
        "reports/actions_context_quality.md",
        "reports/actions_dataset_export_contract.json",
        "reports/actions_dataset_export_contract.md",
        "reports/stack_event_context_quality.json",
        "reports/stack_event_context_quality.md",
        "reports/scenario_sanity.json",
        "reports/scenario_sanity.md",
        "reports/raw_model_status.json",
        "reports/raw_model_status.md",
        "reports/raw_model_challenger.json",
        "reports/raw_model_challenger.md",
        "reports/challenger_strategy_quality.json",
        "reports/challenger_strategy_quality.md",
        "reports/challenger_strategy_quality.json",
        "reports/challenger_strategy_quality.md",
        "reports/client_handoff.json",
        "reports/client_handoff.md",
        "reports/training_cluster_requirements.json",
        "reports/training_cluster_requirements.md",
        "reports/today_acceptance_training.json",
        "reports/today_acceptance_training.md",
        "reports/today_acceptance_production_gate.json",
        "reports/client_gpu_training_response.json",
        "reports/client_gpu_training_response.md",
        "reports/multi_agent_training_status.json",
        "reports/multi_agent_training_status.md",
        "reports/phase3_open_spiel_arena.json",
        "reports/phase3_open_spiel_arena.md",
        "reports/open_spiel_claim_readiness.json",
        "reports/open_spiel_claim_readiness.md",
        "reports/open_spiel_claim_contract.json",
        "reports/open_spiel_claim_contract.md",
        "reports/rl_delivery_boundary.json",
        "reports/rl_delivery_boundary.md",
        "reports/evaluation_metric_contract.json",
        "reports/evaluation_metric_contract.md",
        "reports/test_execution_contract.json",
        "reports/test_execution_contract.md",
        "evaluation/event_extraction_gold.jsonl",
        "evaluation/decision_context_smoke.jsonl",
        "evaluation/decision_context_human_holdout.jsonl",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_bet_timing_calibration.py",
        "scripts/build_behavioral_revalidation.py",
        "scripts/build_behavioral_revalidation_proof.py",
        "scripts/build_human_likeness_evidence.py",
        "scripts/build_human_likeness_claim_gate.py",
        "scripts/build_bet_timing_calibration.py",
        "scripts/build_hole_card_data_quality.py",
        "scripts/build_data_leakage_contract.py",
        "scripts/build_normalized_action_contract.py",
        "scripts/build_actions_context_quality.py",
        "scripts/build_actions_dataset_export_contract.py",
        "scripts/build_stack_event_context_quality.py",
        "scripts/build_scenario_sanity.py",
        "scripts/build_raw_model_status.py",
        "scripts/train_raw_model_challenger.py",
        "scripts/build_challenger_strategy_quality.py",
        "scripts/build_challenger_strategy_quality.py",
        "scripts/build_client_handoff.py",
        "scripts/build_training_cluster_requirements.py",
        "scripts/run_today_acceptance_training.py",
        "scripts/build_client_gpu_training_response.py",
        "scripts/build_multi_agent_training_status.py",
        "scripts/build_phase3_open_spiel_arena.py",
        "scripts/build_open_spiel_claim_readiness.py",
        "scripts/build_open_spiel_claim_contract.py",
        "scripts/build_rl_delivery_boundary.py",
        "scripts/build_evaluation_metric_contract.py",
        "scripts/build_test_execution_contract.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_llm_decision_context.py",
        "scripts/llm_decision_context_eval.py",
        "scripts/build_decision_context_holdout.py",
        "scripts/build_llm_decision_gate.py",
        "scripts/build_llm_architecture_comparison.py",
        "scripts/build_llm_role_boundary.py",
        "scripts/build_llm_policy_experimental.py",
        "scripts/build_qlora_next_stage.py",
        "scripts/build_production_runtime_monitoring.py",
        "scripts/build_scope_contract.py",
        "scripts/build_project_completion.py",
        "scripts/build_final_delivery_acceptance.py",
        "scripts/build_final_strategy_quality_status.py",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/evaluate_policy.py",
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
        "poker_agent/service.py",
        "poker_agent/agents.py",
        "poker_agent/autonomous_agent.py",
        "poker_agent/api_contract.py",
        "poker_agent/approval_boundary.py",
        "poker_agent/scope_contract.py",
        "poker_agent/model_risk_register.py",
        "poker_agent/production_approval.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/bet_timing_calibration.py",
        "poker_agent/behavioral_revalidation.py",
        "poker_agent/behavioral_revalidation_proof.py",
        "poker_agent/human_likeness_evidence.py",
        "poker_agent/human_likeness_claim_gate.py",
        "poker_agent/human_likeness_policy_guard.py",
        "poker_agent/bet_timing_calibration.py",
        "poker_agent/hole_card_data_quality.py",
        "poker_agent/data_leakage_contract.py",
        "poker_agent/action_normalization.py",
        "poker_agent/normalized_action_contract.py",
        "poker_agent/actions_context_quality.py",
        "poker_agent/actions_dataset_export_contract.py",
        "poker_agent/stack_event_context_quality.py",
        "poker_agent/stack_context.py",
        "poker_agent/scenario_sanity.py",
        "poker_agent/raw_model_status.py",
        "poker_agent/raw_model_challenger.py",
        "poker_agent/challenger_strategy_quality.py",
        "poker_agent/challenger_strategy_quality.py",
        "poker_agent/client_handoff.py",
        "poker_agent/training_cluster.py",
        "poker_agent/today_training.py",
        "poker_agent/client_gpu_training_response.py",
        "poker_agent/multi_agent_training_status.py",
        "poker_agent/rl_training_evidence_gate.py",
        "poker_agent/open_spiel_llm_arena.py",
        "poker_agent/open_spiel_claim_readiness.py",
        "poker_agent/open_spiel_claim_contract.py",
        "poker_agent/rl_delivery_boundary.py",
        "poker_agent/strategy_metric_gate.py",
        "poker_agent/evaluation_metric_contract.py",
        "poker_agent/test_execution_contract.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/llm_decision_context.py",
        "poker_agent/llm_decision_benchmark.py",
        "poker_agent/llm_role_boundary.py",
        "poker_agent/llm_policy_experimental.py",
        "poker_agent/qlora_next_stage.py",
        "poker_agent/production_runtime_monitoring.py",
        "poker_agent/llm_decision_gate.py",
        "poker_agent/llm_architecture_comparison.py",
        "poker_agent/project_completion.py",
        "poker_agent/delivery_strategy_boundary.py",
        "poker_agent/final_delivery_acceptance.py",
        "poker_agent/final_strategy_quality_status.py",
        "poker_agent/delivery_readiness.py",
        "poker_agent/features.py",
        "poker_agent/action_planning.py",
        "poker_agent/model.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
        "tests/test_timing_features.py",
        "tests/test_autonomous_agent.py",
        "tests/test_training_cluster.py",
        "tests/test_today_acceptance_training.py",
        "tests/test_client_gpu_training_response.py",
        "tests/test_bet_timing_calibration.py",
        "tests/test_final_delivery_acceptance.py",
        "tests/test_multi_agent_training_status.py",
        "tests/test_rl_training_evidence_gate.py",
        "tests/test_open_spiel_llm_arena.py",
        "tests/test_open_spiel_claim_readiness.py",
        "tests/test_open_spiel_claim_contract.py",
        "tests/test_open_spiel_claim_mode.py",
        "tests/test_open_spiel_claim_command_regression.py",
        "tests/test_rl_delivery_boundary.py",
        "tests/test_strategy_metric_gate.py",
        "tests/test_evaluation_metric_contract.py",
        "tests/test_test_execution_contract.py",
        "tests/test_strategy_stack_maturity.py",
        "tests/test_llm_decision_benchmark.py",
        "tests/test_llm_decision_gate.py",
        "tests/test_llm_architecture_comparison.py",
        "tests/test_llm_role_boundary.py",
        "tests/test_llm_policy_experimental.py",
        "tests/test_qlora_next_stage.py",
        "tests/test_production_runtime_monitoring.py",
        "tests/test_delivery_strategy_boundary.py",
        "tests/test_final_delivery_acceptance.py",
        "tests/test_final_strategy_quality_status.py",
        "tests/test_llm_role_boundary.py",
        "tests/test_strategy_stack_maturity.py",
        "tests/test_behavioral_revalidation.py",
        "tests/test_behavioral_revalidation_proof.py",
        "tests/test_human_likeness_evidence.py",
        "tests/test_human_likeness_claim_gate.py",
        "tests/test_human_likeness_policy_guard.py",
        "tests/test_bet_timing_calibration.py",
        "tests/test_hole_card_data_quality.py",
        "tests/test_data_leakage_contract.py",
        "tests/test_normalized_action_contract.py",
        "tests/test_actions_context_quality.py",
        "tests/test_actions_dataset_export_contract.py",
        "tests/test_stack_event_context_quality.py",
        "tests/test_stack_context.py",
        "tests/test_scenario_sanity.py",
        "tests/test_raw_model_status.py",
        "tests/test_raw_model_challenger.py",
        "tests/test_challenger_strategy_quality.py",
        "tests/test_hole_card_data_quality.py",
        "tests/test_data_leakage_contract.py",
        "tests/test_actions_context_quality.py",
        "tests/test_actions_dataset_export_contract.py",
        "tests/test_stack_event_context_quality.py",
        "tests/test_stack_context.py",
    ]
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")
    return f"{len(required)} required files present"


def compile_sources(root: Path) -> str:
    source_files = [
        "poker_agent/agents.py",
        "poker_agent/autonomous_agent.py",
        "poker_agent/api_contract.py",
        "poker_agent/approval_boundary.py",
        "poker_agent/delivery_readiness.py",
        "poker_agent/evaluator.py",
        "poker_agent/features.py",
        "poker_agent/model.py",
        "poker_agent/schemas.py",
        "poker_agent/scope_contract.py",
        "poker_agent/model_risk_register.py",
        "poker_agent/production_approval.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/behavioral_revalidation.py",
        "poker_agent/behavioral_revalidation_proof.py",
        "poker_agent/hole_card_data_quality.py",
        "poker_agent/data_leakage_contract.py",
        "poker_agent/actions_context_quality.py",
        "poker_agent/actions_dataset_export_contract.py",
        "poker_agent/stack_event_context_quality.py",
        "poker_agent/raw_model_status.py",
        "poker_agent/raw_model_challenger.py",
        "poker_agent/challenger_strategy_quality.py",
        "poker_agent/client_handoff.py",
        "poker_agent/training_cluster.py",
        "poker_agent/today_training.py",
        "poker_agent/client_gpu_training_response.py",
        "poker_agent/multi_agent_training_status.py",
        "poker_agent/open_spiel_llm_arena.py",
        "poker_agent/open_spiel_claim_readiness.py",
        "poker_agent/open_spiel_claim_contract.py",
        "poker_agent/rl_delivery_boundary.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/llm_decision_context.py",
        "poker_agent/llm_decision_benchmark.py",
        "poker_agent/llm_policy_experimental.py",
        "poker_agent/llm_decision_gate.py",
        "poker_agent/llm_architecture_comparison.py",
        "poker_agent/project_completion.py",
        "poker_agent/delivery_strategy_boundary.py",
        "poker_agent/final_delivery_acceptance.py",
        "poker_agent/final_strategy_quality_status.py",
        "poker_agent/service.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/build_scope_contract.py",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_behavioral_revalidation.py",
        "scripts/build_behavioral_revalidation_proof.py",
        "scripts/build_hole_card_data_quality.py",
        "scripts/build_data_leakage_contract.py",
        "scripts/build_actions_context_quality.py",
        "scripts/build_actions_dataset_export_contract.py",
        "scripts/build_stack_event_context_quality.py",
        "scripts/build_raw_model_status.py",
        "scripts/train_raw_model_challenger.py",
        "scripts/build_challenger_strategy_quality.py",
        "scripts/build_client_handoff.py",
        "scripts/build_training_cluster_requirements.py",
        "scripts/run_today_acceptance_training.py",
        "scripts/build_client_gpu_training_response.py",
        "scripts/build_multi_agent_training_status.py",
        "scripts/build_phase3_open_spiel_arena.py",
        "scripts/build_open_spiel_claim_readiness.py",
        "scripts/build_open_spiel_claim_contract.py",
        "scripts/build_rl_delivery_boundary.py",
        "scripts/build_evaluation_metric_contract.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_llm_decision_context.py",
        "scripts/build_llm_role_boundary.py",
        "scripts/build_llm_policy_experimental.py",
        "scripts/build_project_completion.py",
        "scripts/build_final_delivery_acceptance.py",
        "scripts/build_final_strategy_quality_status.py",
        "scripts/check_repo_hygiene.py",
        "scripts/evaluate_policy.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_decision_context_eval.py",
        "scripts/build_decision_context_holdout.py",
        "scripts/build_llm_decision_gate.py",
        "scripts/build_llm_architecture_comparison.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/production_gate.py",
        "scripts/research_experiment.py",
        "scripts/run_hydra_experiment.py",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/verify_delivery.py",
    ]
    for relative in source_files:
        path = root / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return f"{len(source_files)} Python files compile without writing bytecode"


def model_loads(model_path: Path) -> str:
    try:
        model = load_policy(model_path)
    except Exception as exc:
        risk = _read_json(model_path.parents[1] / "reports" / "model_risk_register.json")
        runtime = risk.get("raw_artifact_runtime_status", {})
        if runtime.get("status") == "LOAD_FAILED":
            return f"raw_artifact_load_failed_tracked={type(exc).__name__}"
        raise
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
    agent = get_agent()
    observed = agent.predict(
        PredictionRequest(
            position="BTN",
            street="preflop",
            hole_cards=["Ah", "Kd"],
            board_cards=[],
            pot=2.5,
            to_call=1.0,
            stack=100.0,
            min_raise=2.0,
            player_count=6,
            opponent_wait_before_turn_ms=1800.0,
            betting_history=[
                {"player_position": "BB", "action": "raise", "wait_time_ms": 2100.0},
            ],
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
            stack=100.0,
            min_raise=2.0,
            player_count=6,
        )
    ).to_dict()
    if observed["model_status"] == "missing_card_fallback":
        raise AssertionError("Observed-card request incorrectly used fallback")
    if observed["timing_method"] != "table_tempo_calibrated":
        raise AssertionError("Observed table timing did not calibrate the action plan")
    if isinstance(agent, MLPolicyAgent):
        runtime_policy = getattr(getattr(agent, "model", None), "metadata", {}).get("policy")
        if runtime_policy == "routed_policy_bundle":
            if missing["model_status"] != "routed_policy_bundle":
                raise AssertionError("Missing-card request did not use routed bundle runtime")
        elif missing["model_status"] != "missing_card_fallback":
            raise AssertionError("Missing-card request did not use fallback")
    for payload in (observed, missing):
        total = sum(float(value) for value in payload["probabilities"].values())
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"Probabilities do not sum to 1: {total}")
    return f"agent={type(agent).__name__}, observed={observed['action']} missing={missing['action']}"


def autonomous_agent_contract() -> str:
    controller = get_autonomous_agent()
    hand_id = f"delivery-verification-{uuid.uuid4().hex}"
    payload = {
        "hand_id": hand_id,
        "sequence_number": 0,
        "event_id": "state-0",
        "state": {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["Ah", "Kd"],
            "board_cards": [],
            "pot": 2.5,
            "to_call": 1.0,
            "stack": 100.0,
            "min_raise": 2.0,
            "player_count": 6,
        },
    }
    decision, replayed = controller.decide(payload)
    replay, second_replay = controller.decide(payload)
    if replayed or not second_replay:
        raise AssertionError("Autonomous event idempotency contract failed")
    if decision.decision_id != replay.decision_id:
        raise AssertionError("Autonomous replay changed the decision")
    if decision.action not in decision.legal_actions:
        raise AssertionError("Autonomous controller returned an illegal action")
    if abs(sum(decision.probabilities.values()) - 1.0) > 1e-6:
        raise AssertionError("Autonomous legal probabilities do not sum to 1")
    session = controller.settle(hand_id, {"chip_delta": 0.0})
    if session["status"] != "settled" or session["decision_count"] != 1:
        raise AssertionError(f"Autonomous settlement contract failed: {session}")
    capabilities = controller.capabilities()
    if capabilities.get("status") != "IMPLEMENTED":
        raise AssertionError(f"Unexpected autonomous capability status: {capabilities}")
    return f"agent={capabilities['agent_type']}, action={decision.action}, lifecycle=settled"


def health_contract(model_path: Path) -> str:
    resolved = resolve_model_path()
    accepted_paths = {model_path.resolve()}
    bundle_path = model_path.parent / "poker_policy_bundle.joblib"
    if bundle_path.exists():
        accepted_paths.add(bundle_path.resolve())
    if resolved.resolve() not in accepted_paths:
        raise AssertionError(f"Health resolved unexpected model path: {resolved}")
    payload = health_payload()
    model_status = payload.get("model_status")
    if model_status not in {"loaded", "fallback_rule_based_model_load_failed", "fallback_rule_based"}:
        raise AssertionError(f"Invalid model status: {payload}")
    if model_status == "loaded" and "valid_macro_f1" not in payload:
        raise AssertionError(f"Health payload missing model metric metadata: {payload}")
    if model_status == "fallback_rule_based_model_load_failed" and "model_load_error" not in payload:
        raise AssertionError(f"Fallback health payload does not expose load error: {payload}")
    return json.dumps(payload, sort_keys=True)


def public_openapi_contract() -> str:
    schema = app.openapi()
    paths = set(schema.get("paths") or {})
    expected_paths = {"/predict"}
    swagger_params = app.swagger_ui_parameters or {}
    docs_routes = [route for route in app.routes if getattr(route, "path", None) == "/docs"]
    if app.docs_url is not None:
        raise AssertionError("Default FastAPI Swagger route must be disabled for the client-facing docs shell")
    if len(docs_routes) != 1 or getattr(docs_routes[0], "include_in_schema", True):
        raise AssertionError("Client-facing /docs route must exist and stay hidden from OpenAPI")
    for required_fragment in (
        "client-facing-swagger",
        "compact-public-docs",
        "client-docs-helper",
        "parameters-container",
        "hideEmptyParameterSections",
        'url: "/openapi.json"',
        "SwaggerUIBundle",
        "expandPredictOperation",
        "keepPredictExpanded",
        "onComplete: keepPredictExpanded",
        "tryItOutEnabled: false",
        "supportedSubmitMethods: []",
        "try-out__btn",
    ):
        if required_fragment not in CLIENT_SWAGGER_HTML:
            raise AssertionError(f"Client-facing Swagger HTML is missing {required_fragment!r}")
    if "tryItOutEnabled: true" in CLIENT_SWAGGER_HTML:
        raise AssertionError("Public docs must not open Swagger in editable Try it out mode by default")
    if "supportedSubmitMethods: []" not in CLIENT_SWAGGER_HTML:
        raise AssertionError("Public docs must disable Swagger submit controls by default")
    if swagger_params.get("defaultModelsExpandDepth") != -1:
        raise AssertionError("Swagger UI must hide the Schemas/models panel")
    if swagger_params.get("docExpansion") != "full":
        raise AssertionError("Swagger UI must expand the public /predict operation by default")
    if paths != expected_paths:
        raise AssertionError(f"Public OpenAPI must expose only {sorted(expected_paths)}; got {sorted(paths)}")
    tags = [tag.get("name") for tag in schema.get("tags", []) if isinstance(tag, dict)]
    if tags != ["Prediction"]:
        raise AssertionError(f"Public OpenAPI must expose only the Prediction tag; got {tags}")
    if "/agent/decide" in paths or "/agent/sessions/{hand_id}/settle" in paths:
        raise AssertionError("Controlled session endpoints must remain hidden from public Swagger docs")
    schemas = (schema.get("components") or {}).get("schemas") or {}
    expected_schemas = {
        "ActionProbabilitiesBody",
        "BettingHistoryBody",
        "PredictRequestBody",
        "PredictResponseBody",
        "TimingContextBody",
    }
    if set(schemas) != expected_schemas:
        raise AssertionError(f"Public OpenAPI must expose only predict schemas; got {sorted(schemas)}")
    if {"HTTPValidationError", "ValidationError"} & set(schemas):
        raise AssertionError("Public OpenAPI must not expose validation-error schema clutter")
    if "additionalProp1" in json.dumps(schema, sort_keys=True):
        raise AssertionError("Public OpenAPI must not show generic additionalProp1 examples")
    predict_request_schema = schemas.get("PredictRequestBody") or {}
    request_example = predict_request_schema.get("example") or {}
    compact_example_fields = {
        "position",
        "street",
        "hole_cards",
        "board_cards",
        "pot",
        "to_call",
        "stack",
        "min_raise",
        "player_count",
    }
    if set(request_example) != compact_example_fields:
        raise AssertionError(
            "Public /predict request example must stay compact and client-facing; "
            f"got fields {sorted(request_example)}"
        )
    if {"betting_history", "timing_context"} & set(request_example):
        raise AssertionError("Advanced optional fields must not be shown in the default /predict example")
    predict_operation = ((schema.get("paths") or {}).get("/predict") or {}).get("post") or {}
    description = predict_operation.get("description") or ""
    if "JSON request body" not in description or "No query parameters" not in description:
        raise AssertionError("Public /predict docs must clarify that input is a JSON body, not query parameters")
    probability_schema = json.dumps(schemas.get("ActionProbabilitiesBody") or {}, sort_keys=True)
    for action in ("fold", "call", "check", "bet", "raise"):
        if action not in probability_schema:
            raise AssertionError(f"Action probability schema is missing {action}")
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict) and "422" in (operation.get("responses") or {}):
                raise AssertionError("Public OpenAPI must not show default validation-error schema blocks")
    return f"public_paths={','.join(sorted(paths))};tags={','.join(tags)}"


def api_input_contract() -> str:
    contract = api_contract()
    request_contract = contract.get("prediction_request") or {}
    fields = request_contract.get("request_fields") or {}
    for required in ("betting_history", "timing_context"):
        if required not in fields:
            raise AssertionError(f"Prediction request contract is missing {required}")
    if "observable before the target action" not in str(request_contract.get("leakage_rule", "")):
        raise AssertionError("Prediction request contract does not state the leakage boundary")
    return f"contract_version={contract.get('contract_version')}, request_fields={len(fields)}"


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Required report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def reports_contract(root: Path, require_gate_pass: bool) -> str:
    reports = root / "reports"
    gate = _read_json(reports / "production_gate.json")
    acceptance = _read_json(reports / "policy_acceptance.json")
    self_play = _read_json(reports / "production_self_play.json")
    deployed = _read_json(reports / "deployed_strategy_gate.json")
    delivery = _read_json(reports / "delivery_readiness.json")
    hygiene = _read_json(reports / "repo_hygiene.json")
    gold_payload = _read_json(reports / "llm_event_gold_eval.json")
    decision_context_payload = _read_json(reports / "llm_decision_context.json")
    context_smoke_payload = _read_json(reports / "llm_decision_context_smoke.json")
    qwen_decision_payload = _read_json(reports / "llm_decision_context_qwen25.json")
    llm_decision_gate = _read_json(reports / "llm_decision_gate.json")
    candidate_ranker = _read_json(reports / "llm_decision_candidate_ranker_qwen25.json")
    candidate_gate = _read_json(reports / "llm_decision_candidate_gate.json")
    architecture_comparison = _read_json(reports / "llm_architecture_comparison.json")
    phase2_selection_comparison = _read_json(reports / "phase2_selection_comparison.json")
    llm_role_boundary = _read_json(reports / "llm_role_boundary.json")
    llm_policy_experimental = _read_json(reports / "llm_policy_experimental.json")
    qlora_next_stage = _read_json(reports / "qlora_next_stage.json")
    production_runtime_monitoring = _read_json(reports / "production_runtime_monitoring.json")
    decision_holdout = _read_json(reports / "decision_context_holdout.json")
    scope_payload = _read_json(reports / "scope_contract.json")
    completion_payload = _read_json(reports / "project_completion.json")
    final_delivery_acceptance = _read_json(reports / "final_delivery_acceptance.json")
    final_strategy_quality_status = _read_json(reports / "final_strategy_quality_status.json")
    risk_payload = _read_json(reports / "model_risk_register.json")
    approval_payload = _read_json(reports / "production_approval.json")
    strategy_stack_maturity = _read_json(reports / "strategy_stack_maturity.json")
    behavioral_revalidation = _read_json(reports / "behavioral_revalidation.json")
    behavioral_revalidation_proof = _read_json(reports / "behavioral_revalidation_proof.json")
    human_likeness_evidence = _read_json(reports / "human_likeness_evidence.json")
    human_likeness_claim_gate = _read_json(reports / "human_likeness_claim_gate.json")
    bet_timing_calibration = _read_json(reports / "bet_timing_calibration.json")
    hole_card_data_quality = _read_json(reports / "hole_card_data_quality.json")
    data_leakage_contract = _read_json(reports / "data_leakage_contract.json")
    normalized_action_contract = _read_json(reports / "normalized_action_contract.json")
    actions_context_quality = _read_json(reports / "actions_context_quality.json")
    actions_dataset_export_contract = _read_json(reports / "actions_dataset_export_contract.json")
    stack_event_context_quality = _read_json(reports / "stack_event_context_quality.json")
    scenario_sanity = _read_json(reports / "scenario_sanity.json")
    raw_model_status = _read_json(reports / "raw_model_status.json")
    raw_model_challenger = _read_json(reports / "raw_model_challenger.json")
    challenger_strategy_quality = _read_json(reports / "challenger_strategy_quality.json")
    handoff_payload = _read_json(reports / "client_handoff.json")
    cluster_payload = _read_json(reports / "training_cluster_requirements.json")
    today_training = _read_json(reports / "today_acceptance_training.json")
    client_gpu_response = _read_json(reports / "client_gpu_training_response.json")
    multi_agent_training_status = _read_json(reports / "multi_agent_training_status.json")
    phase3_open_spiel_arena = _read_json(reports / "phase3_open_spiel_arena.json")
    open_spiel_claim_readiness = _read_json(reports / "open_spiel_claim_readiness.json")
    open_spiel_claim_contract = _read_json(reports / "open_spiel_claim_contract.json")
    rl_delivery_boundary = _read_json(reports / "rl_delivery_boundary.json")
    evaluation_metric_contract = _read_json(reports / "evaluation_metric_contract.json")
    test_execution_contract = _read_json(reports / "test_execution_contract.json")
    approval_boundary_payload = build_approval_boundary(root)
    approval_boundary = approval_boundary_payload.get("boundary", {})

    if scope_payload.get("overall_status") != "PASS":
        raise AssertionError(f"Scope contract did not pass: {scope_payload.get('overall_status')}")
    if completion_payload.get("overall_status") != "PASS":
        raise AssertionError(f"Project completion contract did not pass: {completion_payload.get('overall_status')}")
    completion_phases = completion_payload.get("phase_completion", {})
    for phase_name in (
        "phase_1_two_baselines",
        "phase_2_selection_optimization",
        "phase_3_evaluation",
        "phase_4_deployment",
    ):
        if (completion_phases.get(phase_name) or {}).get("status") != "PASS":
            raise AssertionError(f"Project completion phase is not PASS: {phase_name}")
    completion_boundary = completion_payload.get("known_boundary", {})
    if completion_boundary.get("component_risk") is not True or completion_boundary.get("production_blocker") is not False:
        raise AssertionError("Project completion boundary does not preserve component-risk semantics")
    if hygiene.get("status") != "PASS":
        raise AssertionError(f"Repository hygiene did not pass: {hygiene.get('status')}")
    if delivery.get("strategy_policy_status") not in {"APPROVED", None}:
        raise AssertionError(f"Delivery readiness does not preserve strategy approval: {delivery.get('strategy_policy_status')}")
    if deployed.get("status") != "PASS" or deployed.get("strategy_policy_status") != "APPROVED":
        raise AssertionError("Deployed strategy gate is not approved")
    if acceptance.get("overall_status") != "PASS":
        raise AssertionError("Policy acceptance report did not pass")
    if self_play.get("status") != "PASS" or self_play.get("production_scale_status") != "PASS":
        raise AssertionError("Production-scale self-play did not pass")
    if risk_payload.get("deployed_strategy_stack_status") != "APPROVED":
        raise AssertionError("Model risk register does not preserve deployed strategy approval")
    if approval_payload.get("overall_status") != "APPROVED_WITH_COMPONENT_RISK":
        raise AssertionError(f"Unexpected production approval status: {approval_payload.get('overall_status')}")
    maturity_current = strategy_stack_maturity.get("current_strategy_stack") or {}
    maturity_final = strategy_stack_maturity.get("final_engine_boundary") or {}
    if strategy_stack_maturity.get("overall_status") != "PASS":
        raise AssertionError(f"Strategy stack maturity contract did not pass: {strategy_stack_maturity.get('overall_status')}")
    if maturity_current.get("status") != "APPROVED_FOR_DEPLOYMENT_WITH_MONITORING":
        raise AssertionError("Strategy stack must be approved only for deployment with monitoring")
    if maturity_current.get("monitoring_required") is not True:
        raise AssertionError("Strategy stack deployment must require monitoring")
    if maturity_current.get("rollback_plan_required") is not True:
        raise AssertionError("Strategy stack deployment must require a rollback plan")
    if maturity_final.get("status") != "NOT_FINAL_MAXIMALLY_OPTIMIZED_ENGINE":
        raise AssertionError("Strategy stack must not be represented as a final optimized engine")
    if maturity_final.get("final_engine_claim_allowed") is not False:
        raise AssertionError("Final-engine claim must remain blocked")
    if maturity_final.get("maximally_optimized_claim_allowed") is not False:
        raise AssertionError("Maximally optimized engine claim must remain blocked")
    if (strategy_stack_maturity.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Strategy stack maturity invariants failed: {strategy_stack_maturity.get('invariants')}")
    behavioral_scope = behavioral_revalidation.get("current_validation_scope") or {}
    behavioral_boundary = behavioral_revalidation.get("revalidation_boundary") or {}
    if behavioral_revalidation.get("overall_status") != "PASS":
        raise AssertionError(f"Behavioral revalidation contract did not pass: {behavioral_revalidation.get('overall_status')}")
    if behavioral_scope.get("human_likeness_status") != "PASS":
        raise AssertionError("Human-likeness must pass for the current validation scope")
    if behavioral_scope.get("action_distribution_status") != "PASS":
        raise AssertionError("Action-distribution must pass for the current validation scope")
    if behavioral_boundary.get("larger_clean_real_gameplay_revalidation_required") is not True:
        raise AssertionError("Behavioral checks must require revalidation on larger clean real gameplay data")
    if behavioral_boundary.get("generalized_human_likeness_claim_allowed") is not False:
        raise AssertionError("Generalized human-likeness claims must remain blocked")
    if behavioral_boundary.get("generalized_action_distribution_claim_allowed") is not False:
        raise AssertionError("Generalized action-distribution claims must remain blocked")
    if behavioral_boundary.get("production_blocker") is not False:
        raise AssertionError("Behavioral revalidation requirement must not block current monitored deployment")
    if (behavioral_revalidation.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Behavioral revalidation invariants failed: {behavioral_revalidation.get('invariants')}")
    if behavioral_revalidation_proof.get("overall_status") != "PASS":
        raise AssertionError(f"Behavioral revalidation proof did not pass: {behavioral_revalidation_proof.get('overall_status')}")
    if behavioral_revalidation_proof.get("proof_status") != "PASS":
        raise AssertionError("Behavioral revalidation proof cases did not pass")
    proof_cases = {case.get("name"): case for case in behavioral_revalidation_proof.get("proof_cases") or []}
    for required_case in (
        "base_contract_is_valid",
        "blocks_missing_larger_real_gameplay_revalidation",
        "blocks_generalized_human_likeness_claim",
        "blocks_generalized_action_distribution_claim",
        "blocks_wrong_revalidation_scope",
    ):
        if (proof_cases.get(required_case) or {}).get("passed") is not True:
            raise AssertionError(f"Behavioral revalidation proof case failed or missing: {required_case}")
    if (proof_cases.get("base_contract_is_valid") or {}).get("observed_status") != "PASS":
        raise AssertionError("Behavioral proof must show the base contract passes")
    for blocked_case in (
        "blocks_missing_larger_real_gameplay_revalidation",
        "blocks_generalized_human_likeness_claim",
        "blocks_generalized_action_distribution_claim",
        "blocks_wrong_revalidation_scope",
    ):
        if (proof_cases.get(blocked_case) or {}).get("observed_status") != "FAIL":
            raise AssertionError(f"Behavioral proof did not block false claim: {blocked_case}")
    if (behavioral_revalidation_proof.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Behavioral revalidation proof invariants failed: {behavioral_revalidation_proof.get('invariants')}")

    if human_likeness_evidence.get("overall_status") != "PASS":
        raise AssertionError(f"Human-likeness evidence contract did not pass: {human_likeness_evidence.get('overall_status')}")
    if (human_likeness_evidence.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Human-likeness evidence invariants failed: {human_likeness_evidence.get('invariants')}")
    if human_likeness_evidence.get("boundary") != "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF":
        raise AssertionError("Human-likeness evidence must block action-distribution-only proof")
    if human_likeness_evidence.get("human_likeness_fully_proven") is not False:
        raise AssertionError("Human-likeness must not be marked fully proven")
    if human_likeness_evidence.get("final_human_likeness_claim_allowed") is not False:
        raise AssertionError("Final human-likeness claim must remain blocked")
    if human_likeness_evidence.get("current_scope_action_distribution_passed") is not True:
        raise AssertionError("Current-scope action distribution must pass before the limited delivery claim")
    if human_likeness_evidence.get("current_delivery_blocker") is not False:
        raise AssertionError("Human-likeness evidence gap must not block current delivery")
    if human_likeness_evidence.get("model_quality_risk") is not True:
        raise AssertionError("Incomplete human-likeness evidence must remain a model-quality risk")
    expected_behavior_dimensions = {
        "action_distribution",
        "bet_sizing",
        "timing",
        "position_based_behavior",
        "street_level_strategy",
    }
    if set(human_likeness_evidence.get("required_behavior_dimensions") or []) != expected_behavior_dimensions:
        raise AssertionError("Human-likeness evidence must require all behavior dimensions")
    behavior_dimensions = human_likeness_evidence.get("behavior_dimensions") or {}
    for dimension_name in expected_behavior_dimensions:
        dimension = behavior_dimensions.get(dimension_name) or {}
        if dimension.get("required") is not True:
            raise AssertionError(f"Human-likeness behavior dimension must be required: {dimension_name}")
        if dimension.get("final_proof_allowed") is not False:
            raise AssertionError(f"Human-likeness behavior dimension final proof must remain blocked: {dimension_name}")
    human_likeness_proof_cases = human_likeness_evidence.get("proof_cases") or []
    if not human_likeness_proof_cases:
        raise AssertionError("Human-likeness evidence must include proof cases")
    for case in human_likeness_proof_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"Human-likeness evidence proof case failed: {case}")

    if human_likeness_claim_gate.get("overall_status") != "PASS":
        raise AssertionError(
            f"Human-likeness claim gate did not pass: {human_likeness_claim_gate.get('overall_status')}"
        )
    if (human_likeness_claim_gate.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Human-likeness claim gate invariants failed: {human_likeness_claim_gate.get('invariants')}")
    if human_likeness_claim_gate.get("boundary") != "ACTION_DISTRIBUTION_ALONE_IS_NOT_FULL_HUMAN_LIKENESS_PROOF":
        raise AssertionError("Human-likeness claim gate must reject action-distribution-only proof")
    if human_likeness_claim_gate.get("claim") != "FULL_HUMAN_LIKENESS":
        raise AssertionError("Human-likeness claim gate must govern the full human-likeness claim")
    if human_likeness_claim_gate.get("decision") != "BLOCKED":
        raise AssertionError("Full human-likeness claim gate must remain BLOCKED")
    if human_likeness_claim_gate.get("claim_allowed") is not False:
        raise AssertionError("Full human-likeness claim must not be allowed")
    if human_likeness_claim_gate.get("human_likeness_fully_proven") is not False:
        raise AssertionError("Human-likeness claim gate must not mark full proof as complete")
    if human_likeness_claim_gate.get("action_distribution_only_proof_rejected") is not True:
        raise AssertionError("Human-likeness claim gate must reject action-distribution-only proof")
    if human_likeness_claim_gate.get("current_delivery_blocker") is not False:
        raise AssertionError("Human-likeness claim gate must not block current delivery")
    if human_likeness_claim_gate.get("model_quality_risk") is not True:
        raise AssertionError("Human-likeness claim gate must remain a model-quality risk")
    if set(human_likeness_claim_gate.get("required_evidence_dimensions") or []) != expected_behavior_dimensions:
        raise AssertionError("Human-likeness claim gate must require all behavior dimensions")
    claim_requirements = human_likeness_claim_gate.get("evidence_requirements") or {}
    for dimension_name in expected_behavior_dimensions:
        requirement = claim_requirements.get(dimension_name) or {}
        if requirement.get("required_for_final_claim") is not True:
            raise AssertionError(f"Human-likeness claim dimension must be required: {dimension_name}")
        if requirement.get("currently_sufficient_for_final_claim") is not False:
            raise AssertionError(
                f"Human-likeness claim dimension must not be sufficient in current scope: {dimension_name}"
            )
    claim_proof_cases = human_likeness_claim_gate.get("proof_cases") or []
    if not claim_proof_cases:
        raise AssertionError("Human-likeness claim gate must include proof cases")
    for case in claim_proof_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"Human-likeness claim gate proof case failed: {case}")

    bet_timing_current = bet_timing_calibration.get("current_delivery_scope") or {}
    bet_timing_boundary = bet_timing_calibration.get("calibration_boundary") or {}
    bet_timing_label_boundary = bet_timing_calibration.get("timing_label_quality_boundary") or {}
    bet_timing_fields = set(bet_timing_current.get("api_response_fields") or [])
    bet_timing_proof_cases = {case.get("name"): case for case in bet_timing_calibration.get("proof_cases") or []}
    for required_field in ("bet_size", "wait_time_ms", "sizing_method", "timing_method"):
        if required_field not in bet_timing_fields:
            raise AssertionError(f"Bet/timing calibration contract missing response field: {required_field}")
    if bet_timing_calibration.get("overall_status") != "PASS":
        raise AssertionError(f"Bet/timing calibration contract did not pass: {bet_timing_calibration.get('overall_status')}")
    if bet_timing_current.get("implementation_status") != "IMPLEMENTED_AND_MEASURED":
        raise AssertionError("Bet-sizing and timing must remain implemented and measured")
    if bet_timing_current.get("timing_and_bet_size_status") != "PASS":
        raise AssertionError("Timing and bet-size measurement must pass for the current scope")
    if bet_timing_current.get("timing_policy_type") != "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED":
        raise AssertionError("Timing policy must remain marked as heuristic/table-tempo calibrated")
    if bet_timing_current.get("real_human_timing_label_quality") != "TIMING_LABEL_QUALITY_UNCERTAIN":
        raise AssertionError("Real human timing label quality must remain marked uncertain")
    if bet_timing_current.get("real_human_timing_labels_available") is not False:
        raise AssertionError("Real human timing labels must not be claimed available without reviewed labels")
    if bet_timing_current.get("timing_human_likeness_final_proof_allowed") is not False:
        raise AssertionError("Timing human-likeness final proof must remain blocked")
    if bet_timing_current.get("timing_evidence_status") != "HEURISTIC_TIMING_ONLY_NOT_FINAL_HUMAN_LIKENESS_PROOF":
        raise AssertionError("Timing evidence must remain marked as heuristic, not final human-likeness proof")
    if bet_timing_boundary.get("requires_more_real_player_behavior_labels") is not True:
        raise AssertionError("Higher-realism bet/timing calibration must require more real player labels")
    if bet_timing_boundary.get("requires_bet_size_labels") is not True:
        raise AssertionError("Bet-size labels must remain required for higher-realism calibration")
    if bet_timing_boundary.get("requires_decision_timing_labels") is not True:
        raise AssertionError("Decision-timing labels must remain required for higher-realism calibration")
    if bet_timing_boundary.get("final_high_realism_claim_allowed") is not False:
        raise AssertionError("Final high-realism bet/timing claim must remain blocked")
    if bet_timing_boundary.get("production_blocker_for_current_delivery") is not False:
        raise AssertionError("Bet/timing calibration gap must not block current delivery")
    if bet_timing_label_boundary.get("status") != "TIMING_LABEL_QUALITY_UNCERTAIN":
        raise AssertionError("Timing label-quality boundary must remain uncertain")
    if (
        bet_timing_label_boundary.get("boundary")
        != "REAL_HUMAN_TIMING_LABELS_REQUIRED_FOR_FULL_HUMAN_LIKENESS_PROOF"
    ):
        raise AssertionError("Timing boundary must require reviewed real human timing labels for full proof")
    if bet_timing_label_boundary.get("timing_feature_available") is not True:
        raise AssertionError("Timing feature must remain available")
    if bet_timing_label_boundary.get("timing_policy_type") != "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED":
        raise AssertionError("Timing boundary must remain heuristic/table-tempo calibrated")
    if bet_timing_label_boundary.get("real_human_timing_labels_available") is not False:
        raise AssertionError("Timing boundary must not claim real human labels are available")
    if bet_timing_label_boundary.get("requires_real_human_timing_labels") is not True:
        raise AssertionError("Timing boundary must require real human timing labels")
    if bet_timing_label_boundary.get("uses_real_human_timing_labels") is not False:
        raise AssertionError("Timing boundary must not claim reviewed real human timing labels are used")
    required_timing_label_fields = {
        "decision_start_ts",
        "decision_end_ts",
        "human_wait_time_ms",
        "street",
        "position",
        "facing_bet",
        "action",
    }
    if set(bet_timing_label_boundary.get("required_timing_label_fields") or []) != required_timing_label_fields:
        raise AssertionError("Timing boundary must keep the complete required real-human timing label schema")
    if bet_timing_label_boundary.get("heuristic_timing_counts_as_full_human_likeness_proof") is not False:
        raise AssertionError("Heuristic timing must not count as full human-likeness proof")
    if bet_timing_label_boundary.get("final_human_likeness_claim_allowed_from_timing_alone") is not False:
        raise AssertionError("Timing alone must not allow a final human-likeness claim")
    if bet_timing_label_boundary.get("final_production_human_likeness_proof_allowed") is not False:
        raise AssertionError("Timing final production human-likeness proof must remain blocked")
    if bet_timing_label_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("Timing label-quality gap must not block current delivery")
    if bet_timing_label_boundary.get("model_quality_risk") is not True:
        raise AssertionError("Timing label-quality gap must remain a model-quality risk")
    if (bet_timing_calibration.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Bet/timing calibration invariants failed: {bet_timing_calibration.get('invariants')}")
    for required_case in (
        "base_contract_is_valid",
        "blocks_final_timing_human_likeness_claim",
        "blocks_unreviewed_timing_label_availability_claim",
        "blocks_heuristic_timing_relabel_as_supervised",
        "blocks_heuristic_timing_as_full_human_likeness_proof",
        "blocks_missing_real_timing_label_contract",
        "blocks_delivery_blocker_reclassification",
        "blocks_model_quality_risk_removal",
    ):
        if (bet_timing_proof_cases.get(required_case) or {}).get("passed") is not True:
            raise AssertionError(f"Bet/timing proof case did not pass: {required_case}")
    if (bet_timing_proof_cases.get("base_contract_is_valid") or {}).get("observed_status") != "PASS":
        raise AssertionError("Bet/timing base proof case must observe PASS")
    for blocked_case in (
        "blocks_final_timing_human_likeness_claim",
        "blocks_unreviewed_timing_label_availability_claim",
        "blocks_heuristic_timing_relabel_as_supervised",
        "blocks_delivery_blocker_reclassification",
        "blocks_model_quality_risk_removal",
    ):
        if (bet_timing_proof_cases.get(blocked_case) or {}).get("observed_status") != "FAIL":
            raise AssertionError(f"Bet/timing blocked proof case must observe FAIL: {blocked_case}")

    hole_cov = hole_card_data_quality.get("coverage_snapshot") or {}
    hole_mitigation = hole_card_data_quality.get("mitigation_boundary") or {}
    hole_upstream = hole_card_data_quality.get("upstream_data_quality_boundary") or {}
    hole_direct_audit = hole_cov.get("direct_players_csv_audit") or {}
    hole_promotion = hole_card_data_quality.get("promotion_boundary") or {}
    hole_risk = hole_card_data_quality.get("risk_contract") or {}
    hole_feature_policy = hole_risk.get("feature_policy") or {}
    hole_strength = hole_card_data_quality.get("strength_signal_impact") or {}
    if hole_card_data_quality.get("overall_status") != "PASS":
        raise AssertionError(f"Hole-card data-quality contract did not pass: {hole_card_data_quality.get('overall_status')}")
    if hole_risk.get("risk_id") != "hole_card_data_risk":
        raise AssertionError("Hole-card data risk must be explicit in the delivery contract")
    if hole_risk.get("root_cause") != "ocr_hole_card_extraction_missing_or_unreliable":
        raise AssertionError("Hole-card root cause must remain OCR extraction quality")
    if hole_risk.get("primary_dataset_column") != "players.cards":
        raise AssertionError("Hole-card contract must bind the risk to players.cards")
    if hole_risk.get("weakens_primary_poker_signal") is not True:
        raise AssertionError("Hole-card risk must be declared as weakening the primary poker strength signal")
    if hole_risk.get("current_delivery_blocker") is not False:
        raise AssertionError("Hole-card risk must remain a component risk, not a current delivery blocker")
    if hole_risk.get("final_strategy_quality_claim_blocker") is not True:
        raise AssertionError("Hole-card risk must block final strategy-quality claims")
    if hole_feature_policy.get("missing_or_invalid_cards") != "flag_and_route":
        raise AssertionError("Missing or invalid hole cards must be flagged and routed")
    if hole_feature_policy.get("do_not_impute_unknown_cards_as_known_private_cards") is not True:
        raise AssertionError("Unknown hole cards must not be imputed as known private cards")
    if hole_feature_policy.get("do_not_treat_missing_cards_as_reliable_zero_strength") is not True:
        raise AssertionError("Missing hole cards must not be treated as reliable zero-strength evidence")
    if hole_feature_policy.get("train_observed_card_and_public_context_slices_separately") is not True:
        raise AssertionError("Observed-card and missing-card policy slices must remain separated")
    if not {"strength_proxy", "made_hand_score", "draw_pressure"}.issubset(
        set(hole_strength.get("affected_features") or [])
    ):
        raise AssertionError("Hole-card contract must list the affected strength features")
    if float(hole_cov.get("missing_hole_card_rate", 0.0)) <= float(hole_cov.get("complete_hole_card_rate", 1.0)):
        raise AssertionError("Hole-card audit must preserve missing-card dominance over complete-card coverage")
    if hole_direct_audit.get("status") != "PASS":
        raise AssertionError("Hole-card contract must include a passing direct players.csv audit")
    if int(hole_direct_audit.get("rows_scanned") or 0) <= 0:
        raise AssertionError("Direct players.csv hole-card audit must scan at least one row")
    if hole_direct_audit.get("reliable_two_card_rate") is None:
        raise AssertionError("Direct players.csv audit must expose reliable_two_card_rate")
    if hole_direct_audit.get("invalid_card_rate") is None:
        raise AssertionError("Direct players.csv audit must expose invalid_card_rate")
    if hole_mitigation.get("mitigation_status") != "MITIGATED_BY_ROUTED_POLICY_BUNDLE":
        raise AssertionError("Routed policy bundle mitigation must remain active for hole-card missingness")
    if hole_mitigation.get("routed_policy_bundle_handles_missingness") is not True:
        raise AssertionError("Routed policy bundle must explicitly handle missing hole-card paths")
    if hole_mitigation.get("fully_solves_upstream_data_quality_issue") is not False:
        raise AssertionError("Routed policy bundle must not claim to fully solve upstream hole-card data quality")
    if hole_upstream.get("limitation_status") != "OPEN_DATA_QUALITY_LIMITATION":
        raise AssertionError("Hole-card data-quality limitation must remain open")
    if hole_upstream.get("upstream_data_quality_issue_resolved") is not False:
        raise AssertionError("Upstream hole-card data-quality issue cannot be marked resolved")
    if hole_upstream.get("requires_ocr_or_parser_improvement") is not True:
        raise AssertionError("Hole-card boundary must require OCR/parser improvement")
    if hole_upstream.get("requires_larger_reviewed_card_labels") is not True:
        raise AssertionError("Hole-card boundary must require larger reviewed card labels")
    if hole_upstream.get("production_blocker_for_current_deployment") is not False:
        raise AssertionError("Hole-card limitation must remain a component risk, not a current deployment blocker")
    if hole_upstream.get("component_risk") is not True:
        raise AssertionError("Hole-card limitation must remain visible as a component risk")
    if hole_promotion.get("standalone_policy_promotion_allowed") is not False:
        raise AssertionError("Hole-card risk must block standalone policy promotion")
    if hole_promotion.get("model_promotion_blocker") is not True:
        raise AssertionError("Hole-card risk must remain a model promotion blocker")
    if hole_promotion.get("current_deployment_blocker") is not False:
        raise AssertionError("Hole-card risk must not block the current monitored deployment")
    if hole_promotion.get("requires_reviewed_card_label_set") is not True:
        raise AssertionError("Hole-card policy promotion must require reviewed card labels")
    if (hole_card_data_quality.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Hole-card data-quality invariants failed: {hole_card_data_quality.get('invariants')}")
    leakage_boundary = data_leakage_contract.get("leakage_boundary") or {}
    leakage_features = data_leakage_contract.get("feature_name_audit") or {}
    leakage_request = data_leakage_contract.get("prediction_request_audit") or {}
    leakage_models = data_leakage_contract.get("model_artifact_audit") or {}
    leakage_sources = data_leakage_contract.get("source_usage_audit") or {}
    leakage_risk = data_leakage_contract.get("leakage_risk_contract") or {}
    leakage_policy = leakage_risk.get("feature_policy") or {}
    leakage_field_definitions = leakage_risk.get("field_definitions") or {}
    final_board_contract = data_leakage_contract.get("final_board_snapshot_contract") or {}
    final_board_policy = final_board_contract.get("feature_policy") or {}
    final_board_mitigation = final_board_contract.get("required_mitigation") or {}
    final_board_definitions = final_board_contract.get("field_definitions") or {}
    leakage_raw_schema = data_leakage_contract.get("raw_dataset_schema_audit") or {}
    if data_leakage_contract.get("overall_status") != "PASS":
        raise AssertionError(f"Data-leakage contract did not pass: {data_leakage_contract.get('overall_status')}")
    forbidden_outcome_fields = {
        "winner_positions",
        "stack_delta",
        "ending_stack",
        "dealer_winner",
        "dealer_pot",
        "pot_from_stacks",
    }
    if set(data_leakage_contract.get("forbidden_outcome_fields") or []) != forbidden_outcome_fields:
        raise AssertionError("Data-leakage contract must list the full forbidden outcome field set")
    if data_leakage_contract.get("raw_final_board_snapshot_fields") != ["hands.csv::board_cards"]:
        raise AssertionError("Data-leakage contract must list the raw final board snapshot source field")
    if leakage_risk.get("risk_id") != "post_outcome_feature_leakage":
        raise AssertionError("Data-leakage risk id must remain explicit")
    if leakage_risk.get("root_cause") != "post_hand_outcome_fields_available_in_raw_dataset_schema":
        raise AssertionError("Data-leakage root cause must remain post-hand outcome fields in raw schema")
    if leakage_risk.get("temporal_requirement") != "features_must_be_observable_before_target_action":
        raise AssertionError("Training and prediction features must remain decision-time observable")
    if set(leakage_risk.get("forbidden_fields") or []) != forbidden_outcome_fields:
        raise AssertionError("Leakage risk contract forbidden fields must match the delivery contract")
    if set(leakage_field_definitions) != forbidden_outcome_fields:
        raise AssertionError("Every forbidden outcome field must have a temporal availability definition")
    if leakage_field_definitions.get("pot_from_stacks", {}).get("availability") != "post_hand_reconstruction":
        raise AssertionError("pot_from_stacks must remain classified as post-hand reconstruction")
    for field in forbidden_outcome_fields - {"pot_from_stacks"}:
        if leakage_field_definitions.get(field, {}).get("availability") != "post_hand":
            raise AssertionError(f"{field} must remain classified as post-hand outcome data")
    if leakage_policy.get("raw_dataset_schema_presence") != "allowed_for_audit_and_reporting_only":
        raise AssertionError("Raw outcome fields may remain only for audit/reporting")
    if leakage_policy.get("training_feature_use") != "forbidden":
        raise AssertionError("Outcome fields must remain forbidden as training features")
    if leakage_policy.get("prediction_request_use") != "forbidden":
        raise AssertionError("Outcome fields must remain forbidden in prediction requests")
    if leakage_policy.get("model_artifact_feature_use") != "forbidden":
        raise AssertionError("Outcome fields must remain forbidden in model artifacts")
    if leakage_policy.get("detected_violation") != "production_blocker":
        raise AssertionError("Detected outcome-field leakage must remain a production blocker")
    if final_board_contract.get("risk_id") != "final_board_snapshot_leakage":
        raise AssertionError("Final-board leakage risk id must remain explicit")
    if final_board_contract.get("root_cause") != "hands_csv_board_cards_is_final_hand_snapshot":
        raise AssertionError("Final-board leakage root cause must remain the final hands.csv board snapshot")
    if (
        final_board_contract.get("temporal_requirement")
        != "board_features_must_be_truncated_to_cards_visible_at_target_street"
    ):
        raise AssertionError("Final-board leakage contract must require street-visible board truncation")
    if final_board_contract.get("raw_final_board_snapshot_fields") != ["hands.csv::board_cards"]:
        raise AssertionError("Final-board leakage contract must name hands.csv::board_cards")
    if set(final_board_definitions) != {"hands.csv::board_cards"}:
        raise AssertionError("Final-board leakage contract must define hands.csv::board_cards")
    if final_board_definitions.get("hands.csv::board_cards", {}).get("availability") != "post_hand_final_snapshot":
        raise AssertionError("hands.csv::board_cards must remain classified as a post-hand final snapshot")
    if final_board_policy.get("raw_dataset_schema_presence") != "allowed_for_audit_and_street_truncation_only":
        raise AssertionError("Raw final board may remain only for audit and street truncation")
    if final_board_policy.get("direct_training_feature_use") != "forbidden":
        raise AssertionError("Raw final board snapshot must not be allowed as a direct training feature")
    if final_board_policy.get("prediction_request_board_cards") != "allowed_only_as_decision_time_visible_board":
        raise AssertionError("Prediction request board_cards must remain decision-time visible only")
    if final_board_policy.get("model_artifact_direct_final_board_feature_use") != "forbidden":
        raise AssertionError("Raw final board snapshot must not appear as a direct model artifact feature")
    if final_board_policy.get("detected_violation") != "production_blocker":
        raise AssertionError("Detected final-board leakage must remain a production blocker")
    if final_board_mitigation.get("truncate_final_board_by_street") is not True:
        raise AssertionError("Final-board mitigation must truncate raw board cards by street")
    expected_visible_counts = {
        "preflop_visible_board_count": 0,
        "flop_visible_board_count": 3,
        "turn_visible_board_count": 4,
        "river_visible_board_count": 5,
    }
    for key, expected in expected_visible_counts.items():
        if final_board_mitigation.get(key) != expected:
            raise AssertionError(f"Invalid final-board visible-card mitigation count: {key}")
    if leakage_boundary.get("training_feature_use_allowed") is not False:
        raise AssertionError("Outcome-only fields must not be allowed as training features")
    if leakage_boundary.get("prediction_request_use_allowed") is not False:
        raise AssertionError("Outcome-only fields must not be allowed in prediction requests")
    if leakage_boundary.get("model_artifact_feature_use_allowed") is not False:
        raise AssertionError("Outcome-only fields must not be allowed in model artifact features")
    if leakage_boundary.get("direct_final_board_snapshot_feature_use_allowed") is not False:
        raise AssertionError("Raw final board snapshot must not be allowed as a direct feature")
    if leakage_boundary.get("decision_time_visible_board_cards_allowed") is not True:
        raise AssertionError("Decision-time visible board cards must remain allowed")
    if leakage_boundary.get("dataset_schema_presence_allowed") is not True:
        raise AssertionError("Outcome-only fields may remain in raw dataset schema for audit/reporting")
    if leakage_boundary.get("production_blocker_if_detected") is not True:
        raise AssertionError("Detected outcome-field leakage must remain a production blocker")
    if int(leakage_features.get("examples_scanned") or 0) <= 0:
        raise AssertionError("Data-leakage feature audit must scan training examples")
    if leakage_features.get("forbidden_feature_names_detected"):
        raise AssertionError(f"Forbidden training features detected: {leakage_features.get('forbidden_feature_names_detected')}")
    if leakage_request.get("forbidden_feature_names_detected"):
        raise AssertionError(f"Forbidden prediction request features detected: {leakage_request.get('forbidden_feature_names_detected')}")
    if leakage_models.get("forbidden_model_features_detected"):
        raise AssertionError(f"Forbidden model artifact features detected: {leakage_models.get('forbidden_model_features_detected')}")
    if leakage_sources.get("forbidden_source_usages"):
        raise AssertionError(f"Forbidden outcome-field source usage detected: {leakage_sources.get('forbidden_source_usages')}")
    if leakage_raw_schema.get("status") != "PASS":
        raise AssertionError("Raw dataset schema audit must pass")
    if leakage_raw_schema.get("presence_is_not_feature_approval") is not True:
        raise AssertionError("Raw schema presence must not be treated as feature approval")
    if leakage_raw_schema.get("final_board_snapshot_presence_is_not_feature_approval") is not True:
        raise AssertionError("Raw final board schema presence must not be treated as feature approval")
    raw_schema_fields = {item.get("field") for item in leakage_raw_schema.get("outcome_fields_present_in_raw_schema") or []}
    if not forbidden_outcome_fields.issubset(raw_schema_fields):
        raise AssertionError("Raw schema audit must expose all outcome fields present in the CSV schema")
    for item in leakage_raw_schema.get("outcome_fields_present_in_raw_schema") or []:
        if item.get("presence_allowed") is not True:
            raise AssertionError("Outcome fields in raw schema must remain allowed only for audit/reporting")
        if item.get("allowed_use") != "audit_reporting_settlement_only":
            raise AssertionError("Outcome fields in raw schema must not be approved for training use")
    final_board_schema_sources = {
        item.get("source_field") for item in leakage_raw_schema.get("final_board_snapshot_fields_present_in_raw_schema") or []
    }
    if final_board_schema_sources != {"hands.csv::board_cards"}:
        raise AssertionError("Raw schema audit must expose the final board snapshot field")
    for item in leakage_raw_schema.get("final_board_snapshot_fields_present_in_raw_schema") or []:
        if item.get("presence_allowed") is not True:
            raise AssertionError("Raw final board presence must remain allowed for audit/truncation")
        if item.get("allowed_use") != "audit_and_street_truncation_only":
            raise AssertionError("Raw final board must not be approved for direct training use")
        if item.get("direct_training_feature_use_allowed") is not False:
            raise AssertionError("Raw final board direct training feature use must remain forbidden")
    if (data_leakage_contract.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Data-leakage invariants failed: {data_leakage_contract.get('invariants')}")
    normalized_actions_audit = normalized_action_contract.get("actions_csv_audit") or {}
    normalized_training_audit = normalized_action_contract.get("training_label_audit") or {}
    normalized_examples = {
        example.get("raw_action"): example for example in normalized_action_contract.get("noisy_action_examples") or []
    }
    if normalized_action_contract.get("overall_status") != "PASS":
        raise AssertionError(
            f"Normalized action contract did not pass: {normalized_action_contract.get('overall_status')}"
        )
    if (normalized_action_contract.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"Normalized action contract invariants failed: {normalized_action_contract.get('invariants')}"
        )
    if normalized_action_contract.get("normalized_action_status") != "IMPLEMENTED":
        raise AssertionError("Normalized action contract must be implemented")
    if normalized_action_contract.get("raw_action_source_status") != "RAW_OCR_OR_DEALER_TEXT":
        raise AssertionError("actions.csv action source must remain declared as raw OCR/dealer text")
    if set(normalized_action_contract.get("canonical_actions") or []) != {
        "fold",
        "call",
        "check",
        "bet",
        "raise",
        "all_in",
    }:
        raise AssertionError("Normalized action contract must expose the full canonical action set")
    if normalized_action_contract.get("source_field") != "actions.csv::action":
        raise AssertionError("Normalized action source field must be actions.csv::action")
    if normalized_action_contract.get("normalized_field") != "canonical_action":
        raise AssertionError("Normalized action target field must be canonical_action")
    if normalized_action_contract.get("raw_ocr_action_must_not_be_training_label") is not True:
        raise AssertionError("Raw OCR action strings must not be valid training labels")
    for required_flag in (
        "normalization_required_before_training",
        "normalization_required_before_evaluation",
        "normalization_required_before_policy_comparison",
    ):
        if normalized_action_contract.get(required_flag) is not True:
            raise AssertionError(f"Normalized action contract missing required flag: {required_flag}")
    if normalized_action_contract.get("current_delivery_blocker") is not False:
        raise AssertionError("Normalized action contract must not block current delivery")
    if normalized_action_contract.get("model_quality_risk") is not False:
        raise AssertionError("Implemented normalized action contract must not remain an open model-quality risk")
    if normalized_actions_audit.get("action_column_present") is not True:
        raise AssertionError("actions.csv must expose an action column for normalization")
    if int(normalized_actions_audit.get("rows_scanned") or 0) <= 0:
        raise AssertionError("Normalized action audit must scan actions.csv rows")
    if int(normalized_actions_audit.get("canonical_decision_rows") or 0) <= 0:
        raise AssertionError("Normalized action audit must find canonical decision rows")
    if normalized_training_audit.get("status") != "PASS":
        raise AssertionError("Training labels must be normalized into the canonical action set")
    if normalized_training_audit.get("invalid_labels"):
        raise AssertionError(f"Raw or invalid training labels detected: {normalized_training_audit.get('invalid_labels')}")
    for raw_action, expected in {
        "ra1se": "raise",
        "cail": "call",
        "bett": "bet",
        "all-in": "all_in",
    }.items():
        example = normalized_examples.get(raw_action) or {}
        if example.get("observed") != expected or example.get("passed") is not True:
            raise AssertionError(f"Noisy action example failed normalization: {raw_action}")
    actions_schema = actions_context_quality.get("actions_csv_schema_audit") or {}
    actions_risk = actions_context_quality.get("risk_contract") or {}
    actions_mitigation = actions_context_quality.get("derived_context_mitigation") or {}
    actions_features = actions_context_quality.get("training_feature_audit") or {}
    actions_export = actions_context_quality.get("dataset_export_contract") or {}
    missing_action_fields = set(actions_schema.get("missing_explicit_context_fields") or [])
    required_action_fields = {
        "amount",
        "to_call",
        "pot_before_action",
        "min_raise",
        "legal_actions",
        "action_order",
        "last_aggressor",
        "facing_bet",
    }
    if actions_context_quality.get("overall_status") != "PASS":
        raise AssertionError(f"Actions-context quality contract did not pass: {actions_context_quality.get('overall_status')}")
    if actions_risk.get("risk_id") != "actions_csv_betting_context_incomplete":
        raise AssertionError("actions.csv betting-context risk id must remain explicit")
    if actions_risk.get("root_cause") != "actions_csv_lacks_decision_time_betting_context_fields":
        raise AssertionError("actions.csv betting-context root cause must remain explicit")
    if actions_risk.get("source_table") != "actions.csv":
        raise AssertionError("actions.csv betting-context risk must be tied to actions.csv")
    if set(actions_risk.get("missing_or_reconstructed_decision_fields") or []) != required_action_fields:
        raise AssertionError("actions.csv betting-context risk must list all missing or reconstructed decision fields")
    actions_policy = actions_risk.get("decision_time_context_policy") or {}
    if set(actions_policy) != required_action_fields:
        raise AssertionError("actions.csv betting-context policy must cover every required decision-time field")
    if actions_risk.get("target_row_values_are_labels_not_features") is not True:
        raise AssertionError("Target-row action context values must remain labels/evaluation values, not prediction features")
    for field in required_action_fields - {"action_order"}:
        field_policy = actions_policy.get(field) or {}
        if field_policy.get("target_row_value_allowed_as_feature") is not False:
            raise AssertionError(f"{field} target-row value must not be allowed as a decision feature")
        if not field_policy.get("required_semantics") or not field_policy.get("reconstruction_source"):
            raise AssertionError(f"{field} action-context policy must define semantics and reconstruction source")
    if actions_risk.get("mitigation_status") != "LEAKAGE_SAFE_RECONSTRUCTION_REQUIRED":
        raise AssertionError("actions.csv betting-context risk must require leakage-safe reconstruction")
    if actions_risk.get("current_delivery_blocker") is not False:
        raise AssertionError("actions.csv betting-context risk must not block current delivery")
    if actions_risk.get("model_quality_risk") is not True:
        raise AssertionError("actions.csv betting-context risk must remain a model-quality risk")
    if actions_risk.get("final_strategy_quality_claim_blocker_without_richer_action_context") is not True:
        raise AssertionError("actions.csv betting-context risk must block final strategy-quality claims without richer action context")
    if actions_export.get("status") != "EXPLICIT_BETTING_CONTEXT_REQUIRED_FOR_NEXT_DATASET_EXPORT":
        raise AssertionError("actions.csv dataset export contract must require explicit betting context")
    if set(actions_export.get("required_explicit_fields") or []) != required_action_fields:
        raise AssertionError("actions.csv dataset export required fields must match the betting-context contract")
    if actions_export.get("explicit_export_required") is not True:
        raise AssertionError("actions.csv next dataset export must persist explicit decision-time context fields")
    if actions_export.get("reconstructed_context_allowed_for_current_delivery") is not True:
        raise AssertionError("actions.csv reconstructed context must remain allowed for the current delivery")
    if actions_export.get("current_delivery_blocker") is not False:
        raise AssertionError("actions.csv dataset export gap must not block current delivery")
    if actions_export.get("model_quality_risk") is not True:
        raise AssertionError("actions.csv dataset export gap must remain a model-quality risk")
    if actions_schema.get("explicit_context_status") != "INCOMPLETE_EXPLICIT_BETTING_CONTEXT":
        raise AssertionError("actions.csv explicit betting context must remain marked incomplete")
    if not required_action_fields.issubset(missing_action_fields):
        raise AssertionError("actions.csv contract must expose all missing explicit betting-context fields")
    if actions_schema.get("limitation_status") != "OPEN_DATASET_LIMITATION":
        raise AssertionError("actions.csv betting-context limitation must remain open")
    if actions_mitigation.get("status") != "IMPLEMENTED_FROM_PRE_ACTION_EVENT_STREAM":
        raise AssertionError("Derived actions-context mitigation must remain implemented")
    if actions_mitigation.get("uses_target_action_amount_as_feature") is not False:
        raise AssertionError("Target action amount must not be used as a decision feature")
    if actions_mitigation.get("target_action_context_leakage_guard") is not True:
        raise AssertionError("Target-action context leakage guard must remain enabled")
    if actions_mitigation.get("uses_future_outcome_fields") is not False:
        raise AssertionError("Actions-context mitigation must not use future outcome fields")
    if actions_mitigation.get("does_not_fully_replace_explicit_context") is not True:
        raise AssertionError("Derived context must not claim to fully replace explicit action-context labels")
    if actions_mitigation.get("current_delivery_blocker") is not False:
        raise AssertionError("actions.csv context limitation must not block current delivery")
    if actions_mitigation.get("model_quality_risk") is not True:
        raise AssertionError("actions.csv context limitation must remain a model-quality risk")
    if actions_mitigation.get("final_strategy_quality_claim_blocker_without_richer_action_context") is not True:
        raise AssertionError("actions.csv derived context must block final strategy-quality claims without richer action context")
    if actions_features.get("status") != "PASS":
        raise AssertionError("Training features must include leakage-safe derived betting-context features")
    if int(actions_features.get("examples_scanned") or 0) <= 0:
        raise AssertionError("Actions-context feature audit must scan examples")
    if actions_features.get("missing_required_derived_features"):
        raise AssertionError(f"Missing required derived betting-context features: {actions_features.get('missing_required_derived_features')}")
    present_action_features = set(actions_features.get("required_derived_features_present") or [])
    for required_feature in ("facing_bet_derived", "last_aggressor_known", "last_aggressor_derived"):
        if required_feature not in present_action_features:
            raise AssertionError(f"Missing required derived betting-context feature: {required_feature}")
    if (actions_context_quality.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Actions-context invariants failed: {actions_context_quality.get('invariants')}")
    actions_dataset_current = actions_dataset_export_contract.get("current_delivery_boundary") or {}
    actions_dataset_future = actions_dataset_export_contract.get("future_export_boundary") or {}
    if actions_dataset_export_contract.get("overall_status") != "PASS":
        raise AssertionError(
            f"Actions dataset export contract did not pass: {actions_dataset_export_contract.get('overall_status')}"
        )
    if actions_dataset_export_contract.get("status") != "EXPLICIT_BETTING_CONTEXT_REQUIRED_FOR_NEXT_DATASET_EXPORT":
        raise AssertionError("Standalone actions.csv export contract must require explicit betting context")
    if actions_dataset_export_contract.get("source_table") != "actions.csv":
        raise AssertionError("Standalone actions.csv export contract must be tied to actions.csv")
    if set(actions_dataset_export_contract.get("required_explicit_fields") or []) != required_action_fields:
        raise AssertionError("Standalone actions.csv export required fields must match the betting-context contract")
    if set(actions_dataset_export_contract.get("field_contract") or {}) != required_action_fields:
        raise AssertionError("Standalone actions.csv export field contract must cover every required field")
    if actions_dataset_current.get("current_delivery_blocker") is not False:
        raise AssertionError("Standalone actions.csv export gap must not block current delivery")
    if actions_dataset_current.get("reconstructed_context_allowed") is not True:
        raise AssertionError("Standalone actions.csv export contract must allow current reconstructed context")
    if actions_dataset_future.get("explicit_export_required") is not True:
        raise AssertionError("Standalone actions.csv next dataset export must require explicit context fields")
    if actions_dataset_future.get("model_quality_risk_until_export_is_instrumented") is not True:
        raise AssertionError("Standalone actions.csv export gap must remain a model-quality risk")
    if actions_dataset_future.get("must_persist_decision_time_values") is not True:
        raise AssertionError("Standalone actions.csv export must persist decision-time values")
    if actions_dataset_future.get("must_not_use_target_row_values") is not True:
        raise AssertionError("Standalone actions.csv export must forbid target-row values as features")
    if actions_dataset_future.get("must_not_use_future_outcome_fields") is not True:
        raise AssertionError("Standalone actions.csv export must forbid future outcome fields")
    if (actions_dataset_export_contract.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"Standalone actions dataset export invariants failed: {actions_dataset_export_contract.get('invariants')}"
        )
    stack_schema = stack_event_context_quality.get("stack_events_schema_audit") or {}
    stack_risk = stack_event_context_quality.get("risk_contract") or {}
    stack_raw_boundary = stack_event_context_quality.get("raw_stack_event_boundary") or {}
    stack_mitigation = stack_event_context_quality.get("derived_context_mitigation") or {}
    stack_features = stack_event_context_quality.get("training_feature_audit") or {}
    stack_sample_features = stack_features.get("sample_stack_context_feature_values") or {}
    stack_proof_cases = {case.get("name"): case for case in stack_event_context_quality.get("proof_cases") or []}
    if stack_event_context_quality.get("overall_status") != "PASS":
        raise AssertionError(f"Stack-event context quality contract did not pass: {stack_event_context_quality.get('overall_status')}")
    if stack_risk.get("risk_id") != "raw_stack_events_require_decision_context_derivation":
        raise AssertionError("Stack-event context risk id must remain explicit")
    if stack_risk.get("root_cause") != "stack_events_csv_stores_stack_changes_not_decision_time_features":
        raise AssertionError("Stack-event context root cause must remain explicit")
    if stack_risk.get("source_table") != "stack_events.csv":
        raise AssertionError("Stack-event context risk must be tied to stack_events.csv")
    if stack_risk.get("implementation_module") != "poker_agent.stack_context.build_stack_decision_context":
        raise AssertionError("Stack-event context must reference the concrete stack decision context implementation")
    if stack_risk.get("pre_action_event_derivation_helper") != "poker_agent.stack_context.derive_stack_decision_context_from_events":
        raise AssertionError("Stack-event context must reference the pre-action raw event derivation helper")
    if stack_risk.get("raw_events_are_source_data_not_policy_features") is not True:
        raise AssertionError("Raw stack events must remain source data, not direct policy features")
    if stack_risk.get("target_action_stack_delta_is_label_context_not_feature") is not True:
        raise AssertionError("Target action stack delta must remain label/evaluation context, not a prediction feature")
    if stack_risk.get("current_delivery_blocker") is not False:
        raise AssertionError("Stack-event context risk must not block current delivery")
    if stack_risk.get("model_quality_risk") is not True:
        raise AssertionError("Stack-event context risk must remain a model-quality risk")
    if stack_risk.get("final_strategy_quality_claim_blocker_without_explicit_stack_context") is not True:
        raise AssertionError("Stack-event context risk must block final strategy claims without explicit stack context")
    stack_policy = stack_risk.get("derivation_policy") or {}
    if set(stack_policy) != {"pot", "effective_stack", "spr", "bet_size", "pressure"}:
        raise AssertionError("Stack-event derivation policy must cover pot, effective_stack, SPR, bet_size, and pressure")
    stack_required_features = set(stack_features.get("required_stack_context_features_present") or [])
    for context_name, policy in stack_policy.items():
        if not policy.get("required_semantics") or not policy.get("source"):
            raise AssertionError(f"Stack-event derivation policy for {context_name} must define semantics and source")
        if policy.get("target_action_delta_allowed") is not False:
            raise AssertionError(f"Stack-event derivation policy for {context_name} must forbid target action deltas")
        if not set(policy.get("derived_features") or []).issubset(stack_required_features):
            raise AssertionError(f"Stack-event derivation policy for {context_name} references missing derived features")
    if stack_schema.get("status") != "PASS":
        raise AssertionError("stack_events.csv audit must pass")
    if int(stack_schema.get("rows_scanned") or 0) <= 0:
        raise AssertionError("stack_events.csv audit must scan rows")
    if int(stack_schema.get("negative_diff_rows") or 0) <= 0:
        raise AssertionError("stack_events.csv must expose negative contribution rows for betting reconstruction")
    if stack_raw_boundary.get("status") != "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION":
        raise AssertionError("Raw stack events must remain marked as requiring decision-context derivation")
    if stack_raw_boundary.get("raw_stack_events_are_direct_policy_features") is not False:
        raise AssertionError("Raw stack events must not be direct policy features")
    if stack_raw_boundary.get("decision_time_derivation_required") is not True:
        raise AssertionError("Stack events must require decision-time derivation")
    if stack_raw_boundary.get("target_action_stack_delta_allowed_as_feature") is not False:
        raise AssertionError("Target action stack deltas must not be decision features")
    if stack_raw_boundary.get("post_hand_stack_outcome_allowed_as_feature") is not False:
        raise AssertionError("Post-hand stack outcomes must not be decision features")
    if stack_raw_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("Stack-event context gap must not block current delivery")
    if stack_raw_boundary.get("model_quality_risk") is not True:
        raise AssertionError("Stack-event context gap must remain model-quality risk")
    if stack_mitigation.get("status") != "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS":
        raise AssertionError("Derived stack-event context mitigation must remain implemented")
    if stack_mitigation.get("implementation_module") != "poker_agent.stack_context.build_stack_decision_context":
        raise AssertionError("Derived stack-event context mitigation must reference the concrete implementation")
    if stack_mitigation.get("pre_action_event_derivation_helper") != "poker_agent.stack_context.derive_stack_decision_context_from_events":
        raise AssertionError("Derived stack-event context mitigation must reference the pre-action raw event helper")
    if stack_mitigation.get("uses_target_action_stack_delta_as_feature") is not False:
        raise AssertionError("Derived stack context must not use target action stack delta")
    if stack_mitigation.get("target_action_stack_delta_leakage_guard") is not True:
        raise AssertionError("Target stack-delta leakage guard must remain enabled")
    if stack_mitigation.get("uses_post_hand_outcome_fields") is not False:
        raise AssertionError("Derived stack context must not use post-hand outcome fields")
    if stack_mitigation.get("final_strategy_quality_claim_blocker_without_explicit_stack_context") is not True:
        raise AssertionError("Derived stack context must block final strategy claims without explicit stack context")
    if stack_features.get("status") != "PASS":
        raise AssertionError("Training features must include derived stack-event context features")
    if float(stack_sample_features.get("stack_event_target_bet_size_used_as_feature", 0.0) or 0.0) != 0.0:
        raise AssertionError("Target action stack delta leakage guard must remain zero")
    for required_feature in (
        "reconstructed_pot",
        "reconstructed_effective_stack",
        "reconstructed_spr_after_call",
        "reconstructed_current_street_bet_size",
        "reconstructed_call_pressure",
        "reconstructed_raise_pressure",
    ):
        if required_feature not in set(stack_features.get("required_stack_context_features_present") or []):
            raise AssertionError(f"Missing derived stack-event feature: {required_feature}")
    if (stack_event_context_quality.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Stack-event context invariants failed: {stack_event_context_quality.get('invariants')}")
    for required_case in (
        "base_contract_is_valid",
        "blocks_raw_stack_events_as_direct_features",
        "blocks_target_action_stack_delta_feature_leakage",
        "blocks_delivery_blocker_reclassification",
        "blocks_model_quality_risk_removal",
    ):
        if (stack_proof_cases.get(required_case) or {}).get("passed") is not True:
            raise AssertionError(f"Stack-event proof case did not pass: {required_case}")
    if (stack_proof_cases.get("base_contract_is_valid") or {}).get("observed_status") != "PASS":
        raise AssertionError("Stack-event base proof case must observe PASS")
    for blocked_case in (
        "blocks_raw_stack_events_as_direct_features",
        "blocks_target_action_stack_delta_feature_leakage",
        "blocks_delivery_blocker_reclassification",
        "blocks_model_quality_risk_removal",
    ):
        if (stack_proof_cases.get(blocked_case) or {}).get("observed_status") != "FAIL":
            raise AssertionError(f"Stack-event blocked proof case must observe FAIL: {blocked_case}")

    scenario_errors = validate_scenario_sanity(scenario_sanity)
    if scenario_errors:
        raise AssertionError(f"Scenario sanity validation failed: {scenario_errors}")
    scenario_boundary = scenario_sanity.get("boundary") or {}
    if scenario_sanity.get("overall_status") != "PASS":
        raise AssertionError(f"Scenario sanity report did not pass: {scenario_sanity.get('overall_status')}")
    if scenario_sanity.get("passed_scenarios") != scenario_sanity.get("scenario_count"):
        raise AssertionError("All targeted poker sanity scenarios must pass")
    if scenario_boundary.get("full_production_strategy_proof") is not False:
        raise AssertionError("Scenario sanity must not be represented as full production strategy proof")
    if scenario_boundary.get("final_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Scenario sanity alone must not allow final strategy-quality claims")
    if scenario_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("Scenario sanity must not become a software delivery blocker")
    for case in scenario_sanity.get("cases") or []:
        if not case.get("guardrails"):
            raise AssertionError(f"Scenario sanity case did not expose applied guardrail: {case.get('scenario_id')}")

    raw_contract = raw_model_status.get("raw_supervised_model") or {}
    raw_boundary = raw_model_status.get("release_boundary") or {}
    if raw_contract.get("runtime_status") != "LOADABLE":
        raise AssertionError("Raw model status contract does not confirm runtime loadability")
    if raw_contract.get("quality_gate_status") != "FAIL":
        raise AssertionError("Raw model status contract must preserve the failing raw quality gate")
    if raw_contract.get("standalone_status") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Raw model status contract does not preserve standalone non-approval")
    if raw_contract.get("approved_as_standalone_policy") is not False:
        raise AssertionError("Raw model status contract incorrectly approves the raw model as standalone")
    if raw_boundary.get("component_risk") is not True or raw_boundary.get("production_blocker") is not False:
        raise AssertionError("Raw model status contract must track a component risk without creating a production blocker")
    if (raw_model_status.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Raw model status invariants failed: {raw_model_status.get('invariants')}")
    challenger_boundary = raw_model_challenger.get("approval_boundary") or {}
    challenger_best = raw_model_challenger.get("best_candidate") or {}
    challenger_gate = challenger_best.get("gate") or {}
    if (raw_model_challenger.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Raw model challenger invariants failed: {raw_model_challenger.get('invariants')}")
    if challenger_boundary.get("existing_service_delivery_affected") is not False:
        raise AssertionError("Raw model challenger must not break existing service delivery")
    if challenger_gate.get("status") != "PASS" and raw_model_challenger.get("approved_as_standalone_policy") is not False:
        raise AssertionError("Raw model challenger incorrectly approves a failing standalone candidate")
    if challenger_gate.get("status") != "PASS" and raw_model_challenger.get("standalone_status") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Raw model challenger does not preserve standalone non-approval")
    strategy_quality_boundary = challenger_strategy_quality.get("strategy_quality_boundary") or {}
    strategy_quality_challenger = challenger_strategy_quality.get("challenger_result") or {}
    if challenger_strategy_quality.get("overall_status") != "PASS":
        raise AssertionError(f"Challenger strategy-quality contract did not pass: {challenger_strategy_quality.get('overall_status')}")
    if (challenger_strategy_quality.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Challenger strategy-quality invariants failed: {challenger_strategy_quality.get('invariants')}")
    if strategy_quality_boundary.get("challenger_required_before_final_claim") is not True:
        raise AssertionError("Final strategy-quality claims must require a challenger before approval")
    if strategy_quality_boundary.get("challenger_compared_to_raw_model") is not True:
        raise AssertionError("Challenger must be compared before final strategy-quality claims")
    if strategy_quality_boundary.get("final_production_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Final production-level strategy quality must remain blocked until challenger gates pass")
    if strategy_quality_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("Challenger strategy-quality gap must not block current delivery")
    if strategy_quality_boundary.get("deployed_strategy_stack_affected") is not False:
        raise AssertionError("Challenger strategy-quality gap must not affect deployed stack approval")
    if strategy_quality_challenger.get("gate_status") != challenger_gate.get("status"):
        raise AssertionError("Challenger strategy-quality contract does not match raw challenger gate status")
    if approval_payload.get("raw_supervised_model", {}).get("standalone_status") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Production approval does not preserve raw-model standalone boundary")
    if approval_payload.get("risk_position", {}).get("deployment_blockers") != 0:
        raise AssertionError("Production approval incorrectly reports a deployment blocker")
    if approval_boundary.get("release_status") != "READY_WITH_COMPONENT_RISK":
        raise AssertionError(f"Unexpected approval boundary release status: {approval_boundary.get('release_status')}")
    if approval_boundary.get("deployed_strategy_stack") != "APPROVED":
        raise AssertionError("Approval boundary does not preserve deployed strategy approval")
    if approval_boundary.get("raw_supervised_model_runtime") != "LOADABLE":
        raise AssertionError("Approval boundary does not confirm the raw supervised model is loadable")
    if approval_boundary.get("raw_supervised_model_standalone") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Approval boundary does not preserve raw-model standalone boundary")
    if approval_boundary.get("production_blocker"):
        raise AssertionError("Approval boundary incorrectly marks the raw-model component risk as a production blocker")
    if not approval_boundary.get("component_risk"):
        raise AssertionError("Approval boundary does not track raw-model standalone non-approval as a component risk")
    if approval_boundary_payload.get("invariants", {}).get("status") != "PASS":
        raise AssertionError(f"Approval boundary invariants failed: {approval_boundary_payload.get('invariants')}")
    assert_approval_boundary(approval_boundary)
    handoff_position = handoff_payload.get("technical_position", {})
    if handoff_payload.get("handoff_status") != "READY_WITH_COMPONENT_RISK":
        raise AssertionError(f"Unexpected client handoff status: {handoff_payload.get('handoff_status')}")
    if handoff_position.get("service_delivery") != "READY":
        raise AssertionError("Client handoff does not mark service delivery as ready")
    if handoff_position.get("deployed_strategy_stack") != "APPROVED":
        raise AssertionError("Client handoff does not preserve deployed strategy approval")
    if handoff_position.get("raw_supervised_model_runtime") != "LOADABLE":
        raise AssertionError("Client handoff does not confirm the raw supervised model is loadable")
    if handoff_position.get("raw_supervised_model_standalone") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Client handoff does not preserve raw-model standalone boundary")
    if handoff_position.get("production_blocker"):
        raise AssertionError("Client handoff incorrectly marks the component risk as a production blocker")
    if not handoff_position.get("component_risk"):
        raise AssertionError("Client handoff does not track the raw-model limitation as a component risk")
    cluster_estimate = cluster_payload.get("estimate") or {}
    if cluster_payload.get("run_profile") != "immediate_delivery":
        raise AssertionError(f"Unexpected default cluster run profile: {cluster_payload.get('run_profile')}")
    if cluster_estimate.get("status") != "PENDING_CLUSTER_CONFIRMATION":
        raise AssertionError(f"Unexpected default cluster estimate status: {cluster_estimate.get('status')}")
    requested_fields = set(cluster_payload.get("requested_fields") or [])
    for required in ("gpu_type", "gpu_count", "vram_gb_per_gpu", "dedicated_or_shared"):
        if required not in requested_fields:
            raise AssertionError(f"Training cluster report is missing requested field: {required}")
    immediate = cluster_payload.get("immediate_delivery_cluster") or {}
    if immediate.get("expected_completion") != "same_day":
        raise AssertionError("Training cluster report does not preserve same-day A100/H100 delivery validation")
    hours = immediate.get("expected_delivery_validation_hours") or {}
    if hours.get("A100") != 3 or hours.get("H100") != 2:
        raise AssertionError("Training cluster report does not preserve A100/H100 immediate delivery hours")
    full_reference = cluster_payload.get("full_training_reference") or {}
    if full_reference.get("not_required_for_current_delivery") is not True:
        raise AssertionError("Full multi-agent training must remain separate from current delivery approval")
    if today_training.get("profile") != "today_acceptance_training":
        raise AssertionError(f"Unexpected today-training profile: {today_training.get('profile')}")
    if today_training.get("selected_architecture") != "routed_policy_bundle":
        raise AssertionError("Today training must use the routed policy bundle architecture")
    if today_training.get("training_status") != "PASS":
        raise AssertionError("Today acceptance training did not pass")
    if today_training.get("delivery_status") != "READY_FOR_CURRENT_DELIVERY":
        raise AssertionError("Today acceptance training does not mark current delivery ready")
    boundary = today_training.get("approval_boundary") or {}
    if boundary.get("full_multi_agent_training") != "DEFERRED_TO_PRODUCTION_HARDENING":
        raise AssertionError("Today training report must keep full multi-agent training as hardening work")
    metrics = today_training.get("valid_metrics") or {}
    if float(metrics.get("macro_f1", 0.0)) <= 0.0:
        raise AssertionError("Today acceptance training report is missing validation metrics")
    client_training = client_gpu_response.get("current_delivery_training") or {}
    if client_training.get("training_status") != "PASS":
        raise AssertionError("Client GPU response does not preserve passing acceptance training status")
    if "dedicated A100 or H100" not in client_gpu_response.get("recommended_reply", ""):
        raise AssertionError("Client GPU response does not answer the A100/H100 availability question")
    gpu_boundary = client_gpu_response.get("gpu_boundary") or {}
    if gpu_boundary.get("full_multi_agent_training") != "separate production-hardening phase":
        raise AssertionError("Client GPU response must keep full multi-agent training as a separate hardening phase")
    if "Do not represent" not in gpu_boundary.get("do_not_claim", ""):
        raise AssertionError("Client GPU response must preserve the no-false-production-claim boundary")
    multi_agent_boundary = multi_agent_training_status.get("training_boundary") or {}
    multi_agent_validation_boundary = multi_agent_training_status.get("validation_vs_training_boundary") or {}
    multi_agent_hardening_plan = multi_agent_training_status.get("hardening_training_plan") or {}
    multi_agent_approval = multi_agent_training_status.get("approval_boundary") or {}
    if multi_agent_training_status.get("overall_status") != "PASS":
        raise AssertionError(f"Multi-agent training status did not pass: {multi_agent_training_status.get('overall_status')}")
    if multi_agent_boundary.get("delivery_validation_status") != "PASS":
        raise AssertionError("Multi-agent training status must preserve passing delivery validation")
    if multi_agent_boundary.get("acceptance_training_sufficient_for_delivery") is not True:
        raise AssertionError("Acceptance training must remain sufficient for delivery validation")
    if multi_agent_boundary.get("full_production_scale_multi_agent_training_status") != "NOT_COMPLETED":
        raise AssertionError("Full production-scale multi-agent training must not be marked completed")
    if multi_agent_boundary.get("full_long_running_self_play_completed") is not False:
        raise AssertionError("Long-running self-play cannot be completed by the acceptance run")
    if multi_agent_boundary.get("production_blocker") is not False:
        raise AssertionError("Deferred full multi-agent training must not block the current delivery package")
    if multi_agent_validation_boundary.get("acceptance_self_play_is_delivery_validation") is not True:
        raise AssertionError("Acceptance self-play must be reported as delivery validation")
    if multi_agent_validation_boundary.get("acceptance_self_play_counts_as_full_training") is not False:
        raise AssertionError("Acceptance self-play must not count as full production-scale training")
    if multi_agent_hardening_plan.get("required_profile") != "full_multi_agent_training":
        raise AssertionError("Full training hardening plan must require the full_multi_agent_training profile")
    if multi_agent_hardening_plan.get("required_gpu_class") != "single dedicated NVIDIA A100 or H100":
        raise AssertionError("Full training hardening plan must require a dedicated A100/H100 class GPU")
    if multi_agent_hardening_plan.get("seed_stability_required") is not True:
        raise AssertionError("Full training hardening plan must require seed stability")
    if int(multi_agent_hardening_plan.get("minimum_independent_training_seeds") or 0) < 5:
        raise AssertionError("Full training hardening plan requires at least five independent seeds")
    if int(multi_agent_hardening_plan.get("minimum_paired_hands") or 0) <= int(
        multi_agent_validation_boundary.get("acceptance_self_play_paired_hands") or 0
    ):
        raise AssertionError("Full training hardening simulation volume must exceed acceptance self-play volume")
    if int(multi_agent_hardening_plan.get("estimated_duration_days_single_a100_or_h100") or 0) < 5:
        raise AssertionError("Full training hardening plan must preserve the five-day A100/H100 estimate")
    if multi_agent_hardening_plan.get("completion_status") != "NOT_COMPLETED":
        raise AssertionError("Full training hardening plan must remain NOT_COMPLETED until executed")
    if multi_agent_approval.get("full_training_claim_allowed") is not False:
        raise AssertionError("Full multi-agent training claims must remain blocked")
    if (multi_agent_training_status.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Multi-agent training status invariants failed: {multi_agent_training_status.get('invariants')}")
    if phase3_open_spiel_arena.get("overall_status") != "PASS":
        raise AssertionError(f"Phase 3 OpenSpiel arena contract did not pass: {phase3_open_spiel_arena.get('overall_status')}")
    if (phase3_open_spiel_arena.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Phase 3 OpenSpiel arena invariants failed: {phase3_open_spiel_arena.get('invariants')}")
    phase3_contract = phase3_open_spiel_arena.get("arena_contract") or {}
    phase3_quality = phase3_open_spiel_arena.get("quality_boundary") or {}
    if phase3_contract.get("arena_type") != "AGENT_ONLY_OPEN_SPIEL_ARENA":
        raise AssertionError("Phase 3 OpenSpiel arena must be marked as agent-only")
    if phase3_contract.get("agent_only_table") is not True:
        raise AssertionError("Phase 3 OpenSpiel arena must use only agent-controlled seats")
    if phase3_contract.get("all_seats_controlled_by_agents") is not True:
        raise AssertionError("Phase 3 OpenSpiel arena must have one agent policy per seat")
    if phase3_contract.get("human_players_present") is not False:
        raise AssertionError("Phase 3 OpenSpiel arena must not include human players")
    if phase3_contract.get("fixed_scripted_opponents_present") is not False:
        raise AssertionError("Phase 3 OpenSpiel arena must not use fixed scripted opponents")
    if phase3_quality.get("is_reinforcement_learning_stage") is not True:
        raise AssertionError("Phase 3 OpenSpiel arena must be classified as the RL/self-play stage")
    phase3_rl_proof = phase3_open_spiel_arena.get("rl_training_proof_boundary") or {}
    if phase3_rl_proof.get("gate_name") != "phase3_open_spiel_rl_training_evidence_gate":
        raise AssertionError("Phase 3 OpenSpiel RL proof must use the reusable training evidence gate")
    required_rl_evidence = {
        "real_open_spiel_runtime",
        "agent_only_arena",
        "two_phase1_trained_policy_artifacts",
        "long_run_simulation_volume",
        "seed_stability",
        "policy_update_training",
    }
    if set(phase3_rl_proof.get("required_evidence") or []) != required_rl_evidence:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require the complete evidence set")
    if phase3_rl_proof.get("status") != "TRAINING_PROOF_NOT_COMPLETED":
        raise AssertionError("Phase 3 OpenSpiel RL training proof must remain not completed until a real training run is executed")
    if phase3_rl_proof.get("real_open_spiel_runtime_required") is not True:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require real pyspiel runtime")
    if phase3_rl_proof.get("phase1_trained_policy_artifacts_required") is not True:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require two trained Phase 1 policy artifacts")
    if phase3_rl_proof.get("seed_stability_required") is not True:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require seed stability")
    if int(phase3_rl_proof.get("minimum_independent_seeds") or 0) < 5:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require at least five independent seeds")
    if phase3_rl_proof.get("long_run_required") is not True:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require a long run")
    if int(phase3_rl_proof.get("minimum_long_run_episodes") or 0) < 5000:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require at least 5000 episodes")
    if phase3_rl_proof.get("policy_update_training_required") is not True:
        raise AssertionError("Phase 3 OpenSpiel RL proof must require policy-update training")
    if phase3_rl_proof.get("measured_win_rate_claim_allowed") is not False:
        raise AssertionError("Phase 3 OpenSpiel RL win-rate claims must remain blocked without full training proof")
    if phase3_rl_proof.get("current_delivery_blocker") is not False:
        raise AssertionError("Phase 3 OpenSpiel RL training gap must not block current service delivery")
    if phase3_rl_proof.get("model_quality_risk") is not True:
        raise AssertionError("Phase 3 OpenSpiel RL training gap must remain a model-quality risk")
    phase3_proof_cases = phase3_open_spiel_arena.get("proof_cases") or []
    if not phase3_proof_cases:
        raise AssertionError("Phase 3 OpenSpiel report must include proof cases")
    for case in phase3_proof_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"Phase 3 OpenSpiel proof case failed: {case}")
    blocked_case_names = {
        "blocks_win_rate_claim_without_real_open_spiel_runtime",
        "blocks_win_rate_claim_without_two_phase1_artifacts",
        "blocks_win_rate_claim_without_seed_stability",
        "blocks_win_rate_claim_without_long_run",
        "blocks_win_rate_claim_without_policy_update_training",
        "blocks_win_rate_claim_without_agent_only_table",
    }
    observed_blocked_cases = {case.get("name") for case in phase3_proof_cases if case.get("observed_status") == "FAIL"}
    if not blocked_case_names.issubset(observed_blocked_cases):
        raise AssertionError("Phase 3 OpenSpiel proof cases must demonstrate blocked false RL win-rate claims")
    if phase3_open_spiel_arena.get("status") == "READY_PENDING_OPEN_SPIEL_RUNTIME":
        runtime_boundary = phase3_open_spiel_arena.get("runtime_boundary") or {}
        if runtime_boundary.get("run_if_available_required_for_metrics") is not True:
            raise AssertionError("Pending Phase 3 OpenSpiel arena report must require a measured runtime run for metrics")
        if runtime_boundary.get("phase1_adapters_required_for_metrics") is not True:
            raise AssertionError("Pending Phase 3 OpenSpiel arena report must require Phase 1 adapters for metrics")
        if phase3_quality.get("metrics_claim_allowed") is not False:
            raise AssertionError("Pending Phase 3 OpenSpiel arena report must not allow metric claims")
        if "metrics" in phase3_open_spiel_arena:
            raise AssertionError("Pending Phase 3 OpenSpiel arena report must not include measured metrics")
    if open_spiel_claim_readiness.get("overall_status") != "PASS":
        raise AssertionError(
            f"OpenSpiel claim readiness did not pass: {open_spiel_claim_readiness.get('overall_status')}"
        )
    if (open_spiel_claim_readiness.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"OpenSpiel claim readiness invariants failed: {open_spiel_claim_readiness.get('invariants')}"
        )
    if open_spiel_claim_readiness.get("gate_name") != "open_spiel_claim_readiness":
        raise AssertionError("OpenSpiel claim readiness must use the explicit readiness gate name")
    if open_spiel_claim_readiness.get("current_delivery_blocker") is not False:
        raise AssertionError("OpenSpiel readiness gaps must not block current service delivery")
    readiness_runtime = open_spiel_claim_readiness.get("runtime") or {}
    readiness_artifacts = open_spiel_claim_readiness.get("phase1_policy_artifacts") or {}
    readiness_simulation = open_spiel_claim_readiness.get("simulation_profile") or {}
    readiness_policy_update = open_spiel_claim_readiness.get("policy_update_training") or {}
    if readiness_runtime.get("pyspiel_required") is not True:
        raise AssertionError("OpenSpiel claim readiness must require the pyspiel runtime")
    if int(readiness_artifacts.get("required_count") or 0) != 2:
        raise AssertionError("OpenSpiel claim readiness must require exactly two trained Phase 1 artifacts")
    if int(readiness_simulation.get("minimum_episodes") or 0) < 5000:
        raise AssertionError("OpenSpiel claim readiness must require at least 5000 episodes")
    if int(readiness_simulation.get("minimum_independent_seed_count") or 0) < 5:
        raise AssertionError("OpenSpiel claim readiness must require at least five independent seeds")
    if readiness_policy_update.get("required") is not True:
        raise AssertionError("OpenSpiel claim readiness must require PPO/equivalent policy-update training")
    if "PPO" not in str(readiness_policy_update.get("algorithm", "")):
        raise AssertionError("OpenSpiel claim readiness must preserve PPO as the default policy-update algorithm")
    readiness_command = str(open_spiel_claim_readiness.get("claim_command", ""))
    for fragment in (
        "scripts\\build_phase3_open_spiel_arena.py",
        "--claim-mode",
        "--run-if-available",
        "--phase1-adapters-ready",
        "--episodes 5000",
        "--independent-seed-count 5",
        "--policy-update-training-completed",
    ):
        if fragment not in readiness_command:
            raise AssertionError(f"OpenSpiel claim readiness command is missing {fragment!r}")
    if "win-rate" not in str(open_spiel_claim_readiness.get("blocked_claim", "")).lower():
        raise AssertionError("OpenSpiel readiness blocked claim must mention win-rate")
    if "production strategy" not in str(open_spiel_claim_readiness.get("blocked_claim", "")).lower():
        raise AssertionError("OpenSpiel readiness blocked claim must mention production strategy quality")
    if open_spiel_claim_readiness.get("claim_ready") is True:
        if open_spiel_claim_readiness.get("missing_requirements"):
            raise AssertionError("Ready OpenSpiel claim must not list missing requirements")
        if open_spiel_claim_readiness.get("model_quality_risk") is not False:
            raise AssertionError("Ready OpenSpiel claim must clear the readiness model-quality risk")
        if readiness_runtime.get("pyspiel_available") is not True:
            raise AssertionError("Ready OpenSpiel claim requires pyspiel availability")
        if int(readiness_artifacts.get("existing_count") or 0) != 2:
            raise AssertionError("Ready OpenSpiel claim requires both trained Phase 1 artifacts")
        if readiness_simulation.get("long_run_ready") is not True:
            raise AssertionError("Ready OpenSpiel claim requires long-run simulation volume")
        if readiness_simulation.get("seed_stability_ready") is not True:
            raise AssertionError("Ready OpenSpiel claim requires seed-stability coverage")
        if readiness_policy_update.get("ppo_or_equivalent_ready") is not True:
            raise AssertionError("Ready OpenSpiel claim requires completed PPO/equivalent policy-update training")
    else:
        if not open_spiel_claim_readiness.get("missing_requirements"):
            raise AssertionError("Blocked OpenSpiel claim readiness must list missing requirements")
        if open_spiel_claim_readiness.get("model_quality_risk") is not True:
            raise AssertionError("Blocked OpenSpiel claim readiness must remain a model-quality risk")
    readiness_cases = open_spiel_claim_readiness.get("proof_cases") or []
    if not readiness_cases:
        raise AssertionError("OpenSpiel claim readiness must include proof cases")
    for case in readiness_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"OpenSpiel claim readiness proof case failed: {case}")
    if open_spiel_claim_contract.get("overall_status") != "PASS":
        raise AssertionError(
            f"OpenSpiel claim contract did not pass: {open_spiel_claim_contract.get('overall_status')}"
        )
    if (open_spiel_claim_contract.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"OpenSpiel claim contract invariants failed: {open_spiel_claim_contract.get('invariants')}"
        )
    if open_spiel_claim_contract.get("gate_name") != "open_spiel_self_play_claim_contract":
        raise AssertionError("OpenSpiel claim contract must use the explicit claim gate name")
    if open_spiel_claim_contract.get("source_report") != "reports/phase3_open_spiel_arena.json":
        raise AssertionError("OpenSpiel claim contract must be derived from the Phase 3 arena report")
    if open_spiel_claim_contract.get("code_status") != "READY":
        raise AssertionError("OpenSpiel claim contract must mark the arena code path as ready")
    if open_spiel_claim_contract.get("arena_code_ready") is not True:
        raise AssertionError("OpenSpiel claim contract must preserve the code-ready boundary")
    if open_spiel_claim_contract.get("arena_type") != "AGENT_ONLY_OPEN_SPIEL_ARENA":
        raise AssertionError("OpenSpiel claim contract must bind the agent-only arena")
    if open_spiel_claim_contract.get("agent_only_table") is not True:
        raise AssertionError("OpenSpiel claim contract must require an agent-only table")
    if open_spiel_claim_contract.get("human_players_present") is not False:
        raise AssertionError("OpenSpiel claim contract must not allow human players in the arena")
    if set(open_spiel_claim_contract.get("required_evidence_before_self_play_claim") or []) != required_rl_evidence:
        raise AssertionError("OpenSpiel claim contract must require the complete RL evidence set")
    if open_spiel_claim_contract.get("self_play_win_rate_claim_allowed") is not False:
        raise AssertionError("OpenSpiel self-play win-rate claims must remain blocked without full training proof")
    if open_spiel_claim_contract.get("training_proof_completed") is not False:
        raise AssertionError("OpenSpiel claim contract must keep training proof incomplete until executed")
    if open_spiel_claim_contract.get("current_delivery_blocker") is not False:
        raise AssertionError("OpenSpiel training-proof gap must not block current service delivery")
    if open_spiel_claim_contract.get("model_quality_risk") is not True:
        raise AssertionError("OpenSpiel training-proof gap must remain a model-quality risk")
    if not open_spiel_claim_contract.get("missing_requirements"):
        raise AssertionError("Blocked OpenSpiel self-play claim must list missing requirements")
    open_spiel_claim_cases = open_spiel_claim_contract.get("proof_cases") or []
    if not open_spiel_claim_cases:
        raise AssertionError("OpenSpiel claim contract must include proof cases")
    for case in open_spiel_claim_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"OpenSpiel claim proof case failed: {case}")
    if rl_delivery_boundary.get("overall_status") != "PASS":
        raise AssertionError(f"RL delivery boundary did not pass: {rl_delivery_boundary.get('overall_status')}")
    if (rl_delivery_boundary.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"RL delivery boundary invariants failed: {rl_delivery_boundary.get('invariants')}")
    if rl_delivery_boundary.get("gate_name") != "rl_delivery_vs_strategy_claim_boundary":
        raise AssertionError("RL delivery boundary must use the explicit delivery-vs-claim gate name")
    if rl_delivery_boundary.get("boundary") != "DELIVERY_READY_BUT_RL_PROOF_REQUIRED_FOR_STRATEGY_CLAIMS":
        raise AssertionError("RL delivery boundary must preserve the delivery/strategy-claim separation")
    rl_delivery_scope = rl_delivery_boundary.get("delivery_scope") or {}
    rl_permissions = rl_delivery_boundary.get("claim_permissions") or {}
    rl_proof = rl_delivery_boundary.get("rl_training_proof") or {}
    if rl_delivery_scope.get("service_delivery_blocked_by_rl_training_gap") is not False:
        raise AssertionError("RL training gap must not be converted into a service delivery blocker")
    if rl_permissions.get("delivery_readiness_claim_allowed") is not True:
        raise AssertionError("RL delivery boundary must allow the current delivery-readiness claim")
    if rl_permissions.get("self_play_win_rate_claim_allowed") is not False:
        raise AssertionError("RL delivery boundary must block self-play win-rate claims without RL proof")
    if rl_permissions.get("production_strategy_quality_claim_allowed") is not False:
        raise AssertionError("RL delivery boundary must block production strategy-quality claims without RL proof")
    if rl_delivery_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("RL delivery boundary must not block current delivery")
    if rl_delivery_boundary.get("model_quality_risk") is not True:
        raise AssertionError("RL delivery boundary must preserve the pending RL proof as a model-quality risk")
    if set(rl_proof.get("required_evidence") or []) != required_rl_evidence:
        raise AssertionError("RL delivery boundary must reference the complete RL evidence set")
    if not rl_proof.get("missing_requirements"):
        raise AssertionError("RL delivery boundary must list missing proof requirements")
    if "self-play" not in str(rl_delivery_boundary.get("blocked_claim", "")).lower():
        raise AssertionError("RL delivery boundary blocked claim must mention self-play")
    if "production strategy" not in str(rl_delivery_boundary.get("blocked_claim", "")).lower():
        raise AssertionError("RL delivery boundary blocked claim must mention production strategy quality")
    rl_delivery_cases = rl_delivery_boundary.get("proof_cases") or []
    if not rl_delivery_cases:
        raise AssertionError("RL delivery boundary must include proof cases")
    for case in rl_delivery_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"RL delivery boundary proof case failed: {case}")
    if evaluation_metric_contract.get("overall_status") != "PASS":
        raise AssertionError(f"Evaluation metric contract did not pass: {evaluation_metric_contract.get('overall_status')}")
    if (evaluation_metric_contract.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Evaluation metric contract invariants failed: {evaluation_metric_contract.get('invariants')}")
    if evaluation_metric_contract.get("boundary") != "ACCURACY_AND_CROSS_ENTROPY_NOT_SUFFICIENT":
        raise AssertionError("Evaluation contract must block accuracy and cross-entropy only approval")
    if evaluation_metric_contract.get("accuracy_alone_sufficient") is not False:
        raise AssertionError("Accuracy alone must not be sufficient for strategy-quality approval")
    if evaluation_metric_contract.get("accuracy_and_cross_entropy_sufficient") is not False:
        raise AssertionError("Accuracy and cross-entropy must not be sufficient for strategy-quality approval")
    required_metric_families = {
        "action_classification",
        "calibration",
        "action_distribution",
        "bet_sizing",
        "simulation_return",
        "seed_stability",
    }
    if set(evaluation_metric_contract.get("required_metric_families") or []) != required_metric_families:
        raise AssertionError("Evaluation contract must require the full metric family set")
    required_production_metrics = {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "confusion_matrix",
        "calibration_ece",
        "action_distribution_js_divergence",
        "bet_size_mae",
        "expected_value_delta_vs_baseline",
        "win_rate",
        "seed_stability",
    }
    if set(evaluation_metric_contract.get("required_production_metrics") or []) != required_production_metrics:
        raise AssertionError("Evaluation contract must require the full production metric set")
    diagnostic_metrics = set(evaluation_metric_contract.get("diagnostic_metrics_not_sufficient_for_final_claim") or [])
    if not {"accuracy", "cross_entropy"}.issubset(diagnostic_metrics):
        raise AssertionError("Evaluation contract must mark accuracy and cross-entropy as insufficient diagnostics")
    metric_families = evaluation_metric_contract.get("metric_families") or {}
    for family_name in required_metric_families:
        if (metric_families.get(family_name) or {}).get("required") is not True:
            raise AssertionError(f"Evaluation metric family must be required: {family_name}")
    action_metrics = ((metric_families.get("action_classification") or {}).get("metrics") or {})
    for metric_name in ("accuracy", "macro_f1", "balanced_accuracy"):
        if action_metrics.get(metric_name) is None:
            raise AssertionError(f"Evaluation contract missing action-classification metric: {metric_name}")
    if (metric_families.get("action_classification") or {}).get("confusion_matrix_required_for_final_claim") is not True:
        raise AssertionError("Evaluation contract must require confusion matrix for final strategy-quality claims")
    confusion_matrix = action_metrics.get("confusion_matrix")
    if not isinstance(confusion_matrix, dict):
        raise AssertionError("Evaluation contract must include an action-classification confusion matrix")
    confusion_labels = confusion_matrix.get("labels")
    confusion_values = confusion_matrix.get("matrix")
    if not isinstance(confusion_labels, list) or not confusion_labels:
        raise AssertionError("Evaluation contract confusion matrix must include labels")
    if not isinstance(confusion_values, list) or len(confusion_values) != len(confusion_labels):
        raise AssertionError("Evaluation contract confusion matrix must be square")
    for row in confusion_values:
        if not isinstance(row, list) or len(row) != len(confusion_labels):
            raise AssertionError("Evaluation contract confusion matrix row length mismatch")
    calibration_family = metric_families.get("calibration") or {}
    if calibration_family.get("calibration_required_for_final_claim") is not True:
        raise AssertionError("Evaluation contract must require calibration for final claims")
    if calibration_family.get("cross_entropy_only_approval_allowed") is not False:
        raise AssertionError("Evaluation contract must reject cross-entropy-only approval")
    if calibration_family.get("diagnostic_loss_only_approval_allowed") is not False:
        raise AssertionError("Evaluation contract must reject diagnostic-loss-only approval")
    calibration_metrics = calibration_family.get("metrics") or {}
    if calibration_metrics.get("ece_10") is None:
        raise AssertionError("Evaluation contract must include calibration ECE")
    if calibration_metrics.get("cross_entropy") is None:
        raise AssertionError("Evaluation contract must include cross-entropy as a diagnostic metric")
    distribution_metrics = ((metric_families.get("action_distribution") or {}).get("metrics") or {})
    if distribution_metrics.get("js_divergence") is None:
        raise AssertionError("Evaluation contract must include action-distribution divergence")
    bet_family = metric_families.get("bet_sizing") or {}
    if bet_family.get("bet_size_mae_required_for_final_high_realism") is not True:
        raise AssertionError("Evaluation contract must require bet-size MAE or reviewed bet-size labels for final realism")
    simulation_metrics = ((metric_families.get("simulation_return") or {}).get("metrics") or {})
    for metric_name in ("win_rate", "expected_value_delta_vs_baseline"):
        if simulation_metrics.get(metric_name) is None:
            raise AssertionError(f"Evaluation contract missing simulation-return metric: {metric_name}")
    seed_metrics = ((metric_families.get("seed_stability") or {}).get("metrics") or {})
    if seed_metrics.get("full_training_seed_stability_required") is not True:
        raise AssertionError("Evaluation contract must require full-training seed stability")
    if seed_metrics.get("phase3_seed_stability_required") is not True:
        raise AssertionError("Evaluation contract must require Phase 3 seed stability")
    if evaluation_metric_contract.get("final_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Evaluation contract must block final strategy-quality claims until the full metric bundle passes")
    if evaluation_metric_contract.get("current_delivery_blocker") is not False:
        raise AssertionError("Evaluation metric gap must not block current service delivery")
    if evaluation_metric_contract.get("model_quality_risk") is not True:
        raise AssertionError("Incomplete evaluation metric bundle must remain a model-quality risk")
    proof_cases = evaluation_metric_contract.get("proof_cases") or []
    if not proof_cases:
        raise AssertionError("Evaluation metric contract must include proof cases")
    for case in proof_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"Evaluation metric proof case failed: {case}")
    if test_execution_contract.get("overall_status") != "PASS":
        raise AssertionError(f"Test execution contract did not pass: {test_execution_contract.get('overall_status')}")
    if (test_execution_contract.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Test execution contract invariants failed: {test_execution_contract.get('invariants')}")
    if test_execution_contract.get("boundary") != "FULL_PYTEST_TIMEOUT_IS_NOT_DELIVERY_APPROVAL":
        raise AssertionError("Test execution contract must preserve the full-pytest timeout boundary")
    full_pytest = test_execution_contract.get("full_pytest") or {}
    if full_pytest.get("status") == "TIMEOUT" and full_pytest.get("used_as_delivery_approval") is not False:
        raise AssertionError("Timed-out full pytest run must not be used as delivery approval")
    if full_pytest.get("completion_required_for_final_release_hardening") is not True:
        raise AssertionError("Full pytest completion must remain a release-hardening requirement")
    critical_validation = test_execution_contract.get("critical_validation") or {}
    if critical_validation.get("status") != "PASS":
        raise AssertionError("Critical validation suite must pass")
    if critical_validation.get("passed_tests", 0) < 1:
        raise AssertionError("Critical validation suite must report passed tests")
    if critical_validation.get("used_as_delivery_approval") is not True:
        raise AssertionError("Critical validation suite must be delivery approval evidence")
    delivery_verifier = test_execution_contract.get("delivery_verifier") or {}
    if delivery_verifier.get("status") != "PASS":
        raise AssertionError("Delivery verifier must pass in the test execution contract")
    if delivery_verifier.get("used_as_delivery_approval") is not True:
        raise AssertionError("Delivery verifier must be delivery approval evidence")
    test_metric = test_execution_contract.get("metric_contract") or {}
    if test_metric.get("accuracy_alone_sufficient") is not False:
        raise AssertionError("Test execution contract must preserve the accuracy-only rejection")
    if test_metric.get("accuracy_and_cross_entropy_sufficient") is not False:
        raise AssertionError("Test execution contract must preserve the accuracy and cross-entropy rejection")
    if test_metric.get("final_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Test execution contract must preserve final strategy-quality claim block")
    if test_execution_contract.get("current_delivery_blocker") is not False:
        raise AssertionError("Test execution transparency boundary must not block current delivery")
    test_proof_cases = test_execution_contract.get("proof_cases") or []
    if not test_proof_cases:
        raise AssertionError("Test execution contract must include proof cases")
    for case in test_proof_cases:
        if case.get("result") != "PASS":
            raise AssertionError(f"Test execution proof case failed: {case}")
    if risk_payload.get("raw_supervised_model_status") == "NOT_STANDALONE_APPROVED":
        summary = risk_payload.get("risk_summary", {})
        if summary.get("component_risks", 0) < 1:
            raise AssertionError("Raw model non-approval is not tracked as a component risk")
        if summary.get("deployment_blockers", 0) != 0:
            raise AssertionError("Raw model component risk is incorrectly marked as a deployment blocker")
    if gate.get("status") not in {"PASS", "FAIL"}:
        raise AssertionError(f"Invalid gate status: {gate.get('status')}")
    if require_gate_pass and gate.get("status") != "PASS":
        raise AssertionError("Production gate did not pass")
    strict_metrics = gold_payload.get("systems", {}).get("strict_schema_rules", {})
    if strict_metrics.get("event_type", {}).get("macro_f1", 0.0) < 0.90:
        raise AssertionError("Gold event extraction macro F1 is below acceptance threshold")
    if decision_context_payload.get("default_context_mode") != "full_in_context":
        raise AssertionError("LLM decision context does not default to full in-context mode")
    prompt_records = decision_context_payload.get("prompt_records") or []
    if not prompt_records:
        raise AssertionError("LLM decision context report has no prompt records")
    if not any(item.get("contains_rules") for item in prompt_records):
        raise AssertionError("LLM decision context report does not include rules-grounded records")
    if not any(item.get("contains_strategy_guidelines") for item in prompt_records):
        raise AssertionError("LLM decision context report does not include full strategy-guided records")
    if context_smoke_payload.get("quality_claim_allowed") is not False:
        raise AssertionError("Context smoke benchmark must not claim LLM policy quality")
    if context_smoke_payload.get("best_mode") is not None:
        raise AssertionError("Context smoke benchmark must not select a winning prompt")
    if set(context_smoke_payload.get("context_modes") or []) != {
        "minimal_zero_shot",
        "rules_grounded",
        "full_in_context",
    }:
        raise AssertionError("Context smoke benchmark does not cover all context modes")
    if not str(qwen_decision_payload.get("provider", "")).startswith("transformers:Qwen/"):
        raise AssertionError("Measured Qwen decision report is missing a real transformer provider")
    if qwen_decision_payload.get("dataset_kind") != "reconstructed_human_holdout":
        raise AssertionError("Measured Qwen decision report does not use the reconstructed human holdout")
    if qwen_decision_payload.get("comparison_allowed") is not True:
        raise AssertionError("Measured Qwen decision report is not enabled for provisional comparison")
    if qwen_decision_payload.get("quality_claim_allowed") is not False:
        raise AssertionError("Reconstructed human labels must not be treated as reviewed quality evidence")
    if decision_holdout.get("status") != "PASS" or decision_holdout.get("examples") != 20:
        raise AssertionError("Decision-context human holdout is incomplete")
    if llm_decision_gate.get("status") != "BASELINE_NOT_APPROVED":
        raise AssertionError(f"Unexpected LLM decision gate status: {llm_decision_gate.get('status')}")
    if (llm_decision_gate.get("production_boundary") or {}).get("deployed_strategy_stack_affected") is not False:
        raise AssertionError("LLM research gate incorrectly affects deployed strategy approval")
    if not str(candidate_ranker.get("provider", "")).startswith("transformers_candidate_ranker:Qwen/"):
        raise AssertionError("Measured candidate-ranker report is missing the Qwen provider")
    candidate_mode = candidate_ranker.get("provisional_best_mode")
    candidate_metrics = (candidate_ranker.get("systems") or {}).get(candidate_mode) or {}
    if candidate_metrics.get("schema_valid_rate") != 1.0:
        raise AssertionError("Candidate ranker does not preserve schema validity")
    if candidate_metrics.get("legal_action_rate") != 1.0:
        raise AssertionError("Candidate ranker does not preserve legal-action validity")
    if candidate_metrics.get("fallback_rate") != 0.0:
        raise AssertionError("Candidate ranker unexpectedly uses generation fallback")
    if candidate_gate.get("status") != "BASELINE_NOT_APPROVED":
        raise AssertionError(f"Unexpected candidate-ranker gate status: {candidate_gate.get('status')}")
    if architecture_comparison.get("recommended_architecture") != "candidate_ranker":
        raise AssertionError("Measured architecture comparison did not select candidate ranking")
    if architecture_comparison.get("production_approved") is not False:
        raise AssertionError("Architecture comparison incorrectly grants production approval")
    if (architecture_comparison.get("approval_boundary") or {}).get("deployed_strategy_stack_affected") is not False:
        raise AssertionError("LLM architecture comparison incorrectly affects deployed strategy approval")
    phase2_gate = phase2_selection_comparison.get("comparison_gate") or {}
    phase2_candidates = phase2_selection_comparison.get("candidates") or {}
    phase2_evidence_matrix = phase2_gate.get("candidate_evidence_matrix") or {}
    expected_phase2_candidates = {
        "llm_decision_agent",
        "supervised_model",
        "rule_based_fallback",
        "routed_policy_bundle",
        "future_rl_agent",
    }
    if phase2_selection_comparison.get("overall_status") != "PASS":
        raise AssertionError(
            f"Phase 2 selection comparison did not pass: {phase2_selection_comparison.get('overall_status')}"
        )
    if (phase2_selection_comparison.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"Phase 2 selection comparison invariants failed: {phase2_selection_comparison.get('invariants')}"
        )
    if phase2_selection_comparison.get("boundary") != "PHASE_2_SELECTION_REQUIRES_COMMON_HOLDOUT_AND_SIMULATION":
        raise AssertionError("Phase 2 selection comparison must expose the common-condition boundary")
    if set(phase2_selection_comparison.get("required_candidates") or []) != expected_phase2_candidates:
        raise AssertionError("Phase 2 selection comparison must include every required candidate")
    if (phase2_selection_comparison.get("common_holdout_contract") or {}).get("same_holdout_required") is not True:
        raise AssertionError("Phase 2 selection comparison must require a common holdout")
    if (phase2_selection_comparison.get("common_simulation_contract") or {}).get("same_simulation_required") is not True:
        raise AssertionError("Phase 2 selection comparison must require a common simulation")
    if phase2_gate.get("selected_for_current_delivery") != "routed_policy_bundle":
        raise AssertionError("Phase 2 current delivery architecture must remain routed_policy_bundle")
    if phase2_gate.get("final_selection_claim_allowed") is not False:
        raise AssertionError("Phase 2 final selection claim must remain blocked until common-condition comparison")
    if phase2_gate.get("best_approach_claim_allowed") is not False:
        raise AssertionError("Phase 2 best-approach claim must remain blocked until full common-condition evaluation")
    if phase2_gate.get("best_approach_claim_state") != "BLOCKED_PENDING_FULL_COMMON_CONDITION_EVALUATION":
        raise AssertionError("Phase 2 best-approach claim state must explicitly remain blocked")
    if phase2_gate.get("current_delivery_blocker") is not False:
        raise AssertionError("Phase 2 common-condition comparison gap must not block current delivery")
    if phase2_gate.get("model_quality_risk") is not True:
        raise AssertionError("Phase 2 common-condition comparison gap must remain a model-quality risk")
    if phase2_gate.get("all_candidates_compared_on_common_holdout") is not False:
        raise AssertionError("Phase 2 common holdout must not be marked complete yet")
    if phase2_gate.get("all_candidates_compared_in_common_simulation") is not False:
        raise AssertionError("Phase 2 common simulation must not be marked complete yet")
    if phase2_gate.get("all_candidate_metric_bundles_complete") is not False:
        raise AssertionError("Phase 2 metric bundle must not be marked complete until every candidate has all metrics")
    if not phase2_gate.get("missing_common_holdout_candidates"):
        raise AssertionError("Phase 2 missing common holdout candidates must be listed")
    if not phase2_gate.get("missing_common_simulation_candidates"):
        raise AssertionError("Phase 2 missing common simulation candidates must be listed")
    if not phase2_gate.get("missing_metric_bundle_candidates"):
        raise AssertionError("Phase 2 missing metric bundle candidates must be listed")
    if not phase2_gate.get("selection_ineligible_candidates"):
        raise AssertionError("Phase 2 selection-ineligible candidates must be listed")
    if set(phase2_evidence_matrix) != expected_phase2_candidates:
        raise AssertionError("Phase 2 evidence matrix must include every candidate")
    for candidate_name, evidence in phase2_evidence_matrix.items():
        if evidence.get("common_holdout_id") != "phase2_common_grouped_holdout_v1":
            raise AssertionError(f"Phase 2 evidence matrix has wrong holdout id for {candidate_name}")
        if evidence.get("common_simulation_id") != "phase2_common_agent_arena_v1":
            raise AssertionError(f"Phase 2 evidence matrix has wrong simulation id for {candidate_name}")
        if evidence.get("selection_eligible") is True:
            raise AssertionError(f"Phase 2 candidate must not be selection-eligible yet: {candidate_name}")
        if not evidence.get("blocking_reasons"):
            raise AssertionError(f"Phase 2 ineligible candidate must list blocking reasons: {candidate_name}")
    if (phase2_candidates.get("future_rl_agent") or {}).get("implementation_status") != "NOT_AVAILABLE_YET":
        raise AssertionError("Future RL agent must not be claimed available before Phase 3 training proof")

    llm_role = llm_role_boundary.get("current_llm_role") or {}
    llm_event_layer = llm_role.get("event_normalization_layer") or {}
    llm_context_layer = llm_role.get("decision_context_layer") or {}
    llm_term_boundary = llm_role_boundary.get("term_boundary") or {}
    llm_controlled_acceptance = llm_role_boundary.get("controlled_layer_acceptance") or {}
    llm_scope_boundary = llm_role_boundary.get("scope_disambiguation_contract") or {}
    llm_role_taxonomy = llm_role_boundary.get("role_taxonomy") or {}
    llm_autonomous_boundary = llm_role_boundary.get("autonomous_llm_agent_boundary") or {}
    llm_proof_cases = {case.get("name"): case for case in llm_role_boundary.get("proof_cases") or []}
    llm_claim_cases = {case.get("name"): case for case in llm_role_boundary.get("claim_validation_examples") or []}
    llm_production_scope_cases = {
        case.get("name"): case for case in llm_role_boundary.get("production_scope_claim_examples") or []
    }
    if llm_role_boundary.get("overall_status") != "PASS":
        raise AssertionError(f"LLM role boundary did not pass: {llm_role_boundary.get('overall_status')}")
    if llm_role.get("status") != "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER":
        raise AssertionError("LLM work must remain a controlled decision/context and event-normalization layer")
    if llm_term_boundary.get("status") != "LLM_BASED_AGENT_IS_UMBRELLA_TERM":
        raise AssertionError("LLM-based agent term must remain an umbrella term")
    if llm_term_boundary.get("requires_role_specific_qualification") is not True:
        raise AssertionError("LLM-based agent term must require role-specific qualification")
    if llm_term_boundary.get("must_not_imply_fully_autonomous_policy") is not True:
        raise AssertionError("LLM-based agent term must not imply autonomous poker policy")
    if llm_term_boundary.get("ambiguous_unqualified_usage_allowed") is not False:
        raise AssertionError("Unqualified ambiguous LLM-agent usage must remain blocked")
    if llm_controlled_acceptance.get("status") != "CONTROLLED_EVENT_CONTEXT_LAYER_APPROVED":
        raise AssertionError("Controlled LLM layer must be explicitly approved for the current delivery")
    if llm_controlled_acceptance.get("approved_for_current_delivery") is not True:
        raise AssertionError("Controlled LLM event/context layer must remain approved for current delivery")
    if set(llm_controlled_acceptance.get("approved_delivery_scope") or []) != {"event_normalization", "decision_context"}:
        raise AssertionError("Controlled LLM approval must be limited to event_normalization and decision_context")
    if set(llm_controlled_acceptance.get("research_only_scope") or []) != {"candidate_ranking"}:
        raise AssertionError("Candidate ranking must remain research-only in the LLM boundary")
    excluded_llm_scope = set(llm_controlled_acceptance.get("excluded_delivery_scope") or [])
    if "real_policy_agent" not in excluded_llm_scope:
        raise AssertionError("Real LLM policy agent must remain excluded from current delivery scope")
    if "fully_autonomous_poker_playing_llm_policy" not in excluded_llm_scope:
        raise AssertionError("Fully autonomous poker-playing LLM policy must remain excluded from current delivery scope")
    if llm_controlled_acceptance.get("fully_autonomous_poker_playing_llm_policy_status") != "FULLY_AUTONOMOUS_LLM_POLICY_NOT_APPROVED":
        raise AssertionError("Fully autonomous LLM policy status must remain not approved")
    if llm_controlled_acceptance.get("fully_autonomous_poker_playing_llm_policy_approved") is not False:
        raise AssertionError("Fully autonomous LLM policy must not be approved by the controlled-layer contract")
    if llm_controlled_acceptance.get("fully_autonomous_policy_claim_allowed") is not False:
        raise AssertionError("Fully autonomous LLM policy claim must remain blocked by the controlled-layer contract")
    if llm_controlled_acceptance.get("production_blocker_for_current_delivery") is not False:
        raise AssertionError("Controlled LLM boundary must not block current delivery")
    if llm_controlled_acceptance.get("future_policy_agent_requires_separate_approval") is not True:
        raise AssertionError("Future LLM policy-agent work must require separate approval")
    required_llm_roles = {
        "event_normalization",
        "decision_context",
        "candidate_ranking",
        "real_policy_agent",
    }
    required_llm_role_types = {
        "event_normalization": "EVENT_NORMALIZER",
        "decision_context": "DECISION_CONTEXT_AGENT",
        "candidate_ranking": "CANDIDATE_RANKER",
        "real_policy_agent": "POLICY_AGENT",
    }
    if llm_scope_boundary.get("status") != "EXPLICITLY_DISAMBIGUATED":
        raise AssertionError("LLM scope must be explicitly disambiguated")
    if llm_scope_boundary.get("llm_based_agent_requires_explicit_role") is not True:
        raise AssertionError("LLM-based agent must require an explicit role")
    if llm_scope_boundary.get("ambiguous_llm_agent_term_allowed") is not False:
        raise AssertionError("Ambiguous LLM-agent term must not be allowed")
    if set(llm_scope_boundary.get("required_roles") or []) != required_llm_roles:
        raise AssertionError("LLM scope required roles must match the role taxonomy")
    if (llm_scope_boundary.get("role_type_mapping") or {}) != required_llm_role_types:
        raise AssertionError("LLM role type mapping must explicitly separate all role types")
    if set(llm_scope_boundary.get("current_delivery_controlled_roles") or []) != {"event_normalization", "decision_context"}:
        raise AssertionError("LLM controlled delivery roles must be event_normalization and decision_context")
    if set(llm_scope_boundary.get("current_delivery_research_roles") or []) != {"candidate_ranking"}:
        raise AssertionError("LLM research delivery role must be candidate_ranking")
    if set(llm_scope_boundary.get("not_current_delivery_roles") or []) != {"real_policy_agent"}:
        raise AssertionError("Real policy agent must remain outside current delivery roles")
    if set(llm_role_taxonomy) != required_llm_roles:
        raise AssertionError("LLM role taxonomy must explicitly separate event normalization, decision context, candidate ranking, and real policy agent")
    for role_name, role_type in required_llm_role_types.items():
        role_payload = llm_role_taxonomy.get(role_name) or {}
        if role_payload.get("role_type") != role_type:
            raise AssertionError(f"LLM role type mismatch for {role_name}")
        if not role_payload.get("input_contract"):
            raise AssertionError(f"LLM role input contract missing for {role_name}")
        if not role_payload.get("output_contract"):
            raise AssertionError(f"LLM role output contract missing for {role_name}")
        if "may_select_final_poker_action" not in role_payload:
            raise AssertionError(f"LLM policy-action capability missing for {role_name}")
    if (llm_role_taxonomy.get("event_normalization") or {}).get("status") != "CONTROLLED_COMPONENT":
        raise AssertionError("LLM event-normalization role must remain a controlled component")
    if (llm_role_taxonomy.get("event_normalization") or {}).get("can_emit_policy_action") is not False:
        raise AssertionError("LLM event-normalization role must not be a policy-action emitter")
    if (llm_role_taxonomy.get("event_normalization") or {}).get("may_select_final_poker_action") is not False:
        raise AssertionError("LLM event-normalization role must not select final poker actions")
    if (llm_role_taxonomy.get("decision_context") or {}).get("status") != "CONTROLLED_COMPONENT":
        raise AssertionError("LLM decision-context role must remain a controlled component")
    if (llm_role_taxonomy.get("decision_context") or {}).get("may_select_final_poker_action") is not False:
        raise AssertionError("LLM decision-context role must not be marked as the final poker-action selector")
    if (llm_role_taxonomy.get("candidate_ranking") or {}).get("status") != "RESEARCH_BASELINE_COMPONENT":
        raise AssertionError("LLM candidate-ranking role must remain a research baseline component")
    if (llm_role_taxonomy.get("candidate_ranking") or {}).get("may_rank_candidates") is not True:
        raise AssertionError("LLM candidate-ranking role must explicitly rank candidates")
    if (llm_role_taxonomy.get("candidate_ranking") or {}).get("may_select_final_poker_action") is not False:
        raise AssertionError("LLM candidate-ranking role must not select the final poker policy action")
    if (llm_role_taxonomy.get("candidate_ranking") or {}).get("production_policy_approved") is not False:
        raise AssertionError("LLM candidate ranking must not be marked production-approved as a policy")
    if (llm_role_taxonomy.get("real_policy_agent") or {}).get("status") != "NOT_CURRENT_DELIVERY_SCOPE":
        raise AssertionError("Real LLM policy agent must remain outside current delivery scope")
    if (llm_role_taxonomy.get("real_policy_agent") or {}).get("role_type") != "POLICY_AGENT":
        raise AssertionError("Real LLM policy agent must be the only role typed as POLICY_AGENT")
    if (llm_role_taxonomy.get("real_policy_agent") or {}).get("may_select_final_poker_action") is not True:
        raise AssertionError("Real LLM policy agent definition must be the only final-action role")
    if (llm_role_taxonomy.get("real_policy_agent") or {}).get("implemented") is not False:
        raise AssertionError("Real LLM policy agent must not be marked implemented")
    if (llm_role_taxonomy.get("real_policy_agent") or {}).get("production_policy_approved") is not False:
        raise AssertionError("Real LLM policy agent must not be marked production-approved")
    if llm_event_layer.get("implemented") is not True:
        raise AssertionError("LLM role boundary must preserve event-normalization layer evidence")
    if llm_context_layer.get("implemented") is not True:
        raise AssertionError("LLM role boundary must preserve decision-context layer evidence")
    if llm_role.get("llm_decision_path_production_approved") is not False:
        raise AssertionError("LLM decision path must not be production-approved as the deployed poker policy")
    if llm_autonomous_boundary.get("status") != "NOT_FULLY_AUTONOMOUS_POKER_PLAYING_LLM_AGENT":
        raise AssertionError("LLM boundary must explicitly reject autonomous LLM-agent status")
    if llm_autonomous_boundary.get("fully_autonomous_poker_playing_llm_agent_present") is not False:
        raise AssertionError("Fully autonomous poker-playing LLM agent must not be marked present")
    if llm_autonomous_boundary.get("fully_autonomous_llm_agent_claim_allowed") is not False:
        raise AssertionError("Fully autonomous LLM-agent claim must remain blocked")
    if llm_autonomous_boundary.get("deployed_autonomous_endpoint_is_llm") is not False:
        raise AssertionError("Controlled stateful autonomous endpoint must not be labeled as an LLM agent")
    if llm_autonomous_boundary.get("production_blocker_for_current_delivery") is not False:
        raise AssertionError("LLM role boundary must not block current delivery")
    if (llm_role_boundary.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"LLM role boundary invariants failed: {llm_role_boundary.get('invariants')}")
    for required_case in (
        "base_contract_is_valid",
        "blocks_llm_based_agent_as_autonomous_policy",
        "blocks_ambiguous_llm_agent_scope",
        "blocks_event_normalization_as_policy_agent",
        "blocks_candidate_ranking_as_deployed_policy",
        "blocks_real_policy_agent_current_scope_claim",
        "blocks_autonomous_policy_under_controlled_layer_acceptance",
        "blocks_missing_role_taxonomy",
    ):
        if (llm_proof_cases.get(required_case) or {}).get("passed") is not True:
            raise AssertionError(f"LLM role proof case did not pass: {required_case}")
    if (llm_proof_cases.get("base_contract_is_valid") or {}).get("observed_status") != "PASS":
        raise AssertionError("LLM role base proof case must observe PASS")
    for blocked_case in (
        "blocks_llm_based_agent_as_autonomous_policy",
        "blocks_ambiguous_llm_agent_scope",
        "blocks_event_normalization_as_policy_agent",
        "blocks_candidate_ranking_as_deployed_policy",
        "blocks_real_policy_agent_current_scope_claim",
        "blocks_missing_role_taxonomy",
    ):
        if (llm_proof_cases.get(blocked_case) or {}).get("observed_status") != "FAIL":
            raise AssertionError(f"LLM role blocked proof case must observe FAIL: {blocked_case}")
    for required_claim_case, expected_status in {
        "blocks_unqualified_llm_based_agent_production_claim": "FAIL",
        "blocks_event_normalizer_as_policy_agent": "FAIL",
        "allows_decision_context_research_claim": "PASS",
        "blocks_candidate_ranker_as_deployed_policy": "FAIL",
        "blocks_real_policy_agent_current_delivery_claim": "FAIL",
    }.items():
        claim_case = llm_claim_cases.get(required_claim_case) or {}
        if claim_case.get("passed") is not True:
            raise AssertionError(f"LLM claim validation case did not pass: {required_claim_case}")
        if claim_case.get("observed_status") != expected_status:
            raise AssertionError(f"LLM claim validation case observed wrong status: {required_claim_case}")
    for required_scope_case, expected_status in {
        "allows_controlled_event_context_layer_production_claim": "PASS",
        "blocks_autonomous_llm_policy_production_claim": "FAIL",
        "blocks_unqualified_llm_policy_claim_text": "FAIL",
    }.items():
        scope_case = llm_production_scope_cases.get(required_scope_case) or {}
        if scope_case.get("passed") is not True:
            raise AssertionError(f"LLM production-scope claim case did not pass: {required_scope_case}")
        if scope_case.get("observed_status") != expected_status:
            raise AssertionError(f"LLM production-scope claim observed wrong status: {required_scope_case}")

    experimental_guardrails = llm_policy_experimental.get("guardrails") or {}
    experimental_proof_cases = {case.get("name"): case for case in llm_policy_experimental.get("proof_cases") or []}
    if llm_policy_experimental.get("overall_status") != "PASS":
        raise AssertionError(
            f"Experimental LLM policy contract did not pass: {llm_policy_experimental.get('overall_status')}"
        )
    if llm_policy_experimental.get("status") != "EXPERIMENTAL_LLM_POLICY_RESEARCH_ONLY":
        raise AssertionError("Experimental LLM policy must remain research-only")
    if llm_policy_experimental.get("role_type") != "POLICY_AGENT":
        raise AssertionError("Experimental LLM policy must be explicitly typed as a policy adapter")
    if llm_policy_experimental.get("production_policy_approved") is not False:
        raise AssertionError("Experimental LLM policy must not be marked production-approved")
    if llm_policy_experimental.get("autonomous_policy_claim_allowed") is not False:
        raise AssertionError("Experimental LLM policy must not allow autonomous policy claims")
    if llm_policy_experimental.get("served_by_predict_endpoint") is not False:
        raise AssertionError("Experimental LLM policy must not be served by /predict")
    if llm_policy_experimental.get("deployed_strategy_stack_affected") is not False:
        raise AssertionError("Experimental LLM policy must not affect the deployed strategy stack")
    if llm_policy_experimental.get("current_delivery_blocker") is not False:
        raise AssertionError("Experimental LLM policy must not block the current delivery")
    if llm_policy_experimental.get("requires_stakeholder_approval_before_production") is not True:
        raise AssertionError("Experimental LLM policy must require stakeholder approval before production")
    for guardrail in (
        "formal_in_context_learning_required",
        "legal_action_filtering_required",
        "strict_json_output_required",
        "probability_normalization_required",
        "confidence_threshold_required",
        "deterministic_fallback_required",
    ):
        if experimental_guardrails.get(guardrail) is not True:
            raise AssertionError(f"Experimental LLM policy missing guardrail: {guardrail}")
    if experimental_guardrails.get("schema_bypass_allowed") is not False:
        raise AssertionError("Experimental LLM policy must not allow schema bypass")
    if experimental_guardrails.get("unconstrained_action_generation_allowed") is not False:
        raise AssertionError("Experimental LLM policy must not allow unconstrained action generation")
    for required_case in (
        "base_contract_is_valid",
        "blocks_production_approval_without_gates",
        "blocks_serving_by_public_predict_endpoint",
        "blocks_autonomous_unconstrained_policy_claim",
        "blocks_missing_approval_and_gate_contract",
    ):
        if (experimental_proof_cases.get(required_case) or {}).get("passed") is not True:
            raise AssertionError(f"Experimental LLM policy proof case did not pass: {required_case}")
    if (llm_policy_experimental.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"Experimental LLM policy invariants failed: {llm_policy_experimental.get('invariants')}"
        )

    qlora_boundary = qlora_next_stage.get("stage_boundary") or {}
    qlora_targets = qlora_next_stage.get("target_use_cases") or {}
    qlora_delivery = qlora_next_stage.get("delivery_classification") or {}
    qlora_plan = qlora_next_stage.get("recommended_training_plan") or {}
    if qlora_next_stage.get("overall_status") != "PASS":
        raise AssertionError(f"QLoRA next-stage boundary did not pass: {qlora_next_stage.get('overall_status')}")
    if qlora_boundary.get("stage_status") != "NEXT_STAGE_IMPROVEMENT":
        raise AssertionError("QLoRA/larger LLM fine-tuning must remain a next-stage improvement")
    if qlora_boundary.get("milestone_type") != "RESEARCH_QUALITY_IMPROVEMENT_MILESTONE":
        raise AssertionError("QLoRA/larger LLM fine-tuning must remain a research-quality improvement milestone")
    if qlora_boundary.get("fine_tuning_completed") is not False:
        raise AssertionError("QLoRA fine-tuning must not be marked completed in the current delivery")
    if qlora_boundary.get("production_approved") is not False:
        raise AssertionError("QLoRA fine-tuning must not be marked production-approved")
    if qlora_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("QLoRA next-stage improvement must not block current delivery")
    if qlora_boundary.get("delivery_blocker") is not False:
        raise AssertionError("QLoRA delivery blocker must remain false")
    if qlora_boundary.get("approved_current_delivery_component") is not False:
        raise AssertionError("QLoRA must not be marked as an approved current-delivery component")
    if qlora_boundary.get("requires_separate_approval_before_promotion") is not True:
        raise AssertionError("QLoRA promotion must require separate approval")
    if qlora_boundary.get("autonomous_llm_agent_claim_allowed") is not False:
        raise AssertionError("QLoRA plan must not allow autonomous LLM-agent claims")
    if qlora_delivery.get("next_stage_research_milestone") is not True:
        raise AssertionError("QLoRA delivery classification must remain next-stage research milestone")
    if qlora_delivery.get("current_delivery_component") is not False:
        raise AssertionError("QLoRA must not be classified as a current delivery component")
    if qlora_delivery.get("current_delivery_blocker") is not False:
        raise AssertionError("QLoRA delivery classification must not block current delivery")
    if qlora_plan.get("adapter_scope") != "EVENT_NORMALIZATION_STRUCTURED_EXTRACTION_AND_CANDIDATE_RANKING":
        raise AssertionError("QLoRA adapter scope must remain event-normalization/structured-extraction/candidate-ranking")
    for target in (
        "noisy_ocr_dealer_log_normalization",
        "structured_extraction",
        "candidate_ranking",
        "noisy_ocr_dealer_log_handling",
        "json_schema_compliance_improvement",
    ):
        if (qlora_targets.get(target) or {}).get("recommended") is not True:
            raise AssertionError(f"QLoRA target not recommended: {target}")
    if (qlora_targets.get("autonomous_poker_policy") or {}).get("recommended") is not False:
        raise AssertionError("QLoRA must not target autonomous poker policy as the immediate production path")
    if (qlora_next_stage.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"QLoRA next-stage invariants failed: {qlora_next_stage.get('invariants')}")

    runtime_boundary = production_runtime_monitoring.get("runtime_observability_boundary") or {}
    if production_runtime_monitoring.get("overall_status") != "PASS":
        raise AssertionError(f"Production monitoring contract did not pass: {production_runtime_monitoring.get('overall_status')}")
    if runtime_boundary.get("monitoring_required_for_real_traffic") is not True:
        raise AssertionError("Monitoring must be required for real-traffic rollout")
    if runtime_boundary.get("rollback_rules_required_for_real_traffic") is not True:
        raise AssertionError("Rollback rules must be required for real-traffic rollout")
    if runtime_boundary.get("live_drift_tracking_required_for_real_traffic") is not True:
        raise AssertionError("Live drift tracking must be required for real-traffic rollout")
    if runtime_boundary.get("prediction_distribution_tracking_required_for_real_traffic") is not True:
        raise AssertionError("Prediction distribution tracking must be required for real-traffic rollout")
    if runtime_boundary.get("model_confidence_monitoring_required_for_real_traffic") is not True:
        raise AssertionError("Model confidence monitoring must be required for real-traffic rollout")
    if runtime_boundary.get("real_traffic_claim_allowed_without_observability") is not False:
        raise AssertionError("Unmonitored real-traffic production claim must be blocked")
    if runtime_boundary.get("real_production_traffic_approved") is not False:
        raise AssertionError("Real production traffic must not be approved before observability is enabled")
    if runtime_boundary.get("real_production_traffic_approval_status") != "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED":
        raise AssertionError("Real production traffic approval status must require enabled observability")
    if runtime_boundary.get("real_traffic_blocker_if_disabled") is not True:
        raise AssertionError("Disabled observability must block real-traffic rollout")
    if runtime_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("Monitoring contract must not block the current delivery package")
    if (production_runtime_monitoring.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Production monitoring invariants failed: {production_runtime_monitoring.get('invariants')}")

    final_strategy_delivery = final_strategy_quality_status.get("delivery_boundary") or {}
    final_strategy_deployment_vs_competitive = (
        final_strategy_quality_status.get("deployment_vs_competitive_claim_boundary") or {}
    )
    final_strategy_boundary = final_strategy_quality_status.get("final_strategy_quality_boundary") or {}
    final_strategy_remaining = final_strategy_quality_status.get("remaining_work") or {}
    final_strategy_real_traffic = final_strategy_quality_status.get("real_traffic_boundary") or {}
    if final_strategy_quality_status.get("overall_status") != "PASS":
        raise AssertionError(
            f"Final strategy quality status did not pass: {final_strategy_quality_status.get('overall_status')}"
        )
    if final_strategy_delivery.get("software_delivery_ready") is not True:
        raise AssertionError("Final strategy quality status must preserve software delivery readiness")
    if final_strategy_delivery.get("current_delivery_blocker") is not False:
        raise AssertionError("Final strategy quality hardening gap must not block current delivery")
    deployment_components = final_strategy_deployment_vs_competitive.get("deployment_sufficient_components") or {}
    for component in (
        "fastapi_service",
        "docker_packaging",
        "predict_endpoint",
        "health_endpoint",
        "reports_and_verifier",
    ):
        if deployment_components.get(component) is not True:
            raise AssertionError(f"Deployment component must be present for delivery review: {component}")
    if final_strategy_deployment_vs_competitive.get("deployment_delivery_ready") is not True:
        raise AssertionError("Deployment boundary must preserve software delivery readiness")
    if final_strategy_deployment_vs_competitive.get("deployment_claim_allowed") is not True:
        raise AssertionError("Deployment claim must be allowed for FastAPI/Docker service delivery")
    if final_strategy_deployment_vs_competitive.get("competitive_poker_agent_claim_allowed") is not False:
        raise AssertionError("Competitive poker-agent claim must remain blocked")
    if (
        final_strategy_deployment_vs_competitive.get("competitive_poker_agent_claim_state")
        != "BLOCKED_PENDING_MODEL_DATA_AND_TRAINING_HARDENING"
    ):
        raise AssertionError("Competitive poker-agent claim state must remain blocked pending hardening")
    if final_strategy_deployment_vs_competitive.get("current_delivery_blocker") is not False:
        raise AssertionError("Competitive strategy hardening gap must not block current delivery")
    if final_strategy_deployment_vs_competitive.get("deployed_strategy_stack_affected") is not False:
        raise AssertionError("Competitive strategy hardening gap must not affect the deployed stack approval")
    if final_strategy_boundary.get("status") != "NOT_APPROVED_PENDING_HARDENING_GATES":
        raise AssertionError("Final strategy quality must remain not approved pending hardening gates")
    if final_strategy_boundary.get("final_production_strategy_quality_approved") is not False:
        raise AssertionError("Final production-level strategy quality must not be approved")
    if final_strategy_boundary.get("final_production_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Final production-level strategy quality claim must remain blocked")
    required_final_strategy_items = {
        "stronger_challenger_model",
        "hole_card_data_quality",
        "calibration",
        "larger_validation_data",
        "production_scale_multi_agent_training",
    }
    if set(final_strategy_remaining) != required_final_strategy_items:
        raise AssertionError("Final strategy quality status must track every remaining hardening item")
    if (
        set(final_strategy_deployment_vs_competitive.get("required_before_competitive_claim") or [])
        != required_final_strategy_items
    ):
        raise AssertionError("Competitive claim must require every final strategy hardening item")
    for name in required_final_strategy_items:
        if (final_strategy_remaining.get(name) or {}).get("status") != "REQUIRED":
            raise AssertionError(f"Final strategy hardening item must remain required: {name}")
    if final_strategy_real_traffic.get("real_production_traffic_approved") is not False:
        raise AssertionError("Final strategy quality status must not approve real production traffic")
    if (final_strategy_quality_status.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(
            f"Final strategy quality status invariants failed: {final_strategy_quality_status.get('invariants')}"
        )

    final_acceptance_summary = final_delivery_acceptance.get("acceptance_summary") or {}
    final_acceptance_risks = final_delivery_acceptance.get("tracked_component_risks") or {}
    final_raw = final_acceptance_risks.get("raw_supervised_model") or {}
    final_llm = final_acceptance_risks.get("llm_work") or {}
    final_qlora = final_acceptance_risks.get("qlora_larger_llm_fine_tuning") or {}
    final_runtime_monitoring = final_acceptance_risks.get("production_runtime_monitoring") or {}
    final_challenger = final_acceptance_risks.get("challenger_strategy_quality") or {}
    final_hole = final_acceptance_risks.get("hole_card_data_quality") or {}
    final_normalized_action = final_acceptance_risks.get("normalized_action_contract") or {}
    final_actions_context = final_acceptance_risks.get("actions_context_quality") or {}
    final_stack_context = final_acceptance_risks.get("stack_event_context_quality") or {}
    final_bet = final_acceptance_risks.get("bet_timing_calibration") or {}
    final_behavioral = final_acceptance_risks.get("behavioral_revalidation") or {}
    final_multi = final_acceptance_risks.get("multi_agent_training") or {}
    final_phase2_selection = final_acceptance_risks.get("phase2_selection_comparison") or {}
    final_human_likeness_claim_gate = final_acceptance_risks.get("human_likeness_claim_gate") or {}
    final_delivery_strategy_boundary = final_delivery_acceptance.get("delivery_strategy_quality_boundary") or {}
    if final_delivery_acceptance.get("overall_status") != "PASS":
        raise AssertionError(f"Final delivery acceptance did not pass: {final_delivery_acceptance.get('overall_status')}")
    if final_delivery_acceptance.get("final_status") != "READY_WITH_TRACKED_COMPONENT_RISKS":
        raise AssertionError("Final delivery acceptance must preserve tracked component-risk status")
    if final_acceptance_summary.get("service_delivery") != "READY":
        raise AssertionError("Final acceptance must mark service delivery as ready")
    if final_acceptance_summary.get("deployed_strategy_stack") != "APPROVED":
        raise AssertionError("Final acceptance must preserve deployed strategy-stack approval")
    if (
        final_delivery_strategy_boundary.get("boundary")
        != "DEPLOYMENT_READY_IS_NOT_STRATEGY_APPROVED"
    ):
        raise AssertionError("Final acceptance must expose the delivery/strategy-quality boundary")
    if (final_delivery_strategy_boundary.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError("Delivery/strategy-quality boundary invariants must pass")
    if final_delivery_strategy_boundary.get("software_delivery_ready") is not True:
        raise AssertionError("Delivery/strategy boundary must keep software delivery ready")
    if final_delivery_strategy_boundary.get("deployment_ready") is not True:
        raise AssertionError("Delivery/strategy boundary must mark deployment ready")
    if final_delivery_strategy_boundary.get("deployment_ready_does_not_imply_strategy_approved") is not True:
        raise AssertionError("Deployment-ready must not imply strategy-approved")
    if final_delivery_strategy_boundary.get("strategy_approved") is not False:
        raise AssertionError("Deployment-ready boundary must not mark final strategy as approved")
    if final_delivery_strategy_boundary.get("competitive_poker_agent_claim_allowed") is not False:
        raise AssertionError("Competitive poker-agent claim must remain blocked")
    if (
        final_delivery_strategy_boundary.get("competitive_poker_agent_claim_state")
        != "BLOCKED_PENDING_MODEL_DATA_CALIBRATION_AND_MULTI_AGENT_TRAINING"
    ):
        raise AssertionError("Competitive poker-agent claim state must remain blocked pending hardening")
    expected_strategy_gates = {
        "cleaner_real_gameplay_data",
        "stronger_challenger_model",
        "calibration_gate",
        "full_multi_agent_training",
        "full_metric_bundle",
    }
    if set(final_delivery_strategy_boundary.get("required_strategy_approval_gates") or []) != expected_strategy_gates:
        raise AssertionError("Deployment/strategy boundary must require every strategy approval hardening gate")
    if set((final_delivery_strategy_boundary.get("required_before_competitive_claim") or {}).keys()) != expected_strategy_gates:
        raise AssertionError("Deployment/strategy boundary must describe all work required before competitive claims")
    approval_separation = final_delivery_strategy_boundary.get("approval_separation") or {}
    if approval_separation.get("deployment_ready_can_pass_without_strategy_approval") is not True:
        raise AssertionError("Deployment-ready must be allowed to pass without final strategy approval")
    if approval_separation.get("fastapi_docker_predict_are_delivery_evidence_only") is not True:
        raise AssertionError("FastAPI, Docker, and /predict must remain delivery evidence only")
    if approval_separation.get("competitive_claim_requires_model_data_calibration_and_training") is not True:
        raise AssertionError("Competitive claims must require model, data, calibration, and training hardening")
    if final_delivery_strategy_boundary.get("current_delivery_blocker") is not False:
        raise AssertionError("Final strategy metric gap must not become a delivery blocker")
    if final_delivery_strategy_boundary.get("final_metric_bundle_passed") is not False:
        raise AssertionError("Final metric bundle must remain blocked until every required metric passes")
    if final_delivery_strategy_boundary.get("final_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Final strategy-quality claim must remain blocked until the full metric bundle passes")
    if final_delivery_strategy_boundary.get("model_quality_risk") is not True:
        raise AssertionError("Blocked final strategy-quality claim must remain a model-quality risk")
    if final_raw.get("standalone_status") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Final acceptance must not approve the raw supervised model as standalone")
    if final_raw.get("component_risk") is not True or final_raw.get("production_blocker") is not False:
        raise AssertionError("Final acceptance must keep raw model as component risk, not production blocker")
    if final_llm.get("role") != "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER":
        raise AssertionError("Final acceptance must preserve controlled LLM role")
    if final_llm.get("fully_autonomous_llm_agent_claim_allowed") is not False:
        raise AssertionError("Final acceptance must block fully autonomous LLM-agent claims")
    if final_qlora.get("stage_status") != "NEXT_STAGE_IMPROVEMENT":
        raise AssertionError("Final acceptance must keep QLoRA/larger LLM fine-tuning as next-stage improvement")
    if final_qlora.get("milestone_type") != "RESEARCH_QUALITY_IMPROVEMENT_MILESTONE":
        raise AssertionError("Final acceptance must keep QLoRA as a research-quality improvement milestone")
    if final_qlora.get("fine_tuning_completed") is not False:
        raise AssertionError("Final acceptance must not mark QLoRA fine-tuning complete")
    if final_qlora.get("production_approved") is not False:
        raise AssertionError("Final acceptance must not approve QLoRA fine-tuning as production-ready")
    if final_qlora.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make QLoRA a current delivery blocker")
    if final_qlora.get("delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make QLoRA a delivery blocker")
    if final_qlora.get("approved_current_delivery_component") is not False:
        raise AssertionError("Final acceptance must not mark QLoRA as approved current-delivery component")
    if final_qlora.get("adapter_scope") != "EVENT_NORMALIZATION_STRUCTURED_EXTRACTION_AND_CANDIDATE_RANKING":
        raise AssertionError("Final acceptance must preserve QLoRA adapter scope")
    final_qlora_targets = final_qlora.get("targets") or {}
    for target in (
        "noisy_ocr_dealer_log_normalization",
        "structured_extraction",
        "candidate_ranking",
        "json_schema_compliance_improvement",
    ):
        if final_qlora_targets.get(target) is not True:
            raise AssertionError(f"Final acceptance must preserve QLoRA target: {target}")
    if final_qlora_targets.get("autonomous_poker_policy") is not False:
        raise AssertionError("Final acceptance must keep autonomous poker policy out of QLoRA target scope")
    if final_runtime_monitoring.get("monitoring_required_for_real_traffic") is not True:
        raise AssertionError("Final acceptance must require monitoring for real traffic")
    if final_runtime_monitoring.get("rollback_rules_required_for_real_traffic") is not True:
        raise AssertionError("Final acceptance must require rollback rules for real traffic")
    if final_runtime_monitoring.get("live_drift_tracking_required_for_real_traffic") is not True:
        raise AssertionError("Final acceptance must require live drift tracking for real traffic")
    if final_runtime_monitoring.get("prediction_distribution_tracking_required_for_real_traffic") is not True:
        raise AssertionError("Final acceptance must require prediction distribution tracking for real traffic")
    if final_runtime_monitoring.get("model_confidence_monitoring_required_for_real_traffic") is not True:
        raise AssertionError("Final acceptance must require model confidence monitoring for real traffic")
    if final_runtime_monitoring.get("real_traffic_claim_allowed_without_observability") is not False:
        raise AssertionError("Final acceptance must block unmonitored real-traffic claims")
    if final_runtime_monitoring.get("real_production_traffic_approved") is not False:
        raise AssertionError("Final acceptance must not approve real production traffic before observability is enabled")
    if final_runtime_monitoring.get("real_production_traffic_approval_status") != "NOT_APPROVED_UNTIL_OBSERVABILITY_ENABLED":
        raise AssertionError("Final acceptance must keep real production traffic approval pending observability")
    if final_runtime_monitoring.get("real_traffic_blocker_if_disabled") is not True:
        raise AssertionError("Final acceptance must block real traffic when observability is disabled")
    if final_runtime_monitoring.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make monitoring a current delivery blocker")
    if final_challenger.get("challenger_required_before_final_claim") is not True:
        raise AssertionError("Final acceptance must require a challenger before final strategy-quality claims")
    if final_challenger.get("challenger_compared_to_raw_model") is not True:
        raise AssertionError("Final acceptance must preserve challenger comparison evidence")
    if final_challenger.get("final_production_strategy_quality_claim_allowed") is not False:
        raise AssertionError("Final acceptance must block final strategy-quality claims until challenger gates pass")
    if final_challenger.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make challenger strategy-quality gap a delivery blocker")
    if final_challenger.get("deployed_strategy_stack_affected") is not False:
        raise AssertionError("Final acceptance must not let challenger gap affect deployed stack approval")
    if final_hole.get("upstream_resolved") is not False:
        raise AssertionError("Final acceptance must not mark upstream hole-card quality as resolved")
    if final_normalized_action.get("boundary") != "RAW_OCR_ACTIONS_REQUIRE_CANONICAL_LABEL_NORMALIZATION":
        raise AssertionError("Final acceptance must expose the normalized-action boundary")
    if final_normalized_action.get("normalized_action_status") != "IMPLEMENTED":
        raise AssertionError("Final acceptance must keep normalized action contract implemented")
    if final_normalized_action.get("raw_action_source_status") != "RAW_OCR_OR_DEALER_TEXT":
        raise AssertionError("Final acceptance must declare raw action source as OCR/dealer text")
    if set(final_normalized_action.get("canonical_actions") or []) != {
        "fold",
        "call",
        "check",
        "bet",
        "raise",
        "all_in",
    }:
        raise AssertionError("Final acceptance must preserve the full canonical action set")
    if final_normalized_action.get("raw_ocr_action_must_not_be_training_label") is not True:
        raise AssertionError("Final acceptance must block raw OCR action labels")
    if final_normalized_action.get("normalization_required_before_training") is not True:
        raise AssertionError("Final acceptance must require action normalization before training")
    if final_normalized_action.get("normalization_required_before_evaluation") is not True:
        raise AssertionError("Final acceptance must require action normalization before evaluation")
    if final_normalized_action.get("normalization_required_before_policy_comparison") is not True:
        raise AssertionError("Final acceptance must require action normalization before policy comparison")
    if final_normalized_action.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make normalized action contract a delivery blocker")
    if final_normalized_action.get("model_quality_risk") is not False:
        raise AssertionError("Final acceptance must not keep normalized action as an open model-quality risk")
    if final_normalized_action.get("training_label_status") != "PASS":
        raise AssertionError("Final acceptance must expose canonical training labels")
    if final_normalized_action.get("invalid_training_labels"):
        raise AssertionError("Final acceptance must not expose raw OCR labels in training labels")
    if final_actions_context.get("explicit_context_status") != "INCOMPLETE_EXPLICIT_BETTING_CONTEXT":
        raise AssertionError("Final acceptance must keep actions.csv explicit context marked incomplete")
    final_missing_actions_context = set(final_actions_context.get("missing_explicit_context_fields") or [])
    for required_field in (
        "amount",
        "to_call",
        "pot_before_action",
        "min_raise",
        "legal_actions",
        "action_order",
        "last_aggressor",
        "facing_bet",
    ):
        if required_field not in final_missing_actions_context:
            raise AssertionError(f"Final acceptance must expose missing actions.csv context field: {required_field}")
    if final_actions_context.get("future_dataset_explicit_export_required") is not True:
        raise AssertionError("Final acceptance must require explicit actions.csv context in the next dataset export")
    if set(final_actions_context.get("future_dataset_required_explicit_fields") or []) != final_missing_actions_context:
        raise AssertionError("Final acceptance future actions.csv export fields must match missing explicit context fields")
    if final_actions_context.get("reconstructed_context_allowed_for_current_delivery") is not True:
        raise AssertionError("Final acceptance must allow reconstructed actions.csv context for current delivery")
    if final_actions_context.get("does_not_fully_replace_explicit_context") is not True:
        raise AssertionError("Final acceptance must not claim derived context fully replaces explicit action-context labels")
    if final_actions_context.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make actions.csv context limitation a delivery blocker")
    if final_actions_context.get("model_quality_risk") is not True:
        raise AssertionError("Final acceptance must keep actions.csv context limitation as model-quality risk")
    if final_stack_context.get("raw_stack_event_status") != "RAW_EVENTS_REQUIRE_DECISION_CONTEXT_DERIVATION":
        raise AssertionError("Final acceptance must keep raw stack events marked as requiring decision-context derivation")
    if final_stack_context.get("raw_stack_events_are_direct_policy_features") is not False:
        raise AssertionError("Final acceptance must not mark raw stack events as direct policy features")
    if final_stack_context.get("decision_time_derivation_required") is not True:
        raise AssertionError("Final acceptance must require decision-time derivation for stack events")
    if final_stack_context.get("target_action_stack_delta_allowed_as_feature") is not False:
        raise AssertionError("Final acceptance must block target action stack deltas as features")
    if final_stack_context.get("post_hand_stack_outcome_allowed_as_feature") is not False:
        raise AssertionError("Final acceptance must block post-hand stack outcomes as features")
    if final_stack_context.get("derived_context_status") != "IMPLEMENTED_FROM_PRE_ACTION_STACK_DELTAS":
        raise AssertionError("Final acceptance must keep derived stack context implemented")
    if final_stack_context.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make stack-event context gap a delivery blocker")
    if final_stack_context.get("model_quality_risk") is not True:
        raise AssertionError("Final acceptance must keep stack-event context gap as model-quality risk")
    if final_bet.get("final_high_realism_claim_allowed") is not False:
        raise AssertionError("Final acceptance must block final high-realism bet/timing claims")
    if final_behavioral.get("larger_clean_real_gameplay_revalidation_required") is not True:
        raise AssertionError("Final acceptance must require larger clean gameplay revalidation")
    if final_multi.get("full_production_scale_multi_agent_training_status") != "NOT_COMPLETED":
        raise AssertionError("Final acceptance must not mark full production-scale multi-agent training complete")
    if final_phase2_selection.get("status") != "STRICT_SELECTION_GATE_IMPLEMENTED":
        raise AssertionError("Final acceptance must expose the strict Phase 2 selection comparison")
    if final_phase2_selection.get("selected_for_current_delivery") != "routed_policy_bundle":
        raise AssertionError("Final acceptance must keep routed_policy_bundle as the current delivery architecture")
    if final_phase2_selection.get("final_selection_claim_allowed") is not False:
        raise AssertionError("Final acceptance must block Phase 2 final selection until common-condition comparison")
    if final_phase2_selection.get("all_candidates_compared_on_common_holdout") is not False:
        raise AssertionError("Final acceptance must not mark Phase 2 common holdout complete")
    if final_phase2_selection.get("all_candidates_compared_in_common_simulation") is not False:
        raise AssertionError("Final acceptance must not mark Phase 2 common simulation complete")
    if not final_phase2_selection.get("missing_common_holdout_candidates"):
        raise AssertionError("Final acceptance must list missing Phase 2 common holdout candidates")
    if not final_phase2_selection.get("missing_common_simulation_candidates"):
        raise AssertionError("Final acceptance must list missing Phase 2 common simulation candidates")
    if final_phase2_selection.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make Phase 2 common-condition gap a delivery blocker")
    if final_phase2_selection.get("model_quality_risk") is not True:
        raise AssertionError("Final acceptance must keep Phase 2 common-condition gap as model-quality risk")
    if final_human_likeness_claim_gate.get("claim") != "FULL_HUMAN_LIKENESS":
        raise AssertionError("Final acceptance must expose the full human-likeness claim gate")
    if final_human_likeness_claim_gate.get("decision") != "BLOCKED":
        raise AssertionError("Final acceptance must keep the full human-likeness claim blocked")
    if final_human_likeness_claim_gate.get("claim_allowed") is not False:
        raise AssertionError("Final acceptance must not allow the full human-likeness claim")
    if final_human_likeness_claim_gate.get("action_distribution_only_proof_rejected") is not True:
        raise AssertionError("Final acceptance must reject action-distribution-only human-likeness proof")
    if final_human_likeness_claim_gate.get("current_delivery_blocker") is not False:
        raise AssertionError("Final acceptance must not make the human-likeness claim gate a delivery blocker")
    if final_human_likeness_claim_gate.get("model_quality_risk") is not True:
        raise AssertionError("Final acceptance must keep the human-likeness claim gap as model-quality risk")
    if (final_delivery_acceptance.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Final delivery acceptance invariants failed: {final_delivery_acceptance.get('invariants')}")
    return (
        f"delivery={delivery.get('overall_status')}, deployed={deployed.get('strategy_policy_status')}, "
        f"raw_gate={gate.get('status')}, handoff={handoff_payload.get('handoff_status')}, "
        f"component_risks={risk_payload.get('risk_summary', {}).get('component_risks')}, "
        f"gold_examples={gold_payload.get('examples')}"
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
    required_configs = [
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/training/group_holdout.yaml",
        "configs/evaluation/standard.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_decision_context.yaml",
        "configs/experiments/llm_decision_context_smoke.yaml",
        "configs/experiments/llm_decision_context_qwen25.yaml",
        "configs/experiments/build_decision_context_holdout.yaml",
        "configs/experiments/llm_decision_gate.yaml",
        "configs/experiments/llm_decision_candidate_ranker_qwen25.yaml",
        "configs/experiments/llm_decision_candidate_gate.yaml",
        "configs/experiments/llm_architecture_comparison.yaml",
        "configs/experiments/phase2_selection_comparison.yaml",
        "configs/experiments/llm_policy_experimental.yaml",
        "configs/experiments/project_completion.yaml",
        "configs/experiments/training_cluster_requirements.yaml",
        "configs/experiments/today_acceptance_training.yaml",
        "configs/experiments/client_gpu_training_response.yaml",
        "configs/experiments/multi_agent_training_status.yaml",
        "configs/experiments/phase3_open_spiel_arena.yaml",
        "configs/experiments/phase3_open_spiel_claim.yaml",
        "configs/experiments/open_spiel_claim_readiness.yaml",
        "configs/experiments/open_spiel_claim_contract.yaml",
        "configs/experiments/rl_delivery_boundary.yaml",
        "configs/experiments/evaluation_metric_contract.yaml",
        "configs/experiments/test_execution_contract.yaml",
        "configs/experiments/strategy_stack_maturity.yaml",
        "configs/experiments/behavioral_revalidation.yaml",
        "configs/experiments/behavioral_revalidation_proof.yaml",
        "configs/experiments/human_likeness_evidence.yaml",
        "configs/experiments/human_likeness_claim_gate.yaml",
        "configs/experiments/hole_card_data_quality.yaml",
        "configs/experiments/data_leakage_contract.yaml",
        "configs/experiments/normalized_action_contract.yaml",
        "configs/experiments/actions_context_quality.yaml",
        "configs/experiments/actions_dataset_export_contract.yaml",
        "configs/experiments/stack_event_context_quality.yaml",
        "configs/experiments/scenario_sanity.yaml",
        "configs/experiments/production_gate.yaml",
        "configs/experiments/challenger_strategy_quality.yaml",
        "configs/experiments/final_strategy_quality_status.yaml",
        "configs/experiments/verify_delivery.yaml",
    ]
    missing = [relative for relative in required_configs if not (root / relative).exists()]
    if missing:
        raise AssertionError(f"Hydra configuration hierarchy is incomplete: {missing}")
    return f"hydra_configs={len(required_configs)}"


def zip_contract(root: Path, zip_path: Path) -> str:
    required = {
        "models/poker_policy.joblib",
        "models/poker_policy_bundle.joblib",
        "activate_env.cmd",
        "install.ps1",
        "run_server.ps1",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/model/text_event_smol.yaml",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/prompts/poker_decision_minimal_zero_shot.txt",
        "configs/prompts/poker_decision_rules_grounded.txt",
        "configs/prompts/poker_decision_full_context.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_decision_context.yaml",
        "configs/experiments/llm_decision_context_smoke.yaml",
        "configs/experiments/llm_decision_context_qwen25.yaml",
        "configs/experiments/build_decision_context_holdout.yaml",
        "configs/experiments/llm_decision_gate.yaml",
        "configs/experiments/llm_decision_candidate_ranker_qwen25.yaml",
        "configs/experiments/llm_decision_candidate_gate.yaml",
        "configs/experiments/llm_architecture_comparison.yaml",
        "configs/experiments/phase2_selection_comparison.yaml",
        "configs/experiments/llm_policy_experimental.yaml",
        "configs/experiments/challenger_strategy_quality.yaml",
        "configs/experiments/final_strategy_quality_status.yaml",
        "configs/experiments/phase3_open_spiel_arena.yaml",
        "configs/experiments/phase3_open_spiel_claim.yaml",
        "configs/experiments/open_spiel_claim_readiness.yaml",
        "configs/experiments/open_spiel_claim_contract.yaml",
        "configs/experiments/rl_delivery_boundary.yaml",
        "configs/experiments/evaluation_metric_contract.yaml",
        "configs/experiments/test_execution_contract.yaml",
        "configs/experiments/human_likeness_evidence.yaml",
        "configs/experiments/human_likeness_claim_gate.yaml",
        "configs/experiments/data_leakage_contract.yaml",
        "configs/experiments/normalized_action_contract.yaml",
        "configs/experiments/actions_context_quality.yaml",
        "configs/experiments/actions_dataset_export_contract.yaml",
        "configs/experiments/stack_event_context_quality.yaml",
        "evaluation/decision_context_smoke.jsonl",
        "evaluation/decision_context_human_holdout.jsonl",
        "configs/experiments/project_completion.yaml",
        "evaluation/event_extraction_gold.jsonl",
        "reports/production_gate.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_eval.md",
        "reports/llm_decision_context.json",
        "reports/llm_decision_context.md",
        "reports/llm_decision_context_smoke.json",
        "reports/llm_decision_context_smoke_predictions.jsonl",
        "reports/llm_decision_context_smoke.md",
        "reports/decision_context_holdout.json",
        "reports/llm_decision_context_qwen25.json",
        "reports/llm_decision_context_qwen25_predictions.jsonl",
        "reports/llm_decision_context_qwen25.md",
        "reports/llm_decision_gate.json",
        "reports/llm_decision_gate.md",
        "reports/llm_decision_candidate_ranker_qwen25.json",
        "reports/llm_decision_candidate_ranker_qwen25_predictions.jsonl",
        "reports/llm_decision_candidate_ranker_qwen25.md",
        "reports/llm_decision_candidate_gate.json",
        "reports/llm_decision_candidate_gate.md",
        "reports/llm_architecture_comparison.json",
        "reports/llm_architecture_comparison.md",
        "reports/phase2_selection_comparison.json",
        "reports/phase2_selection_comparison.md",
        "reports/llm_policy_experimental.json",
        "reports/llm_policy_experimental.md",
        "reports/policy_acceptance.json",
        "reports/production_self_play.json",
        "reports/deployed_strategy_gate.json",
        "reports/delivery_readiness.json",
        "reports/repo_hygiene.json",
        "reports/scope_contract.json",
        "reports/scope_contract.md",
        "reports/project_completion.json",
        "reports/project_completion.md",
        "reports/model_risk_register.json",
        "reports/model_risk_register.md",
        "reports/production_approval.json",
        "reports/production_approval.md",
        "reports/strategy_stack_maturity.json",
        "reports/strategy_stack_maturity.md",
        "reports/strategy_stack_maturity.json",
        "reports/strategy_stack_maturity.md",
        "reports/behavioral_revalidation.json",
        "reports/behavioral_revalidation.md",
        "reports/behavioral_revalidation_proof.json",
        "reports/behavioral_revalidation_proof.md",
        "reports/raw_model_status.json",
        "reports/raw_model_status.md",
        "reports/raw_model_challenger.json",
        "reports/raw_model_challenger.md",
        "reports/challenger_strategy_quality.json",
        "reports/challenger_strategy_quality.md",
        "reports/final_strategy_quality_status.json",
        "reports/final_strategy_quality_status.md",
        "reports/client_handoff.json",
        "reports/client_handoff.md",
        "reports/training_cluster_requirements.json",
        "reports/training_cluster_requirements.md",
        "reports/today_acceptance_training.json",
        "reports/today_acceptance_training.md",
        "reports/today_acceptance_production_gate.json",
        "reports/client_gpu_training_response.json",
        "reports/client_gpu_training_response.md",
        "reports/multi_agent_training_status.json",
        "reports/multi_agent_training_status.md",
        "reports/phase3_open_spiel_arena.json",
        "reports/phase3_open_spiel_arena.md",
        "reports/open_spiel_claim_readiness.json",
        "reports/open_spiel_claim_readiness.md",
        "reports/open_spiel_claim_contract.json",
        "reports/open_spiel_claim_contract.md",
        "reports/rl_delivery_boundary.json",
        "reports/rl_delivery_boundary.md",
        "reports/evaluation_metric_contract.json",
        "reports/evaluation_metric_contract.md",
        "reports/test_execution_contract.json",
        "reports/test_execution_contract.md",
        "reports/human_likeness_evidence.json",
        "reports/human_likeness_evidence.md",
        "reports/human_likeness_claim_gate.json",
        "reports/human_likeness_claim_gate.md",
        "reports/data_leakage_contract.json",
        "reports/data_leakage_contract.md",
        "reports/normalized_action_contract.json",
        "reports/normalized_action_contract.md",
        "reports/actions_context_quality.json",
        "reports/actions_context_quality.md",
        "reports/actions_dataset_export_contract.json",
        "reports/actions_dataset_export_contract.md",
        "reports/stack_event_context_quality.json",
        "reports/stack_event_context_quality.md",
        "reports/scenario_sanity.json",
        "reports/scenario_sanity.md",
        "poker_agent/autonomous_agent.py",
        "poker_agent/model_risk_register.py",
        "poker_agent/production_approval.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/behavioral_revalidation.py",
        "poker_agent/behavioral_revalidation_proof.py",
        "poker_agent/raw_model_status.py",
        "poker_agent/raw_model_challenger.py",
        "poker_agent/client_handoff.py",
        "poker_agent/training_cluster.py",
        "poker_agent/today_training.py",
        "poker_agent/client_gpu_training_response.py",
        "poker_agent/multi_agent_training_status.py",
        "poker_agent/rl_training_evidence_gate.py",
        "poker_agent/open_spiel_llm_arena.py",
        "poker_agent/open_spiel_claim_readiness.py",
        "poker_agent/open_spiel_claim_contract.py",
        "poker_agent/rl_delivery_boundary.py",
        "poker_agent/strategy_metric_gate.py",
        "poker_agent/evaluation_metric_contract.py",
        "poker_agent/test_execution_contract.py",
        "poker_agent/human_likeness_evidence.py",
        "poker_agent/human_likeness_claim_gate.py",
        "poker_agent/human_likeness_policy_guard.py",
        "poker_agent/data_leakage_contract.py",
        "poker_agent/action_normalization.py",
        "poker_agent/normalized_action_contract.py",
        "poker_agent/actions_context_quality.py",
        "poker_agent/actions_dataset_export_contract.py",
        "poker_agent/stack_event_context_quality.py",
        "poker_agent/stack_context.py",
        "poker_agent/scenario_sanity.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/approval_boundary.py",
        "poker_agent/llm_decision_context.py",
        "poker_agent/llm_decision_benchmark.py",
        "poker_agent/llm_decision_gate.py",
        "poker_agent/llm_architecture_comparison.py",
        "poker_agent/phase2_selection_comparison.py",
        "poker_agent/llm_policy_experimental.py",
        "poker_agent/project_completion.py",
        "poker_agent/final_delivery_acceptance.py",
        "poker_agent/final_strategy_quality_status.py",
        "poker_agent/api_contract.py",
        "poker_agent/delivery_readiness.py",
        "poker_agent/scope_contract.py",
        "scripts/check_repo_hygiene.py",
        "scripts/audit_repository.py",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_behavioral_revalidation.py",
        "scripts/build_behavioral_revalidation_proof.py",
        "scripts/build_raw_model_status.py",
        "scripts/train_raw_model_challenger.py",
        "scripts/build_client_handoff.py",
        "scripts/build_training_cluster_requirements.py",
        "scripts/run_today_acceptance_training.py",
        "scripts/build_client_gpu_training_response.py",
        "scripts/build_multi_agent_training_status.py",
        "scripts/build_phase3_open_spiel_arena.py",
        "scripts/build_open_spiel_claim_readiness.py",
        "scripts/build_open_spiel_claim_contract.py",
        "scripts/build_rl_delivery_boundary.py",
        "scripts/build_evaluation_metric_contract.py",
        "scripts/build_test_execution_contract.py",
        "scripts/build_human_likeness_evidence.py",
        "scripts/build_human_likeness_claim_gate.py",
        "scripts/build_data_leakage_contract.py",
        "scripts/build_normalized_action_contract.py",
        "scripts/build_actions_context_quality.py",
        "scripts/build_actions_dataset_export_contract.py",
        "scripts/build_stack_event_context_quality.py",
        "scripts/build_scenario_sanity.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_llm_decision_context.py",
        "scripts/llm_decision_context_eval.py",
        "scripts/build_decision_context_holdout.py",
        "scripts/build_llm_decision_gate.py",
        "scripts/build_llm_architecture_comparison.py",
        "scripts/build_phase2_selection_comparison.py",
        "scripts/build_llm_policy_experimental.py",
        "scripts/build_project_completion.py",
        "scripts/build_final_delivery_acceptance.py",
        "scripts/build_final_strategy_quality_status.py",
        "scripts/build_scope_contract.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
        "tests/test_autonomous_agent.py",
        "tests/test_training_cluster.py",
        "tests/test_today_acceptance_training.py",
        "tests/test_client_gpu_training_response.py",
        "tests/test_multi_agent_training_status.py",
        "tests/test_rl_training_evidence_gate.py",
        "tests/test_open_spiel_llm_arena.py",
        "tests/test_open_spiel_claim_readiness.py",
        "tests/test_open_spiel_claim_contract.py",
        "tests/test_open_spiel_claim_mode.py",
        "tests/test_open_spiel_claim_command_regression.py",
        "tests/test_rl_delivery_boundary.py",
        "tests/test_strategy_metric_gate.py",
        "tests/test_evaluation_metric_contract.py",
        "tests/test_test_execution_contract.py",
        "tests/test_human_likeness_evidence.py",
        "tests/test_human_likeness_claim_gate.py",
        "tests/test_human_likeness_policy_guard.py",
        "tests/test_data_leakage_contract.py",
        "tests/test_normalized_action_contract.py",
        "tests/test_actions_context_quality.py",
        "tests/test_actions_dataset_export_contract.py",
        "tests/test_stack_event_context_quality.py",
        "tests/test_stack_context.py",
        "tests/test_scenario_sanity.py",
        "tests/test_phase2_selection_comparison.py",
        "tests/test_llm_policy_experimental.py",
        "tests/test_challenger_strategy_quality.py",
        "tests/test_final_strategy_quality_status.py",
        "tests/test_strategy_stack_maturity.py",
        "tests/test_hole_card_data_quality.py",
        "verify_delivery.ps1",
    }
    if not zip_path.exists():
        raise AssertionError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = {name.replace("\\", "/") for name in archive.namelist()}
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
    return f"zip_entries={len(names)}"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checks = [
        run_check("required_files", lambda: require_files(root)),
        run_check("compile_sources", lambda: compile_sources(root)),
        run_check("model_loads", lambda: model_loads(args.model)),
        run_check("inference_contract", lambda: inference_contract(args.model)),
        run_check("autonomous_agent_contract", autonomous_agent_contract),
        run_check("health_contract", lambda: health_contract(args.model)),
        run_check("public_openapi_contract", public_openapi_contract),
        run_check("api_input_contract", api_input_contract),
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

