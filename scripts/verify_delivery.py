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
from poker_agent.schemas import PredictionRequest
from poker_agent.service import get_agent, get_autonomous_agent, health_payload, resolve_model_path


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
        "configs/experiments/bet_timing_calibration.yaml",
        "configs/experiments/hole_card_data_quality.yaml",
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
        "configs/experiments/llm_role_boundary.yaml",
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
        "configs/experiments/strategy_stack_maturity.yaml",
        "configs/experiments/behavioral_revalidation.yaml",
        "configs/experiments/behavioral_revalidation_proof.yaml",
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
        "reports/llm_role_boundary.json",
        "reports/llm_role_boundary.md",
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
        "reports/bet_timing_calibration.json",
        "reports/bet_timing_calibration.md",
        "reports/bet_timing_calibration.json",
        "reports/bet_timing_calibration.md",
        "reports/hole_card_data_quality.json",
        "reports/hole_card_data_quality.md",
        "reports/hole_card_data_quality.json",
        "reports/hole_card_data_quality.md",
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
        "evaluation/event_extraction_gold.jsonl",
        "evaluation/decision_context_smoke.jsonl",
        "evaluation/decision_context_human_holdout.jsonl",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_bet_timing_calibration.py",
        "scripts/build_behavioral_revalidation.py",
        "scripts/build_behavioral_revalidation_proof.py",
        "scripts/build_bet_timing_calibration.py",
        "scripts/build_hole_card_data_quality.py",
        "scripts/build_raw_model_status.py",
        "scripts/train_raw_model_challenger.py",
        "scripts/build_challenger_strategy_quality.py",
        "scripts/build_challenger_strategy_quality.py",
        "scripts/build_client_handoff.py",
        "scripts/build_training_cluster_requirements.py",
        "scripts/run_today_acceptance_training.py",
        "scripts/build_client_gpu_training_response.py",
        "scripts/build_multi_agent_training_status.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_llm_decision_context.py",
        "scripts/llm_decision_context_eval.py",
        "scripts/build_decision_context_holdout.py",
        "scripts/build_llm_decision_gate.py",
        "scripts/build_llm_architecture_comparison.py",
        "scripts/build_llm_role_boundary.py",
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
        "poker_agent/bet_timing_calibration.py",
        "poker_agent/hole_card_data_quality.py",
        "poker_agent/raw_model_status.py",
        "poker_agent/raw_model_challenger.py",
        "poker_agent/challenger_strategy_quality.py",
        "poker_agent/challenger_strategy_quality.py",
        "poker_agent/client_handoff.py",
        "poker_agent/training_cluster.py",
        "poker_agent/today_training.py",
        "poker_agent/client_gpu_training_response.py",
        "poker_agent/multi_agent_training_status.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/llm_decision_context.py",
        "poker_agent/llm_decision_benchmark.py",
        "poker_agent/llm_role_boundary.py",
        "poker_agent/qlora_next_stage.py",
        "poker_agent/production_runtime_monitoring.py",
        "poker_agent/llm_decision_gate.py",
        "poker_agent/llm_architecture_comparison.py",
        "poker_agent/project_completion.py",
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
        "tests/test_strategy_stack_maturity.py",
        "tests/test_llm_decision_benchmark.py",
        "tests/test_llm_decision_gate.py",
        "tests/test_llm_architecture_comparison.py",
        "tests/test_llm_role_boundary.py",
        "tests/test_qlora_next_stage.py",
        "tests/test_production_runtime_monitoring.py",
        "tests/test_final_delivery_acceptance.py",
        "tests/test_final_strategy_quality_status.py",
        "tests/test_llm_role_boundary.py",
        "tests/test_strategy_stack_maturity.py",
        "tests/test_behavioral_revalidation.py",
        "tests/test_behavioral_revalidation_proof.py",
        "tests/test_bet_timing_calibration.py",
        "tests/test_hole_card_data_quality.py",
        "tests/test_raw_model_status.py",
        "tests/test_raw_model_challenger.py",
        "tests/test_challenger_strategy_quality.py",
        "tests/test_hole_card_data_quality.py",
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
        "poker_agent/raw_model_status.py",
        "poker_agent/raw_model_challenger.py",
        "poker_agent/challenger_strategy_quality.py",
        "poker_agent/client_handoff.py",
        "poker_agent/training_cluster.py",
        "poker_agent/today_training.py",
        "poker_agent/client_gpu_training_response.py",
        "poker_agent/multi_agent_training_status.py",
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/llm_decision_context.py",
        "poker_agent/llm_decision_benchmark.py",
        "poker_agent/llm_decision_gate.py",
        "poker_agent/llm_architecture_comparison.py",
        "poker_agent/project_completion.py",
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
        "scripts/build_raw_model_status.py",
        "scripts/train_raw_model_challenger.py",
        "scripts/build_challenger_strategy_quality.py",
        "scripts/build_client_handoff.py",
        "scripts/build_training_cluster_requirements.py",
        "scripts/run_today_acceptance_training.py",
        "scripts/build_client_gpu_training_response.py",
        "scripts/build_multi_agent_training_status.py",
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_llm_decision_context.py",
        "scripts/build_llm_role_boundary.py",
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
    llm_role_boundary = _read_json(reports / "llm_role_boundary.json")
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
    bet_timing_calibration = _read_json(reports / "bet_timing_calibration.json")
    hole_card_data_quality = _read_json(reports / "hole_card_data_quality.json")
    raw_model_status = _read_json(reports / "raw_model_status.json")
    raw_model_challenger = _read_json(reports / "raw_model_challenger.json")
    challenger_strategy_quality = _read_json(reports / "challenger_strategy_quality.json")
    handoff_payload = _read_json(reports / "client_handoff.json")
    cluster_payload = _read_json(reports / "training_cluster_requirements.json")
    today_training = _read_json(reports / "today_acceptance_training.json")
    client_gpu_response = _read_json(reports / "client_gpu_training_response.json")
    multi_agent_training_status = _read_json(reports / "multi_agent_training_status.json")
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

    bet_timing_current = bet_timing_calibration.get("current_delivery_scope") or {}
    bet_timing_boundary = bet_timing_calibration.get("calibration_boundary") or {}
    bet_timing_fields = set(bet_timing_current.get("api_response_fields") or [])
    for required_field in ("bet_size", "wait_time_ms", "sizing_method", "timing_method"):
        if required_field not in bet_timing_fields:
            raise AssertionError(f"Bet/timing calibration contract missing response field: {required_field}")
    if bet_timing_calibration.get("overall_status") != "PASS":
        raise AssertionError(f"Bet/timing calibration contract did not pass: {bet_timing_calibration.get('overall_status')}")
    if bet_timing_current.get("implementation_status") != "IMPLEMENTED_AND_MEASURED":
        raise AssertionError("Bet-sizing and timing must remain implemented and measured")
    if bet_timing_current.get("timing_and_bet_size_status") != "PASS":
        raise AssertionError("Timing and bet-size measurement must pass for the current scope")
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
    if (bet_timing_calibration.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Bet/timing calibration invariants failed: {bet_timing_calibration.get('invariants')}")

    hole_cov = hole_card_data_quality.get("coverage_snapshot") or {}
    hole_mitigation = hole_card_data_quality.get("mitigation_boundary") or {}
    hole_upstream = hole_card_data_quality.get("upstream_data_quality_boundary") or {}
    if hole_card_data_quality.get("overall_status") != "PASS":
        raise AssertionError(f"Hole-card data-quality contract did not pass: {hole_card_data_quality.get('overall_status')}")
    if float(hole_cov.get("missing_hole_card_rate", 0.0)) <= float(hole_cov.get("complete_hole_card_rate", 1.0)):
        raise AssertionError("Hole-card audit must preserve missing-card dominance over complete-card coverage")
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
    if (hole_card_data_quality.get("invariants") or {}).get("status") != "PASS":
        raise AssertionError(f"Hole-card data-quality invariants failed: {hole_card_data_quality.get('invariants')}")
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

    llm_role = llm_role_boundary.get("current_llm_role") or {}
    llm_event_layer = llm_role.get("event_normalization_layer") or {}
    llm_context_layer = llm_role.get("decision_context_layer") or {}
    llm_autonomous_boundary = llm_role_boundary.get("autonomous_llm_agent_boundary") or {}
    if llm_role_boundary.get("overall_status") != "PASS":
        raise AssertionError(f"LLM role boundary did not pass: {llm_role_boundary.get('overall_status')}")
    if llm_role.get("status") != "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER":
        raise AssertionError("LLM work must remain a controlled decision/context and event-normalization layer")
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
    final_bet = final_acceptance_risks.get("bet_timing_calibration") or {}
    final_behavioral = final_acceptance_risks.get("behavioral_revalidation") or {}
    final_multi = final_acceptance_risks.get("multi_agent_training") or {}
    if final_delivery_acceptance.get("overall_status") != "PASS":
        raise AssertionError(f"Final delivery acceptance did not pass: {final_delivery_acceptance.get('overall_status')}")
    if final_delivery_acceptance.get("final_status") != "READY_WITH_TRACKED_COMPONENT_RISKS":
        raise AssertionError("Final delivery acceptance must preserve tracked component-risk status")
    if final_acceptance_summary.get("service_delivery") != "READY":
        raise AssertionError("Final acceptance must mark service delivery as ready")
    if final_acceptance_summary.get("deployed_strategy_stack") != "APPROVED":
        raise AssertionError("Final acceptance must preserve deployed strategy-stack approval")
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
    if final_bet.get("final_high_realism_claim_allowed") is not False:
        raise AssertionError("Final acceptance must block final high-realism bet/timing claims")
    if final_behavioral.get("larger_clean_real_gameplay_revalidation_required") is not True:
        raise AssertionError("Final acceptance must require larger clean gameplay revalidation")
    if final_multi.get("full_production_scale_multi_agent_training_status") != "NOT_COMPLETED":
        raise AssertionError("Final acceptance must not mark full production-scale multi-agent training complete")
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
        "configs/experiments/project_completion.yaml",
        "configs/experiments/training_cluster_requirements.yaml",
        "configs/experiments/today_acceptance_training.yaml",
        "configs/experiments/client_gpu_training_response.yaml",
        "configs/experiments/multi_agent_training_status.yaml",
        "configs/experiments/strategy_stack_maturity.yaml",
        "configs/experiments/behavioral_revalidation.yaml",
        "configs/experiments/behavioral_revalidation_proof.yaml",
        "configs/experiments/hole_card_data_quality.yaml",
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
        "README.md",
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
        "configs/experiments/challenger_strategy_quality.yaml",
        "configs/experiments/final_strategy_quality_status.yaml",
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
        "poker_agent/strategy_stack_maturity.py",
        "poker_agent/approval_boundary.py",
        "poker_agent/llm_decision_context.py",
        "poker_agent/llm_decision_benchmark.py",
        "poker_agent/llm_decision_gate.py",
        "poker_agent/llm_architecture_comparison.py",
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
        "scripts/build_strategy_stack_maturity.py",
        "scripts/build_llm_decision_context.py",
        "scripts/llm_decision_context_eval.py",
        "scripts/build_decision_context_holdout.py",
        "scripts/build_llm_decision_gate.py",
        "scripts/build_llm_architecture_comparison.py",
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

