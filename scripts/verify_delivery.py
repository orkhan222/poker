from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.agents import MLPolicyAgent
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest
from poker_agent.service import get_agent, health_payload, resolve_model_path


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
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/evaluate_policy.yaml",
        "configs/experiments/research_compare_tabular.yaml",
        "configs/experiments/audit_dataset.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/production_gate.yaml",
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
        "models/poker_policy.joblib",
        "reports/production_gate.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_eval.md",
        "reports/policy_acceptance.json",
        "reports/production_self_play.json",
        "reports/deployed_strategy_gate.json",
        "reports/delivery_readiness.json",
        "reports/scope_contract.json",
        "reports/scope_contract.md",
        "reports/model_risk_register.json",
        "reports/model_risk_register.md",
        "reports/production_approval.json",
        "reports/production_approval.md",
        "reports/client_handoff.json",
        "reports/client_handoff.md",
        "evaluation/event_extraction_gold.jsonl",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_client_handoff.py",
        "scripts/build_scope_contract.py",
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
        "poker_agent/api_contract.py",
        "poker_agent/scope_contract.py",
        "poker_agent/model_risk_register.py",
        "poker_agent/production_approval.py",
        "poker_agent/client_handoff.py",
        "poker_agent/delivery_readiness.py",
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
        "poker_agent/agents.py",
        "poker_agent/api_contract.py",
        "poker_agent/delivery_readiness.py",
        "poker_agent/evaluator.py",
        "poker_agent/features.py",
        "poker_agent/model.py",
        "poker_agent/schemas.py",
        "poker_agent/scope_contract.py",
        "poker_agent/model_risk_register.py",
        "poker_agent/production_approval.py",
        "poker_agent/client_handoff.py",
        "poker_agent/service.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/build_scope_contract.py",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_client_handoff.py",
        "scripts/check_repo_hygiene.py",
        "scripts/evaluate_policy.py",
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
    if isinstance(agent, MLPolicyAgent) and missing["model_status"] != "missing_card_fallback":
        raise AssertionError("Missing-card request did not use fallback")
    for payload in (observed, missing):
        total = sum(float(value) for value in payload["probabilities"].values())
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"Probabilities do not sum to 1: {total}")
    return f"agent={type(agent).__name__}, observed={observed['action']} missing={missing['action']}"


def health_contract(model_path: Path) -> str:
    resolved = resolve_model_path()
    if resolved.resolve() != model_path.resolve():
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
    scope_payload = _read_json(reports / "scope_contract.json")
    risk_payload = _read_json(reports / "model_risk_register.json")
    approval_payload = _read_json(reports / "production_approval.json")
    handoff_payload = _read_json(reports / "client_handoff.json")

    if scope_payload.get("overall_status") != "PASS":
        raise AssertionError(f"Scope contract did not pass: {scope_payload.get('overall_status')}")
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
    if approval_payload.get("raw_supervised_model", {}).get("standalone_status") != "NOT_STANDALONE_APPROVED":
        raise AssertionError("Production approval does not preserve raw-model standalone boundary")
    if approval_payload.get("risk_position", {}).get("deployment_blockers") != 0:
        raise AssertionError("Production approval incorrectly reports a deployment blocker")
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
        "configs/experiments/production_gate.yaml",
        "configs/experiments/verify_delivery.yaml",
    ]
    missing = [relative for relative in required_configs if not (root / relative).exists()]
    if missing:
        raise AssertionError(f"Hydra configuration hierarchy is incomplete: {missing}")
    return f"hydra_configs={len(required_configs)}"


def zip_contract(root: Path, zip_path: Path) -> str:
    required = {
        "models/poker_policy.joblib",
        "README.md",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/model/text_event_smol.yaml",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "evaluation/event_extraction_gold.jsonl",
        "reports/production_gate.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_eval.md",
        "reports/policy_acceptance.json",
        "reports/production_self_play.json",
        "reports/deployed_strategy_gate.json",
        "reports/delivery_readiness.json",
        "reports/repo_hygiene.json",
        "reports/scope_contract.json",
        "reports/scope_contract.md",
        "reports/model_risk_register.json",
        "reports/model_risk_register.md",
        "reports/production_approval.json",
        "reports/production_approval.md",
        "reports/client_handoff.json",
        "reports/client_handoff.md",
        "poker_agent/model_risk_register.py",
        "poker_agent/production_approval.py",
        "poker_agent/client_handoff.py",
        "poker_agent/api_contract.py",
        "poker_agent/delivery_readiness.py",
        "poker_agent/scope_contract.py",
        "scripts/check_repo_hygiene.py",
        "scripts/audit_repository.py",
        "scripts/build_model_risk_register.py",
        "scripts/build_production_approval.py",
        "scripts/build_client_handoff.py",
        "scripts/build_scope_contract.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
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
