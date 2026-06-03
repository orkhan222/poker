from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.agents import MLPolicyAgent
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest
from poker_agent.service import health_payload, resolve_model_path


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
        "configs/training/group_holdout.yaml",
        "configs/training/smoke.yaml",
        "configs/evaluation/standard.yaml",
        "configs/inference/local_service.yaml",
        "configs/logging/local.yaml",
        "configs/prompts/event_extraction_prompt.txt",
        "configs/prompts/event_extraction_minimal.txt",
        "configs/prompts/event_extraction_permissive.txt",
        "configs/prompts/event_extraction_strict.txt",
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
        "configs/experiments/verify_delivery.yaml",
        "Dockerfile",
        "docker-compose.yml",
        "install.ps1",
        "run_server.ps1",
        "complete_delivery.ps1",
        "verify_delivery.ps1",
        "models/poker_policy.joblib",
        "reports/dataset_audit.json",
        "reports/repository_audit.json",
        "reports/production_gate.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_report.md",
        "evaluation/event_extraction_gold.jsonl",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/evaluate_policy.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/check_repo_hygiene.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/production_gate.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
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
        "poker_agent/agents.py",
        "poker_agent/evaluator.py",
        "poker_agent/features.py",
        "poker_agent/model.py",
        "poker_agent/schemas.py",
        "poker_agent/service.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/check_repo_hygiene.py",
        "scripts/evaluate_policy.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/production_gate.py",
        "scripts/research_experiment.py",
        "scripts/run_hydra_experiment.py",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/verify_delivery.py",
    ]
    for relative in source_files:
        py_compile.compile(str(root / relative), doraise=True)
    return f"{len(source_files)} Python files compile"


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
    agent = MLPolicyAgent.from_path(model_path)
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
    if missing["model_status"] != "missing_card_fallback":
        raise AssertionError("Missing-card request did not use fallback")
    for payload in (observed, missing):
        total = sum(float(value) for value in payload["probabilities"].values())
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"Probabilities do not sum to 1: {total}")
    return f"observed={observed['action']} missing={missing['action']}"


def health_contract(model_path: Path) -> str:
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
    if "findings" not in audit:
        raise AssertionError("Audit report has no findings key")
    if repo_audit.get("status") != "PASS":
        raise AssertionError("Repository audit did not pass")
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
    return (
        f"audit_findings={len(audit.get('findings', []))}, "
        f"repo_audit={repo_audit.get('status')}, gate={gate.get('status')}{benchmark_detail}"
    )


def repo_hygiene_contract(root: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_repo_hygiene.py"), "--root", str(root)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise AssertionError(detail[:2000])
    payload = json.loads(completed.stdout)
    return f"hygiene={payload['status']}"


def zip_contract(root: Path, zip_path: Path) -> str:
    required = {
        "models/poker_policy.joblib",
        "README.md",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "evaluation/event_extraction_gold.jsonl",
        "reports/dataset_audit.json",
        "reports/repository_audit.json",
        "reports/production_gate.json",
        "reports/llm_event_benchmark.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_report.md",
        "reports/llm_event_methodology.md",
        "scripts/check_repo_hygiene.py",
        "scripts/audit_repository.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
        "verify_delivery.ps1",
    }
    if not zip_path.exists():
        raise AssertionError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
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
