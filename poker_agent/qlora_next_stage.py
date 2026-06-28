from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QLORA_NEXT_STAGE_VERSION = "2026-06-28"
NEXT_STAGE_IMPROVEMENT = "NEXT_STAGE_IMPROVEMENT"
CONTROLLED_LAYER = "CONTROLLED_DECISION_CONTEXT_AND_EVENT_NORMALIZATION_LAYER"
NOT_COMPLETED = "NOT_COMPLETED"
NOT_PRODUCTION_APPROVED = "NOT_PRODUCTION_APPROVED"


def build_qlora_next_stage(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    llm_role = _read_optional_json(reports / "llm_role_boundary.json")
    architecture = _read_optional_json(reports / "llm_architecture_comparison.json")
    candidate_gate = _read_optional_json(reports / "llm_decision_candidate_gate.json")
    candidate_ranker = _read_optional_json(reports / "llm_decision_candidate_ranker_qwen25.json")
    event_eval = _read_optional_json(reports / "llm_event_gold_eval.json")

    llm_current = llm_role.get("current_llm_role") or {}
    llm_boundary = llm_role.get("autonomous_llm_agent_boundary") or {}
    architecture_boundary = architecture.get("approval_boundary") or {}
    candidate_boundary = candidate_gate.get("production_boundary") or {}

    payload: dict[str, Any] = {
        "version": QLORA_NEXT_STAGE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "QLoRA and larger-LLM fine-tuning next-stage boundary",
        "client_statement": (
            "QLoRA or larger LLM fine-tuning remains a next-stage improvement, especially for better "
            "structured extraction, candidate ranking, and noisy OCR/dealer-log handling. It is not presented "
            "as completed fine-tuning and it is not required to approve the current delivery package."
        ),
        "stage_boundary": {
            "stage_status": NEXT_STAGE_IMPROVEMENT,
            "fine_tuning_status": NOT_COMPLETED,
            "fine_tuning_completed": False,
            "production_status": NOT_PRODUCTION_APPROVED,
            "production_approved": False,
            "current_delivery_blocker": False,
            "current_llm_role": llm_current.get("status"),
            "autonomous_llm_agent_claim_allowed": llm_boundary.get("fully_autonomous_llm_agent_claim_allowed", False),
            "deployed_strategy_stack_affected": architecture_boundary.get("deployed_strategy_stack_affected", False),
            "llm_agent_production_approved": candidate_boundary.get("llm_agent_production_approved", False),
        },
        "target_use_cases": {
            "structured_extraction": {"recommended": True, "priority": "high"},
            "candidate_ranking": {"recommended": True, "priority": "high"},
            "noisy_ocr_dealer_log_handling": {"recommended": True, "priority": "high"},
            "autonomous_poker_policy": {"recommended": False, "priority": "deferred"},
        },
        "recommended_training_plan": {
            "primary_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "secondary_model": "Qwen/Qwen3-1.7B",
            "lightweight_baseline": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "larger_llm_role": "Research benchmark only unless latency and cost gates justify promotion.",
            "method": "QLoRA",
            "quantization": "4-bit NF4",
            "lora_rank": [8, 16],
            "lora_alpha": [16, 32],
            "lora_dropout": 0.05,
            "learning_rate": [0.0001, 0.0002],
            "epochs": [2, 4],
            "max_seq_len": [1024, 4096],
            "train_target": "controlled event extraction and candidate ranking, not unconstrained poker decision making",
        },
        "data_requirements": {
            "initial_validation_examples": 500,
            "reliable_benchmark_examples": 5000,
            "production_candidate_examples": 20000,
            "required_slices": [
                "clean dealer messages",
                "corrupted OCR actions",
                "card recognition noise",
                "amount recognition noise",
                "hard negatives and unmatched text",
            ],
        },
        "acceptance_gates": [
            {"name": "schema_validity", "threshold": ">= 0.99"},
            {"name": "macro_f1", "threshold": "> current candidate ranker and deterministic baseline"},
            {"name": "event_type_exact_match", "threshold": ">= 0.90 on reviewed holdout"},
            {"name": "action_exact_match", "threshold": ">= 0.90 on reviewed action subset"},
            {"name": "card_exact_match", "threshold": ">= 0.85 on reviewed card subset"},
            {"name": "amount_exact_match_or_tolerance", "threshold": "within configured monetary tolerance"},
            {"name": "latency_memory", "threshold": "within deployment budget"},
            {"name": "baseline_comparison", "threshold": "statistically improves controlled candidate ranker"},
            {"name": "deterministic_fallback", "threshold": "available for invalid or low-confidence outputs"},
        ],
        "allowed_claims": [
            "QLoRA/larger LLM fine-tuning is a planned next-stage improvement.",
            "The target is controlled structured extraction, candidate ranking, and noisy OCR/dealer-log handling.",
            "The current delivery remains valid without marking QLoRA as completed or production-approved.",
        ],
        "blocked_claims": [
            "QLoRA fine-tuning is complete and production-approved.",
            "A larger LLM is the deployed poker policy.",
            "The LLM is a fully autonomous poker-playing agent.",
            "Fine-tuning solves upstream missing-card and label-quality issues by itself.",
        ],
        "evidence": {
            "llm_role_boundary": "reports/llm_role_boundary.json",
            "llm_architecture_comparison": "reports/llm_architecture_comparison.json",
            "candidate_ranker_gate": "reports/llm_decision_candidate_gate.json",
            "candidate_ranker_report": "reports/llm_decision_candidate_ranker_qwen25.json",
            "event_normalization_eval": "reports/llm_event_gold_eval.json",
            "current_llm_role": llm_current.get("status"),
            "candidate_gate_status": candidate_gate.get("status"),
            "candidate_ranker_provider": candidate_ranker.get("provider"),
            "event_gold_examples": event_eval.get("examples"),
        },
        "promotion_requirements": [
            "Stakeholder approval for a separate LLM fine-tuning milestone.",
            "Reviewed train/validation/test sets grouped by source hand or source file.",
            "Reproducible Hydra config and saved training configuration.",
            "Baseline comparison against deterministic parser and measured candidate ranker.",
            "Production serving benchmark with rollback to deterministic fallback.",
        ],
    }
    payload["invariants"] = validate_qlora_next_stage(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload

def validate_qlora_next_stage(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    boundary = payload.get("stage_boundary") or {}
    targets = payload.get("target_use_cases") or {}
    plan = payload.get("recommended_training_plan") or {}
    gate_names = {str(gate.get("name")) for gate in payload.get("acceptance_gates", []) if isinstance(gate, dict)}

    if payload.get("overall_status") == "PASS":
        violations.append("overall_status_must_be_assigned_after_invariant_validation")
    if boundary.get("stage_status") != NEXT_STAGE_IMPROVEMENT:
        violations.append("qlora_must_remain_next_stage_improvement")
    if boundary.get("fine_tuning_completed") is not False:
        violations.append("qlora_fine_tuning_must_not_be_marked_completed")
    if boundary.get("production_approved") is not False:
        violations.append("qlora_must_not_be_marked_production_approved")
    if boundary.get("current_delivery_blocker") is not False:
        violations.append("qlora_next_stage_must_not_block_current_delivery")
    if boundary.get("current_llm_role") != CONTROLLED_LAYER:
        violations.append("qlora_plan_must_preserve_controlled_llm_role")
    if boundary.get("autonomous_llm_agent_claim_allowed") is not False:
        violations.append("qlora_plan_must_block_autonomous_llm_claims")
    if boundary.get("deployed_strategy_stack_affected") is not False:
        violations.append("qlora_plan_must_not_change_deployed_strategy_stack_approval")
    if boundary.get("llm_agent_production_approved") is not False:
        violations.append("qlora_plan_must_not_approve_llm_agent_policy")

    for key in ("structured_extraction", "candidate_ranking", "noisy_ocr_dealer_log_handling"):
        if (targets.get(key) or {}).get("recommended") is not True:
            violations.append(f"{key}_must_be_recommended_target")
    if (targets.get("autonomous_poker_policy") or {}).get("recommended") is not False:
        violations.append("autonomous_poker_policy_must_not_be_immediate_qlora_target")

    if plan.get("method") != "QLoRA":
        violations.append("training_plan_must_use_qlora")
    if plan.get("quantization") != "4-bit NF4":
        violations.append("training_plan_must_use_nf4_quantization")
    if not str(plan.get("primary_model", "")).startswith("Qwen/"):
        violations.append("primary_model_must_be_qwen_family")

    required_gates = {
        "schema_validity",
        "macro_f1",
        "event_type_exact_match",
        "action_exact_match",
        "latency_memory",
        "baseline_comparison",
        "deterministic_fallback",
    }
    missing_gates = sorted(required_gates - gate_names)
    if missing_gates:
        violations.append(f"missing_acceptance_gates:{','.join(missing_gates)}")

    blocked_claims = " ".join(payload.get("blocked_claims") or [])
    if "production-approved" not in blocked_claims and "production approved" not in blocked_claims:
        violations.append("blocked_claims_must_reject_production_approval_overclaim")
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_qlora_next_stage(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_qlora_next_stage(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_qlora_next_stage_markdown(payload), encoding="utf-8")
    return payload


def render_qlora_next_stage_markdown(payload: dict[str, Any]) -> str:
    boundary = payload["stage_boundary"]
    plan = payload["recommended_training_plan"]
    lines = [
        "# QLoRA Next-Stage Boundary Contract",
        "",
        payload["client_statement"],
        "",
        "## Stage Boundary",
        "",
        f"- Stage status: `{boundary['stage_status']}`",
        f"- Fine-tuning completed: `{boundary['fine_tuning_completed']}`",
        f"- Production approved: `{boundary['production_approved']}`",
        f"- Current delivery blocker: `{boundary['current_delivery_blocker']}`",
        f"- Current LLM role: `{boundary['current_llm_role']}`",
        f"- Autonomous LLM-agent claim allowed: `{boundary['autonomous_llm_agent_claim_allowed']}`",
        "",
        "## Recommended Training Plan",
        "",
        f"- Primary model: `{plan['primary_model']}`",
        f"- Secondary model: `{plan['secondary_model']}`",
        f"- Method: `{plan['method']}`",
        f"- Quantization: `{plan['quantization']}`",
        f"- Train target: `{plan['train_target']}`",
        "",
        "## Target Use Cases",
        "",
    ]
    for name, target in payload["target_use_cases"].items():
        lines.append(f"- `{name}`: recommended=`{target['recommended']}`, priority=`{target['priority']}`")
    lines.extend(["", "## Acceptance Gates", ""])
    lines.extend(f"- `{gate['name']}`: {gate['threshold']}" for gate in payload["acceptance_gates"])
    lines.extend(["", "## Blocked Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
