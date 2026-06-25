from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_llm_decision_report(
    benchmark: dict[str, Any],
    holdout: dict[str, Any],
    *,
    min_examples: int,
    min_macro_f1: float,
    min_schema_valid_rate: float,
    min_legal_action_rate: float,
    max_average_latency_ms: float,
) -> dict[str, Any]:
    systems = benchmark.get("systems") or {}
    provisional_mode = benchmark.get("provisional_best_mode")
    selected = systems.get(provisional_mode) or {}
    checks = [
        _check(
            "real_transformer_provider",
            str(benchmark.get("provider", "")).startswith("transformers"),
            benchmark.get("provider"),
        ),
        _check("human_log_holdout", benchmark.get("dataset_kind") == "reconstructed_human_holdout", benchmark.get("dataset_kind")),
        _check("balanced_holdout", holdout.get("status") == "PASS", holdout.get("class_distribution")),
        _check("minimum_examples", int(selected.get("examples", 0)) >= min_examples, selected.get("examples")),
        _check("macro_f1", float(selected.get("macro_f1", 0.0)) >= min_macro_f1, selected.get("macro_f1")),
        _check(
            "schema_valid_rate",
            float(selected.get("schema_valid_rate", 0.0)) >= min_schema_valid_rate,
            selected.get("schema_valid_rate"),
        ),
        _check(
            "legal_action_rate",
            float(selected.get("legal_action_rate", 0.0)) >= min_legal_action_rate,
            selected.get("legal_action_rate"),
        ),
        _check(
            "average_latency_ms",
            float(selected.get("average_latency_ms", float("inf"))) <= max_average_latency_ms,
            selected.get("average_latency_ms"),
        ),
        _check(
            "manual_reviewed_labels",
            benchmark.get("quality_claim_allowed") is True,
            benchmark.get("quality_claim_allowed"),
        ),
    ]
    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    return {
        "gate_version": "2026-06-25",
        "status": "APPROVED" if not failed else "BASELINE_NOT_APPROVED",
        "provider": benchmark.get("provider"),
        "selected_context_mode": provisional_mode,
        "selection_is_provisional": benchmark.get("quality_claim_allowed") is not True,
        "metrics": selected,
        "thresholds": {
            "min_examples": min_examples,
            "min_macro_f1": min_macro_f1,
            "min_schema_valid_rate": min_schema_valid_rate,
            "min_legal_action_rate": min_legal_action_rate,
            "max_average_latency_ms": max_average_latency_ms,
        },
        "checks": checks,
        "failed_checks": failed,
        "production_boundary": {
            "deployed_strategy_stack_affected": False,
            "llm_agent_production_approved": not failed,
            "reason": (
                "The LLM decision experiment is an independently gated research baseline and does not "
                "override the deployed strategy-stack approval."
            ),
        },
        "recommendation": (
            "Use the measured model as a research baseline only. Add constrained candidate ranking or "
            "QLoRA schema tuning, manually review the holdout labels, and repeat the gate before enabling "
            "the LLM decision path in production."
            if failed
            else "The measured LLM decision path satisfies the configured acceptance thresholds."
        ),
    }


def build_llm_decision_gate(
    benchmark_path: Path,
    holdout_path: Path,
    *,
    min_examples: int = 20,
    min_macro_f1: float = 0.40,
    min_schema_valid_rate: float = 0.95,
    min_legal_action_rate: float = 0.99,
    max_average_latency_ms: float = 5000.0,
) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    return evaluate_llm_decision_report(
        benchmark,
        holdout,
        min_examples=min_examples,
        min_macro_f1=min_macro_f1,
        min_schema_valid_rate=min_schema_valid_rate,
        min_legal_action_rate=min_legal_action_rate,
        max_average_latency_ms=max_average_latency_ms,
    )


def write_llm_decision_gate(
    payload: dict[str, Any],
    out_path: Path,
    markdown_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_llm_decision_gate_markdown(payload), encoding="utf-8")


def render_llm_decision_gate_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics") or {}
    lines = [
        "# LLM Decision Model Gate",
        "",
        f"- Status: `{payload['status']}`",
        f"- Provider: `{payload['provider']}`",
        f"- Selected context: `{payload['selected_context_mode']}`",
        f"- Selection is provisional: `{payload['selection_is_provisional']}`",
        "",
        "## Metrics",
        "",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- Macro F1: `{metrics.get('macro_f1')}`",
        f"- Raw schema validity: `{metrics.get('schema_valid_rate')}`",
        f"- Legal action rate: `{metrics.get('legal_action_rate')}`",
        f"- Average latency ms: `{metrics.get('average_latency_ms')}`",
        f"- Peak GPU memory MB: `{metrics.get('peak_memory_mb')}`",
        "",
        "## Gate Checks",
        "",
        "| Check | Status | Observed |",
        "| --- | --- | --- |",
    ]
    for item in payload["checks"]:
        lines.append(f"| `{item['name']}` | `{item['status']}` | `{item['observed']}` |")
    lines.extend(["", "## Recommendation", "", payload["recommendation"], ""])
    return "\n".join(lines)


def _check(name: str, passed: bool, observed: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
    }
