from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_llm_decision_architectures(
    generation: dict[str, Any],
    candidate_ranker: dict[str, Any],
    generation_gate: dict[str, Any],
    candidate_gate: dict[str, Any],
) -> dict[str, Any]:
    generation_mode, generation_metrics = _selected_metrics(generation)
    ranker_mode, ranker_metrics = _selected_metrics(candidate_ranker)
    latency_speedup = _safe_ratio(
        generation_metrics.get("average_latency_ms"),
        ranker_metrics.get("average_latency_ms"),
    )
    deltas = {
        "accuracy": _delta(ranker_metrics, generation_metrics, "accuracy"),
        "macro_f1": _delta(ranker_metrics, generation_metrics, "macro_f1"),
        "schema_valid_rate": _delta(ranker_metrics, generation_metrics, "schema_valid_rate"),
        "legal_action_rate": _delta(ranker_metrics, generation_metrics, "legal_action_rate"),
        "fallback_rate": _delta(ranker_metrics, generation_metrics, "fallback_rate"),
        "average_latency_ms": _delta(
            ranker_metrics,
            generation_metrics,
            "average_latency_ms",
        ),
        "latency_speedup": latency_speedup,
    }
    ranker_is_research_choice = all(
        [
            float(ranker_metrics.get("schema_valid_rate", 0.0)) >= 0.95,
            float(ranker_metrics.get("legal_action_rate", 0.0)) >= 0.99,
            float(ranker_metrics.get("average_latency_ms", float("inf")))
            < float(generation_metrics.get("average_latency_ms", float("inf"))),
            float(ranker_metrics.get("macro_f1", 0.0))
            >= float(generation_metrics.get("macro_f1", 0.0)) - 0.02,
        ]
    )
    production_approved = (
        generation_gate.get("status") == "APPROVED"
        or candidate_gate.get("status") == "APPROVED"
    )
    return {
        "comparison_version": "2026-06-25",
        "status": (
            "CANDIDATE_RANKER_SELECTED_FOR_RESEARCH"
            if ranker_is_research_choice
            else "NO_ARCHITECTURE_SELECTED"
        ),
        "production_approved": production_approved,
        "recommended_architecture": (
            "candidate_ranker" if ranker_is_research_choice else None
        ),
        "selection_scope": "research_iteration",
        "systems": {
            "free_generation": {
                "provider": generation.get("provider"),
                "context_mode": generation_mode,
                "metrics": generation_metrics,
                "gate_status": generation_gate.get("status"),
            },
            "candidate_ranker": {
                "provider": candidate_ranker.get("provider"),
                "context_mode": ranker_mode,
                "metrics": ranker_metrics,
                "gate_status": candidate_gate.get("status"),
            },
        },
        "candidate_ranker_deltas": deltas,
        "decision": {
            "why_selected": [
                "Legal actions are constrained before model scoring.",
                "Schema-valid probabilities are constructed deterministically.",
                "No free-form generation fallback is required.",
                "Measured latency is materially lower than free generation.",
                "Measured Macro F1 is not materially worse than the generative baseline.",
            ]
            if ranker_is_research_choice
            else [],
            "why_not_production": sorted(
                set(generation_gate.get("failed_checks", []))
                | set(candidate_gate.get("failed_checks", []))
            ),
            "next_experiment": (
                "Train a LoRA/QLoRA action-ranking adapter on manually reviewed state-action pairs, "
                "then rerun candidate ranking on a reviewed hand-group holdout."
            ),
        },
        "approval_boundary": {
            "deployed_strategy_stack_affected": False,
            "llm_decision_path_enabled_in_production": production_approved,
        },
    }


def build_llm_architecture_comparison(
    generation_path: Path,
    candidate_ranker_path: Path,
    generation_gate_path: Path,
    candidate_gate_path: Path,
) -> dict[str, Any]:
    return compare_llm_decision_architectures(
        _read_json(generation_path),
        _read_json(candidate_ranker_path),
        _read_json(generation_gate_path),
        _read_json(candidate_gate_path),
    )


def write_llm_architecture_comparison(
    payload: dict[str, Any],
    out_path: Path,
    markdown_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_comparison_markdown(payload), encoding="utf-8")


def render_comparison_markdown(payload: dict[str, Any]) -> str:
    generation = payload["systems"]["free_generation"]
    ranker = payload["systems"]["candidate_ranker"]
    lines = [
        "# LLM Decision Architecture Comparison",
        "",
        f"- Status: `{payload['status']}`",
        f"- Recommended research architecture: `{payload['recommended_architecture']}`",
        f"- Production approved: `{payload['production_approved']}`",
        "",
        "| Architecture | Context | Accuracy | Macro F1 | Schema | Fallback | Latency ms | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, system in (("free_generation", generation), ("candidate_ranker", ranker)):
        metrics = system["metrics"]
        lines.append(
            f"| `{name}` | `{system['context_mode']}` | {metrics.get('accuracy', 0):.4f} | "
            f"{metrics.get('macro_f1', 0):.4f} | {metrics.get('schema_valid_rate', 0):.4f} | "
            f"{metrics.get('fallback_rate', 0):.4f} | {metrics.get('average_latency_ms', 0):.2f} | "
            f"`{system['gate_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in payload["decision"]["why_selected"])
    lines.extend(
        [
            "",
            "## Next Experiment",
            "",
            payload["decision"]["next_experiment"],
            "",
        ]
    )
    return "\n".join(lines)


def _selected_metrics(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    mode = payload.get("provisional_best_mode") or payload.get("best_mode")
    return mode, (payload.get("systems") or {}).get(mode) or {}


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    return float(left.get(key, 0.0)) - float(right.get(key, 0.0))


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    top = float(numerator or 0.0)
    bottom = float(denominator or 0.0)
    return top / bottom if bottom > 0 else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
