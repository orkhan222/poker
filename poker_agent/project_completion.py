from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.approval_boundary import build_approval_boundary
from poker_agent.scope_contract import DATASET_CONTRACT, SOURCE_DOCUMENTS, build_scope_contract


COMPLETION_VERSION = "2026-06-23"


FEATURE_SPACE = [
    {
        "name": "opponent_action_timing",
        "status": "IMPLEMENTED",
        "evidence": "poker_agent/schemas.py + poker_agent/features.py + poker_agent/action_planning.py",
        "description": (
            "The request contract accepts observed opponent timing, historical frame gaps are extracted "
            "without target-action leakage, and the output delay is calibrated to table tempo."
        ),
    },
    {
        "name": "opponent_betting_history",
        "status": "IMPLEMENTED",
        "evidence": "poker_agent/features.py",
        "description": "Betting history is converted into action counts, last-action indicators, aggression, and pressure features.",
    },
    {
        "name": "cards_and_board",
        "status": "IMPLEMENTED",
        "evidence": "poker_agent/features.py",
        "description": "Hero hole cards, board cards, made-hand signals, draw pressure, and card-count features are extracted without future-card leakage.",
    },
    {
        "name": "stacks_and_pot_dynamics",
        "status": "IMPLEMENTED",
        "evidence": "poker_agent/features.py",
        "description": "Stack, pot, stack-to-pot ratio, to-call, min-raise, and historical stack deltas are part of the feature contract.",
    },
]


ACTION_SPACE = [
    {
        "name": "action",
        "status": "IMPLEMENTED",
        "values": ["fold", "call", "check", "bet", "raise"],
        "evidence": "poker_agent/schemas.py",
    },
    {
        "name": "bet_size",
        "status": "IMPLEMENTED",
        "evidence": "poker_agent/action_planning.py",
    },
    {
        "name": "wait_time_ms",
        "status": "IMPLEMENTED",
        "evidence": "poker_agent/action_planning.py",
    },
]


def build_project_completion(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    scope = build_scope_contract(project_root)
    delivery_verification = _read_json(reports / "delivery_verification.json")
    delivery_readiness = _read_json(reports / "delivery_readiness.json")
    policy_acceptance = _read_json(reports / "policy_acceptance.json")
    self_play = _read_json(reports / "production_self_play.json")
    deployed_gate = _read_json(reports / "deployed_strategy_gate.json")
    production_gate = _read_json(reports / "production_gate.json")
    decision_context = _read_json(reports / "llm_decision_context.json")
    llm_decision_benchmark = _read_json(reports / "llm_decision_context_qwen25.json")
    llm_decision_gate = _read_json(reports / "llm_decision_gate.json")
    llm_candidate_ranker = _read_json(reports / "llm_decision_candidate_ranker_qwen25.json")
    llm_candidate_gate = _read_json(reports / "llm_decision_candidate_gate.json")
    llm_architecture_comparison = _read_json(reports / "llm_architecture_comparison.json")
    client_handoff = _read_json(reports / "client_handoff.json")
    boundary = build_approval_boundary(project_root)["boundary"]

    phase_statuses = scope.get("phase_statuses", {})
    overall_status = _overall_status(
        scope=scope,
        delivery_readiness=delivery_readiness,
        deployed_gate=deployed_gate,
        policy_acceptance=policy_acceptance,
        self_play=self_play,
        decision_context=decision_context,
    )
    return {
        "version": COMPLETION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_documents": SOURCE_DOCUMENTS,
        "overall_status": overall_status,
        "executive_summary": _executive_summary(overall_status, client_handoff),
        "feature_space": FEATURE_SPACE,
        "action_space": ACTION_SPACE,
        "dataset_model": _dataset_model(scope),
        "phase_completion": {
            "phase_1_two_baselines": {
                "status": phase_statuses.get("phase_1_two_baselines", "MISSING"),
                "evidence": [
                    "reports/llm_event_gold_eval.json",
                    "reports/llm_decision_context.json",
                    "reports/llm_decision_context_smoke.json",
                    "reports/llm_decision_context_qwen25.json",
                    "reports/llm_decision_gate.json",
                    "reports/llm_decision_candidate_ranker_qwen25.json",
                    "reports/llm_decision_candidate_gate.json",
                    "reports/llm_architecture_comparison.json",
                    "reports/production_gate.json",
                    "reports/policy_acceptance.json",
                ],
                "implemented_outputs": [
                    "LLM decision context contract",
                    "context-ablation benchmark runner with explicit quality-claim boundary",
                    "measured Qwen2.5-1.5B 4-bit GPU baseline",
                    "independent LLM decision-model acceptance gate",
                    "constrained legal-action candidate ranker",
                    "measured generation-versus-ranking architecture comparison",
                    "structured event extraction benchmark",
                    "supervised policy artifact",
                    "action probability output",
                ],
            },
            "phase_2_selection_optimization": {
                "status": phase_statuses.get("phase_2_selection_optimization", "MISSING"),
                "evidence": [
                    "reports/deployed_strategy_gate.json",
                    "reports/production_self_play.json",
                    "reports/model_risk_register.json",
                    "reports/production_approval.json",
                ],
                "implemented_outputs": [
                    "deployed strategy gate",
                    "production-scale self-play comparison",
                    "component risk register",
                    "client handoff boundary",
                ],
            },
            "phase_3_evaluation": {
                "status": phase_statuses.get("phase_3_evaluation", "MISSING"),
                "evidence": [
                    "reports/policy_acceptance.json",
                    "reports/production_self_play.json",
                    "reports/production_gate.json",
                ],
                "metrics": _evaluation_metrics(policy_acceptance, self_play, production_gate),
            },
            "phase_4_deployment": {
                "status": phase_statuses.get("phase_4_deployment", "MISSING"),
                "evidence": [
                    "poker_agent/service.py",
                    "Dockerfile",
                    "docker-compose.yml",
                    "release/poker-decision-agent.zip",
                    "reports/delivery_verification.json",
                ],
                "api_contract": {
                    "predict": "/predict",
                    "health": "/health.json",
                    "scope": "/scope-contract.json",
                    "completion": "/project-completion.json",
                },
            },
        },
        "delivery_status": {
            "delivery_verification": delivery_verification.get("status", "MISSING"),
            "delivery_readiness": delivery_readiness.get("overall_status", "MISSING"),
            "deployed_strategy_stack": deployed_gate.get("strategy_policy_status", "MISSING"),
            "client_handoff": client_handoff.get("handoff_status", "MISSING"),
        },
        "known_boundary": {
            "raw_supervised_model_gate": boundary["raw_production_gate"],
            "raw_supervised_model_status": boundary["raw_supervised_model_standalone"],
            "production_blocker": boundary["production_blocker"],
            "component_risk": boundary["component_risk"],
            "evidence": "reports/model_risk_register.json",
            "approval_boundary": "/approval-boundary.json",
        },
        "llm_decision_research": {
            "status": llm_decision_gate.get("status", "MISSING"),
            "provider": llm_decision_benchmark.get("provider"),
            "selected_context_mode": llm_decision_gate.get("selected_context_mode"),
            "selection_is_provisional": llm_decision_gate.get("selection_is_provisional"),
            "metrics": llm_decision_gate.get("metrics", {}),
            "failed_checks": llm_decision_gate.get("failed_checks", []),
            "deployed_strategy_stack_affected": (
                llm_decision_gate.get("production_boundary") or {}
            ).get("deployed_strategy_stack_affected"),
            "candidate_ranker": {
                "provider": llm_candidate_ranker.get("provider"),
                "gate_status": llm_candidate_gate.get("status"),
                "metrics": llm_candidate_gate.get("metrics", {}),
            },
            "architecture_selection": {
                "status": llm_architecture_comparison.get("status"),
                "recommended_architecture": llm_architecture_comparison.get("recommended_architecture"),
                "production_approved": llm_architecture_comparison.get("production_approved"),
            },
        },
    }


def write_project_completion(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_project_completion(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_project_completion_markdown(payload), encoding="utf-8")
    return payload


def render_project_completion_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Project Completion Contract",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Delivery verification: `{payload['delivery_status']['delivery_verification']}`",
        f"- Delivery readiness: `{payload['delivery_status']['delivery_readiness']}`",
        f"- Deployed strategy stack: `{payload['delivery_status']['deployed_strategy_stack']}`",
        f"- Client handoff: `{payload['delivery_status']['client_handoff']}`",
        "",
        "## Executive Summary",
        "",
        payload["executive_summary"],
        "",
        "## Feature Space",
        "",
    ]
    lines.extend(f"- `{item['name']}`: {item['status']} ({item['evidence']})" for item in payload["feature_space"])
    lines.extend(["", "## Action Space", ""])
    lines.extend(f"- `{item['name']}`: {item['status']} ({item['evidence']})" for item in payload["action_space"])
    lines.extend(
        [
            "",
            "## Phase Completion",
            "",
            "| Phase | Status | Evidence count |",
            "| --- | --- | --- |",
        ]
    )
    for name, phase in payload["phase_completion"].items():
        lines.append(f"| `{name}` | `{phase['status']}` | {len(phase['evidence'])} |")
    metrics = payload["phase_completion"]["phase_3_evaluation"]["metrics"]
    lines.extend(
        [
            "",
            "## Evaluation Snapshot",
            "",
            f"- Human action accuracy: `{metrics['human_action_accuracy']}`",
            f"- Human action macro F1: `{metrics['human_action_macro_f1']}`",
            f"- Cross-entropy: `{metrics['cross_entropy']}`",
            f"- Self-play mean win-rate: `{metrics['self_play_mean_win_rate']}`",
            f"- Self-play run count: `{metrics['self_play_run_count']}`",
            "",
            "## Known Boundary",
            "",
            f"- Raw supervised model gate: `{payload['known_boundary']['raw_supervised_model_gate']}`",
            f"- Raw supervised model status: `{payload['known_boundary']['raw_supervised_model_status']}`",
            f"- Production blocker: `{payload['known_boundary']['production_blocker']}`",
            f"- Component risk: `{payload['known_boundary']['component_risk']}`",
            "",
            "## LLM Decision Research",
            "",
            f"- Status: `{payload['llm_decision_research']['status']}`",
            f"- Provider: `{payload['llm_decision_research']['provider']}`",
            f"- Provisional context selection: `{payload['llm_decision_research']['selected_context_mode']}`",
            f"- Deployed stack affected: `{payload['llm_decision_research']['deployed_strategy_stack_affected']}`",
            f"- Recommended research architecture: `{payload['llm_decision_research']['architecture_selection']['recommended_architecture']}`",
            f"- LLM production approved: `{payload['llm_decision_research']['architecture_selection']['production_approved']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _dataset_model(scope: dict[str, Any]) -> dict[str, Any]:
    dataset_status = scope.get("dataset_contract", {})
    tables = dataset_status.get("tables", {})
    return {
        "status": dataset_status.get("status", "MISSING"),
        "tables": {
            table: {
                "required_columns": columns,
                "status": (tables.get(table) or {}).get("status", "MISSING"),
                "path": (tables.get(table) or {}).get("path", ""),
            }
            for table, columns in DATASET_CONTRACT.items()
        },
        "relationships": [
            "hands.hand_id -> players.hand_id",
            "hands.hand_id -> actions.hand_id",
            "hands.hand_id -> stack_events.hand_id",
        ],
    }


def _evaluation_metrics(
    policy_acceptance: dict[str, Any],
    self_play: dict[str, Any],
    production_gate: dict[str, Any],
) -> dict[str, Any]:
    alignment = policy_acceptance.get("human_action_alignment") or {}
    raw_metrics = production_gate.get("valid_metrics") or {}
    return {
        "human_action_accuracy": alignment.get("accuracy"),
        "human_action_macro_f1": alignment.get("macro_f1"),
        "cross_entropy": raw_metrics.get("cross_entropy"),
        "self_play_mean_win_rate": self_play.get("mean_policy_win_rate"),
        "self_play_min_win_rate": self_play.get("min_policy_win_rate"),
        "self_play_max_win_rate": self_play.get("max_policy_win_rate"),
        "self_play_run_count": self_play.get("run_count"),
        "self_play_paired_hands": self_play.get("paired_hands"),
        "stability_status": self_play.get("status"),
    }


def _overall_status(
    *,
    scope: dict[str, Any],
    delivery_readiness: dict[str, Any],
    deployed_gate: dict[str, Any],
    policy_acceptance: dict[str, Any],
    self_play: dict[str, Any],
    decision_context: dict[str, Any],
) -> str:
    checks = [
        scope.get("overall_status") == "PASS",
        delivery_readiness.get("overall_status") == "READY_FOR_PRODUCTION_POLICY",
        deployed_gate.get("status") == "PASS",
        deployed_gate.get("strategy_policy_status") == "APPROVED",
        policy_acceptance.get("overall_status") == "PASS",
        self_play.get("status") == "PASS",
        self_play.get("production_scale_status") == "PASS",
        decision_context.get("default_context_mode") == "full_in_context",
    ]
    return "PASS" if all(checks) else "PARTIAL"


def _executive_summary(status: str, client_handoff: dict[str, Any]) -> str:
    if status == "PASS":
        return (
            "The documented scope is implemented for delivery: data contracts, evaluation, deployment, "
            "LLM decision context, action planning, service packaging, and production-stack approval evidence "
            "are present. The raw supervised model limitation remains a tracked component risk, not a deployment blocker."
        )
    statement = client_handoff.get("client_statement")
    if statement:
        return str(statement)
    return "The project has implemented evidence, but at least one completion gate still requires attention."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
