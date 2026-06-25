from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any


SCOPE_CONTRACT_VERSION = "2026-06-21"


SOURCE_DOCUMENTS = [
    {
        "name": "Poker ML Project.docx",
        "path": "C:/Users/user/Desktop/AllFile/Poker ML Project.docx",
        "role": "primary_scope",
    },
    {
        "name": "Poker_Agent_Development_EN_detailed.pdf",
        "path": "C:/Users/user/Desktop/AllFile/Poker_Agent_Development_EN_detailed.pdf",
        "role": "detailed_delivery_reference",
    },
]


SCOPE_PHASES: list[dict[str, Any]] = [
    {
        "id": "phase_1_two_baselines",
        "title": "Two baselines",
        "requirements": [
            "llm_based_decision_agent",
            "structured_state_to_prompt",
            "discrete_action_parsing",
            "llm_accuracy_latency_cost_metrics",
            "supervised_policy_pretraining",
            "action_probability_output",
        ],
        "required_evidence": [
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
    },
    {
        "id": "phase_2_selection_optimization",
        "title": "Selection and optimization",
        "requirements": [
            "compare_llm_and_policy_approaches",
            "action_accuracy_comparison",
            "simulation_win_rate_comparison",
            "stability_metrics",
            "latency_or_inference_contract",
            "reproducible_pipeline",
        ],
        "required_evidence": [
            "reports/deployed_strategy_gate.json",
            "reports/model_risk_register.json",
            "reports/production_approval.json",
            "reports/client_handoff.json",
            "reports/production_self_play.json",
            "reports/delivery_readiness.json",
        ],
    },
    {
        "id": "phase_3_evaluation",
        "title": "Evaluation",
        "requirements": [
            "heldout_human_action_alignment",
            "cross_entropy_or_probability_metrics",
            "win_rate_in_simulation",
            "stability_across_seeds",
        ],
        "required_evidence": [
            "reports/policy_acceptance.json",
            "reports/production_self_play.json",
        ],
    },
    {
        "id": "phase_4_deployment",
        "title": "Deployment",
        "requirements": [
            "fastapi_service",
            "predict_endpoint",
            "structured_game_state_input",
            "action_and_probabilities_output",
            "bet_size_and_timing_output",
            "docker_packaging",
            "machine_readable_readiness_reports",
        ],
        "required_evidence": [
            "poker_agent/service.py",
            "Dockerfile",
            "docker-compose.yml",
            "reports/delivery_readiness.json",
            "reports/model_risk_register.json",
            "reports/production_approval.json",
            "reports/client_handoff.json",
            "release/poker-decision-agent.zip",
        ],
    },
]


DATASET_CONTRACT: dict[str, list[str]] = {
    "hands.csv": [
        "hand_id",
        "hand_index",
        "local_hand_index",
        "source_file",
        "start_frame",
        "end_frame",
        "board_cards",
        "total_actions",
        "total_stack_events",
        "winner_positions",
        "pot_from_stacks",
        "pot_from_recognition",
        "dealer_hand_number",
        "dealer_winner",
        "dealer_pot",
    ],
    "players.csv": [
        "hand_id",
        "hand_index",
        "local_hand_index",
        "source_file",
        "position",
        "nickname",
        "cards",
        "starting_stack",
        "ending_stack",
        "stack_delta",
    ],
    "actions.csv": [
        "hand_id",
        "hand_index",
        "local_hand_index",
        "source_file",
        "frame_id",
        "player_position",
        "player_nickname",
        "action",
        "street",
    ],
    "stack_events.csv": [
        "hand_id",
        "hand_index",
        "local_hand_index",
        "source_file",
        "frame_id",
        "player_position",
        "event",
        "stack",
        "diff",
        "stack_after_event",
    ],
}


def build_scope_contract(project_root: Path) -> dict[str, Any]:
    phase_reports = [_phase_status(project_root, phase) for phase in SCOPE_PHASES]
    dataset_report = _dataset_status(project_root)
    delivery = _read_json(project_root / "reports" / "delivery_readiness.json")
    deployed_gate = _read_json(project_root / "reports" / "deployed_strategy_gate.json")
    raw_gate = _read_json(project_root / "reports" / "production_gate.json")
    return {
        "version": SCOPE_CONTRACT_VERSION,
        "source_documents": SOURCE_DOCUMENTS,
        "overall_status": _overall_status(phase_reports, dataset_report),
        "delivery_status": delivery.get("overall_status", "MISSING"),
        "strategy_policy_status": delivery.get("strategy_policy_status", "UNKNOWN"),
        "deployed_strategy_gate_status": deployed_gate.get("status", "MISSING"),
        "raw_supervised_model_status": deployed_gate.get("raw_supervised_model_status", "MISSING"),
        "raw_production_gate_status": raw_gate.get("status", "MISSING"),
        "phase_statuses": {phase["id"]: phase["status"] for phase in phase_reports},
        "phases": phase_reports,
        "dataset_contract": dataset_report,
        "approval_boundary": {
            "scope_delivery": "All client-visible phases have implemented evidence or an explicit risk entry.",
            "deployed_strategy_stack": "Approved by the deployed strategy gate when policy acceptance and production-scale self-play pass.",
            "standalone_raw_model": "Still independent from deployed-stack approval and currently not standalone approved unless the raw gate passes.",
        },
        "remaining_risks": _remaining_risks(deployed_gate, raw_gate),
    }


def write_scope_contract(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_scope_contract(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_scope_contract_markdown(payload), encoding="utf-8")
    return payload


def render_scope_contract_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Scope Contract",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Delivery status: `{payload['delivery_status']}`",
        f"- Strategy policy status: `{payload['strategy_policy_status']}`",
        f"- Deployed strategy gate: `{payload['deployed_strategy_gate_status']}`",
        f"- Raw supervised model status: `{payload['raw_supervised_model_status']}`",
        "",
        "## Phase Status",
        "",
        "| Phase | Status | Missing Evidence |",
        "| --- | --- | --- |",
    ]
    for phase in payload["phases"]:
        missing = ", ".join(phase["missing_evidence"]) or "none"
        lines.append(f"| {phase['title']} | {phase['status']} | {missing} |")
    lines.extend(["", "## Remaining Risks", ""])
    if payload["remaining_risks"]:
        for risk in payload["remaining_risks"]:
            lines.append(f"- `{risk['id']}`: {risk['description']}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _phase_status(project_root: Path, phase: dict[str, Any]) -> dict[str, Any]:
    missing = [path for path in phase["required_evidence"] if not (project_root / path).exists()]
    status = "PASS" if not missing else "PARTIAL"
    return {
        "id": phase["id"],
        "title": phase["title"],
        "status": status,
        "requirements": list(phase["requirements"]),
        "required_evidence": list(phase["required_evidence"]),
        "missing_evidence": missing,
    }


def _dataset_status(project_root: Path) -> dict[str, Any]:
    data_dir = project_root / "data"
    dataset_dir = project_root / "dataset"
    tables = {}
    for filename, columns in DATASET_CONTRACT.items():
        path = data_dir / filename
        fallback = dataset_dir / filename
        existing = path if path.exists() else fallback
        header = _csv_header(existing) if existing.exists() else []
        missing_columns = [column for column in columns if column not in header]
        tables[filename] = {
            "status": "PASS" if existing.exists() and not missing_columns else "MISSING" if not existing.exists() else "INVALID_SCHEMA",
            "path": str(existing) if existing.exists() else str(path),
            "required_columns": columns,
            "observed_columns": header,
            "missing_columns": missing_columns,
        }
    status = "PASS" if all(table["status"] == "PASS" for table in tables.values()) else "PARTIAL"
    return {
        "status": status,
        "tables": tables,
    }


def _overall_status(phase_reports: list[dict[str, Any]], dataset_report: dict[str, Any]) -> str:
    if all(phase["status"] == "PASS" for phase in phase_reports) and dataset_report["status"] == "PASS":
        return "PASS"
    if any(phase["status"] == "PASS" for phase in phase_reports):
        return "PARTIAL"
    return "FAIL"


def _remaining_risks(deployed_gate: dict[str, Any], raw_gate: dict[str, Any]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    if deployed_gate.get("raw_supervised_model_status") == "NOT_STANDALONE_APPROVED":
        risks.append(
            {
                "id": "raw_supervised_model_not_standalone_approved",
                "description": "The deployed stack is approved, but the raw supervised artifact still fails standalone production thresholds.",
            }
        )
    if raw_gate.get("status") == "FAIL":
        risks.append(
            {
                "id": "raw_production_gate_fail",
                "description": "Raw production gate remains FAIL and should be resolved by training a stronger challenger artifact.",
            }
        )
    return risks


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [column.strip() for column in next(reader)]
        except StopIteration:
            return []
