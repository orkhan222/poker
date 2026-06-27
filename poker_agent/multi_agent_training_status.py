from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MULTI_AGENT_TRAINING_STATUS_VERSION = "2026-06-27"
FULL_TRAINING_STATUS_NOT_COMPLETED = "NOT_COMPLETED"
DELIVERY_VALIDATION_SCOPE = "delivery_validation_only"
PRODUCTION_HARDENING_SCOPE = "production_hardening"


def build_multi_agent_training_status(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    today_training = _read_optional_json(reports / "today_acceptance_training.json")
    client_gpu = _read_optional_json(reports / "client_gpu_training_response.json")
    cluster = _read_optional_json(reports / "training_cluster_requirements.json")
    self_play = _read_optional_json(reports / "production_self_play.json")

    training_status = today_training.get("training_status", "UNKNOWN")
    delivery_status = today_training.get("delivery_status", "UNKNOWN")
    delivery_validation_status = (
        "PASS"
        if training_status == "PASS" and delivery_status == "READY_FOR_CURRENT_DELIVERY"
        else "FAIL"
    )
    validation_metrics = today_training.get("valid_metrics") or {}

    payload: dict[str, Any] = {
        "version": MULTI_AGENT_TRAINING_STATUS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Multi-agent training completion boundary",
        "client_statement": (
            "Full production-scale multi-agent training has not been completed yet. "
            "The current acceptance training is sufficient for delivery validation, "
            "but it is not a full long-running self-play training cycle."
        ),
        "training_boundary": {
            "current_acceptance_training_profile": today_training.get("profile", "today_acceptance_training"),
            "selected_architecture": today_training.get("selected_architecture", "routed_policy_bundle"),
            "acceptance_training_status": training_status,
            "delivery_status": delivery_status,
            "delivery_validation_status": delivery_validation_status,
            "acceptance_training_sufficient_for_delivery": delivery_validation_status == "PASS",
            "full_production_scale_multi_agent_training_status": FULL_TRAINING_STATUS_NOT_COMPLETED,
            "full_long_running_self_play_completed": False,
            "scope_completed": DELIVERY_VALIDATION_SCOPE,
            "scope_deferred": PRODUCTION_HARDENING_SCOPE,
            "production_blocker": False,
        },
        "evidence": {
            "today_acceptance_training_report": "reports/today_acceptance_training.json",
            "client_gpu_training_response_report": "reports/client_gpu_training_response.json",
            "training_cluster_requirements_report": "reports/training_cluster_requirements.json",
            "production_self_play_report": "reports/production_self_play.json",
            "acceptance_metrics": {
                "accuracy": validation_metrics.get("accuracy"),
                "macro_f1": validation_metrics.get("macro_f1"),
                "balanced_accuracy": validation_metrics.get("balanced_accuracy"),
                "cross_entropy": validation_metrics.get("cross_entropy"),
            },
            "production_self_play_status": self_play.get("status", "UNKNOWN"),
            "production_scale_self_play_status": self_play.get("production_scale_status", "UNKNOWN"),
            "cluster_run_profile": cluster.get("run_profile", "immediate_delivery"),
            "client_gpu_boundary": (client_gpu.get("gpu_boundary") or {}).get("full_multi_agent_training"),
        },
        "completion_requirements_for_full_training": [
            "Confirmed dedicated training cluster profile, preferably A100/H100 class GPU capacity.",
            "Long-running multi-agent self-play cycle executed under the approved training plan.",
            "Seed-stability analysis across independent training seeds.",
            "Win-rate, EV, action-distribution, and human-likeness confidence intervals.",
            "Regression comparison against the current deployed strategy stack.",
            "Explicit promotion review before changing the full-training status to completed.",
        ],
        "approval_boundary": {
            "current_delivery_allowed": delivery_validation_status == "PASS",
            "full_training_claim_allowed": False,
            "false_completion_guard": "ENABLED",
            "non_override_rule": (
                "Acceptance training cannot be used to claim completion of full production-scale "
                "multi-agent training."
            ),
        },
    }
    payload["invariants"] = validate_multi_agent_training_status(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_multi_agent_training_status(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    boundary = payload.get("training_boundary") or {}
    approval = payload.get("approval_boundary") or {}

    if boundary.get("delivery_validation_status") == "PASS":
        if boundary.get("acceptance_training_sufficient_for_delivery") is not True:
            violations.append("Passing delivery validation must mark acceptance training sufficient for delivery.")
        if boundary.get("production_blocker") is not False:
            violations.append("Current acceptance training boundary must not create a production blocker.")

    full_status = boundary.get("full_production_scale_multi_agent_training_status")
    if full_status != FULL_TRAINING_STATUS_NOT_COMPLETED:
        violations.append("Full production-scale multi-agent training must remain NOT_COMPLETED until a separate hardening run is executed.")
    if boundary.get("full_long_running_self_play_completed") is not False:
        violations.append("Long-running self-play cannot be marked completed by the acceptance run.")
    if boundary.get("scope_completed") != DELIVERY_VALIDATION_SCOPE:
        violations.append("Completed scope must remain delivery_validation_only.")
    if boundary.get("scope_deferred") != PRODUCTION_HARDENING_SCOPE:
        violations.append("Deferred scope must remain production_hardening.")
    if approval.get("full_training_claim_allowed") is not False:
        violations.append("Full-training claims must be blocked until the hardening run has evidence.")
    if approval.get("false_completion_guard") != "ENABLED":
        violations.append("False completion guard must remain enabled.")

    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def write_multi_agent_training_status(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_multi_agent_training_status(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_multi_agent_training_status_markdown(payload), encoding="utf-8")
    return payload


def render_multi_agent_training_status_markdown(payload: dict[str, Any]) -> str:
    boundary = payload["training_boundary"]
    evidence = payload["evidence"]
    metrics = evidence.get("acceptance_metrics") or {}
    lines = [
        "# Multi-Agent Training Status",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Current Boundary",
        "",
        f"- Acceptance training status: `{boundary['acceptance_training_status']}`",
        f"- Delivery validation status: `{boundary['delivery_validation_status']}`",
        f"- Acceptance training sufficient for delivery: `{boundary['acceptance_training_sufficient_for_delivery']}`",
        f"- Full production-scale multi-agent training: `{boundary['full_production_scale_multi_agent_training_status']}`",
        f"- Full long-running self-play completed: `{boundary['full_long_running_self_play_completed']}`",
        f"- Production blocker: `{boundary['production_blocker']}`",
        "",
        "## Acceptance Metrics",
        "",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- Macro F1: `{metrics.get('macro_f1')}`",
        f"- Balanced accuracy: `{metrics.get('balanced_accuracy')}`",
        f"- Cross entropy: `{metrics.get('cross_entropy')}`",
        "",
        "## Full Training Completion Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["completion_requirements_for_full_training"])
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            payload["approval_boundary"]["non_override_rule"],
            "",
            f"Invariant status: `{payload['invariants']['status']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
