from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATEGY_STACK_MATURITY_VERSION = "2026-06-28"
DEPLOYMENT_APPROVED_WITH_MONITORING = "APPROVED_FOR_DEPLOYMENT_WITH_MONITORING"
NOT_FINAL_ENGINE = "NOT_FINAL_MAXIMALLY_OPTIMIZED_ENGINE"
NOT_DEPLOYMENT_APPROVED = "NOT_APPROVED_FOR_DEPLOYMENT"


def build_strategy_stack_maturity(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    deployed_gate = _read_optional_json(reports / "deployed_strategy_gate.json")
    production_approval = _read_optional_json(reports / "production_approval.json")
    policy_acceptance = _read_optional_json(reports / "policy_acceptance.json")
    self_play = _read_optional_json(reports / "production_self_play.json")
    raw_challenger = _read_optional_json(reports / "raw_model_challenger.json")

    deployed_approved = (
        deployed_gate.get("status") == "PASS"
        and deployed_gate.get("strategy_policy_status") == "APPROVED"
        and production_approval.get("overall_status") in {"APPROVED", "APPROVED_WITH_COMPONENT_RISK"}
    )
    current_status = DEPLOYMENT_APPROVED_WITH_MONITORING if deployed_approved else NOT_DEPLOYMENT_APPROVED

    payload: dict[str, Any] = {
        "version": STRATEGY_STACK_MATURITY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Strategy stack maturity and deployment boundary",
        "client_statement": (
            "The current strategy stack is approved for deployment with monitoring, "
            "not as a final maximally optimized poker engine."
        ),
        "current_strategy_stack": {
            "status": current_status,
            "deployment_mode": "monitored_rollout" if deployed_approved else "blocked",
            "monitoring_required": deployed_approved,
            "rollback_plan_required": deployed_approved,
            "approved_for_client_delivery": deployed_approved,
            "source_reports": [
                "reports/deployed_strategy_gate.json",
                "reports/production_approval.json",
                "reports/policy_acceptance.json",
                "reports/production_self_play.json",
            ],
        },
        "final_engine_boundary": {
            "status": NOT_FINAL_ENGINE,
            "final_engine_claim_allowed": False,
            "maximally_optimized_claim_allowed": False,
            "autonomous_profitable_engine_claim_allowed": False,
            "reason": (
                "The deployed stack has passed delivery and strategy-stack gates, but long-horizon optimization, "
                "standalone raw-model promotion, broader opponent modeling, and extended production hardening remain "
                "future work."
            ),
        },
        "evidence": {
            "deployed_strategy_gate_status": deployed_gate.get("status", "UNKNOWN"),
            "strategy_policy_status": deployed_gate.get("strategy_policy_status", "UNKNOWN"),
            "production_approval_status": production_approval.get("overall_status", "UNKNOWN"),
            "human_action_alignment_status": policy_acceptance.get("human_action_alignment_status", "UNKNOWN"),
            "human_likeness_status": (policy_acceptance.get("human_likeness") or {}).get("status", "UNKNOWN"),
            "production_self_play_status": self_play.get("status", "UNKNOWN"),
            "production_scale_self_play_status": self_play.get("production_scale_status", "UNKNOWN"),
            "raw_challenger_standalone_status": raw_challenger.get("standalone_status", "UNKNOWN"),
        },
        "allowed_claims": [
            "The service and deployed strategy stack are ready for monitored deployment.",
            "The stack has passed the current deployment and acceptance gates.",
            "The system must be operated with monitoring, rollback, and continued model evaluation.",
        ],
        "not_allowed_claims": [
            "The system is a final maximally optimized poker engine.",
            "The deployed strategy stack is guaranteed to be strategically optimal.",
            "The raw supervised model is a standalone production-approved policy.",
            "Production hardening and long-running optimization are complete.",
        ],
        "next_optimization_milestones": [
            "Run extended multi-agent training and seed-stability analysis under the approved cluster profile.",
            "Promote a standalone challenger model only after it clears the raw production model gate.",
            "Expand opponent modeling, bet-sizing calibration, and long-horizon self-play evaluation.",
            "Keep production monitoring active for action distribution, EV drift, latency, and rollback triggers.",
        ],
    }
    payload["invariants"] = validate_strategy_stack_maturity(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_strategy_stack_maturity(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    current = payload.get("current_strategy_stack") or {}
    final_boundary = payload.get("final_engine_boundary") or {}

    if current.get("status") == DEPLOYMENT_APPROVED_WITH_MONITORING:
        if current.get("monitoring_required") is not True:
            violations.append("Approved deployment must require monitoring.")
        if current.get("rollback_plan_required") is not True:
            violations.append("Approved deployment must require a rollback plan.")
        if current.get("deployment_mode") != "monitored_rollout":
            violations.append("Approved deployment must be represented as a monitored rollout.")

    if final_boundary.get("status") != NOT_FINAL_ENGINE:
        violations.append("Strategy stack must not be represented as a final maximally optimized poker engine.")
    if final_boundary.get("final_engine_claim_allowed") is not False:
        violations.append("Final-engine claims must remain blocked.")
    if final_boundary.get("maximally_optimized_claim_allowed") is not False:
        violations.append("Maximally optimized claims must remain blocked.")
    if final_boundary.get("autonomous_profitable_engine_claim_allowed") is not False:
        violations.append("Autonomous profitable-engine claims must remain blocked.")

    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def write_strategy_stack_maturity(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_strategy_stack_maturity(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_strategy_stack_maturity_markdown(payload), encoding="utf-8")
    return payload


def render_strategy_stack_maturity_markdown(payload: dict[str, Any]) -> str:
    current = payload["current_strategy_stack"]
    final_boundary = payload["final_engine_boundary"]
    evidence = payload["evidence"]
    lines = [
        "# Strategy Stack Maturity Contract",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Deployment Boundary",
        "",
        f"- Current strategy stack: `{current['status']}`",
        f"- Deployment mode: `{current['deployment_mode']}`",
        f"- Monitoring required: `{current['monitoring_required']}`",
        f"- Rollback plan required: `{current['rollback_plan_required']}`",
        f"- Final engine status: `{final_boundary['status']}`",
        f"- Final engine claim allowed: `{final_boundary['final_engine_claim_allowed']}`",
        f"- Maximally optimized claim allowed: `{final_boundary['maximally_optimized_claim_allowed']}`",
        "",
        "## Evidence",
        "",
        f"- Deployed strategy gate: `{evidence.get('deployed_strategy_gate_status')}`",
        f"- Strategy policy status: `{evidence.get('strategy_policy_status')}`",
        f"- Production approval: `{evidence.get('production_approval_status')}`",
        f"- Human action alignment: `{evidence.get('human_action_alignment_status')}`",
        f"- Human likeness: `{evidence.get('human_likeness_status')}`",
        f"- Production-scale self-play: `{evidence.get('production_scale_self_play_status')}`",
        "",
        "## Not Allowed Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", "## Next Optimization Milestones", ""])
    lines.extend(f"- {item}" for item in payload["next_optimization_milestones"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
