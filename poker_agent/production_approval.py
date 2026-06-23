from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.approval_boundary import build_approval_boundary


APPROVAL_VERSION = "2026-06-22"


def build_production_approval(project_root: Path) -> dict[str, Any]:
    boundary = build_approval_boundary(project_root)["boundary"]
    delivery_ready = boundary["service_delivery"] == "READY"
    deployed_approved = boundary["deployed_strategy_stack"] == "APPROVED"
    deployment_blockers = int(boundary["deployment_blockers"])
    component_risks = int(boundary["component_risks"])

    status = _approval_status(delivery_ready, deployed_approved, deployment_blockers, component_risks)
    return {
        "version": APPROVAL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": status,
        "delivery_ready": delivery_ready,
        "deployed_strategy_stack": {
            "status": "APPROVED" if deployed_approved else "NOT_APPROVED",
            "source": "reports/deployed_strategy_gate.json",
        },
        "raw_supervised_model": {
            "runtime_status": boundary["raw_supervised_model_runtime"],
            "standalone_status": boundary["raw_supervised_model_standalone"],
            "raw_production_gate": boundary["raw_production_gate"],
            "source": "reports/production_gate.json",
        },
        "risk_position": {
            "deployment_blockers": deployment_blockers,
            "component_risks": component_risks,
            "component_risk_is_production_blocker": deployment_blockers > 0,
            "source": "reports/model_risk_register.json",
        },
        "approval_boundary": {
            "source": "/approval-boundary.json",
            "release_status": boundary["release_status"],
            "production_blocker": boundary["production_blocker"],
            "component_risk": boundary["component_risk"],
        },
        "approval_claims": {
            "allowed": [
                "The service delivery package is ready for production-policy rollout.",
                "The deployed strategy stack is approved as a composed runtime policy.",
                "The raw supervised model is loadable and usable only inside the approved deployed stack.",
                "The remaining raw-model weakness is tracked as a component risk, not a production blocker.",
            ],
            "not_allowed": [
                "The raw supervised artifact is not approved as a standalone production poker policy.",
                "The raw production gate must not be converted into a false pass.",
                "Deployed-stack approval must not be presented as standalone raw-model approval.",
            ],
        },
        "release_decision": _release_decision(status, component_risks),
        "next_required_milestone": {
            "name": "standalone supervised challenger",
            "objective": "Train a challenger artifact that passes the raw production gate while preserving the deployed-stack approval boundary.",
            "gate": "reports/production_gate.json",
        },
    }


def write_production_approval(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_production_approval(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_production_approval_markdown(payload), encoding="utf-8")
    return payload


def render_production_approval_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Production Approval Contract",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Delivery ready: `{payload['delivery_ready']}`",
        f"- Deployed strategy stack: `{payload['deployed_strategy_stack']['status']}`",
        f"- Raw supervised model: `{payload['raw_supervised_model']['standalone_status']}`",
        f"- Raw artifact runtime: `{payload['raw_supervised_model']['runtime_status']}`",
        f"- Deployment blockers: `{payload['risk_position']['deployment_blockers']}`",
        f"- Component risks: `{payload['risk_position']['component_risks']}`",
        "",
        "## Allowed Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in payload["approval_claims"]["allowed"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["approval_claims"]["not_allowed"])
    lines.extend(
        [
            "",
            "## Release Decision",
            "",
            payload["release_decision"],
            "",
            "## Next Required Milestone",
            "",
            f"- {payload['next_required_milestone']['objective']}",
            "",
        ]
    )
    return "\n".join(lines)


def _approval_status(
    delivery_ready: bool,
    deployed_approved: bool,
    deployment_blockers: int,
    component_risks: int,
) -> str:
    if not delivery_ready or not deployed_approved or deployment_blockers:
        return "NOT_APPROVED"
    if component_risks:
        return "APPROVED_WITH_COMPONENT_RISK"
    return "APPROVED"


def _release_decision(status: str, component_risks: int) -> str:
    if status == "APPROVED_WITH_COMPONENT_RISK":
        return (
            "Release can proceed for the deployed strategy stack. The raw supervised model is loadable but remains "
            f"bounded by {component_risks} tracked component risk until a challenger clears the raw production gate."
        )
    if status == "APPROVED":
        return "Release can proceed without open production-blocking model risks."
    return "Release is not approved until delivery, deployed-stack, and deployment-blocker checks pass."
