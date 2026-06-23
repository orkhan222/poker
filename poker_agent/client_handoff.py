from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.approval_boundary import build_approval_boundary


HANDOFF_VERSION = "2026-06-22"


def build_client_handoff(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    boundary_payload = build_approval_boundary(project_root)
    boundary = boundary_payload["boundary"]

    return {
        "version": HANDOFF_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "handoff_status": boundary["release_status"],
        "client_statement": (
            "The service and deployed strategy stack are ready for delivery. The raw supervised model is "
            "loadable and integrated into the service, but it is not approved as a standalone production policy. "
            "This is tracked as an official component risk and is not a production blocker for the approved "
            "deployed strategy stack."
        ),
        "technical_position": {
            "service_delivery": boundary["service_delivery"],
            "deployed_strategy_stack": boundary["deployed_strategy_stack"],
            "raw_supervised_model_runtime": boundary["raw_supervised_model_runtime"],
            "raw_supervised_model_standalone": boundary["raw_supervised_model_standalone"],
            "raw_production_gate": boundary["raw_production_gate"],
            "production_blocker": boundary["production_blocker"],
            "component_risk": boundary["component_risk"],
            "deployment_blockers": boundary["deployment_blockers"],
            "component_risks": boundary["component_risks"],
        },
        "evidence": {
            "delivery_verification": "reports/delivery_verification.json",
            "delivery_readiness": "reports/delivery_readiness.json",
            "production_approval": "reports/production_approval.json",
            "model_risk_register": "reports/model_risk_register.json",
            "production_gate": "reports/production_gate.json",
            "approval_boundary": "/approval-boundary.json",
        },
        "allowed_external_claims": [
            "Service delivery is ready.",
            "The deployed strategy stack is approved for the delivered runtime boundary.",
            "The raw supervised model is loadable and integrated into the service.",
            "The raw supervised model limitation is an official component risk, not a deployment blocker.",
        ],
        "disallowed_external_claims": [
            "The raw supervised model is a standalone production-approved poker policy.",
            "The raw production gate passed if reports/production_gate.json says FAIL.",
            "The deployed strategy approval automatically approves every internal component independently.",
        ],
        "next_engineering_milestone": {
            "name": "standalone supervised challenger",
            "objective": (
                "Train and validate a stronger raw supervised artifact that can pass the production gate "
                "without relying on deployed-stack safeguards."
            ),
            "acceptance_gate": "reports/production_gate.json",
        },
    }


def write_client_handoff(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_client_handoff(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_client_handoff_markdown(payload), encoding="utf-8")
    return payload


def render_client_handoff_markdown(payload: dict[str, Any]) -> str:
    position = payload["technical_position"]
    lines = [
        "# Client Handoff Statement",
        "",
        f"- Handoff status: `{payload['handoff_status']}`",
        f"- Service delivery: `{position['service_delivery']}`",
        f"- Deployed strategy stack: `{position['deployed_strategy_stack']}`",
        f"- Raw supervised model runtime: `{position['raw_supervised_model_runtime']}`",
        f"- Raw supervised model standalone status: `{position['raw_supervised_model_standalone']}`",
        f"- Production blocker: `{position['production_blocker']}`",
        f"- Component risk: `{position['component_risk']}`",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Allowed External Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in payload["allowed_external_claims"])
    lines.extend(["", "## Disallowed External Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["disallowed_external_claims"])
    lines.extend(
        [
            "",
            "## Evidence",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{path}`" for name, path in payload["evidence"].items())
    lines.extend(
        [
            "",
            "## Next Engineering Milestone",
            "",
            f"- {payload['next_engineering_milestone']['objective']}",
            "",
        ]
    )
    return "\n".join(lines)
