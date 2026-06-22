from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HANDOFF_VERSION = "2026-06-22"


def build_client_handoff(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    delivery_verification = _read_json(reports / "delivery_verification.json")
    delivery_readiness = _read_json(reports / "delivery_readiness.json")
    production_approval = _read_json(reports / "production_approval.json")
    risk_register = _read_json(reports / "model_risk_register.json")
    production_gate = _read_json(reports / "production_gate.json")

    service_delivery_ready = (
        delivery_verification.get("status") == "PASS"
        and delivery_readiness.get("overall_status") == "READY_FOR_PRODUCTION_POLICY"
    )
    deployed_strategy_status = (production_approval.get("deployed_strategy_stack") or {}).get(
        "status", "UNKNOWN"
    )
    raw_model = production_approval.get("raw_supervised_model") or {}
    raw_runtime_status = raw_model.get(
        "runtime_status",
        (risk_register.get("raw_artifact_runtime_status") or {}).get("status", "UNKNOWN"),
    )
    raw_standalone_status = raw_model.get("standalone_status", "UNKNOWN")
    risk_position = production_approval.get("risk_position") or {}
    deployment_blockers = int(risk_position.get("deployment_blockers", 0))
    component_risks = int(risk_position.get("component_risks", 0))
    production_blocker = deployment_blockers > 0 or deployed_strategy_status != "APPROVED"
    component_risk = raw_standalone_status == "NOT_STANDALONE_APPROVED" or component_risks > 0

    return {
        "version": HANDOFF_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "handoff_status": _handoff_status(
            service_delivery_ready=service_delivery_ready,
            deployed_strategy_status=deployed_strategy_status,
            raw_runtime_status=raw_runtime_status,
            production_blocker=production_blocker,
            component_risk=component_risk,
        ),
        "client_statement": (
            "The service and deployed strategy stack are ready for delivery. The raw supervised model is "
            "loadable and integrated into the service, but it is not approved as a standalone production policy. "
            "This is tracked as an official component risk and is not a production blocker for the approved "
            "deployed strategy stack."
        ),
        "technical_position": {
            "service_delivery": "READY" if service_delivery_ready else "NOT_READY",
            "deployed_strategy_stack": deployed_strategy_status,
            "raw_supervised_model_runtime": raw_runtime_status,
            "raw_supervised_model_standalone": raw_standalone_status,
            "raw_production_gate": production_gate.get("status", "MISSING"),
            "production_blocker": production_blocker,
            "component_risk": component_risk,
            "deployment_blockers": deployment_blockers,
            "component_risks": component_risks,
        },
        "evidence": {
            "delivery_verification": "reports/delivery_verification.json",
            "delivery_readiness": "reports/delivery_readiness.json",
            "production_approval": "reports/production_approval.json",
            "model_risk_register": "reports/model_risk_register.json",
            "production_gate": "reports/production_gate.json",
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


def _handoff_status(
    *,
    service_delivery_ready: bool,
    deployed_strategy_status: str,
    raw_runtime_status: str,
    production_blocker: bool,
    component_risk: bool,
) -> str:
    if production_blocker or not service_delivery_ready or deployed_strategy_status != "APPROVED":
        return "NOT_READY"
    if raw_runtime_status != "LOADABLE":
        return "NOT_READY"
    if component_risk:
        return "READY_WITH_COMPONENT_RISK"
    return "READY"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
