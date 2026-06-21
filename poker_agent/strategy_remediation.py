from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_strategy_remediation(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    deployed = _read_json(reports / "deployed_strategy_gate.json")
    raw_gate = _read_json(reports / "production_gate.json")
    delivery = _read_json(reports / "delivery_readiness.json")
    blocking_items = list(deployed.get("blocking_items", []))
    component_risks = list(deployed.get("component_risks", []))
    if raw_gate.get("status") != "PASS" and deployed.get("status") != "PASS":
        blocking_items.append(
            {
                "id": "production_model_gate",
                "severity": "critical",
                "evidence": f"production_gate={raw_gate.get('status', 'MISSING')}",
                "required_fix": "Clear the raw or deployed strategy gate before production strategy approval.",
            }
        )
    approved = not blocking_items and deployed.get("status") == "PASS"
    return {
        "version": "2026-06-20",
        "service_delivery_status": delivery.get("service_delivery_status", "UNKNOWN"),
        "delivery_overall_status": delivery.get("overall_status", "UNKNOWN"),
        "strategy_policy_status": "APPROVED" if approved else "NOT_APPROVED",
        "production_claim_allowed": approved,
        "release_mode": "production_policy" if approved else "technical_handoff_only",
        "approval_boundary": deployed.get("approval_boundary", {}),
        "approval_invariants": deployed.get("approval_invariants", {}),
        "client_message": (
            "Deployed strategy stack is approved for production rollout with monitoring. "
            "Standalone raw supervised model risk remains separately reported."
            if approved
            else "Strategy production approval is blocked by measurable gates."
        ),
        "blocking_items": blocking_items,
        "component_risks": component_risks,
    }


def write_strategy_remediation(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_strategy_remediation(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strategy Remediation",
        "",
        f"- Strategy policy status: `{payload['strategy_policy_status']}`",
        f"- Release mode: `{payload['release_mode']}`",
        "",
        "## Blocking Items",
        "",
    ]
    if payload["blocking_items"]:
        for item in payload["blocking_items"]:
            lines.append(f"- `{item['id']}`: {item['evidence']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Component Risks", ""])
    if payload["component_risks"]:
        for risk in payload["component_risks"]:
            lines.append(f"- `{risk['component']}`: {risk['evidence']}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
