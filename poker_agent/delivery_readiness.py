from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_agent.strategy_readiness import load_combined_strategy_readiness


def summarize_delivery_readiness(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    verification = _read_json(reports / "delivery_verification.json")
    hygiene = _read_json(reports / "repo_hygiene.json")
    remediation = _read_json(reports / "strategy_remediation.json")
    scope_contract = _read_json(reports / "scope_contract.json")
    model_risk_register = _read_json(reports / "model_risk_register.json")
    strategy = load_combined_strategy_readiness(
        reports / "production_gate.json",
        reports / "deployed_strategy_gate.json",
    )
    service_ready = verification.get("status") == "PASS" and hygiene.get("status") == "PASS"
    strategy_approved = strategy.get("strategy_policy_status") == "APPROVED"
    return {
        "version": "2026-06-20",
        "overall_status": _overall_status(service_ready, strategy_approved),
        "service_delivery_status": "READY" if service_ready else "NOT_READY",
        "strategy_policy_status": strategy.get("strategy_policy_status", "UNKNOWN"),
        "deployment_mode": (
            "production_policy"
            if service_ready and strategy_approved
            else "technical_handoff_only"
            if service_ready
            else "not_ready"
        ),
        "client_message": _client_message(service_ready, strategy_approved),
        "service_evidence": {
            "delivery_verification": verification.get("status", "MISSING"),
            "repo_hygiene": hygiene.get("status", "MISSING"),
            "scope_contract": scope_contract.get("overall_status", "MISSING"),
            "model_risk_register": model_risk_register.get("overall_status", "MISSING"),
        },
        "strategy_evidence": {
            "deployed_strategy_gate_status": strategy.get("deployed_strategy_gate_status"),
            "raw_production_gate_status": strategy.get("raw_production_gate_status"),
            "raw_supervised_model_status": strategy.get("raw_supervised_model_status"),
            "strategy_remediation_status": remediation.get("strategy_policy_status", "MISSING"),
            "strategy_remediation_blockers": len(remediation.get("blocking_items", [])),
            "approval_boundary": strategy.get("approval_boundary", {}),
            "approval_invariants": strategy.get("approval_invariants", {}),
            "metric_snapshot": strategy.get("metric_snapshot", {}),
            "blocking_reasons": strategy.get("blocking_reasons", []),
            "component_risks": strategy.get("component_risks", []),
            "model_risk_summary": model_risk_register.get("risk_summary", {}),
        },
    }


def write_delivery_readiness(project_root: Path, out_path: Path) -> dict[str, Any]:
    payload = summarize_delivery_readiness(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _overall_status(service_ready: bool, strategy_approved: bool) -> str:
    if service_ready and strategy_approved:
        return "READY_FOR_PRODUCTION_POLICY"
    if service_ready:
        return "READY_FOR_TECHNICAL_HANDOFF"
    return "NOT_READY_FOR_HANDOFF"


def _client_message(service_ready: bool, strategy_approved: bool) -> str:
    if service_ready and strategy_approved:
        return "The service and deployed strategy stack are approved for production rollout with monitoring."
    if service_ready:
        return "The service is ready, but strategy production approval is not complete."
    return "The service is not ready for handoff until delivery checks pass."
