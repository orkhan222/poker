from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_raw_strategy_readiness(report: dict[str, Any]) -> dict[str, Any]:
    passed = str(report.get("status")).upper() == "PASS"
    failing = [gate for gate in report.get("gates", []) if not gate.get("passed")]
    return {
        "strategy_policy_status": "APPROVED" if passed and not failing else "NOT_APPROVED",
        "production_gate_status": "PASS" if passed and not failing else "FAIL",
        "raw_production_gate_status": report.get("status", "MISSING"),
        "deployment_mode": "production_policy" if passed and not failing else "technical_handoff_only",
        "metric_snapshot": report.get("valid_metrics", {}),
        "blocking_reasons": [
            {
                "gate": gate.get("name", "unknown"),
                "observed": gate.get("observed"),
                "threshold": gate.get("threshold"),
                "impact": gate.get("impact"),
            }
            for gate in failing
        ],
        "component_risks": [],
        "recommended_next_milestone": {
            "name": "raw production model gate",
            "objective": "Clear the standalone supervised model gate.",
        },
    }


def summarize_deployed_strategy_readiness(report: dict[str, Any]) -> dict[str, Any]:
    approved = str(report.get("status")).upper() == "PASS"
    return {
        "strategy_policy_status": "APPROVED" if approved else "NOT_APPROVED",
        "production_gate_status": "PASS" if approved else "FAIL",
        "deployed_strategy_gate_status": report.get("status", "MISSING"),
        "raw_production_gate_status": report.get("raw_supervised_model_gate_status", "MISSING"),
        "raw_supervised_model_status": report.get("raw_supervised_model_status", "MISSING"),
        "deployment_mode": "production_policy" if approved else "technical_handoff_only",
        "decision": report.get("decision", ""),
        "approval_boundary": report.get("approval_boundary", {}),
        "approval_invariants": report.get("approval_invariants", {}),
        "production_claim_allowed": report.get("production_claim_allowed", approved),
        "metric_snapshot": report.get("metric_snapshot", {}),
        "blocking_reasons": report.get("blocking_items", []),
        "component_risks": report.get("component_risks", []),
        "recommended_next_milestone": report.get("recommended_next_milestone", {}),
    }


def load_strategy_readiness(raw_gate_path: Path) -> dict[str, Any]:
    if not raw_gate_path.exists():
        return {
            "strategy_policy_status": "UNKNOWN",
            "production_gate_status": "MISSING_REPORT",
            "deployment_mode": "unavailable",
            "blocking_reasons": [],
            "component_risks": [],
        }
    return summarize_raw_strategy_readiness(json.loads(raw_gate_path.read_text(encoding="utf-8")))


def load_combined_strategy_readiness(raw_gate_path: Path, deployed_gate_path: Path | None = None) -> dict[str, Any]:
    if deployed_gate_path is not None and deployed_gate_path.exists():
        return summarize_deployed_strategy_readiness(json.loads(deployed_gate_path.read_text(encoding="utf-8")))
    return load_strategy_readiness(raw_gate_path)
