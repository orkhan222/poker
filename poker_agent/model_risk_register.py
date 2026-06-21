from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTER_VERSION = "2026-06-21"


def build_model_risk_register(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    deployed_gate = _read_json(reports / "deployed_strategy_gate.json")
    production_gate = _read_json(reports / "production_gate.json")
    delivery = _read_json(reports / "delivery_readiness.json")
    raw_artifact = _raw_artifact_status(project_root)

    deployed_approved = deployed_gate.get("strategy_policy_status") == "APPROVED"
    raw_gate_status = str(production_gate.get("status", "MISSING")).upper()
    raw_model_status = deployed_gate.get("raw_supervised_model_status", "UNKNOWN")
    risks = _component_risks(deployed_gate, production_gate, raw_artifact)
    open_risks = [risk for risk in risks if risk["status"] not in {"CLOSED", "ACCEPTED"}]

    return {
        "version": REGISTER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if deployed_approved and risks else "PASS" if deployed_approved else "FAIL",
        "deployed_strategy_stack_status": "APPROVED" if deployed_approved else "NOT_APPROVED",
        "delivery_status": delivery.get("overall_status", "MISSING"),
        "raw_supervised_model_status": raw_model_status,
        "raw_production_gate_status": raw_gate_status,
        "raw_artifact_runtime_status": raw_artifact,
        "approval_boundary": {
            "approved": "The deployed strategy stack is approved only as a composed runtime policy.",
            "not_approved": "The raw supervised artifact is not approved as a standalone production policy unless the raw production gate passes.",
            "non_override_rule": "Deployed-stack approval must not be used to claim standalone raw-model approval.",
        },
        "risk_summary": {
            "total": len(risks),
            "open": len(open_risks),
            "deployment_blockers": sum(1 for risk in open_risks if risk["deployment_blocker"]),
            "component_risks": sum(1 for risk in open_risks if not risk["deployment_blocker"]),
        },
        "risks": risks,
        "challenger_training_plan": _challenger_training_plan(production_gate),
        "release_position": _release_position(deployed_approved, open_risks),
    }


def write_model_risk_register(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_model_risk_register(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_model_risk_register_markdown(payload), encoding="utf-8")
    return payload


def render_model_risk_register_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Model Risk Register",
        "",
        f"- Deployed strategy stack: `{payload['deployed_strategy_stack_status']}`",
        f"- Raw supervised model: `{payload['raw_supervised_model_status']}`",
        f"- Raw production gate: `{payload['raw_production_gate_status']}`",
        f"- Open component risks: `{payload['risk_summary']['component_risks']}`",
        f"- Deployment blockers: `{payload['risk_summary']['deployment_blockers']}`",
        "",
        "## Approval Boundary",
        "",
        f"- Approved: {payload['approval_boundary']['approved']}",
        f"- Not approved: {payload['approval_boundary']['not_approved']}",
        f"- Rule: {payload['approval_boundary']['non_override_rule']}",
        "",
        "## Open Risks",
        "",
    ]
    if payload["risks"]:
        for risk in payload["risks"]:
            lines.extend(
                [
                    f"### {risk['id']}",
                    "",
                    f"- Severity: `{risk['severity']}`",
                    f"- Owner: `{risk['owner']}`",
                    f"- Deployment blocker: `{risk['deployment_blocker']}`",
                    f"- Scope: `{risk['blocking_scope']}`",
                    f"- Evidence: {risk['evidence']}",
                    f"- Mitigation: {risk['mitigation']}",
                    f"- Acceptance criteria: {risk['acceptance_criteria']}",
                    "",
                ]
            )
    else:
        lines.append("- none")
        lines.append("")
    plan = payload["challenger_training_plan"]
    lines.extend(
        [
            "## Challenger Training Plan",
            "",
            f"- Objective: {plan['objective']}",
            f"- Primary gate: `{plan['primary_gate']}`",
            f"- Minimum evidence: {', '.join(plan['minimum_evidence'])}",
            "",
            "## Release Position",
            "",
            payload["release_position"],
            "",
        ]
    )
    return "\n".join(lines)


def _component_risks(
    deployed_gate: dict[str, Any],
    production_gate: dict[str, Any],
    raw_artifact: dict[str, str],
) -> list[dict[str, Any]]:
    raw_status = str(production_gate.get("status", "MISSING")).upper()
    if raw_status == "PASS" and raw_artifact.get("status") == "LOADABLE":
        return []

    reported = deployed_gate.get("component_risks") or []
    evidence = _risk_evidence(reported, production_gate, raw_artifact)
    return [
        {
            "id": "raw_supervised_model_not_standalone_approved",
            "component": "raw_supervised_model_artifact",
            "severity": "high",
            "owner": "modeling",
            "status": "OPEN",
            "deployment_blocker": False,
            "blocking_scope": "standalone_raw_model_only",
            "evidence": evidence,
            "impact": (
                "The deployed strategy stack remains approved, but the raw supervised artifact must not be used, "
                "sold, or documented as an independently production-approved poker policy."
            ),
            "mitigation": (
                "Keep the deployed gated stack in production with monitoring, and train a challenger supervised "
                "artifact under the same production gate before promoting standalone model claims."
            ),
            "acceptance_criteria": (
                "A challenger artifact must pass the raw production gate, including macro-F1, balanced accuracy, "
                "majority-baseline lift, calibration, and held-out action-distribution checks."
            ),
        }
    ]


def _risk_evidence(
    reported: list[dict[str, Any]],
    production_gate: dict[str, Any],
    raw_artifact: dict[str, str],
) -> str:
    if raw_artifact.get("status") == "LOAD_FAILED":
        return f"artifact_load_status=LOAD_FAILED, error={raw_artifact.get('error', '')}"
    if reported:
        evidence = reported[0].get("evidence")
        if evidence:
            return str(evidence)

    readiness = production_gate.get("strategy_readiness") or {}
    reasons = readiness.get("blocking_reasons") or []
    if reasons:
        return ", ".join(
            f"{item.get('gate')}={item.get('observed')}"
            for item in reasons[:6]
            if item.get("gate")
        )
    metrics = production_gate.get("valid_metrics") or {}
    if metrics:
        return ", ".join(
            f"{key}={value}"
            for key, value in sorted(metrics.items())
            if key in {"macro_f1", "balanced_accuracy", "lift_vs_majority", "accuracy"}
        )
    return f"production_gate={production_gate.get('status', 'MISSING')}"


def _raw_artifact_status(project_root: Path) -> dict[str, str]:
    path = project_root / "models" / "poker_policy.joblib"
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    try:
        from poker_agent.model import load_policy

        load_policy(path)
    except Exception as exc:
        return {
            "status": "LOAD_FAILED",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "LOADABLE", "path": str(path)}


def _challenger_training_plan(production_gate: dict[str, Any]) -> dict[str, Any]:
    current_metrics = production_gate.get("valid_metrics") or {}
    return {
        "objective": "Train and validate a stronger standalone supervised policy artifact without weakening the deployed-stack approval boundary.",
        "primary_gate": "reports/production_gate.json",
        "current_metrics": current_metrics,
        "minimum_evidence": [
            "grouped held-out validation split",
            "macro-F1 and balanced-accuracy lift over majority baseline",
            "action distribution similarity",
            "calibration report",
            "production-scale self-play comparison",
            "latency and inference contract check",
        ],
        "candidate_methods": [
            "class-weighted gradient boosting challenger",
            "routed tabular ensemble with calibrated probabilities",
            "sequence-aware policy challenger once clean action histories are available",
        ],
    }


def _release_position(deployed_approved: bool, open_risks: list[dict[str, Any]]) -> str:
    blockers = [risk for risk in open_risks if risk["deployment_blocker"]]
    if deployed_approved and not blockers:
        return (
            "Release can proceed for the deployed strategy stack. The raw supervised model weakness remains "
            "a tracked component risk and must be resolved by a challenger artifact before standalone model approval."
        )
    if deployed_approved:
        return "Release requires mitigation because at least one open model risk is marked as a deployment blocker."
    return "Release should not proceed until the deployed strategy stack is approved."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
