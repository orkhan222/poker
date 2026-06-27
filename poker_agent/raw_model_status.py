from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_MODEL_STATUS_VERSION = "2026-06-27"
RAW_MODEL_COMPONENT = "raw_supervised_model_artifact"
NOT_STANDALONE_APPROVED = "NOT_STANDALONE_APPROVED"
STANDALONE_APPROVED = "STANDALONE_APPROVED"
LOADABLE = "LOADABLE"


def build_raw_model_status(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    reports_dir = project_root / "reports"
    model_path = project_root / "models" / "poker_policy.joblib"

    production_gate = _read_optional_json(reports_dir / "production_gate.json")
    risk_register = _read_optional_json(reports_dir / "model_risk_register.json")
    production_approval = _read_optional_json(reports_dir / "production_approval.json")
    deployed_gate = _read_optional_json(reports_dir / "deployed_strategy_gate.json")
    challenger = _read_optional_json(reports_dir / "raw_model_challenger.json")

    runtime_status = _runtime_status(model_path, risk_register)
    gate_status = _first_non_empty(
        risk_register.get("raw_production_gate_status"),
        production_gate.get("status"),
        "UNKNOWN",
    )
    standalone_status = _first_non_empty(
        risk_register.get("raw_supervised_model_status"),
        (production_approval.get("raw_supervised_model") or {}).get("standalone_status"),
        STANDALONE_APPROVED if gate_status == "PASS" else NOT_STANDALONE_APPROVED,
    )

    risk_summary = risk_register.get("risk_summary") or {}
    deployment_blockers = int(risk_summary.get("deployment_blockers") or 0)
    component_risks = int(risk_summary.get("component_risks") or len(risk_register.get("risks") or []))
    approved_as_standalone = gate_status == "PASS" and standalone_status == STANDALONE_APPROVED
    service_loadable = runtime_status == LOADABLE
    failed_gates = _failed_gates(production_gate)
    metrics = production_gate.get("valid_metrics") or {}

    payload = {
        "version": RAW_MODEL_STATUS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS",
        "client_statement": (
            "The raw supervised model is loadable and works inside the service, but it is not approved "
            "as a standalone production poker policy. Its standalone quality gate still fails, so the "
            "limitation is tracked as a component risk rather than hidden or converted into a false PASS."
        ),
        "raw_supervised_model": {
            "component": RAW_MODEL_COMPONENT,
            "artifact_path": str(model_path),
            "runtime_status": runtime_status,
            "service_loadable": service_loadable,
            "quality_gate_status": gate_status,
            "standalone_status": standalone_status,
            "approved_as_standalone_policy": approved_as_standalone,
        },
        "release_boundary": {
            "service_delivery_allowed": deployment_blockers == 0,
            "deployed_strategy_stack_status": _first_non_empty(
                risk_register.get("deployed_strategy_stack_status"),
                deployed_gate.get("strategy_policy_status"),
                deployed_gate.get("status"),
                "UNKNOWN",
            ),
            "component_risk": not approved_as_standalone,
            "production_blocker": deployment_blockers > 0,
            "approved_usage": "May be loaded and used inside the approved deployed strategy stack.",
            "prohibited_usage": "Must not be sold, documented, or represented as a standalone production-approved poker policy.",
            "non_override_rule": "Deployed-stack approval must not be used to claim standalone raw-model approval.",
        },
        "quality_evidence": {
            "gate_report": "reports/production_gate.json",
            "risk_report": "reports/model_risk_register.json",
            "challenger_report": "reports/raw_model_challenger.json" if challenger else None,
            "failed_gates": failed_gates,
            "metrics": {
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
                "lift_vs_majority": metrics.get("lift_vs_majority"),
                "ece_10": metrics.get("ece_10"),
            },
            "best_challenger": _best_challenger_summary(challenger),
        },
        "next_step": {
            "name": "standalone supervised challenger hardening",
            "objective": "Close the remaining challenger gate failures, especially macro F1, calibration, observed-card and facing-bet slices, and dataset audit blockers.",
            "acceptance_rule": "Standalone approval is allowed only after production_gate.status and raw_model_challenger.best_candidate.gate.status both become PASS.",
        },
    }
    payload["invariants"] = {
        "status": "PASS" if not validate_raw_model_status(payload) else "FAIL",
        "violations": validate_raw_model_status(payload),
    }
    return payload


def write_raw_model_status(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_raw_model_status(project_root)
    assert_raw_model_status(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_raw_model_status_markdown(payload), encoding="utf-8")
    return payload


def render_raw_model_status_markdown(payload: dict[str, Any]) -> str:
    raw = payload["raw_supervised_model"]
    boundary = payload["release_boundary"]
    metrics = payload["quality_evidence"]["metrics"]
    failed_gates = payload["quality_evidence"].get("failed_gates") or []
    challenger = payload["quality_evidence"].get("best_challenger") or {}
    lines = [
        "# Raw Supervised Model Status",
        "",
        "## Client Statement",
        "",
        payload["client_statement"],
        "",
        "## Status",
        "",
        f"- Runtime status: `{raw['runtime_status']}`",
        f"- Service loadable: `{raw['service_loadable']}`",
        f"- Raw production gate: `{raw['quality_gate_status']}`",
        f"- Standalone status: `{raw['standalone_status']}`",
        f"- Approved as standalone policy: `{raw['approved_as_standalone_policy']}`",
        f"- Component risk: `{boundary['component_risk']}`",
        f"- Production blocker: `{boundary['production_blocker']}`",
        "",
        "## Quality Evidence",
        "",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- Macro F1: `{metrics.get('macro_f1')}`",
        f"- Balanced accuracy: `{metrics.get('balanced_accuracy')}`",
        f"- Majority baseline accuracy: `{metrics.get('majority_baseline_accuracy')}`",
        f"- Lift vs majority: `{metrics.get('lift_vs_majority')}`",
        f"- Failed gates: `{', '.join(failed_gates) if failed_gates else 'none'}`",
        "",
        "## Best Challenger",
        "",
        f"- Candidate: `{challenger.get('name')}`",
        f"- Gate status: `{challenger.get('gate_status')}`",
        f"- Standalone status: `{challenger.get('standalone_status')}`",
        f"- Accuracy: `{(challenger.get('metrics') or {}).get('accuracy')}`",
        f"- Macro F1: `{(challenger.get('metrics') or {}).get('macro_f1')}`",
        f"- Balanced accuracy: `{(challenger.get('metrics') or {}).get('balanced_accuracy')}`",
        f"- Failed challenger gates: `{', '.join(challenger.get('failed_gates') or []) if challenger else 'not run'}`",
        "",
        "## Boundary",
        "",
        f"- Approved usage: {boundary['approved_usage']}",
        f"- Prohibited usage: {boundary['prohibited_usage']}",
        f"- Rule: {boundary['non_override_rule']}",
        "",
        "## Next Step",
        "",
        f"- {payload['next_step']['objective']}",
    ]
    return "\n".join(lines) + "\n"


def validate_raw_model_status(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    raw = payload.get("raw_supervised_model") or {}
    boundary = payload.get("release_boundary") or {}
    gate_status = raw.get("quality_gate_status")
    standalone_status = raw.get("standalone_status")

    if gate_status != "PASS" and raw.get("approved_as_standalone_policy"):
        violations.append("raw_model_cannot_be_standalone_approved_when_quality_gate_fails")
    if gate_status != "PASS" and standalone_status == STANDALONE_APPROVED:
        violations.append("standalone_status_cannot_be_approved_when_quality_gate_fails")
    if raw.get("runtime_status") == LOADABLE and raw.get("service_loadable") is not True:
        violations.append("loadable_runtime_must_mark_service_loadable_true")
    if standalone_status == NOT_STANDALONE_APPROVED and boundary.get("component_risk") is not True:
        violations.append("not_standalone_approved_must_be_tracked_as_component_risk")
    if boundary.get("production_blocker") and boundary.get("service_delivery_allowed"):
        violations.append("production_blocker_cannot_allow_service_delivery")
    return violations


def assert_raw_model_status(payload: dict[str, Any]) -> None:
    violations = validate_raw_model_status(payload)
    if violations:
        raise ValueError(f"Invalid raw model status contract: {violations}")


def _best_challenger_summary(challenger: dict[str, Any]) -> dict[str, Any] | None:
    if not challenger:
        return None
    best = challenger.get("best_candidate") or {}
    gate = best.get("gate") or {}
    metrics = best.get("valid_metrics") or {}
    return {
        "name": best.get("name"),
        "standalone_status": challenger.get("standalone_status"),
        "approved_as_standalone_policy": challenger.get("approved_as_standalone_policy"),
        "gate_status": gate.get("status"),
        "failed_gates": gate.get("failed_gates"),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
            "lift_vs_majority": metrics.get("lift_vs_majority"),
            "ece_10": metrics.get("ece_10"),
        },
    }


def _runtime_status(model_path: Path, risk_register: dict[str, Any]) -> str:
    reported = (risk_register.get("raw_artifact_runtime_status") or {}).get("status")
    if reported:
        return str(reported)
    return LOADABLE if model_path.exists() else "MISSING"


def _failed_gates(production_gate: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    for gate in production_gate.get("gates") or []:
        if gate.get("passed") is False:
            failed.append(str(gate.get("name")))
    return failed


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None
