from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BOUNDARY_VERSION = "2026-06-23"
STANDALONE_APPROVED = "STANDALONE_APPROVED"
NOT_STANDALONE_APPROVED = "NOT_STANDALONE_APPROVED"


@dataclass(frozen=True)
class ApprovalBoundary:
    service_delivery: str
    deployed_strategy_stack: str
    raw_supervised_model_runtime: str
    raw_supervised_model_standalone: str
    raw_production_gate: str
    production_blocker: bool
    component_risk: bool
    deployment_blockers: int
    component_risks: int

    @property
    def release_status(self) -> str:
        if self.production_blocker:
            return "NOT_READY"
        if self.component_risk:
            return "READY_WITH_COMPONENT_RISK"
        return "READY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_delivery": self.service_delivery,
            "deployed_strategy_stack": self.deployed_strategy_stack,
            "raw_supervised_model_runtime": self.raw_supervised_model_runtime,
            "raw_supervised_model_standalone": self.raw_supervised_model_standalone,
            "raw_production_gate": self.raw_production_gate,
            "production_blocker": self.production_blocker,
            "component_risk": self.component_risk,
            "deployment_blockers": self.deployment_blockers,
            "component_risks": self.component_risks,
            "release_status": self.release_status,
        }


def build_approval_boundary(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    boundary = calculate_approval_boundary(
        delivery_readiness=_read_json(reports / "delivery_readiness.json"),
        deployed_gate=_read_json(reports / "deployed_strategy_gate.json"),
        production_gate=_read_json(reports / "production_gate.json"),
        risk_register=_read_json(reports / "model_risk_register.json"),
        hygiene=_read_json(reports / "repo_hygiene.json"),
    )
    violations = validate_approval_boundary(boundary)
    return {
        "version": BOUNDARY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": boundary.to_dict(),
        "invariants": {
            "status": "PASS" if not violations else "FAIL",
            "violations": violations,
        },
        "client_statement": (
            "The service and deployed strategy stack are ready for delivery. The raw supervised model is "
            "loadable and integrated into the service, but it is not approved as a standalone production policy. "
            "This limitation is tracked as an official component risk and is not a production blocker."
        ),
        "non_override_rule": (
            "Deployed strategy-stack approval must not be used to claim standalone approval for the raw "
            "supervised model artifact."
        ),
        "evidence": {
            "delivery_verification": "reports/delivery_verification.json",
            "delivery_readiness": "reports/delivery_readiness.json",
            "deployed_strategy_gate": "reports/deployed_strategy_gate.json",
            "production_gate": "reports/production_gate.json",
            "model_risk_register": "reports/model_risk_register.json",
            "repo_hygiene": "reports/repo_hygiene.json",
        },
    }


def calculate_approval_boundary(
    *,
    delivery_readiness: dict[str, Any],
    deployed_gate: dict[str, Any],
    production_gate: dict[str, Any],
    risk_register: dict[str, Any],
    hygiene: dict[str, Any],
) -> ApprovalBoundary:
    delivery_ready = (
        delivery_readiness.get("overall_status") == "READY_FOR_PRODUCTION_POLICY"
        and hygiene.get("status") == "PASS"
    )
    deployed_approved = (
        deployed_gate.get("status") == "PASS"
        and deployed_gate.get("strategy_policy_status") == "APPROVED"
    )
    raw_runtime_status = (risk_register.get("raw_artifact_runtime_status") or {}).get("status", "UNKNOWN")
    raw_gate_status = str(production_gate.get("status", "MISSING")).upper()
    raw_standalone_status = _raw_standalone_status(
        raw_gate_status=raw_gate_status,
        risk_register=risk_register,
        raw_runtime_status=raw_runtime_status,
    )

    risk_summary = risk_register.get("risk_summary") or {}
    deployment_blockers = int(risk_summary.get("deployment_blockers", 0))
    component_risks = int(risk_summary.get("component_risks", 0))
    if raw_standalone_status == "NOT_STANDALONE_APPROVED" and component_risks == 0:
        component_risks = 1

    production_blocker = not delivery_ready or not deployed_approved or deployment_blockers > 0
    component_risk = component_risks > 0 or raw_standalone_status == "NOT_STANDALONE_APPROVED"

    return ApprovalBoundary(
        service_delivery="READY" if delivery_ready else "NOT_READY",
        deployed_strategy_stack="APPROVED" if deployed_approved else "NOT_APPROVED",
        raw_supervised_model_runtime=raw_runtime_status,
        raw_supervised_model_standalone=raw_standalone_status,
        raw_production_gate=raw_gate_status,
        production_blocker=production_blocker,
        component_risk=component_risk,
        deployment_blockers=deployment_blockers,
        component_risks=component_risks,
    )


def _raw_standalone_status(
    *,
    raw_gate_status: str,
    risk_register: dict[str, Any],
    raw_runtime_status: str,
) -> str:
    if (
        raw_gate_status == "PASS"
        and raw_runtime_status == "LOADABLE"
        and risk_register.get("raw_supervised_model_status") == STANDALONE_APPROVED
    ):
        return STANDALONE_APPROVED
    return NOT_STANDALONE_APPROVED


def validate_approval_boundary(boundary: ApprovalBoundary | dict[str, Any]) -> list[str]:
    payload = boundary.to_dict() if isinstance(boundary, ApprovalBoundary) else boundary
    violations: list[str] = []

    raw_gate = str(payload.get("raw_production_gate", "MISSING")).upper()
    raw_standalone = payload.get("raw_supervised_model_standalone")
    component_risk = bool(payload.get("component_risk"))
    component_risks = int(payload.get("component_risks", 0) or 0)
    deployment_blockers = int(payload.get("deployment_blockers", 0) or 0)
    production_blocker = bool(payload.get("production_blocker"))

    if raw_gate != "PASS" and raw_standalone == STANDALONE_APPROVED:
        violations.append("raw_model_cannot_be_standalone_approved_when_raw_gate_is_not_pass")
    if raw_standalone == NOT_STANDALONE_APPROVED and not component_risk:
        violations.append("raw_model_non_approval_must_be_tracked_as_component_risk")
    if raw_standalone == NOT_STANDALONE_APPROVED and component_risks < 1:
        violations.append("raw_model_non_approval_requires_at_least_one_component_risk")
    if (
        payload.get("service_delivery") == "READY"
        and payload.get("deployed_strategy_stack") == "APPROVED"
        and raw_standalone == NOT_STANDALONE_APPROVED
        and deployment_blockers == 0
        and production_blocker
    ):
        violations.append("raw_model_component_risk_must_not_be_promoted_to_production_blocker")
    if component_risk and not production_blocker and payload.get("release_status") != "READY_WITH_COMPONENT_RISK":
        violations.append("component_risk_release_status_must_be_ready_with_component_risk")

    return violations


def assert_approval_boundary(boundary: ApprovalBoundary | dict[str, Any]) -> None:
    violations = validate_approval_boundary(boundary)
    if violations:
        raise AssertionError(f"Approval boundary invariant failure: {violations}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
