from __future__ import annotations

from dataclasses import dataclass
from typing import Any

USAGE_BOUNDARY_CONTRACT_VERSION = "usage_boundary.v1"

ALLOWED_USAGE = ("offline_research", "simulation", "authorized_environment")
BLOCKED_USAGE = ("real_money_platform", "unauthorized_platform", "stealth_automation", "tos_bypass")

USAGE_ALIASES = {
    "research": "offline_research",
    "offline": "offline_research",
    "offline_research": "offline_research",
    "simulation": "simulation",
    "sim": "simulation",
    "sandbox": "simulation",
    "authorized": "authorized_environment",
    "authorised": "authorized_environment",
    "authorized_environment": "authorized_environment",
    "authorised_environment": "authorized_environment",
    "real_money": "real_money_platform",
    "real_money_platform": "real_money_platform",
    "unauthorized": "unauthorized_platform",
    "unauthorised": "unauthorized_platform",
    "unauthorized_platform": "unauthorized_platform",
    "unauthorised_platform": "unauthorized_platform",
    "stealth": "stealth_automation",
    "stealth_automation": "stealth_automation",
    "tos_bypass": "tos_bypass",
    "terms_bypass": "tos_bypass",
}


def _normalize(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return USAGE_ALIASES.get(text, text)


def _as_bool(raw: Any, default: bool | None = None) -> bool | None:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _raw_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("usage_boundary")
    if isinstance(raw, str):
        return {"declared_use": raw}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _candidate_values(boundary: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    keys = (
        "declared_use",
        "intended_use",
        "use_case",
        "purpose",
        "environment",
        "platform_type",
        "prohibited_use",
        "blocked_use",
        "automation_mode",
    )
    values: list[str] = []
    for source in (boundary, payload):
        for key in keys:
            raw = source.get(key)
            if isinstance(raw, (list, tuple, set)):
                values.extend(_normalize(item) for item in raw)
            elif raw not in (None, ""):
                values.append(_normalize(raw))
    return values


@dataclass(frozen=True)
class UsageBoundaryDecision:
    allowed: bool
    declared_use: str
    reason_code: str
    message: str
    blocked_signals: tuple[str, ...] = ()

    def to_error_details(self) -> dict[str, Any]:
        return {
            "schema_version": USAGE_BOUNDARY_CONTRACT_VERSION,
            "allowed_usage": list(ALLOWED_USAGE),
            "blocked_usage": list(BLOCKED_USAGE),
            "declared_use": self.declared_use,
            "reason_code": self.reason_code,
            "blocked_signals": list(self.blocked_signals),
        }

    def response_context(self) -> dict[str, Any]:
        return {
            "schema_version": USAGE_BOUNDARY_CONTRACT_VERSION,
            "declared_use": self.declared_use,
            "allowed": self.allowed,
            "reason_code": self.reason_code,
        }


def evaluate_usage_boundary(payload: dict[str, Any]) -> UsageBoundaryDecision:
    boundary = _raw_boundary(payload)
    candidates = _candidate_values(boundary, payload)
    declared_use = _normalize(
        boundary.get("declared_use")
        or boundary.get("intended_use")
        or boundary.get("use_case")
        or boundary.get("purpose")
        or boundary.get("environment")
        or ""
    )

    if not boundary:
        return UsageBoundaryDecision(
            allowed=False,
            declared_use="",
            reason_code="missing_usage_boundary",
            message="usage_boundary is required and must declare offline_research, simulation, or authorized_environment.",
        )

    blocked = sorted({value for value in candidates if value in BLOCKED_USAGE})
    real_money = _as_bool(boundary.get("real_money") or payload.get("real_money"), False)
    if real_money:
        blocked.append("real_money_platform")
    terms_ok = _as_bool(
        boundary.get("terms_compliant")
        if "terms_compliant" in boundary
        else boundary.get("tos_compliant", payload.get("tos_compliant")),
        True,
    )
    if terms_ok is False:
        blocked.append("tos_bypass")
    authorized = _as_bool(
        boundary.get("authorized")
        if "authorized" in boundary
        else boundary.get("authorization_confirmed", payload.get("authorization_confirmed")),
        None,
    )
    if authorized is False:
        blocked.append("unauthorized_platform")

    blocked = tuple(sorted(set(blocked)))
    if blocked:
        return UsageBoundaryDecision(
            allowed=False,
            declared_use=declared_use,
            reason_code="blocked_usage",
            message="This service is limited to offline research, simulation, and explicitly authorized environments.",
            blocked_signals=blocked,
        )

    if declared_use not in ALLOWED_USAGE:
        return UsageBoundaryDecision(
            allowed=False,
            declared_use=declared_use,
            reason_code="unsupported_declared_use",
            message="declared_use must be offline_research, simulation, or authorized_environment.",
        )

    return UsageBoundaryDecision(
        allowed=True,
        declared_use=declared_use,
        reason_code="allowed_usage",
        message="Usage boundary accepted.",
    )


def describe_usage_boundary_contract() -> dict[str, Any]:
    return {
        "schema_version": USAGE_BOUNDARY_CONTRACT_VERSION,
        "required_request_field": "usage_boundary",
        "allowed_usage": list(ALLOWED_USAGE),
        "blocked_usage": list(BLOCKED_USAGE),
        "blocked_http_status": 403,
        "blocked_error_code": "USAGE_BOUNDARY_VIOLATION",
        "enforced_for": ["/predict", "autonomous_agent"],
    }
