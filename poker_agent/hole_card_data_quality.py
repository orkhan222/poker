from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOLE_CARD_DATA_QUALITY_VERSION = "2026-06-28"
OPEN_DATA_QUALITY_LIMITATION = "OPEN_DATA_QUALITY_LIMITATION"
MITIGATED_BY_ROUTED_POLICY_BUNDLE = "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
UPSTREAM_NOT_RESOLVED = "UPSTREAM_NOT_RESOLVED"
DEGRADED_STRENGTH_SIGNAL = "DEGRADED_BY_MISSING_HOLE_CARDS"
MISSING_RATE_RISK_THRESHOLD = 0.50
STRENGTH_PROXY_ZERO_RATE_THRESHOLD = 0.50


def build_hole_card_data_quality(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    audit = _read_optional_json(reports / "dataset_audit.json")
    production_gate = _read_optional_json(reports / "production_gate.json")
    today_training = _read_optional_json(reports / "today_acceptance_training.json")
    raw_challenger = _read_optional_json(reports / "raw_model_challenger.json")

    players = audit.get("players") or {}
    features = audit.get("features") or {}
    zero_rates = features.get("critical_feature_zero_rates") or {}
    audit_findings = audit.get("findings") or production_gate.get("audit_findings") or []
    selected_architecture = today_training.get("selected_architecture", "UNKNOWN")
    routed_active = selected_architecture == "routed_policy_bundle"
    hole_card_finding = _find_hole_card_finding(audit_findings)
    strength_proxy_finding = _find_strength_proxy_finding(audit_findings)
    production_observed_gate = _gate_by_name(production_gate.get("gates") or [], "observed_hole_cards_macro_f1")
    raw_challenger_gate = ((raw_challenger.get("best_candidate") or {}).get("gate") or {})
    raw_challenger_slices = (
        (raw_challenger.get("best_candidate") or {}).get("valid_slice_metrics")
        or (raw_challenger.get("best_candidate") or {}).get("slice_metrics")
        or {}
    )

    payload: dict[str, Any] = {
        "version": HOLE_CARD_DATA_QUALITY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Hole-card data-quality limitation and routed-policy mitigation boundary",
        "client_statement": (
            "Missing or unreliable hole-card data remains a core dataset limitation. "
            "The routed policy bundle handles this better, but it does not fully solve the upstream data-quality issue."
        ),
        "coverage_snapshot": {
            "players_rows": players.get("rows"),
            "missing_hole_card_rate": players.get("missing_hole_card_rate"),
            "partial_hole_card_rate": players.get("partial_hole_card_rate"),
            "complete_hole_card_rate": players.get("complete_hole_card_rate"),
            "card_count_distribution": players.get("card_count_distribution", {}),
            "audit_finding": hole_card_finding,
        },
        "strength_signal_impact": {
            "status": DEGRADED_STRENGTH_SIGNAL,
            "strength_proxy_zero_rate": zero_rates.get("strength_proxy"),
            "missing_hole_card_rate": players.get("missing_hole_card_rate"),
            "complete_hole_card_rate": players.get("complete_hole_card_rate"),
            "primary_hand_strength_signal_reliable_for_standalone_policy": False,
            "observed_hole_cards_macro_f1": production_observed_gate.get("observed"),
            "observed_hole_cards_threshold": production_observed_gate.get("threshold"),
            "observed_hole_cards_gate_passed": production_observed_gate.get("passed"),
            "challenger_observed_hole_cards_macro_f1": (
                raw_challenger_slices.get("observed_hole_cards") or {}
            ).get("macro_f1"),
            "strength_proxy_audit_finding": strength_proxy_finding,
            "expected_model_impact": (
                "Card-strength features cannot be treated as the primary reliable signal while most "
                "player rows have missing hole cards and strength_proxy is zero-dominant."
            ),
        },
        "mitigation_boundary": {
            "selected_architecture": selected_architecture,
            "mitigation_status": MITIGATED_BY_ROUTED_POLICY_BUNDLE if routed_active else "MITIGATION_NOT_ACTIVE",
            "routed_policy_bundle_handles_missingness": routed_active,
            "observed_card_policy_path": "uses private-card/card-texture features when two hole cards are observed",
            "public_context_policy_path": "removes private-card features when hole cards are missing or unreliable",
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
            "fully_solves_upstream_data_quality_issue": False,
        },
        "upstream_data_quality_boundary": {
            "limitation_status": OPEN_DATA_QUALITY_LIMITATION,
            "upstream_status": UPSTREAM_NOT_RESOLVED,
            "upstream_data_quality_issue_resolved": False,
            "requires_ocr_or_parser_improvement": True,
            "requires_larger_reviewed_card_labels": True,
            "production_blocker_for_current_deployment": False,
            "component_risk": True,
            "raw_standalone_policy_affected": True,
        },
        "evidence": {
            "dataset_audit": "reports/dataset_audit.json",
            "production_gate": "reports/production_gate.json",
            "today_acceptance_training": "reports/today_acceptance_training.json",
            "raw_model_challenger": "reports/raw_model_challenger.json",
            "raw_challenger_failed_gates": raw_challenger_gate.get("failed_gates", []),
        },
        "required_upstream_fixes": [
            "Improve OCR/card extraction and dealer-log reconciliation for player hole cards.",
            "Create a larger reviewed card-label set with complete, partial, and missing-card slices.",
            "Report observed-card and missing-card model quality separately in every production gate.",
            "Promote a standalone challenger only after observed-card gates and dataset audit blockers pass.",
        ],
        "allowed_claims": [
            "The routed policy bundle mitigates missing-card data better than a single raw supervised model.",
            "The deployed service can route observed-card and missing-card requests through separate policy paths.",
        ],
        "not_allowed_claims": [
            "The upstream hole-card data-quality issue is fully solved.",
            "The routed policy bundle makes missing or unreliable hole-card data irrelevant.",
            "The raw supervised model is standalone production-approved despite the hole-card blocker.",
        ],
    }
    payload["invariants"] = validate_hole_card_data_quality(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_hole_card_data_quality(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    coverage = payload.get("coverage_snapshot") or {}
    strength = payload.get("strength_signal_impact") or {}
    mitigation = payload.get("mitigation_boundary") or {}
    upstream = payload.get("upstream_data_quality_boundary") or {}

    missing_rate = _as_float(coverage.get("missing_hole_card_rate"))
    complete_rate = _as_float(coverage.get("complete_hole_card_rate"))
    strength_zero_rate = _as_float(strength.get("strength_proxy_zero_rate"))
    if missing_rate is None:
        violations.append("missing_hole_card_rate_is_required")
    if complete_rate is None:
        violations.append("complete_hole_card_rate_is_required")
    if missing_rate is not None and complete_rate is not None and missing_rate <= complete_rate:
        violations.append("current_audit_must_show_missing_hole_cards_as_the_dominant_condition")
    if not coverage.get("audit_finding"):
        violations.append("hole_card_audit_finding_must_remain_visible")

    high_missingness = missing_rate is not None and missing_rate >= MISSING_RATE_RISK_THRESHOLD
    high_strength_zero_rate = (
        strength_zero_rate is not None and strength_zero_rate >= STRENGTH_PROXY_ZERO_RATE_THRESHOLD
    )
    if high_missingness and strength.get("status") != DEGRADED_STRENGTH_SIGNAL:
        violations.append("high_hole_card_missingness_must_degrade_strength_signal")
    if high_strength_zero_rate and strength.get("status") != DEGRADED_STRENGTH_SIGNAL:
        violations.append("high_strength_proxy_zero_rate_must_degrade_strength_signal")
    if (high_missingness or high_strength_zero_rate) and (
        strength.get("primary_hand_strength_signal_reliable_for_standalone_policy") is not False
    ):
        violations.append("degraded_strength_signal_cannot_be_standalone_reliable")
    if high_strength_zero_rate and not strength.get("strength_proxy_audit_finding"):
        violations.append("strength_proxy_audit_finding_must_remain_visible")

    if mitigation.get("mitigation_status") != MITIGATED_BY_ROUTED_POLICY_BUNDLE:
        violations.append("routed_policy_bundle_mitigation_must_be_active")
    if mitigation.get("routed_policy_bundle_handles_missingness") is not True:
        violations.append("routed_policy_bundle_must_handle_missingness")
    if mitigation.get("fully_solves_upstream_data_quality_issue") is not False:
        violations.append("routed_policy_bundle_must_not_claim_to_fully_solve_upstream_data_quality")
    if mitigation.get("mitigation_scope") != "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR":
        violations.append("routed_policy_bundle_scope_must_remain_runtime_mitigation_not_data_repair")
    if mitigation.get("requires_slice_specific_monitoring") is not True:
        violations.append("hole_card_routes_must_require_slice_specific_monitoring")

    if upstream.get("limitation_status") != OPEN_DATA_QUALITY_LIMITATION:
        violations.append("hole_card_limitation_must_remain_open")
    if upstream.get("upstream_status") != UPSTREAM_NOT_RESOLVED:
        violations.append("upstream_hole_card_issue_must_remain_not_resolved")
    if upstream.get("upstream_data_quality_issue_resolved") is not False:
        violations.append("upstream_data_quality_issue_cannot_be_marked_resolved")
    if upstream.get("requires_ocr_or_parser_improvement") is not True:
        violations.append("ocr_or_parser_improvement_must_remain_required")
    if upstream.get("requires_larger_reviewed_card_labels") is not True:
        violations.append("larger_reviewed_card_labels_must_remain_required")
    if upstream.get("production_blocker_for_current_deployment") is not False:
        violations.append("hole_card_limitation_must_not_block_current_monitored_deployment")
    if upstream.get("component_risk") is not True:
        violations.append("hole_card_limitation_must_remain_a_component_risk")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_hole_card_data_quality(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_hole_card_data_quality(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_hole_card_data_quality_markdown(payload), encoding="utf-8")
    return payload


def render_hole_card_data_quality_markdown(payload: dict[str, Any]) -> str:
    coverage = payload["coverage_snapshot"]
    strength = payload["strength_signal_impact"]
    mitigation = payload["mitigation_boundary"]
    upstream = payload["upstream_data_quality_boundary"]
    lines = [
        "# Hole-Card Data Quality Contract",
        "",
        payload["client_statement"],
        "",
        "## Coverage Snapshot",
        "",
        f"- Missing hole-card rate: `{coverage.get('missing_hole_card_rate')}`",
        f"- Partial hole-card rate: `{coverage.get('partial_hole_card_rate')}`",
        f"- Complete hole-card rate: `{coverage.get('complete_hole_card_rate')}`",
        "",
        "## Strength Signal Impact",
        "",
        f"- Strength signal status: `{strength['status']}`",
        f"- Strength proxy zero rate: `{strength.get('strength_proxy_zero_rate')}`",
        f"- Primary hand-strength signal reliable for standalone policy: `{strength['primary_hand_strength_signal_reliable_for_standalone_policy']}`",
        f"- Observed-hole-card macro F1: `{strength.get('observed_hole_cards_macro_f1')}`",
        f"- Observed-hole-card threshold: `{strength.get('observed_hole_cards_threshold')}`",
        f"- Challenger observed-hole-card macro F1: `{strength.get('challenger_observed_hole_cards_macro_f1')}`",
        f"- Expected impact: {strength['expected_model_impact']}",
        "",
        "## Mitigation Boundary",
        "",
        f"- Selected architecture: `{mitigation['selected_architecture']}`",
        f"- Mitigation status: `{mitigation['mitigation_status']}`",
        f"- Mitigation scope: `{mitigation['mitigation_scope']}`",
        f"- Requires slice-specific monitoring: `{mitigation['requires_slice_specific_monitoring']}`",
        f"- Fully solves upstream data-quality issue: `{mitigation['fully_solves_upstream_data_quality_issue']}`",
        "",
        "## Upstream Boundary",
        "",
        f"- Limitation status: `{upstream['limitation_status']}`",
        f"- Upstream status: `{upstream['upstream_status']}`",
        f"- Upstream data-quality issue resolved: `{upstream['upstream_data_quality_issue_resolved']}`",
        f"- Production blocker for current deployment: `{upstream['production_blocker_for_current_deployment']}`",
        f"- Component risk: `{upstream['component_risk']}`",
        "",
        "## Required Upstream Fixes",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["required_upstream_fixes"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _find_hole_card_finding(findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for finding in findings:
        issue = str(finding.get("issue", "")).lower()
        if "hole-card" in issue or "hole card" in issue:
            return finding
    return None


def _find_strength_proxy_finding(findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for finding in findings:
        issue = str(finding.get("issue", "")).lower()
        if "strength_proxy" in issue or "strength proxy" in issue:
            return finding
    return None


def _gate_by_name(gates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for gate in gates:
        if gate.get("name") == name:
            return gate
    return {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
