from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIN_PRODUCTION_PAIRED_HANDS = 5000


APPROVAL_BOUNDARY: dict[str, str] = {
    "deployed_strategy_stack": (
        "Approves the deployed runtime stack: model wrapper, DeploymentGatedPolicy behavior, action planning, "
        "human-alignment evidence, human-likeness evidence, service delivery, and production-scale self-play."
    ),
    "raw_supervised_model_artifact": (
        "Approves only the standalone supervised artifact. This remains independent and must not be inferred "
        "from deployed-stack approval."
    ),
}


def build_deployed_strategy_gate(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    acceptance = _read_json(reports / "policy_acceptance.json")
    self_play = _read_json(reports / "production_self_play.json")
    raw_gate = _read_json(reports / "production_gate.json")
    delivery = _read_json(reports / "delivery_readiness.json")
    hygiene = _read_json(reports / "repo_hygiene.json")

    gates = [
        _gate("policy_acceptance", acceptance.get("overall_status") == "PASS", acceptance.get("overall_status"), "PASS"),
        _gate(
            "human_action_alignment",
            acceptance.get("human_action_alignment_status") == "PASS",
            acceptance.get("human_action_alignment_status"),
            "PASS",
        ),
        _gate(
            "human_likeness",
            (acceptance.get("human_likeness") or {}).get("status") == "PASS",
            (acceptance.get("human_likeness") or {}).get("status"),
            "PASS",
        ),
        _gate(
            "production_scale_self_play",
            self_play.get("status") == "PASS"
            and self_play.get("production_scale_status") == "PASS"
            and int(self_play.get("paired_hands", 0)) >= MIN_PRODUCTION_PAIRED_HANDS,
            {
                "status": self_play.get("status"),
                "production_scale_status": self_play.get("production_scale_status"),
                "paired_hands": self_play.get("paired_hands"),
            },
            {"status": "PASS", "production_scale_status": "PASS", "paired_hands": f">={MIN_PRODUCTION_PAIRED_HANDS}"},
        ),
        _gate("service_delivery", delivery.get("service_delivery_status") == "READY", delivery.get("service_delivery_status"), "READY"),
        _gate("repository_hygiene", hygiene.get("status") == "PASS", hygiene.get("status"), "PASS"),
    ]
    blocking_items = [_blocker(gate) for gate in gates if not gate["passed"]]
    raw_status = str(raw_gate.get("status", "MISSING")).upper()
    status = "PASS" if not blocking_items else "FAIL"
    component_risks = _raw_model_risks(raw_gate) if raw_status != "PASS" else []
    standalone_raw_model_approved = raw_status == "PASS"
    deployed_strategy_approved = status == "PASS"
    return {
        "version": "2026-06-20",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "strategy_policy_status": "APPROVED" if status == "PASS" else "NOT_APPROVED",
        "deployment_mode": "production_policy" if status == "PASS" else "technical_handoff_only",
        "production_claim_allowed": status == "PASS",
        "approval_boundary": APPROVAL_BOUNDARY,
        "approval_invariants": {
            "deployed_strategy_approved": deployed_strategy_approved,
            "standalone_raw_model_approved": standalone_raw_model_approved,
            "deployed_stack_approval_overrides_raw_model_gate": False,
            "raw_gate_failure_requires_component_risk": raw_status != "PASS",
            "component_risk_present": bool(component_risks),
        },
        "raw_supervised_model_gate_status": raw_status,
        "raw_supervised_model_status": "STANDALONE_APPROVED" if raw_status == "PASS" else "NOT_STANDALONE_APPROVED",
        "decision": (
            "Approved for deployed strategy-stack rollout with monitoring."
            if status == "PASS"
            else "Blocked from deployed strategy-stack rollout until deployed-stack gates pass."
        ),
        "gates": gates,
        "blocking_items": blocking_items,
        "component_risks": component_risks,
        "metric_snapshot": {
            "human_action_accuracy": (acceptance.get("human_action_alignment") or {}).get("accuracy"),
            "human_action_macro_f1": (acceptance.get("human_action_alignment") or {}).get("macro_f1"),
            "human_likeness_js_divergence": (acceptance.get("human_likeness") or {}).get("js_divergence"),
            "production_self_play_paired_hands": self_play.get("paired_hands"),
            "production_self_play_mean_win_rate": self_play.get("mean_policy_win_rate"),
            "raw_model_macro_f1": (raw_gate.get("valid_metrics") or {}).get("macro_f1"),
            "raw_model_balanced_accuracy": (raw_gate.get("valid_metrics") or {}).get("balanced_accuracy"),
            "raw_model_accuracy_lift": (raw_gate.get("valid_metrics") or {}).get("lift_vs_majority"),
        },
        "recommended_next_milestone": {
            "name": "standalone challenger artifact",
            "objective": "Train a new supervised artifact that clears the raw production model gate.",
        },
    }


def write_deployed_strategy_gate(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_deployed_strategy_gate(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Deployed Strategy Gate",
        "",
        f"- Status: `{payload['status']}`",
        f"- Strategy policy status: `{payload['strategy_policy_status']}`",
        f"- Raw supervised model status: `{payload['raw_supervised_model_status']}`",
        "",
        "## Component Risks",
        "",
    ]
    if payload["component_risks"]:
        for risk in payload["component_risks"]:
            lines.append(f"- `{risk['component']}`: {risk['evidence']}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _gate(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "threshold": threshold}


def _blocker(gate: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(gate["name"]),
        "severity": "critical",
        "evidence": f"observed={gate['observed']}, threshold={gate['threshold']}",
        "required_fix": "Clear this deployed-stack gate before production strategy approval.",
    }


def _raw_model_risks(raw_gate: dict[str, Any]) -> list[dict[str, str]]:
    reasons = (raw_gate.get("strategy_readiness") or {}).get("blocking_reasons") or []
    evidence = ", ".join(f"{item.get('gate')}={item.get('observed')}" for item in reasons[:6] if item.get("gate"))
    if not evidence:
        evidence = f"production_gate={raw_gate.get('status', 'MISSING')}"
    return [
        {
            "component": "raw_supervised_model_artifact",
            "severity": "high",
            "owner": "modeling",
            "status": "NOT_STANDALONE_APPROVED",
            "deployment_blocker": False,
            "blocking_scope": "standalone_raw_model_only",
            "evidence": evidence,
            "required_fix": "Train and gate a challenger artifact that clears the raw production model thresholds.",
        }
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
