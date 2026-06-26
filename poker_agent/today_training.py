from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.training_cluster import build_training_cluster_requirements


TODAY_TRAINING_VERSION = "2026-06-26"
TODAY_TRAINING_PROFILE = "today_acceptance_training"
SELECTED_ARCHITECTURE = "routed_policy_bundle"


def build_today_training_plan(
    project_root: Path,
    *,
    dataset: Path | None = None,
    model_out: Path | None = None,
    max_examples: int = 1000,
    cluster: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = dataset or project_root / "data"
    model_out = model_out or project_root / "models" / "poker_policy_bundle.joblib"
    cluster_contract = build_training_cluster_requirements(
        project_root,
        cluster=cluster,
        run_profile="immediate_delivery",
    )
    return {
        "version": TODAY_TRAINING_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": TODAY_TRAINING_PROFILE,
        "selected_architecture": SELECTED_ARCHITECTURE,
        "delivery_decision": "RUN_NOW_FOR_CURRENT_DELIVERY",
        "dataset": str(dataset),
        "model_out": str(model_out),
        "max_examples": max_examples,
        "why_this_training": [
            "Routed policy bundle is the current best delivery architecture for missing-card data.",
            "It trains an observed-card policy and a public-context fallback policy instead of forcing one weak model to handle both regimes.",
            "It can be validated today and loaded by the deployed FastAPI service.",
        ],
        "not_selected_today": {
            "full_multi_agent_training": {
                "reason": "Long production-hardening run, not required to close the current delivery package.",
                "status": "DEFERRED_TO_HARDENING",
            },
            "raw_single_supervised_model": {
                "reason": "Known component risk; kept loadable but not approved as standalone production policy.",
                "status": "COMPONENT_RISK_ONLY",
            },
        },
        "cluster_contract": cluster_contract,
        "training_command": [
            "python",
            "scripts/train_policy_bundle.py",
            "--dataset",
            str(dataset),
            "--model-out",
            str(model_out),
            "--max-examples",
            str(max_examples),
            "--observed-policy",
            "hist_gradient_boosting",
            "--context-policy",
            "hist_gradient_boosting",
            "--class-weighting",
            "sqrt_balanced",
            "--max-iter",
            "25",
            "--n-estimators",
            "100",
        ],
    }


def build_today_training_report(
    plan: dict[str, Any],
    *,
    training_result: dict[str, Any],
    model_metadata: dict[str, Any] | None = None,
    gate_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_metadata = model_metadata or {}
    gate_result = gate_result or {}
    training_passed = training_result.get("returncode") == 0
    valid_metrics = model_metadata.get("valid_metrics", {}) if isinstance(model_metadata, dict) else {}
    return {
        "version": TODAY_TRAINING_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": plan["profile"],
        "selected_architecture": plan["selected_architecture"],
        "delivery_status": "READY_FOR_CURRENT_DELIVERY" if training_passed else "TRAINING_FAILED",
        "training_status": "PASS" if training_passed else "FAIL",
        "training_result": training_result,
        "model_out": plan["model_out"],
        "dataset": plan["dataset"],
        "max_examples": plan["max_examples"],
        "valid_metrics": valid_metrics,
        "production_gate_status": gate_result.get("status", "NOT_RUN"),
        "production_gate_decision": gate_result.get("decision"),
        "approval_boundary": {
            "current_delivery": "APPROVED_WHEN_TRAINING_PASSES",
            "full_multi_agent_training": "DEFERRED_TO_PRODUCTION_HARDENING",
            "raw_supervised_model_standalone": "NOT_STANDALONE_APPROVED",
        },
        "cluster_contract": plan["cluster_contract"],
        "senior_summary": (
            "Today we run the routed policy bundle acceptance training for the current delivery. "
            "Full multi-agent training remains a separate production-hardening phase and is not a blocker for closing this package."
        ),
    }


def write_today_training_report(payload: dict[str, Any], out: Path, markdown_out: Path | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_today_training_markdown(payload), encoding="utf-8")


def render_today_training_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("valid_metrics") or {}
    lines = [
        "# Today Acceptance Training",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Selected architecture: `{payload['selected_architecture']}`",
        f"- Delivery status: `{payload['delivery_status']}`",
        f"- Training status: `{payload['training_status']}`",
        f"- Production gate status: `{payload['production_gate_status']}`",
        f"- Model output: `{payload['model_out']}`",
        "",
        "## Validation Metrics",
        "",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- Macro F1: `{metrics.get('macro_f1')}`",
        f"- Balanced accuracy: `{metrics.get('balanced_accuracy')}`",
        f"- Cross entropy: `{metrics.get('cross_entropy')}`",
        "",
        "## Boundary",
        "",
        "- Current delivery uses the routed policy bundle acceptance run.",
        "- Full multi-agent training remains a separate production-hardening run.",
        "- The raw supervised model is loadable but not standalone production-approved.",
        "",
        "## Senior Summary",
        "",
        payload["senior_summary"],
        "",
    ]
    return "\n".join(lines)