from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLIENT_GPU_RESPONSE_VERSION = "2026-06-26"


def build_client_gpu_training_response(project_root: Path) -> dict[str, Any]:
    reports_dir = project_root / "reports"
    today_training = _read_optional_json(reports_dir / "today_acceptance_training.json")
    cluster_contract = _read_optional_json(reports_dir / "training_cluster_requirements.json")

    validation_metrics = today_training.get("valid_metrics", {}) if today_training else {}
    cluster_estimate = (cluster_contract.get("estimate") or {}) if cluster_contract else {}
    return {
        "version": CLIENT_GPU_RESPONSE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "A100/H100 availability and current delivery training plan",
        "client_question_addressed": "Whether a dedicated A100 or H100 GPU is available for the current training milestone.",
        "recommended_reply": (
            "Yes, we need to confirm whether a dedicated A100 or H100 GPU is available. "
            "For the current delivery milestone, a single dedicated A100 or H100 is sufficient "
            "to run the immediate acceptance training and validation package, including routed "
            "policy bundle training, simulation sanity checks, service verification, and report refresh. "
            "The full production-scale multi-agent training run should remain a separate hardening phase "
            "with its own runtime estimate, evaluation scope, acceptance criteria, and final approval process."
        ),
        "short_reply": (
            "Yes, we are asking whether a dedicated A100 or H100 is available. One such GPU is enough "
            "for the current acceptance training and validation; full production-scale multi-agent training "
            "should be scheduled separately as the next hardening phase."
        ),
        "current_delivery_training": {
            "selected_architecture": today_training.get("selected_architecture") if today_training else "routed_policy_bundle",
            "profile": today_training.get("profile") if today_training else "today_acceptance_training",
            "training_status": today_training.get("training_status") if today_training else "UNKNOWN",
            "delivery_status": today_training.get("delivery_status") if today_training else "UNKNOWN",
            "model_out": today_training.get("model_out") if today_training else str(project_root / "models" / "poker_policy_bundle.joblib"),
            "validation_metrics": {
                "accuracy": validation_metrics.get("accuracy"),
                "macro_f1": validation_metrics.get("macro_f1"),
                "balanced_accuracy": validation_metrics.get("balanced_accuracy"),
            },
        },
        "gpu_boundary": {
            "minimum_for_current_delivery": "single dedicated NVIDIA A100 or H100",
            "current_delivery_scope": [
                "routed policy bundle acceptance training",
                "validation refresh",
                "simulation sanity checks",
                "service verification",
                "delivery report refresh",
            ],
            "full_multi_agent_training": "separate production-hardening phase",
            "do_not_claim": "Do not represent the full production-scale multi-agent training cycle as completed by the current acceptance run.",
        },
        "cluster_status": {
            "run_profile": cluster_contract.get("run_profile") if cluster_contract else "immediate_delivery",
            "status": cluster_estimate.get("status", "PENDING_CLUSTER_CONFIRMATION"),
            "estimated_hours": cluster_estimate.get("estimated_hours"),
            "estimated_days": cluster_estimate.get("estimated_days"),
            "confidence": cluster_estimate.get("confidence"),
        },
        "approval_boundary": {
            "current_delivery": "READY when acceptance training and delivery verification pass",
            "full_multi_agent_training": "deferred to hardening",
            "raw_supervised_model_standalone": "not standalone production-approved",
        },
    }


def write_client_gpu_training_response(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_client_gpu_training_response(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_client_gpu_training_response_markdown(payload), encoding="utf-8")
    return payload


def render_client_gpu_training_response_markdown(payload: dict[str, Any]) -> str:
    training = payload["current_delivery_training"]
    metrics = training.get("validation_metrics") or {}
    lines = [
        "# Client GPU Training Response",
        "",
        "## Recommended Reply",
        "",
        payload["recommended_reply"],
        "",
        "## Current Delivery Training",
        "",
        f"- Selected architecture: `{training.get('selected_architecture')}`",
        f"- Training status: `{training.get('training_status')}`",
        f"- Delivery status: `{training.get('delivery_status')}`",
        f"- Model artifact: `{training.get('model_out')}`",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- Macro F1: `{metrics.get('macro_f1')}`",
        f"- Balanced accuracy: `{metrics.get('balanced_accuracy')}`",
        "",
        "## Boundary",
        "",
        f"- Minimum GPU for current delivery: `{payload['gpu_boundary']['minimum_for_current_delivery']}`",
        f"- Full multi-agent training: `{payload['gpu_boundary']['full_multi_agent_training']}`",
        f"- Do not claim: {payload['gpu_boundary']['do_not_claim']}",
        "",
    ]
    return "\n".join(lines)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))