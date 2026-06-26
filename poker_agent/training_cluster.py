from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLUSTER_REQUIREMENTS_VERSION = "2026-06-26"
DEFAULT_RUN_PROFILE = "immediate_delivery"


GPU_PROFILES: dict[str, dict[str, Any]] = {
    "A100": {
        "recommended_vram_gb": 40,
        "full_training_days": 5.0,
        "immediate_delivery_hours": 3.0,
        "status": "RECOMMENDED",
    },
    "H100": {
        "recommended_vram_gb": 80,
        "full_training_days": 5.0,
        "immediate_delivery_hours": 2.0,
        "status": "RECOMMENDED",
    },
    "L40S": {
        "recommended_vram_gb": 48,
        "full_training_days": 8.0,
        "immediate_delivery_hours": 5.0,
        "status": "ACCEPTABLE_WITH_LONGER_RUNTIME",
    },
    "RTX4090": {
        "recommended_vram_gb": 24,
        "full_training_days": 10.0,
        "immediate_delivery_hours": 8.0,
        "status": "RESEARCH_ONLY",
    },
}

RUN_PROFILES: dict[str, dict[str, Any]] = {
    "immediate_delivery": {
        "status": "READY_FOR_IMMEDIATE_DELIVERY_VALIDATION",
        "scope": "same-day acceptance run: smoke training, inference contract, simulation sanity checks, and report refresh",
        "commitment": "Use this profile to finish the current delivery package now without claiming full production-scale training is complete.",
    },
    "full_multi_agent_training": {
        "status": "FULL_TRAINING_ESTIMATE",
        "scope": "full multi-agent training cycle with production-scale self-play",
        "commitment": "Use this profile only when the client approves the longer production-hardening training run.",
    },
}

REQUESTED_CLUSTER_FIELDS = [
    "gpu_type",
    "gpu_count",
    "vram_gb_per_gpu",
    "cpu_cores",
    "system_ram_gb",
    "storage_gb",
    "interconnect",
    "dedicated_or_shared",
]


@dataclass(frozen=True)
class ClusterSpec:
    gpu_type: str | None = None
    gpu_count: int | None = None
    vram_gb_per_gpu: float | None = None
    cpu_cores: int | None = None
    system_ram_gb: float | None = None
    storage_gb: float | None = None
    interconnect: str | None = None
    dedicated_or_shared: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ClusterSpec":
        payload = payload or {}
        return cls(
            gpu_type=_clean_string(payload.get("gpu_type")),
            gpu_count=_optional_int(payload.get("gpu_count")),
            vram_gb_per_gpu=_optional_float(payload.get("vram_gb_per_gpu")),
            cpu_cores=_optional_int(payload.get("cpu_cores")),
            system_ram_gb=_optional_float(payload.get("system_ram_gb")),
            storage_gb=_optional_float(payload.get("storage_gb")),
            interconnect=_clean_string(payload.get("interconnect")),
            dedicated_or_shared=_clean_string(payload.get("dedicated_or_shared")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "vram_gb_per_gpu": self.vram_gb_per_gpu,
            "cpu_cores": self.cpu_cores,
            "system_ram_gb": self.system_ram_gb,
            "storage_gb": self.storage_gb,
            "interconnect": self.interconnect,
            "dedicated_or_shared": self.dedicated_or_shared,
        }


def build_training_cluster_requirements(
    project_root: Path,
    cluster: dict[str, Any] | None = None,
    run_profile: str = DEFAULT_RUN_PROFILE,
) -> dict[str, Any]:
    profile_name = normalize_run_profile(run_profile)
    spec = ClusterSpec.from_dict(cluster)
    estimate = estimate_multi_agent_training(spec, run_profile=profile_name)
    return {
        "version": CLUSTER_REQUIREMENTS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": "Confirm compute capacity and select the correct training run profile.",
        "run_profile": profile_name,
        "available_run_profiles": RUN_PROFILES,
        "requested_fields": REQUESTED_CLUSTER_FIELDS,
        "immediate_delivery_cluster": {
            "gpu": "single NVIDIA A100 or H100",
            "gpu_count": 1,
            "vram_gb_per_gpu": "40GB+ for A100, 80GB preferred for H100",
            "expected_completion": "same_day",
            "expected_delivery_validation_hours": {"A100": 3, "H100": 2},
            "scope": RUN_PROFILES["immediate_delivery"]["scope"],
        },
        "full_training_reference": {
            "gpu": "single NVIDIA A100 or H100",
            "gpu_count": 1,
            "expected_training_duration_days": 5,
            "scope": RUN_PROFILES["full_multi_agent_training"]["scope"],
            "not_required_for_current_delivery": True,
        },
        "provided_cluster": spec.to_dict(),
        "estimate": estimate,
        "stakeholder_question": (
            "Please confirm the available training cluster: GPU type/count, VRAM, CPU/RAM, "
            "storage, interconnect, and whether the environment is dedicated or shared."
        ),
        "senior_response": (
            "For the current delivery, a dedicated single A100 or H100 is sufficient to complete "
            "the immediate acceptance run now: smoke training, simulation sanity checks, validation, "
            "and report refresh. The longer full multi-agent production training cycle remains a "
            "separate hardening run and must not be represented as already completed."
        ),
        "assumptions": [
            "The immediate profile is for same-day delivery validation, not full production-scale training.",
            "The environment should be dedicated or have predictable scheduling.",
            "Dataset preprocessing and experiment configuration are already available.",
            "Full multi-agent training is tracked separately as production hardening.",
        ],
        "operational_risks": estimate["risks"],
        "project_root": str(project_root),
    }


def estimate_multi_agent_training(
    spec: ClusterSpec,
    run_profile: str = DEFAULT_RUN_PROFILE,
) -> dict[str, Any]:
    profile_name = normalize_run_profile(run_profile)
    missing = [field for field, value in spec.to_dict().items() if value in (None, "")]
    if not spec.gpu_type:
        return {
            "status": "PENDING_CLUSTER_CONFIRMATION",
            "run_profile": profile_name,
            "estimated_hours": None,
            "estimated_days": None,
            "confidence": "LOW",
            "basis": "Cluster details have not been provided yet.",
            "missing_fields": missing,
            "risks": [
                "Delivery runtime cannot be committed until GPU type, GPU count, and environment ownership are known.",
            ],
        }

    normalized_gpu = normalize_gpu_type(spec.gpu_type)
    profile = GPU_PROFILES.get(normalized_gpu)
    risks: list[str] = []
    if profile is None:
        return {
            "status": "UNSUPPORTED_OR_UNBENCHMARKED_GPU",
            "run_profile": profile_name,
            "estimated_hours": None,
            "estimated_days": None,
            "confidence": "LOW",
            "basis": f"No delivery estimate is registered for GPU type `{spec.gpu_type}`.",
            "missing_fields": missing,
            "risks": [
                "A benchmark smoke run is required before committing to the delivery timeline.",
            ],
        }

    gpu_count = max(int(spec.gpu_count or 1), 1)
    speedup = min(gpu_count, 4) ** 0.7
    if profile_name == "immediate_delivery":
        estimated_hours = max(1.0, float(profile["immediate_delivery_hours"]) / speedup)
        estimated_days = round(estimated_hours / 24.0, 2)
        status = RUN_PROFILES[profile_name]["status"]
        basis = f"{gpu_count}x {normalized_gpu}; immediate delivery validation baseline is {profile['immediate_delivery_hours']} hours."
    else:
        base_days = float(profile["full_training_days"])
        estimated_days = max(2.0, base_days / speedup)
        estimated_hours = round(estimated_days * 24.0, 2)
        status = profile["status"]
        basis = f"{gpu_count}x {normalized_gpu}; full training baseline is {profile['full_training_days']} days."

    if spec.dedicated_or_shared and spec.dedicated_or_shared.lower() == "shared":
        estimated_hours *= 1.2
        estimated_days *= 1.2
        risks.append("Shared cluster scheduling can increase wall-clock runtime.")
    if spec.vram_gb_per_gpu is not None and spec.vram_gb_per_gpu < float(profile["recommended_vram_gb"]):
        risks.append("Available VRAM is below the recommended profile for this GPU class.")
    if spec.storage_gb is not None and spec.storage_gb < 500:
        risks.append("Storage below 500GB may constrain replay buffers, checkpoints, and simulation logs.")
    if spec.cpu_cores is not None and spec.cpu_cores < 16:
        risks.append("Low CPU core count may bottleneck environment simulation throughput.")
    if not spec.interconnect:
        risks.append("Interconnect details are missing; multi-GPU scaling cannot be guaranteed.")

    if normalized_gpu in {"A100", "H100"} and gpu_count == 1 and not risks:
        if profile_name == "immediate_delivery":
            estimated_hours = float(profile["immediate_delivery_hours"])
            estimated_days = round(estimated_hours / 24.0, 2)
        else:
            estimated_days = 5.0
            estimated_hours = 120.0

    return {
        "status": status,
        "run_profile": profile_name,
        "gpu_profile": normalized_gpu,
        "estimated_hours": round(estimated_hours, 2),
        "estimated_days": round(estimated_days, 2),
        "confidence": "MEDIUM" if missing else "HIGH",
        "basis": basis,
        "missing_fields": missing,
        "risks": risks,
    }


def write_training_cluster_requirements(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    cluster: dict[str, Any] | None = None,
    run_profile: str = DEFAULT_RUN_PROFILE,
) -> dict[str, Any]:
    payload = build_training_cluster_requirements(project_root, cluster=cluster, run_profile=run_profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_training_cluster_markdown(payload), encoding="utf-8")
    return payload


def render_training_cluster_markdown(payload: dict[str, Any]) -> str:
    estimate = payload["estimate"]
    lines = [
        "# Training Cluster Requirements",
        "",
        f"- Run profile: `{payload['run_profile']}`",
        f"- Status: `{estimate['status']}`",
        f"- Estimated hours: `{estimate['estimated_hours']}`",
        f"- Estimated days: `{estimate['estimated_days']}`",
        f"- Confidence: `{estimate['confidence']}`",
        f"- Immediate delivery cluster: `{payload['immediate_delivery_cluster']['gpu']}`",
        "",
        "## Required Cluster Details",
        "",
    ]
    lines.extend(f"- `{field}`" for field in payload["requested_fields"])
    lines.extend(
        [
            "",
            "## Stakeholder Response",
            "",
            payload["senior_response"],
            "",
            "## Run Profile Boundary",
            "",
            "- `immediate_delivery` finishes the current acceptance package.",
            "- `full_multi_agent_training` is a separate production-hardening run.",
            "",
            "## Risks",
            "",
        ]
    )
    risks = payload.get("operational_risks") or []
    lines.extend(f"- {risk}" for risk in risks) if risks else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def normalize_run_profile(value: str | None) -> str:
    if value in (None, ""):
        return DEFAULT_RUN_PROFILE
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in RUN_PROFILES:
        raise ValueError(f"Unsupported training run profile: {value}")
    return normalized


def normalize_gpu_type(value: str) -> str:
    raw = value.upper().replace("NVIDIA", "").replace(" ", "").replace("-", "")
    if "H100" in raw:
        return "H100"
    if "A100" in raw:
        return "A100"
    if "L40S" in raw:
        return "L40S"
    if "4090" in raw:
        return "RTX4090"
    return raw


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)