from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.rl_training_evidence_gate import REQUIRED_RL_EVIDENCE


RL_DELIVERY_BOUNDARY_VERSION = "2026-07-09"
RL_DELIVERY_BOUNDARY_NAME = "rl_delivery_vs_strategy_claim_boundary"
OPEN_SPIEL_CLAIM_CONTRACT_PATH = Path("reports") / "open_spiel_claim_contract.json"

BLOCKED_CLAIMS_WITHOUT_RL_PROOF = (
    "Measured OpenSpiel self-play win-rate is complete.",
    "Phase 3 RL training has produced production strategy evidence.",
    "The deployed strategy is a final production-quality poker policy.",
)


def build_rl_delivery_boundary(project_root: Path) -> dict[str, Any]:
    source_contract = _read_json(project_root / OPEN_SPIEL_CLAIM_CONTRACT_PATH)
    return build_rl_delivery_boundary_from_claim(source_contract)


def build_rl_delivery_boundary_from_claim(source_contract: dict[str, Any]) -> dict[str, Any]:
    proof_completed = source_contract.get("training_proof_completed") is True
    source_self_play_claim_allowed = source_contract.get("self_play_win_rate_claim_allowed") is True
    complete_rl_evidence = proof_completed and source_self_play_claim_allowed
    production_strategy_claim_allowed = (
        complete_rl_evidence and source_contract.get("model_quality_risk") is False
    )

    payload: dict[str, Any] = {
        "version": RL_DELIVERY_BOUNDARY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_name": RL_DELIVERY_BOUNDARY_NAME,
        "source_contract": OPEN_SPIEL_CLAIM_CONTRACT_PATH.as_posix(),
        "source_contract_status": source_contract.get("overall_status"),
        "boundary": "DELIVERY_READY_BUT_RL_PROOF_REQUIRED_FOR_STRATEGY_CLAIMS",
        "delivery_scope": {
            "service_delivery_blocked_by_rl_training_gap": False,
            "delivery_claim_allowed": True,
            "allowed_current_delivery_claim": (
                "The service package, API surface, Docker assets, and agent-only arena code are ready "
                "for delivery. The RL training-proof gap is tracked as a model-quality risk, not as a "
                "service-delivery blocker."
            ),
        },
        "rl_training_proof": {
            "training_proof_completed": proof_completed,
            "source_self_play_win_rate_claim_allowed": source_self_play_claim_allowed,
            "required_evidence": list(source_contract.get("required_evidence_before_self_play_claim") or []),
            "missing_requirements": list(source_contract.get("missing_requirements") or []),
            "real_open_spiel_runtime_available": source_contract.get(
                "real_open_spiel_runtime_available"
            )
            is True,
            "phase1_trained_policy_artifact_count": int(
                source_contract.get("phase1_trained_policy_artifact_count") or 0
            ),
            "seed_stability_evaluated": source_contract.get("seed_stability_evaluated") is True,
            "long_run_completed": source_contract.get("long_run_completed") is True,
            "ppo_or_equivalent_policy_update_completed": source_contract.get(
                "ppo_or_equivalent_policy_update_completed"
            )
            is True,
        },
        "claim_permissions": {
            "delivery_readiness_claim_allowed": True,
            "self_play_win_rate_claim_allowed": complete_rl_evidence,
            "production_strategy_quality_claim_allowed": production_strategy_claim_allowed,
        },
        "blocked_claims_without_rl_proof": list(BLOCKED_CLAIMS_WITHOUT_RL_PROOF),
        "current_delivery_blocker": False,
        "model_quality_risk": not production_strategy_claim_allowed,
        "allowed_current_claim": (
            "Delivery can proceed with the current service and arena code. Do not present the pending "
            "RL/OpenSpiel stage as measured self-play performance."
        ),
        "blocked_claim": (
            "Do not claim self-play win-rate or production strategy quality until real OpenSpiel runtime "
            "execution, exactly two trained Phase 1 policy artifacts, long-run simulation volume, seed "
            "stability, and PPO/equivalent policy-update training are complete."
        ),
        "invariants": {},
    }
    payload["proof_cases"] = build_rl_delivery_boundary_proof_cases(payload)
    payload["invariants"] = validate_rl_delivery_boundary(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_rl_delivery_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    delivery_scope = payload.get("delivery_scope") or {}
    proof = payload.get("rl_training_proof") or {}
    permissions = payload.get("claim_permissions") or {}
    required_evidence = set(proof.get("required_evidence") or [])
    missing_requirements = list(proof.get("missing_requirements") or [])
    proof_completed = proof.get("training_proof_completed") is True
    source_self_play_allowed = proof.get("source_self_play_win_rate_claim_allowed") is True
    complete_rl_evidence = proof_completed and source_self_play_allowed
    self_play_allowed = permissions.get("self_play_win_rate_claim_allowed") is True
    production_strategy_allowed = permissions.get("production_strategy_quality_claim_allowed") is True

    if payload.get("gate_name") != RL_DELIVERY_BOUNDARY_NAME:
        violations.append("rl_delivery_boundary_gate_name_must_be_explicit")
    if payload.get("boundary") != "DELIVERY_READY_BUT_RL_PROOF_REQUIRED_FOR_STRATEGY_CLAIMS":
        violations.append("rl_delivery_boundary_statement_must_be_explicit")
    if delivery_scope.get("service_delivery_blocked_by_rl_training_gap") is not False:
        violations.append("rl_training_gap_must_not_block_service_delivery")
    if delivery_scope.get("delivery_claim_allowed") is not True:
        violations.append("delivery_claim_must_remain_allowed")
    if permissions.get("delivery_readiness_claim_allowed") is not True:
        violations.append("delivery_readiness_claim_must_remain_allowed")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("rl_delivery_boundary_must_not_be_current_delivery_blocker")
    if required_evidence != set(REQUIRED_RL_EVIDENCE):
        violations.append("rl_delivery_boundary_must_reference_complete_rl_evidence_set")
    if self_play_allowed and not complete_rl_evidence:
        violations.append("self_play_win_rate_claim_requires_completed_rl_training_proof")
    if production_strategy_allowed and not self_play_allowed:
        violations.append("production_strategy_quality_claim_requires_self_play_win_rate_claim")
    if production_strategy_allowed and payload.get("model_quality_risk") is not False:
        violations.append("approved_production_strategy_claim_must_clear_model_quality_risk")
    if not production_strategy_allowed and payload.get("model_quality_risk") is not True:
        violations.append("blocked_production_strategy_claim_must_remain_model_quality_risk")
    if not complete_rl_evidence and not missing_requirements:
        violations.append("blocked_rl_claim_must_list_missing_requirements")
    if complete_rl_evidence and missing_requirements:
        violations.append("completed_rl_evidence_must_not_list_missing_requirements")
    if "self-play" not in str(payload.get("blocked_claim", "")).lower():
        violations.append("blocked_claim_must_reference_self_play")
    if "production strategy" not in str(payload.get("blocked_claim", "")).lower():
        violations.append("blocked_claim_must_reference_production_strategy_quality")

    blocked_claims = set(payload.get("blocked_claims_without_rl_proof") or [])
    if set(BLOCKED_CLAIMS_WITHOUT_RL_PROOF) != blocked_claims:
        violations.append("blocked_claims_without_rl_proof_must_match_contract")

    proof_case_results = [case.get("result") for case in payload.get("proof_cases") or []]
    if proof_case_results and any(result != "PASS" for result in proof_case_results):
        violations.append("rl_delivery_boundary_proof_cases_must_pass")

    return {"status": "PASS" if not violations else "FAIL", "violations": violations}


def build_rl_delivery_boundary_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_rl_delivery_boundary(candidate)
        observed = candidate["invariants"]["status"]
        cases.append(
            {
                "name": name,
                "expected_status": expected_status,
                "observed_status": observed,
                "result": "PASS" if observed == expected_status else "FAIL",
                "violations": candidate["invariants"]["violations"],
            }
        )

    record("base_contract_valid", _clone(payload), "PASS")

    candidate = _clone(payload)
    candidate["rl_training_proof"]["training_proof_completed"] = False
    candidate["rl_training_proof"]["source_self_play_win_rate_claim_allowed"] = False
    candidate["rl_training_proof"]["missing_requirements"] = ["real_open_spiel_runtime"]
    candidate["claim_permissions"]["self_play_win_rate_claim_allowed"] = True
    candidate["claim_permissions"]["production_strategy_quality_claim_allowed"] = False
    candidate["model_quality_risk"] = True
    record("blocks_self_play_win_rate_claim_without_rl_proof", candidate, "FAIL")

    candidate = _clone(payload)
    candidate["claim_permissions"]["self_play_win_rate_claim_allowed"] = False
    candidate["claim_permissions"]["production_strategy_quality_claim_allowed"] = True
    candidate["model_quality_risk"] = False
    record("blocks_production_strategy_claim_without_self_play_evidence", candidate, "FAIL")

    candidate = _clone(payload)
    candidate["delivery_scope"]["service_delivery_blocked_by_rl_training_gap"] = True
    candidate["current_delivery_blocker"] = True
    record("blocks_turning_rl_gap_into_delivery_blocker", candidate, "FAIL")

    candidate = _clone(payload)
    candidate["rl_training_proof"]["training_proof_completed"] = False
    candidate["rl_training_proof"]["source_self_play_win_rate_claim_allowed"] = False
    candidate["rl_training_proof"]["missing_requirements"] = []
    candidate["claim_permissions"]["self_play_win_rate_claim_allowed"] = False
    candidate["claim_permissions"]["production_strategy_quality_claim_allowed"] = False
    candidate["model_quality_risk"] = True
    record("blocks_silent_missing_requirements_for_pending_rl_proof", candidate, "FAIL")

    completed = _clone(payload)
    completed["rl_training_proof"]["training_proof_completed"] = True
    completed["rl_training_proof"]["source_self_play_win_rate_claim_allowed"] = True
    completed["rl_training_proof"]["missing_requirements"] = []
    completed["claim_permissions"]["self_play_win_rate_claim_allowed"] = True
    completed["claim_permissions"]["production_strategy_quality_claim_allowed"] = True
    completed["model_quality_risk"] = False
    record("allows_claim_after_completed_rl_proof", completed, "PASS")

    return cases


def write_rl_delivery_boundary(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_rl_delivery_boundary(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_rl_delivery_boundary_markdown(payload), encoding="utf-8")
    return payload


def render_rl_delivery_boundary_markdown(payload: dict[str, Any]) -> str:
    permissions = payload["claim_permissions"]
    delivery_scope = payload["delivery_scope"]
    proof = payload["rl_training_proof"]
    lines = [
        "# RL Delivery Boundary",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Boundary: `{payload['boundary']}`",
        f"- Delivery claim allowed: `{permissions['delivery_readiness_claim_allowed']}`",
        f"- Self-play win-rate claim allowed: `{permissions['self_play_win_rate_claim_allowed']}`",
        (
            "- Production strategy-quality claim allowed: "
            f"`{permissions['production_strategy_quality_claim_allowed']}`"
        ),
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## Delivery Scope",
        "",
        delivery_scope["allowed_current_delivery_claim"],
        "",
        "## RL Training Proof",
        "",
        f"- Training proof completed: `{proof['training_proof_completed']}`",
        f"- Missing requirements: `{', '.join(proof['missing_requirements']) or 'none'}`",
        "",
        "## Blocked Claim",
        "",
        payload["blocked_claim"],
        "",
        "## Proof Cases",
        "",
    ]
    for case in payload.get("proof_cases") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, result `{case['result']}`"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required RL delivery boundary source is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))
