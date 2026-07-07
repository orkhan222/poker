from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.rl_training_evidence_gate import (
    REQUIRED_RL_EVIDENCE,
    RL_TRAINING_PROOF_COMPLETED_STATUS,
)


OPEN_SPIEL_CLAIM_CONTRACT_VERSION = "2026-07-07"
OPEN_SPIEL_CLAIM_GATE_NAME = "open_spiel_self_play_claim_contract"
PHASE3_REPORT_PATH = Path("reports") / "phase3_open_spiel_arena.json"


def build_open_spiel_claim_contract(project_root: Path) -> dict[str, Any]:
    phase3_report_path = project_root / PHASE3_REPORT_PATH
    phase3_report = _read_json(phase3_report_path)
    proof = phase3_report.get("rl_training_proof_boundary") or {}
    arena_contract = phase3_report.get("arena_contract") or {}
    runtime_boundary = phase3_report.get("runtime_boundary") or {}
    training_proof_completed = proof.get("status") == RL_TRAINING_PROOF_COMPLETED_STATUS
    win_rate_claim_allowed = proof.get("measured_win_rate_claim_allowed") is True

    payload = {
        "version": OPEN_SPIEL_CLAIM_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_name": OPEN_SPIEL_CLAIM_GATE_NAME,
        "source_report": PHASE3_REPORT_PATH.as_posix(),
        "source_report_status": phase3_report.get("status"),
        "source_report_overall_status": phase3_report.get("overall_status"),
        "code_status": "READY",
        "arena_code_ready": True,
        "arena_type": arena_contract.get("arena_type"),
        "agent_only_table": arena_contract.get("agent_only_table") is True,
        "all_seats_controlled_by_agents": arena_contract.get("all_seats_controlled_by_agents") is True,
        "human_players_present": arena_contract.get("human_players_present"),
        "fixed_scripted_opponents_present": arena_contract.get("fixed_scripted_opponents_present"),
        "rl_training_proof_status": proof.get("status"),
        "training_proof_completed": training_proof_completed,
        "self_play_win_rate_claim_allowed": win_rate_claim_allowed,
        "measured_open_spiel_metrics_present": "metrics" in phase3_report,
        "real_open_spiel_runtime_available": proof.get("real_open_spiel_runtime_available") is True,
        "phase1_trained_policy_artifacts_attached": (
            proof.get("phase1_trained_policy_artifacts_attached") is True
        ),
        "seed_stability_evaluated": proof.get("seed_stability_evaluated") is True,
        "long_run_completed": proof.get("long_run_completed") is True,
        "policy_update_training_completed": proof.get("policy_update_training_completed") is True,
        "required_evidence_before_self_play_claim": list(REQUIRED_RL_EVIDENCE),
        "minimum_independent_seeds": proof.get("minimum_independent_seeds"),
        "minimum_long_run_episodes": proof.get("minimum_long_run_episodes"),
        "configured_episodes": proof.get("episodes"),
        "configured_independent_seed_count": proof.get("independent_seed_count"),
        "missing_requirements": list(proof.get("missing_requirements") or []),
        "runtime_boundary": runtime_boundary,
        "current_delivery_blocker": False,
        "model_quality_risk": not win_rate_claim_allowed,
        "allowed_current_claim": (
            "The OpenSpiel/RL code path and agent-only arena contract are implemented and ready "
            "for a measured run."
        ),
        "blocked_claim": (
            "Do not claim Phase 3 self-play win-rate, RL training success, or production strategy "
            "quality until a real OpenSpiel runtime run uses two trained Phase 1 policy artifacts, "
            "long-run volume, seed stability, and policy-update training."
        ),
        "invariants": {},
    }
    payload["invariants"] = validate_open_spiel_claim_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    payload["proof_cases"] = build_open_spiel_claim_proof_cases(payload)
    payload["invariants"] = validate_open_spiel_claim_contract(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def write_open_spiel_claim_contract(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_open_spiel_claim_contract(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_open_spiel_claim_contract_markdown(payload), encoding="utf-8")
    return payload


def validate_open_spiel_claim_contract(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    required_evidence = set(payload.get("required_evidence_before_self_play_claim") or [])
    required_training_flags = (
        payload.get("real_open_spiel_runtime_available") is True,
        payload.get("phase1_trained_policy_artifacts_attached") is True,
        payload.get("agent_only_table") is True,
        payload.get("seed_stability_evaluated") is True,
        payload.get("long_run_completed") is True,
        payload.get("policy_update_training_completed") is True,
    )
    complete_training_proof = all(required_training_flags)

    if payload.get("gate_name") != OPEN_SPIEL_CLAIM_GATE_NAME:
        violations.append("open_spiel_claim_gate_name_must_be_explicit")
    if payload.get("code_status") != "READY":
        violations.append("open_spiel_arena_code_must_be_marked_ready")
    if payload.get("arena_code_ready") is not True:
        violations.append("open_spiel_arena_code_ready_flag_must_be_true")
    if payload.get("source_report") != PHASE3_REPORT_PATH.as_posix():
        violations.append("open_spiel_claim_contract_must_reference_phase3_report")
    if payload.get("source_report_overall_status") != "PASS":
        violations.append("source_phase3_report_must_pass_its_own_invariants")
    if payload.get("arena_type") != "AGENT_ONLY_OPEN_SPIEL_ARENA":
        violations.append("open_spiel_claim_contract_must_bind_agent_only_arena")
    if payload.get("agent_only_table") is not True:
        violations.append("open_spiel_claim_contract_must_require_agent_only_table")
    if payload.get("all_seats_controlled_by_agents") is not True:
        violations.append("all_open_spiel_seats_must_be_agent_controlled")
    if payload.get("human_players_present") is not False:
        violations.append("human_players_must_not_be_present_in_arena_claim")
    if payload.get("fixed_scripted_opponents_present") is not False:
        violations.append("fixed_scripted_opponents_must_not_be_present_in_arena_claim")
    if required_evidence != set(REQUIRED_RL_EVIDENCE):
        violations.append("open_spiel_claim_contract_must_require_complete_rl_evidence_set")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("missing_rl_training_proof_must_not_block_current_delivery")
    if int(payload.get("minimum_independent_seeds") or 0) < 5:
        violations.append("open_spiel_claim_contract_seed_threshold_too_low")
    if int(payload.get("minimum_long_run_episodes") or 0) < 5000:
        violations.append("open_spiel_claim_contract_episode_threshold_too_low")
    if payload.get("self_play_win_rate_claim_allowed") is True and not complete_training_proof:
        violations.append("self_play_win_rate_claim_requires_complete_training_proof")
    if payload.get("training_proof_completed") is True and not complete_training_proof:
        violations.append("training_proof_completed_requires_all_evidence")
    if complete_training_proof and payload.get("self_play_win_rate_claim_allowed") is not True:
        violations.append("complete_training_proof_must_allow_measured_self_play_claim")
    if not complete_training_proof and payload.get("model_quality_risk") is not True:
        violations.append("incomplete_open_spiel_training_proof_must_remain_model_quality_risk")
    if complete_training_proof and payload.get("model_quality_risk") is not False:
        violations.append("completed_open_spiel_training_proof_must_clear_model_quality_risk")
    if payload.get("self_play_win_rate_claim_allowed") is False and not payload.get("missing_requirements"):
        violations.append("blocked_self_play_claim_must_list_missing_requirements")
    if "win-rate" not in str(payload.get("blocked_claim", "")).lower():
        violations.append("blocked_claim_must_explicitly_reference_win_rate")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def build_open_spiel_claim_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    def cloned() -> dict[str, Any]:
        return json.loads(json.dumps(payload))

    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_open_spiel_claim_contract(candidate)
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

    record("base_contract_valid", cloned(), "PASS")
    for name, field in (
        ("blocks_claim_without_real_open_spiel_runtime", "real_open_spiel_runtime_available"),
        ("blocks_claim_without_two_phase1_policy_artifacts", "phase1_trained_policy_artifacts_attached"),
        ("blocks_claim_without_agent_only_table", "agent_only_table"),
        ("blocks_claim_without_seed_stability", "seed_stability_evaluated"),
        ("blocks_claim_without_long_run", "long_run_completed"),
        ("blocks_claim_without_policy_update_training", "policy_update_training_completed"),
    ):
        candidate = cloned()
        candidate[field] = False
        candidate["self_play_win_rate_claim_allowed"] = True
        candidate["training_proof_completed"] = True
        candidate["model_quality_risk"] = False
        record(name, candidate, "FAIL")
    return cases


def render_open_spiel_claim_contract_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OpenSpiel/RL Claim Contract",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Code status: `{payload['code_status']}`",
        f"- Agent-only table: `{payload['agent_only_table']}`",
        f"- RL training proof status: `{payload['rl_training_proof_status']}`",
        f"- Self-play win-rate claim allowed: `{payload['self_play_win_rate_claim_allowed']}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## Allowed Current Claim",
        "",
        payload["allowed_current_claim"],
        "",
        "## Blocked Claim",
        "",
        payload["blocked_claim"],
        "",
        "## Required Evidence Before Self-Play Claim",
        "",
    ]
    for item in payload["required_evidence_before_self_play_claim"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Missing Requirements", ""])
    for item in payload["missing_requirements"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Proof Cases", ""])
    for case in payload.get("proof_cases") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, result `{case['result']}`"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required OpenSpiel claim source report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
