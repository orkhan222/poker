from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.rl_training_evidence_gate import (
    DEFAULT_POLICY_UPDATE_ALGORITHM,
    MINIMUM_RL_LONG_RUN_EPISODES,
    MINIMUM_RL_STABILITY_SEEDS,
    REQUIRED_PHASE1_POLICY_ARTIFACTS,
    SUPPORTED_POLICY_UPDATE_ALGORITHMS,
)


OPEN_SPIEL_CLAIM_READINESS_VERSION = "2026-07-09"
OPEN_SPIEL_CLAIM_READINESS_GATE = "open_spiel_claim_readiness"
DEFAULT_AGENT_A_ARTIFACT = Path("models") / "phase1_llm_policy_a.joblib"
DEFAULT_AGENT_B_ARTIFACT = Path("models") / "phase1_llm_policy_b.joblib"


def build_open_spiel_claim_readiness(
    project_root: Path,
    *,
    agent_a_model_path: str | Path | None = None,
    agent_b_model_path: str | Path | None = None,
    episodes: int = MINIMUM_RL_LONG_RUN_EPISODES,
    independent_seed_count: int = MINIMUM_RL_STABILITY_SEEDS,
    policy_update_training_completed: bool = False,
    policy_update_algorithm: str = DEFAULT_POLICY_UPDATE_ALGORITHM,
    pyspiel_runtime_available: bool | None = None,
) -> dict[str, Any]:
    agent_a = _resolve_project_path(project_root, agent_a_model_path or DEFAULT_AGENT_A_ARTIFACT)
    agent_b = _resolve_project_path(project_root, agent_b_model_path or DEFAULT_AGENT_B_ARTIFACT)
    artifact_paths = [agent_a, agent_b]
    existing_artifacts = [path for path in artifact_paths if path.exists()]
    runtime_available = _pyspiel_available() if pyspiel_runtime_available is None else bool(pyspiel_runtime_available)
    algorithm = _canonical_algorithm(policy_update_algorithm)
    algorithm_supported = algorithm in SUPPORTED_POLICY_UPDATE_ALGORITHMS
    long_run_ready = int(episodes) >= MINIMUM_RL_LONG_RUN_EPISODES
    seed_stability_ready = int(independent_seed_count) >= MINIMUM_RL_STABILITY_SEEDS
    two_artifacts_ready = len(existing_artifacts) == REQUIRED_PHASE1_POLICY_ARTIFACTS
    ppo_update_ready = bool(policy_update_training_completed) and algorithm_supported

    missing_requirements = _missing_requirements(
        runtime_available=runtime_available,
        two_artifacts_ready=two_artifacts_ready,
        long_run_ready=long_run_ready,
        seed_stability_ready=seed_stability_ready,
        ppo_update_ready=ppo_update_ready,
    )
    claim_ready = not missing_requirements
    payload: dict[str, Any] = {
        "version": OPEN_SPIEL_CLAIM_READINESS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_name": OPEN_SPIEL_CLAIM_READINESS_GATE,
        "status": "READY_TO_CLAIM" if claim_ready else "NOT_READY_TO_CLAIM",
        "claim_ready": claim_ready,
        "current_delivery_blocker": False,
        "model_quality_risk": not claim_ready,
        "runtime": {
            "pyspiel_required": True,
            "pyspiel_available": runtime_available,
        },
        "phase1_policy_artifacts": {
            "required_count": REQUIRED_PHASE1_POLICY_ARTIFACTS,
            "existing_count": len(existing_artifacts),
            "agent_a_model_path": _display_path(project_root, agent_a),
            "agent_b_model_path": _display_path(project_root, agent_b),
            "agent_a_exists": agent_a.exists(),
            "agent_b_exists": agent_b.exists(),
        },
        "simulation_profile": {
            "episodes": int(episodes),
            "minimum_episodes": MINIMUM_RL_LONG_RUN_EPISODES,
            "long_run_ready": long_run_ready,
            "independent_seed_count": int(independent_seed_count),
            "minimum_independent_seed_count": MINIMUM_RL_STABILITY_SEEDS,
            "seed_stability_ready": seed_stability_ready,
        },
        "policy_update_training": {
            "required": True,
            "completed": bool(policy_update_training_completed),
            "algorithm": algorithm,
            "algorithm_supported": algorithm_supported,
            "ppo_or_equivalent_ready": ppo_update_ready,
        },
        "missing_requirements": missing_requirements,
        "claim_command": build_claim_command(
            agent_a_model_path=_display_path(project_root, agent_a),
            agent_b_model_path=_display_path(project_root, agent_b),
            episodes=int(episodes),
            independent_seed_count=int(independent_seed_count),
            policy_update_algorithm=algorithm,
        ),
        "allowed_current_claim": (
            "The service delivery can remain approved. OpenSpiel/RL self-play win-rate and final "
            "production strategy-quality claims require this readiness gate to pass and the claim-mode "
            "arena run to complete successfully."
        ),
        "blocked_claim": (
            "Do not claim OpenSpiel self-play win-rate or production strategy quality until pyspiel, "
            "two trained Phase 1 policy artifacts, 5000+ episodes, 5+ independent seeds, and "
            "PPO/equivalent policy-update training are all complete."
        ),
        "invariants": {},
    }
    payload["proof_cases"] = build_open_spiel_claim_readiness_proof_cases(payload)
    payload["invariants"] = validate_open_spiel_claim_readiness(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def build_claim_command(
    *,
    agent_a_model_path: str,
    agent_b_model_path: str,
    episodes: int,
    independent_seed_count: int,
    policy_update_algorithm: str,
) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe scripts\\build_phase3_open_spiel_arena.py "
        "--claim-mode --run-if-available --phase1-adapters-ready "
        f"--agent-a-model-path {agent_a_model_path} "
        f"--agent-b-model-path {agent_b_model_path} "
        f"--episodes {episodes} "
        f"--independent-seed-count {independent_seed_count} "
        "--policy-update-training-completed "
        f"--policy-update-algorithm {policy_update_algorithm}"
    )


def validate_open_spiel_claim_readiness(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    runtime = payload.get("runtime") or {}
    artifacts = payload.get("phase1_policy_artifacts") or {}
    simulation = payload.get("simulation_profile") or {}
    update = payload.get("policy_update_training") or {}
    missing = list(payload.get("missing_requirements") or [])
    claim_ready = payload.get("claim_ready") is True

    required_ready = (
        runtime.get("pyspiel_available") is True,
        int(artifacts.get("existing_count") or 0) == REQUIRED_PHASE1_POLICY_ARTIFACTS,
        artifacts.get("agent_a_exists") is True,
        artifacts.get("agent_b_exists") is True,
        simulation.get("long_run_ready") is True,
        simulation.get("seed_stability_ready") is True,
        update.get("ppo_or_equivalent_ready") is True,
    )
    all_ready = all(required_ready)

    if payload.get("gate_name") != OPEN_SPIEL_CLAIM_READINESS_GATE:
        violations.append("open_spiel_claim_readiness_gate_name_must_be_explicit")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("open_spiel_claim_readiness_must_not_block_service_delivery")
    if int(artifacts.get("required_count") or 0) != REQUIRED_PHASE1_POLICY_ARTIFACTS:
        violations.append("claim_readiness_must_require_exactly_two_phase1_artifacts")
    if int(simulation.get("minimum_episodes") or 0) < MINIMUM_RL_LONG_RUN_EPISODES:
        violations.append("claim_readiness_episode_threshold_too_low")
    if int(simulation.get("minimum_independent_seed_count") or 0) < MINIMUM_RL_STABILITY_SEEDS:
        violations.append("claim_readiness_seed_threshold_too_low")
    if update.get("required") is not True:
        violations.append("claim_readiness_must_require_policy_update_training")
    if update.get("algorithm_supported") is not (str(update.get("algorithm")) in SUPPORTED_POLICY_UPDATE_ALGORITHMS):
        violations.append("claim_readiness_algorithm_support_must_match_supported_set")
    if claim_ready != all_ready:
        violations.append("claim_ready_must_match_all_required_evidence")
    if claim_ready and missing:
        violations.append("ready_claim_must_not_list_missing_requirements")
    if not claim_ready and not missing:
        violations.append("blocked_claim_must_list_missing_requirements")
    if claim_ready and payload.get("model_quality_risk") is not False:
        violations.append("ready_claim_must_clear_model_quality_risk")
    if not claim_ready and payload.get("model_quality_risk") is not True:
        violations.append("blocked_claim_must_remain_model_quality_risk")
    if "build_phase3_open_spiel_arena.py" not in str(payload.get("claim_command", "")):
        violations.append("claim_readiness_must_emit_claim_mode_command")
    if "--claim-mode" not in str(payload.get("claim_command", "")):
        violations.append("claim_readiness_command_must_use_claim_mode")
    if "win-rate" not in str(payload.get("blocked_claim", "")).lower():
        violations.append("blocked_claim_must_reference_win_rate")
    if "production strategy" not in str(payload.get("blocked_claim", "")).lower():
        violations.append("blocked_claim_must_reference_production_strategy_quality")

    proof_case_results = [case.get("result") for case in payload.get("proof_cases") or []]
    if proof_case_results and any(result != "PASS" for result in proof_case_results):
        violations.append("open_spiel_claim_readiness_proof_cases_must_pass")

    return {"status": "PASS" if not violations else "FAIL", "violations": violations}


def build_open_spiel_claim_readiness_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_open_spiel_claim_readiness(candidate)
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

    false_ready = _clone(payload)
    false_ready["claim_ready"] = True
    false_ready["model_quality_risk"] = False
    false_ready["missing_requirements"] = []
    false_ready["runtime"]["pyspiel_available"] = False
    record("blocks_claim_ready_without_pyspiel", false_ready, "FAIL")

    delivery_blocker = _clone(payload)
    delivery_blocker["current_delivery_blocker"] = True
    record("blocks_turning_claim_gap_into_delivery_blocker", delivery_blocker, "FAIL")

    complete = _clone(payload)
    complete["claim_ready"] = True
    complete["status"] = "READY_TO_CLAIM"
    complete["model_quality_risk"] = False
    complete["missing_requirements"] = []
    complete["runtime"]["pyspiel_available"] = True
    complete["phase1_policy_artifacts"]["existing_count"] = REQUIRED_PHASE1_POLICY_ARTIFACTS
    complete["phase1_policy_artifacts"]["agent_a_exists"] = True
    complete["phase1_policy_artifacts"]["agent_b_exists"] = True
    complete["simulation_profile"]["long_run_ready"] = True
    complete["simulation_profile"]["seed_stability_ready"] = True
    complete["policy_update_training"]["completed"] = True
    complete["policy_update_training"]["algorithm"] = "PPO"
    complete["policy_update_training"]["algorithm_supported"] = True
    complete["policy_update_training"]["ppo_or_equivalent_ready"] = True
    record("allows_claim_ready_after_complete_evidence", complete, "PASS")

    return cases


def write_open_spiel_claim_readiness(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_open_spiel_claim_readiness(project_root, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_open_spiel_claim_readiness_markdown(payload), encoding="utf-8")
    return payload


def render_open_spiel_claim_readiness_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OpenSpiel Claim Readiness",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Claim ready: `{payload['claim_ready']}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        f"- Model-quality risk: `{payload['model_quality_risk']}`",
        "",
        "## Missing Requirements",
        "",
    ]
    for item in payload["missing_requirements"]:
        lines.append(f"- `{item}`")
    if not payload["missing_requirements"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Claim Command",
            "",
            "```powershell",
            payload["claim_command"],
            "```",
            "",
            "## Blocked Claim",
            "",
            payload["blocked_claim"],
            "",
            "## Proof Cases",
            "",
        ]
    )
    for case in payload.get("proof_cases") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, result `{case['result']}`"
        )
    return "\n".join(lines) + "\n"


def _missing_requirements(
    *,
    runtime_available: bool,
    two_artifacts_ready: bool,
    long_run_ready: bool,
    seed_stability_ready: bool,
    ppo_update_ready: bool,
) -> list[str]:
    missing: list[str] = []
    if not runtime_available:
        missing.append("pyspiel_runtime")
    if not two_artifacts_ready:
        missing.append("two_phase1_trained_policy_artifacts")
    if not long_run_ready:
        missing.append("5000_or_more_episodes")
    if not seed_stability_ready:
        missing.append("5_or_more_independent_seeds")
    if not ppo_update_ready:
        missing.append("ppo_or_equivalent_policy_update_training")
    return missing


def _pyspiel_available() -> bool:
    return importlib.util.find_spec("pyspiel") is not None


def _canonical_algorithm(value: str | None) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_POLICY_UPDATE_ALGORITHM
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))
