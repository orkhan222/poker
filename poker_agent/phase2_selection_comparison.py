from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE2_SELECTION_VERSION = "2026-07-04"
PHASE2_SELECTION_BOUNDARY = "PHASE_2_SELECTION_REQUIRES_COMMON_HOLDOUT_AND_SIMULATION"
PHASE2_SELECTION_STATUS = "STRICT_SELECTION_GATE_IMPLEMENTED"
COMMON_HOLDOUT_ID = "phase2_common_grouped_holdout_v1"
COMMON_SIMULATION_ID = "phase2_common_agent_arena_v1"
CURRENT_DELIVERY_ARCHITECTURE = "routed_policy_bundle"

REQUIRED_CANDIDATES = (
    "llm_decision_agent",
    "supervised_model",
    "rule_based_fallback",
    "routed_policy_bundle",
    "future_rl_agent",
)

REQUIRED_METRICS = (
    "action_accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "calibration_ece",
    "action_distribution_js",
    "bet_size_mae",
    "win_rate",
    "expected_value",
    "latency_ms",
    "seed_stability",
)


def build_phase2_selection_comparison(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    llm_gate = _read_optional_json(reports / "llm_decision_gate.json")
    llm_comparison = _read_optional_json(reports / "llm_architecture_comparison.json")
    production_gate = _read_optional_json(reports / "production_gate.json")
    raw_model = _read_optional_json(reports / "raw_model_status.json")
    deployed_gate = _read_optional_json(reports / "deployed_strategy_gate.json")
    self_play = _read_optional_json(reports / "production_self_play.json")

    candidates = {
        "llm_decision_agent": _candidate(
            name="llm_decision_agent",
            implementation_status="RESEARCH_BASELINE_AVAILABLE" if llm_gate or llm_comparison else "MISSING",
            role="out_of_box_llm_decision_baseline",
            model_family=llm_comparison.get("recommended_model_family") or llm_comparison.get("model_family"),
            selected_for_current_delivery=False,
            metrics={
                "action_accuracy": _coalesce(
                    llm_comparison.get("best_accuracy"),
                    llm_comparison.get("accuracy"),
                    (llm_gate.get("benchmark_metrics") or {}).get("accuracy"),
                ),
                "macro_f1": _coalesce(
                    llm_comparison.get("best_macro_f1"),
                    llm_comparison.get("macro_f1"),
                    (llm_gate.get("benchmark_metrics") or {}).get("macro_f1"),
                ),
                "latency_ms": _coalesce(llm_comparison.get("latency_ms"), llm_gate.get("latency_ms")),
            },
            compared_on_common_holdout=_uses_common_holdout(llm_gate) or _uses_common_holdout(llm_comparison),
            compared_in_common_simulation=_uses_common_simulation(llm_gate) or _uses_common_simulation(llm_comparison),
            limitations=[
                "Out-of-box LLM comparison is research evidence only until it is rerun on the Phase 2 common holdout and common arena.",
                "Prompt/context performance must be compared against non-LLM baselines under identical conditions.",
            ],
        ),
        "supervised_model": _candidate(
            name="supervised_model",
            implementation_status=(raw_model.get("raw_supervised_model") or {}).get("runtime_status", "AVAILABLE"),
            role="standalone_supervised_policy_artifact",
            selected_for_current_delivery=False,
            metrics={
                "action_accuracy": _coalesce(production_gate.get("accuracy"), production_gate.get("valid_accuracy")),
                "macro_f1": _coalesce(production_gate.get("macro_f1"), production_gate.get("valid_macro_f1")),
                "balanced_accuracy": _coalesce(
                    production_gate.get("balanced_accuracy"),
                    production_gate.get("valid_balanced_accuracy"),
                ),
                "calibration_ece": _coalesce(production_gate.get("ece"), production_gate.get("calibration_ece")),
            },
            compared_on_common_holdout=_uses_common_holdout(production_gate) or _uses_common_holdout(raw_model),
            compared_in_common_simulation=_uses_common_simulation(production_gate) or _uses_common_simulation(raw_model),
            limitations=[
                "The raw supervised artifact remains loadable but not standalone production-approved.",
                "It must be compared against routed, rule-based, LLM, and RL candidates on the same holdout and arena before final selection.",
            ],
        ),
        "rule_based_fallback": _candidate(
            name="rule_based_fallback",
            implementation_status="AVAILABLE_AS_SERVICE_FALLBACK",
            role="deterministic_rule_based_baseline",
            selected_for_current_delivery=False,
            metrics={},
            compared_on_common_holdout=False,
            compared_in_common_simulation=False,
            limitations=[
                "Fallback is implemented for safe inference degradation, but it has not been promoted as the final Phase 2 winner.",
                "It must be evaluated as a first-class baseline in the common holdout and simulation run.",
            ],
        ),
        "routed_policy_bundle": _candidate(
            name="routed_policy_bundle",
            implementation_status="CURRENT_DEPLOYED_STACK",
            role="observed_card_policy_plus_public_context_fallback",
            selected_for_current_delivery=True,
            metrics={
                "win_rate": _coalesce(self_play.get("mean_win_rate"), self_play.get("win_rate")),
                "expected_value": _coalesce(self_play.get("expected_value"), self_play.get("ev")),
                "action_distribution_js": _coalesce(
                    deployed_gate.get("action_distribution_js"),
                    deployed_gate.get("human_likeness_js_divergence"),
                ),
            },
            compared_on_common_holdout=_uses_common_holdout(deployed_gate) or _uses_common_holdout(self_play),
            compared_in_common_simulation=_uses_common_simulation(deployed_gate) or _uses_common_simulation(self_play),
            limitations=[
                "This remains the selected delivery stack because it handles missing-card conditions better than a single raw model.",
                "It is not a final Phase 2 architecture winner until every required candidate is rerun under identical selection conditions.",
            ],
        ),
        "future_rl_agent": _candidate(
            name="future_rl_agent",
            implementation_status="NOT_AVAILABLE_YET",
            role="open_spiel_policy_update_agent",
            selected_for_current_delivery=False,
            metrics={},
            compared_on_common_holdout=False,
            compared_in_common_simulation=False,
            limitations=[
                "Requires Phase 3 OpenSpiel runtime, two trained policy artifacts, seed stability, and policy-update training.",
                "No win-rate or final selection claim is allowed before the RL training proof is complete.",
            ],
        ),
    }

    missing_holdout = [
        name for name, candidate in candidates.items() if candidate["compared_on_common_holdout"] is not True
    ]
    missing_simulation = [
        name for name, candidate in candidates.items() if candidate["compared_in_common_simulation"] is not True
    ]

    comparison_gate = {
        "same_holdout_required": True,
        "same_simulation_required": True,
        "all_required_candidates_present": set(candidates) == set(REQUIRED_CANDIDATES),
        "all_candidates_compared_on_common_holdout": not missing_holdout,
        "all_candidates_compared_in_common_simulation": not missing_simulation,
        "missing_common_holdout_candidates": missing_holdout,
        "missing_common_simulation_candidates": missing_simulation,
        "selected_for_current_delivery": CURRENT_DELIVERY_ARCHITECTURE,
        "final_selected_architecture": None,
        "final_selection_claim_allowed": False,
        "current_delivery_blocker": False,
        "model_quality_risk": True,
    }

    payload: dict[str, Any] = {
        "version": PHASE2_SELECTION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boundary": PHASE2_SELECTION_BOUNDARY,
        "status": PHASE2_SELECTION_STATUS,
        "subject": "Phase 2 strict architecture selection comparison",
        "required_candidates": list(REQUIRED_CANDIDATES),
        "required_metrics": list(REQUIRED_METRICS),
        "common_holdout_contract": {
            "id": COMMON_HOLDOUT_ID,
            "same_holdout_required": True,
            "split_type": "stratified_hand_group_holdout",
            "leakage_rule": "Only pre-action observable features may be used for all candidates.",
        },
        "common_simulation_contract": {
            "id": COMMON_SIMULATION_ID,
            "same_simulation_required": True,
            "arena": "agent_only_common_simulation",
            "seed_policy": "same seed list for every candidate",
            "volume_policy": "same number of hands/episodes for every candidate",
        },
        "candidates": candidates,
        "comparison_gate": comparison_gate,
        "allowed_claims": [
            "The current delivery uses routed_policy_bundle as the deployed stack.",
            "Phase 2 final architecture selection requires all required candidates on the same holdout and common simulation.",
            "Current missing common-condition comparisons are model-quality risks, not current delivery blockers.",
        ],
        "blocked_claims": [
            "Phase 2 final architecture selection is complete without comparing LLM, supervised, rule-based fallback, routed policy, and future RL agent under the same holdout and simulation conditions.",
            "The routed policy bundle is the final global winner before every required candidate is evaluated on the common Phase 2 selection contract.",
            "Future RL agent performance is known before the OpenSpiel/agent-only training proof is complete.",
        ],
        "next_actions": [
            "Freeze the common grouped holdout and common simulation configuration.",
            "Run LLM, supervised, rule-based fallback, routed policy, and future RL agent adapters against the same holdout.",
            "Run all candidates in the same agent-only simulation arena with identical seed lists and episode counts.",
            "Select a final Phase 2 winner only after the complete metric bundle is available for every candidate.",
        ],
    }
    payload["invariants"] = validate_phase2_selection_comparison(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    payload["final_contract_result"] = phase2_final_contract_result(payload)
    return payload


def phase2_final_contract_result(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("comparison_gate") or {}
    final_selection_allowed = gate.get("final_selection_claim_allowed") is True
    return {
        "phase2_status": payload.get("overall_status", "UNKNOWN"),
        "strict_comparison_mechanism_ready": payload.get("overall_status") == "PASS",
        "current_delivery_stack": gate.get("selected_for_current_delivery"),
        "final_selection_claim_allowed": final_selection_allowed,
        "final_winner_claim_state": (
            "ALLOWED_AFTER_COMMON_CONDITION_COMPARISON"
            if final_selection_allowed
            else "BLOCKED_PENDING_COMMON_HOLDOUT_AND_SIMULATION"
        ),
        "reason": (
            "The strict comparison mechanism is implemented and passing, but the final architecture "
            "winner claim remains blocked until every required candidate is evaluated on the same "
            "grouped holdout and the same agent-only simulation."
        ),
        "required_before_unlock": [
            "Evaluate LLM decision agent on the common grouped holdout and common simulation.",
            "Evaluate supervised model on the common grouped holdout and common simulation.",
            "Evaluate rule-based fallback on the common grouped holdout and common simulation.",
            "Evaluate routed policy bundle on the common grouped holdout and common simulation.",
            "Evaluate future RL agent after the OpenSpiel/agent-only training proof is complete.",
        ],
    }


def validate_phase2_selection_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    required = set(REQUIRED_CANDIDATES)
    candidates = payload.get("candidates") or {}
    gate = payload.get("comparison_gate") or {}
    holdout = payload.get("common_holdout_contract") or {}
    simulation = payload.get("common_simulation_contract") or {}

    if payload.get("boundary") != PHASE2_SELECTION_BOUNDARY:
        violations.append("phase2_selection_boundary_must_be_present")
    if payload.get("status") != PHASE2_SELECTION_STATUS:
        violations.append("phase2_selection_status_must_be_strict_gate_implemented")
    if set(payload.get("required_candidates") or []) != required:
        violations.append("phase2_selection_required_candidates_must_be_complete")
    if set(candidates) != required:
        violations.append("phase2_selection_candidates_must_match_required_set")
    if holdout.get("id") != COMMON_HOLDOUT_ID or holdout.get("same_holdout_required") is not True:
        violations.append("phase2_selection_must_require_common_holdout")
    if simulation.get("id") != COMMON_SIMULATION_ID or simulation.get("same_simulation_required") is not True:
        violations.append("phase2_selection_must_require_common_simulation")
    if gate.get("same_holdout_required") is not True:
        violations.append("phase2_selection_gate_must_require_same_holdout")
    if gate.get("same_simulation_required") is not True:
        violations.append("phase2_selection_gate_must_require_same_simulation")
    if gate.get("all_required_candidates_present") is not True:
        violations.append("phase2_selection_gate_must_include_all_candidates")
    if gate.get("selected_for_current_delivery") != CURRENT_DELIVERY_ARCHITECTURE:
        violations.append("phase2_selection_current_delivery_architecture_must_be_routed_bundle")
    if (candidates.get("routed_policy_bundle") or {}).get("selected_for_current_delivery") is not True:
        violations.append("phase2_selection_routed_bundle_must_remain_delivery_stack")
    future_rl = candidates.get("future_rl_agent") or {}
    if future_rl.get("implementation_status") != "NOT_AVAILABLE_YET":
        violations.append("phase2_selection_future_rl_must_not_be_claimed_available")
    if future_rl.get("compared_in_common_simulation") is not False:
        violations.append("phase2_selection_future_rl_must_not_have_common_simulation_result_yet")
    if gate.get("final_selection_claim_allowed") is not False:
        violations.append("phase2_selection_final_claim_must_be_blocked_until_common_conditions")
    if gate.get("final_selected_architecture") is not None:
        violations.append("phase2_selection_final_architecture_must_not_be_selected_yet")
    if gate.get("current_delivery_blocker") is not False:
        violations.append("phase2_selection_gap_must_not_block_current_delivery")
    if gate.get("model_quality_risk") is not True:
        violations.append("phase2_selection_gap_must_remain_model_quality_risk")
    if gate.get("all_candidates_compared_on_common_holdout") is not False:
        violations.append("phase2_selection_common_holdout_must_not_be_marked_complete_yet")
    if gate.get("all_candidates_compared_in_common_simulation") is not False:
        violations.append("phase2_selection_common_simulation_must_not_be_marked_complete_yet")
    if not gate.get("missing_common_holdout_candidates"):
        violations.append("phase2_selection_missing_common_holdout_candidates_must_be_listed")
    if not gate.get("missing_common_simulation_candidates"):
        violations.append("phase2_selection_missing_common_simulation_candidates_must_be_listed")
    for candidate_name in required:
        candidate = candidates.get(candidate_name) or {}
        if candidate.get("name") != candidate_name:
            violations.append(f"phase2_selection_candidate_name_mismatch:{candidate_name}")
        if candidate.get("common_holdout_id") != COMMON_HOLDOUT_ID:
            violations.append(f"phase2_selection_candidate_must_reference_common_holdout:{candidate_name}")
        if candidate.get("common_simulation_id") != COMMON_SIMULATION_ID:
            violations.append(f"phase2_selection_candidate_must_reference_common_simulation:{candidate_name}")
        if "metrics" not in candidate:
            violations.append(f"phase2_selection_candidate_metrics_missing:{candidate_name}")
    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def write_phase2_selection_comparison(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_phase2_selection_comparison(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_phase2_selection_comparison_markdown(payload), encoding="utf-8")
    return payload


def render_phase2_selection_comparison_markdown(payload: dict[str, Any]) -> str:
    gate = payload["comparison_gate"]
    final_result = payload.get("final_contract_result") or phase2_final_contract_result(payload)
    lines = [
        "# Phase 2 Strict Selection Comparison",
        "",
        "Phase 2 selection is not considered final until every required candidate is evaluated on the same grouped holdout and the same simulation arena.",
        "",
        f"- Boundary: `{payload['boundary']}`",
        f"- Status: `{payload['status']}`",
        f"- Current delivery architecture: `{gate['selected_for_current_delivery']}`",
        f"- Final selection claim allowed: `{gate['final_selection_claim_allowed']}`",
        f"- Current delivery blocker: `{gate['current_delivery_blocker']}`",
        f"- Model-quality risk: `{gate['model_quality_risk']}`",
        "",
        "## Final Contract Result",
        "",
        f"- Phase 2 status: `{final_result['phase2_status']}`",
        f"- Strict comparison mechanism ready: `{final_result['strict_comparison_mechanism_ready']}`",
        f"- Final selection claim allowed: `{final_result['final_selection_claim_allowed']}`",
        f"- Final winner claim state: `{final_result['final_winner_claim_state']}`",
        f"- Reason: {final_result['reason']}",
        "",
        "## Required Candidates",
        "",
    ]
    for candidate_name, candidate in payload["candidates"].items():
        lines.extend(
            [
                f"### {candidate_name}",
                "",
                f"- Implementation status: `{candidate['implementation_status']}`",
                f"- Common holdout complete: `{candidate['compared_on_common_holdout']}`",
                f"- Common simulation complete: `{candidate['compared_in_common_simulation']}`",
                f"- Selected for current delivery: `{candidate['selected_for_current_delivery']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Missing Common-Condition Evidence",
            "",
            f"- Missing common holdout candidates: `{gate['missing_common_holdout_candidates']}`",
            f"- Missing common simulation candidates: `{gate['missing_common_simulation_candidates']}`",
            "",
            "## Blocked Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _candidate(
    *,
    name: str,
    implementation_status: str,
    role: str,
    selected_for_current_delivery: bool,
    metrics: dict[str, Any],
    compared_on_common_holdout: bool,
    compared_in_common_simulation: bool,
    limitations: list[str],
    model_family: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "implementation_status": implementation_status,
        "model_family": model_family,
        "common_holdout_id": COMMON_HOLDOUT_ID,
        "common_simulation_id": COMMON_SIMULATION_ID,
        "compared_on_common_holdout": compared_on_common_holdout,
        "compared_in_common_simulation": compared_in_common_simulation,
        "selected_for_current_delivery": selected_for_current_delivery,
        "metrics": {metric: metrics.get(metric) for metric in REQUIRED_METRICS},
        "limitations": limitations,
    }


def _uses_common_holdout(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    split = payload.get("split") or payload.get("split_info") or payload.get("holdout") or {}
    return (
        payload.get("common_holdout_id") == COMMON_HOLDOUT_ID
        or split.get("common_holdout_id") == COMMON_HOLDOUT_ID
        or split.get("id") == COMMON_HOLDOUT_ID
    )


def _uses_common_simulation(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    simulation = payload.get("simulation") or payload.get("arena") or {}
    return (
        payload.get("common_simulation_id") == COMMON_SIMULATION_ID
        or simulation.get("common_simulation_id") == COMMON_SIMULATION_ID
        or simulation.get("id") == COMMON_SIMULATION_ID
    )


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
