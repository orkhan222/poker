from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from poker_agent.features import card_rank, card_suit, made_hand_category
from poker_agent.schemas import PredictionRequest


ACTION_ORDER = ("fold", "call", "check", "bet", "raise")
REPORT_VERSION = "2026-07-05"
SCENARIO_SANITY_PROBLEM = {
    "problem": (
        "The deployed strategy stack can satisfy broad delivery checks while still making obvious mistakes "
        "in high-confidence poker spots."
    ),
    "examples": [
        "Premium pairs may retain too much fold probability when facing a preflop raise.",
        "Nut flush draws may be folded too often without an explicit continue invariant.",
        "Missed weak river hands may over-continue against large bets without a river discipline invariant.",
    ],
    "implemented_control": "critical_spot_guardrail",
    "control_scope": "narrow_high_confidence_holdem_invariants",
}
SCENARIO_SANITY_REQUIRED_FINAL_PROOF = [
    "larger grouped holdout action-alignment evaluation",
    "macro F1 and balanced accuracy by action slice",
    "calibration and expected calibration error",
    "bet-sizing MAE by street and position",
    "EV and win-rate in self-play",
    "seed-stability across repeated runs",
]


class PolicyAgent(Protocol):
    def predict(self, request: PredictionRequest) -> Any:
        ...


@dataclass(frozen=True)
class ScenarioExpectation:
    accepted_actions: tuple[str, ...]
    max_fold_probability: float | None = None
    min_fold_probability: float | None = None
    min_aggressive_probability: float | None = None
    max_aggressive_probability: float | None = None
    min_continue_probability: float | None = None
    max_continue_probability: float | None = None


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    description: str
    payload: dict[str, Any]
    expected_behavior: str
    expectation: ScenarioExpectation


def apply_critical_spot_guardrail(
    request: PredictionRequest,
    probabilities: dict[str, float],
) -> tuple[str, dict[str, float], list[str]]:
    """Apply conservative poker invariants for high-confidence sanity spots.

    The guardrail is intentionally narrow. It does not replace model training;
    it prevents obvious strategy failures in spots where standard Hold'em play
    is unambiguous enough to enforce at inference time.
    """
    normalized = _normalize_probabilities(probabilities)
    guardrails: list[str] = []

    if _is_premium_pair_facing_raise(request):
        guardrails.append("premium_pair_preflop_3bet")
        normalized = _blend(
            normalized,
            {"fold": 0.01, "call": 0.12, "check": 0.01, "bet": 0.04, "raise": 0.82},
            strength=0.95,
        )
    elif _is_trash_hand_facing_raise(request):
        guardrails.append("trash_hand_preflop_fold")
        normalized = _blend(
            normalized,
            {"fold": 0.90, "call": 0.06, "check": 0.01, "bet": 0.01, "raise": 0.02},
            strength=0.90,
        )
    elif _is_nut_flush_draw_facing_bet(request):
        guardrails.append("nut_flush_draw_continue")
        normalized = _blend(
            normalized,
            {"fold": 0.03, "call": 0.64, "check": 0.02, "bet": 0.08, "raise": 0.23},
            strength=0.88,
        )
    elif _is_missed_river_facing_large_bet(request):
        guardrails.append("missed_river_fold")
        normalized = _blend(
            normalized,
            {"fold": 0.88, "call": 0.07, "check": 0.01, "bet": 0.02, "raise": 0.02},
            strength=0.90,
        )

    action = max(normalized, key=normalized.get)
    return action, normalized, guardrails


def build_scenario_definitions() -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            scenario_id="pocket_aces_preflop_bb_facing_raise",
            name="Pocket Aces preflop facing raise",
            description="Ah As in BB position facing a 3.0 BB open raise.",
            payload={
                "position": "BB",
                "street": "preflop",
                "hole_cards": ["Ah", "As"],
                "board_cards": [],
                "pot": 4.5,
                "to_call": 3.0,
                "stack": 100.0,
                "min_raise": 6.0,
                "player_count": 6,
                "betting_history": [
                    {"player_position": "UTG", "action": "raise", "amount": 3.0, "street": "preflop"}
                ],
            },
            expected_behavior="Aggressive re-raise/3-bet; fold probability must be close to zero.",
            expectation=ScenarioExpectation(
                accepted_actions=("raise", "bet", "all_in"),
                max_fold_probability=0.05,
                min_aggressive_probability=0.55,
            ),
        ),
        ScenarioDefinition(
            scenario_id="trash_72o_preflop_sb_facing_raise",
            name="Trash hand preflop facing raise",
            description="7c 2d in SB position facing a 4.0 BB raise.",
            payload={
                "position": "SB",
                "street": "preflop",
                "hole_cards": ["7c", "2d"],
                "board_cards": [],
                "pot": 5.5,
                "to_call": 4.0,
                "stack": 100.0,
                "min_raise": 8.0,
                "player_count": 6,
                "betting_history": [
                    {"player_position": "BTN", "action": "raise", "amount": 4.0, "street": "preflop"}
                ],
            },
            expected_behavior="High-confidence fold with the weakest offsuit starting hand.",
            expectation=ScenarioExpectation(
                accepted_actions=("fold",),
                min_fold_probability=0.60,
                max_aggressive_probability=0.25,
            ),
        ),
        ScenarioDefinition(
            scenario_id="nut_flush_draw_flop_facing_bet",
            name="Nut flush draw on flop facing bet",
            description="Kd Qd on Ad 8d 2c facing a 6.0 BB flop bet.",
            payload={
                "position": "BTN",
                "street": "flop",
                "hole_cards": ["Kd", "Qd"],
                "board_cards": ["Ad", "8d", "2c"],
                "pot": 14.0,
                "to_call": 6.0,
                "stack": 94.0,
                "min_raise": 12.0,
                "player_count": 6,
                "betting_history": [
                    {"player_position": "CO", "action": "bet", "amount": 6.0, "street": "flop"}
                ],
            },
            expected_behavior="Continue against the bet, usually call with optional semi-bluff raise frequency.",
            expectation=ScenarioExpectation(
                accepted_actions=("call", "raise", "bet"),
                max_fold_probability=0.15,
                min_continue_probability=0.55,
            ),
        ),
        ScenarioDefinition(
            scenario_id="missed_river_facing_large_bet",
            name="Missed river facing large bet",
            description="7c 3c on Ks Jh 5d 2s 9c facing a 20.0 BB river bet.",
            payload={
                "position": "BB",
                "street": "river",
                "hole_cards": ["7c", "3c"],
                "board_cards": ["Ks", "Jh", "5d", "2s", "9c"],
                "pot": 42.0,
                "to_call": 20.0,
                "stack": 80.0,
                "min_raise": 40.0,
                "player_count": 6,
                "betting_history": [
                    {"player_position": "BTN", "action": "bet", "amount": 20.0, "street": "river"}
                ],
            },
            expected_behavior="Fold; do not bluff-catch with a completely missed weak hand.",
            expectation=ScenarioExpectation(
                accepted_actions=("fold",),
                min_fold_probability=0.60,
                max_continue_probability=0.30,
            ),
        ),
    ]


def evaluate_scenario_sanity(agent: PolicyAgent) -> dict[str, Any]:
    cases = [_evaluate_case(agent, scenario) for scenario in build_scenario_definitions()]
    passed = all(case["passed"] for case in cases)
    return {
        "version": REPORT_VERSION,
        "overall_status": "PASS" if passed else "NEEDS_CALIBRATION",
        "scenario_count": len(cases),
        "passed_scenarios": sum(1 for case in cases if case["passed"]),
        "failed_scenarios": [case["scenario_id"] for case in cases if not case["passed"]],
        "problem_statement": SCENARIO_SANITY_PROBLEM,
        "boundary": {
            "validation_type": "TARGETED_SCENARIO_SANITY",
            "evidence_level": "targeted_regression_gate",
            "claim_allowed": "critical_spot_sanity_passed",
            "full_production_strategy_proof": False,
            "final_strategy_quality_claim_allowed": False,
            "current_delivery_blocker": False,
            "model_quality_risk_if_failed": True,
            "claim_not_allowed": [
                "full_production_strategy_quality",
                "profitable_poker_policy",
                "complete_human_likeness_proof",
                "self_play_ev_approval",
            ],
            "required_for_final_strategy_claim": SCENARIO_SANITY_REQUIRED_FINAL_PROOF,
            "reason": (
                "These scenarios verify critical poker invariants, but they do not replace broad holdout, "
                "calibration, self-play, EV, and seed-stability evaluation."
            ),
        },
        "cases": cases,
    }


def build_scenario_sanity(project_root: Path, model_path: Path | None = None) -> dict[str, Any]:
    from poker_agent.agents import MLPolicyAgent

    resolved_model = model_path or _default_model_path(project_root)
    agent = MLPolicyAgent.from_path(resolved_model)
    payload = evaluate_scenario_sanity(agent)
    payload["model_path"] = str(resolved_model)
    return payload


def write_scenario_sanity(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    payload = build_scenario_sanity(project_root, model_path=model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def validate_scenario_sanity(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("overall_status") != "PASS":
        errors.append(f"overall_status={payload.get('overall_status')}")
    boundary = payload.get("boundary") or {}
    if boundary.get("full_production_strategy_proof") is not False:
        errors.append("targeted scenario sanity must not be treated as full production proof")
    if boundary.get("current_delivery_blocker") is not False:
        errors.append("scenario sanity must not block software delivery by itself")
    if boundary.get("final_strategy_quality_claim_allowed") is not False:
        errors.append("scenario sanity alone must not allow final strategy-quality claims")
    if boundary.get("evidence_level") != "targeted_regression_gate":
        errors.append("scenario sanity must be represented as targeted regression evidence")
    if boundary.get("claim_allowed") != "critical_spot_sanity_passed":
        errors.append("scenario sanity must only allow a critical-spot sanity claim")
    if not boundary.get("claim_not_allowed"):
        errors.append("scenario sanity must list claims that are not allowed")
    required = set(SCENARIO_SANITY_REQUIRED_FINAL_PROOF)
    observed_required = set(boundary.get("required_for_final_strategy_claim") or [])
    if not required.issubset(observed_required):
        errors.append("scenario sanity boundary must preserve final strategy proof requirements")
    problem = payload.get("problem_statement") or {}
    if problem.get("implemented_control") != "critical_spot_guardrail":
        errors.append("scenario sanity must document the implemented critical-spot control")
    if problem.get("control_scope") != "narrow_high_confidence_holdem_invariants":
        errors.append("scenario sanity must document the narrow control scope")
    cases = payload.get("cases") or []
    expected_ids = {scenario.scenario_id for scenario in build_scenario_definitions()}
    observed_ids = {case.get("scenario_id") for case in cases}
    if observed_ids != expected_ids:
        errors.append(f"scenario set mismatch: expected={sorted(expected_ids)} observed={sorted(observed_ids)}")
    for case in cases:
        if case.get("passed") is not True:
            errors.append(f"scenario failed: {case.get('scenario_id')}")
        if not case.get("guardrails"):
            errors.append(f"scenario did not exercise a guardrail: {case.get('scenario_id')}")
    return errors


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Scenario Sanity Validation",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Passed scenarios: `{payload['passed_scenarios']}/{payload['scenario_count']}`",
        f"- Model path: `{payload.get('model_path', 'unknown')}`",
        "",
        "This is targeted strategy sanity evidence. It does not replace full production strategy approval.",
        "",
        "## Problem Boundary",
        "",
        f"- Problem: {payload['problem_statement']['problem']}",
        f"- Implemented control: `{payload['problem_statement']['implemented_control']}`",
        f"- Control scope: `{payload['problem_statement']['control_scope']}`",
        f"- Allowed claim: `{payload['boundary']['claim_allowed']}`",
        "- Not allowed claims: "
        + ", ".join(f"`{claim}`" for claim in payload["boundary"]["claim_not_allowed"]),
        "",
        "Final strategy-quality claims still require:",
        "",
    ]
    for requirement in payload["boundary"]["required_for_final_strategy_claim"]:
        lines.append(f"- {requirement}")
    lines.extend(
        [
            "",
        "## Scenarios",
        "",
        ]
    )
    for case in payload["cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                f"- Result: `{'PASS' if case['passed'] else 'FAIL'}`",
                f"- Action: `{case['actual_action']}`",
                f"- Expected: {case['expected_behavior']}",
                f"- Guardrails: `{', '.join(case['guardrails']) or 'none'}`",
                f"- Fold probability: `{case['probabilities'].get('fold', 0.0):.4f}`",
                f"- Continue probability: `{case['derived_metrics']['continue_probability']:.4f}`",
                f"- Aggressive probability: `{case['derived_metrics']['aggressive_probability']:.4f}`",
                "",
            ]
        )
        if case["failures"]:
            for failure in case["failures"]:
                lines.append(f"- Failure: {failure}")
            lines.append("")
    return "\n".join(lines) + "\n"


def _evaluate_case(agent: PolicyAgent, scenario: ScenarioDefinition) -> dict[str, Any]:
    request = PredictionRequest.from_dict(scenario.payload)
    response = agent.predict(request).to_dict()
    action = _canonical_action(response.get("action"))
    probabilities = _normalize_probabilities(response.get("probabilities") or {})
    expectation = scenario.expectation
    aggressive_probability = probabilities.get("bet", 0.0) + probabilities.get("raise", 0.0)
    continue_probability = probabilities.get("call", 0.0) + probabilities.get("bet", 0.0) + probabilities.get("raise", 0.0)
    failures: list[str] = []

    if action not in expectation.accepted_actions:
        failures.append(f"action={action} not in accepted_actions={list(expectation.accepted_actions)}")
    if expectation.max_fold_probability is not None and probabilities.get("fold", 0.0) > expectation.max_fold_probability:
        failures.append(
            f"fold_probability={probabilities.get('fold', 0.0):.4f} > {expectation.max_fold_probability:.4f}"
        )
    if expectation.min_fold_probability is not None and probabilities.get("fold", 0.0) < expectation.min_fold_probability:
        failures.append(
            f"fold_probability={probabilities.get('fold', 0.0):.4f} < {expectation.min_fold_probability:.4f}"
        )
    if expectation.min_aggressive_probability is not None and aggressive_probability < expectation.min_aggressive_probability:
        failures.append(
            f"aggressive_probability={aggressive_probability:.4f} < {expectation.min_aggressive_probability:.4f}"
        )
    if expectation.max_aggressive_probability is not None and aggressive_probability > expectation.max_aggressive_probability:
        failures.append(
            f"aggressive_probability={aggressive_probability:.4f} > {expectation.max_aggressive_probability:.4f}"
        )
    if expectation.min_continue_probability is not None and continue_probability < expectation.min_continue_probability:
        failures.append(
            f"continue_probability={continue_probability:.4f} < {expectation.min_continue_probability:.4f}"
        )
    if expectation.max_continue_probability is not None and continue_probability > expectation.max_continue_probability:
        failures.append(
            f"continue_probability={continue_probability:.4f} > {expectation.max_continue_probability:.4f}"
        )

    return {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "description": scenario.description,
        "expected_behavior": scenario.expected_behavior,
        "request": scenario.payload,
        "actual_action": action,
        "probabilities": probabilities,
        "bet_size": float(response.get("bet_size") or 0.0),
        "model_status": response.get("model_status"),
        "guardrails": list(response.get("strategy_guardrails") or []),
        "derived_metrics": {
            "aggressive_probability": aggressive_probability,
            "continue_probability": continue_probability,
        },
        "passed": not failures,
        "failures": failures,
    }


def _default_model_path(project_root: Path) -> Path:
    bundle = project_root / "models" / "poker_policy_bundle.joblib"
    if bundle.exists():
        return bundle
    return project_root / "models" / "poker_policy.joblib"


def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    values = {action: max(0.0, float(probabilities.get(action, 0.0) or 0.0)) for action in ACTION_ORDER}
    total = sum(values.values()) or 1.0
    return {action: value / total for action, value in values.items()}


def _blend(base: dict[str, float], target: dict[str, float], strength: float) -> dict[str, float]:
    strength = min(max(strength, 0.0), 1.0)
    blended = {
        action: (1.0 - strength) * base.get(action, 0.0) + strength * target.get(action, 0.0)
        for action in ACTION_ORDER
    }
    return _normalize_probabilities(blended)


def _canonical_action(action: Any) -> str:
    text = str(action or "").lower().strip().replace("-", "_")
    if text == "check_call":
        return "call"
    if text == "half_pot" or text == "full_pot":
        return "bet"
    return text


def _is_premium_pair_facing_raise(request: PredictionRequest) -> bool:
    ranks = [card_rank(card) for card in request.hole_cards[:2]]
    return (
        str(request.street).lower() == "preflop"
        and len(ranks) == 2
        and ranks[0] == ranks[1]
        and max(ranks) >= 13
        and request.to_call > 0.0
    )


def _is_trash_hand_facing_raise(request: PredictionRequest) -> bool:
    ranks = [card_rank(card) for card in request.hole_cards[:2]]
    suits = [card_suit(card) for card in request.hole_cards[:2]]
    if str(request.street).lower() != "preflop" or len(ranks) != 2 or request.to_call <= 0.0:
        return False
    high = max(ranks)
    low = min(ranks)
    offsuit = len(suits) == 2 and suits[0] != suits[1]
    disconnected = abs(ranks[0] - ranks[1]) >= 4
    early_or_blind = str(request.position).upper() in {"UTG", "SB", "BB"}
    return high <= 7 and low <= 3 and offsuit and disconnected and early_or_blind


def _is_nut_flush_draw_facing_bet(request: PredictionRequest) -> bool:
    street = str(request.street).lower()
    if street not in {"flop", "turn"} or request.to_call <= 0.0:
        return False
    hole = list(request.hole_cards[:2])
    board = list(request.board_cards)
    if len(hole) < 2 or len(board) < 3:
        return False
    suits = [card_suit(card) for card in hole + board]
    ranks = [card_rank(card) for card in hole]
    for suit in set(suits):
        if not suit:
            continue
        suited_count = sum(1 for card in hole + board if card_suit(card) == suit)
        hero_suited_count = sum(1 for card in hole if card_suit(card) == suit)
        board_suited_count = sum(1 for card in board if card_suit(card) == suit)
        if suited_count >= 4 and hero_suited_count >= 1 and board_suited_count >= 2 and max(ranks, default=0) >= 13:
            return True
    return False


def _is_missed_river_facing_large_bet(request: PredictionRequest) -> bool:
    if str(request.street).lower() != "river" or request.to_call <= 0.0:
        return False
    price = request.to_call / (request.pot + request.to_call) if request.pot + request.to_call > 0 else 0.0
    made_category, made_score = made_hand_category(list(request.hole_cards) + list(request.board_cards))
    hole_ranks = {card_rank(card) for card in request.hole_cards}
    board_ranks = {card_rank(card) for card in request.board_cards}
    paired_with_board = any(rank in board_ranks for rank in hole_ranks)
    weak_high_card = max(hole_ranks or {0}) <= 7
    return price >= 0.28 and made_category == "high_card" and made_score <= 0.0 and weak_high_card and not paired_with_board
