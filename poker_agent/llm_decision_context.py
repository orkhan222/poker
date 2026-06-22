from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from poker_agent.action_planning import build_action_plan
from poker_agent.schemas import PredictionRequest, PredictionResponse


CONTEXT_VERSION = "2026-06-22"
ContextMode = Literal["minimal_zero_shot", "rules_grounded", "full_in_context"]
CANONICAL_ACTIONS = ("fold", "check", "call", "bet", "raise")


CONTEXT_MODE_SUMMARY: dict[str, str] = {
    "minimal_zero_shot": "Task, legal actions, and strict JSON output only.",
    "rules_grounded": "Task, legal actions, core Hold'em rules, betting constraints, and strict JSON output.",
    "full_in_context": "Rules-grounded context plus strategy heuristics, risk constraints, and output calibration guidance.",
}


OUTPUT_SCHEMA: dict[str, Any] = {
    "action": "fold | check | call | bet | raise",
    "probabilities": {
        "fold": "float between 0 and 1",
        "check": "float between 0 and 1",
        "call": "float between 0 and 1",
        "bet": "float between 0 and 1",
        "raise": "float between 0 and 1",
    },
    "confidence": "float between 0 and 1",
    "bet_size": "float, 0 unless action is call, bet, or raise",
    "reason_code": "pot_odds | position | hand_strength | pressure | uncertain",
}


@dataclass(frozen=True)
class DecisionPrompt:
    version: str
    context_mode: ContextMode
    legal_actions: tuple[str, ...]
    system_context: str
    user_context: str
    output_schema: dict[str, Any]

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_context},
            {"role": "user", "content": self.user_context},
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "context_mode": self.context_mode,
            "legal_actions": list(self.legal_actions),
            "messages": self.messages(),
            "output_schema": self.output_schema,
        }


def legal_actions_for_request(request: PredictionRequest) -> tuple[str, ...]:
    if request.to_call > 0:
        return ("fold", "call", "raise")
    return ("check", "bet")


def build_decision_prompt(
    request: PredictionRequest,
    context_mode: ContextMode = "full_in_context",
) -> DecisionPrompt:
    legal_actions = legal_actions_for_request(request)
    return DecisionPrompt(
        version=CONTEXT_VERSION,
        context_mode=context_mode,
        legal_actions=legal_actions,
        system_context=_system_context(context_mode, legal_actions),
        user_context=_user_context(request, legal_actions),
        output_schema=OUTPUT_SCHEMA,
    )


def parse_decision_output(raw_text: str, request: PredictionRequest) -> PredictionResponse:
    payload = _extract_json_object(raw_text)
    legal_actions = set(legal_actions_for_request(request))
    warnings: list[str] = []

    action = str(payload.get("action", "")).lower().strip()
    if action == "all_in":
        action = "raise"
    if action not in legal_actions:
        warnings.append(f"Illegal or missing action from LLM output: {action or 'missing'}")
        action = "fold" if request.to_call > 0 else "check"

    probabilities = _normalize_probabilities(payload.get("probabilities"), legal_actions, action)
    confidence = _bounded_float(payload.get("confidence"), max(probabilities.values(), default=0.0))
    requested_bet_size = _nonnegative_float(payload.get("bet_size"), 0.0)
    plan = build_action_plan(request, action, confidence)
    bet_size = requested_bet_size if action in {"call", "bet", "raise"} and requested_bet_size > 0 else plan.bet_size

    return PredictionResponse(
        action=action,
        probabilities=probabilities,
        confidence=confidence,
        bet_size=bet_size,
        wait_time_ms=plan.wait_time_ms,
        sizing_method=plan.sizing_method,
        timing_method=plan.timing_method,
        model_status="llm_context_validated",
        warnings=warnings,
    )


def build_decision_context_report() -> dict[str, Any]:
    examples = _example_requests()
    prompt_records: list[dict[str, Any]] = []
    for name, request in examples.items():
        for mode in CONTEXT_MODE_SUMMARY:
            prompt = build_decision_prompt(request, mode)  # type: ignore[arg-type]
            prompt_records.append(
                {
                    "example": name,
                    "context_mode": mode,
                    "legal_actions": list(prompt.legal_actions),
                    "system_context_chars": len(prompt.system_context),
                    "user_context_chars": len(prompt.user_context),
                    "contains_rules": "Texas Hold'em rules" in prompt.system_context,
                    "contains_strategy_guidelines": "Decision guidelines" in prompt.system_context,
                    "contains_output_constraints": "Output constraints" in prompt.system_context,
                }
            )
    return {
        "version": CONTEXT_VERSION,
        "objective": (
            "Define the in-context learning contract for out-of-box LLM poker decision experiments."
        ),
        "supported_context_modes": CONTEXT_MODE_SUMMARY,
        "default_context_mode": "full_in_context",
        "output_schema": OUTPUT_SCHEMA,
        "required_controls": [
            "legal action filtering",
            "strict JSON-only output",
            "probability normalization",
            "bet-size and timing post-processing",
            "fallback action when an invalid action is returned",
        ],
        "prompt_records": prompt_records,
    }


def write_decision_context_report(out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_decision_context_report()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_decision_context_markdown(payload), encoding="utf-8")
    return payload


def render_decision_context_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LLM Decision Context Contract",
        "",
        f"- Version: `{payload['version']}`",
        f"- Default context mode: `{payload['default_context_mode']}`",
        "",
        "## Objective",
        "",
        payload["objective"],
        "",
        "## Context Modes",
        "",
    ]
    for name, description in payload["supported_context_modes"].items():
        lines.append(f"- `{name}`: {description}")
    lines.extend(
        [
            "",
            "## Required Controls",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["required_controls"])
    lines.extend(
        [
            "",
            "## Prompt Coverage",
            "",
            "| Example | Context mode | Legal actions | Rules | Strategy | Constraints |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in payload["prompt_records"]:
        lines.append(
            "| {example} | `{context_mode}` | {actions} | {rules} | {strategy} | {constraints} |".format(
                example=record["example"],
                context_mode=record["context_mode"],
                actions=", ".join(record["legal_actions"]),
                rules=record["contains_rules"],
                strategy=record["contains_strategy_guidelines"],
                constraints=record["contains_output_constraints"],
            )
        )
    return "\n".join(lines) + "\n"


def _system_context(context_mode: ContextMode, legal_actions: tuple[str, ...]) -> str:
    base = [
        "You are a poker decision model for No-Limit Texas Hold'em.",
        "Task: choose one legal poker action for the supplied structured game state.",
        "Return JSON only. Do not include prose outside the JSON object.",
        f"Legal actions for this decision: {', '.join(legal_actions)}.",
        "Output constraints:",
        "- The action must be one of the legal actions.",
        "- Probabilities must include fold, check, call, bet, and raise and sum to 1.",
        "- Confidence must be between 0 and 1.",
        "- Bet size must be 0 for fold and check.",
        "- Do not invent hidden opponent cards.",
    ]
    if context_mode in {"rules_grounded", "full_in_context"}:
        base.extend(
            [
                "",
                "Texas Hold'em rules:",
                "- A hand has four betting streets: preflop, flop, turn, and river.",
                "- Hole cards are private to the hero; board cards are public community cards.",
                "- If facing a bet or raise, legal actions are fold, call, or raise.",
                "- If not facing a bet, legal actions are check or bet.",
                "- A raise must be at least the minimum raise when stack size allows it.",
                "- Pot odds are to_call / (pot + to_call) when facing a call price.",
            ]
        )
    if context_mode == "full_in_context":
        base.extend(
            [
                "",
                "Decision guidelines:",
                "- Prefer call over fold when pot odds are favorable and the hand has showdown value.",
                "- Prefer raise with strong made hands, high-card premium preflop holdings, or clear pressure spots.",
                "- Prefer check with weak hands when there is no cost to continue.",
                "- Avoid speculative raises when stack-to-pot ratio is low.",
                "- Use position as a secondary signal; late position can support more betting pressure.",
                "- When card information is incomplete, reduce confidence and avoid over-aggressive actions.",
            ]
        )
    base.extend(["", "Required JSON schema:", json.dumps(OUTPUT_SCHEMA, indent=2, sort_keys=True)])
    return "\n".join(base)


def _user_context(request: PredictionRequest, legal_actions: tuple[str, ...]) -> str:
    payload = {
        "game_state": {
            "street": request.street,
            "hero_position": request.position,
            "hero_cards": request.hole_cards,
            "board_cards": request.board_cards,
            "pot_size": request.pot,
            "to_call": request.to_call,
            "hero_stack": request.stack,
            "min_raise": request.min_raise,
            "player_count": request.player_count,
            "betting_history": request.betting_history,
        },
        "derived_context": {
            "legal_actions": list(legal_actions),
            "stack_to_pot_ratio": _safe_ratio(request.stack, request.pot),
            "pot_odds": _safe_ratio(request.to_call, request.pot + request.to_call),
            "board_card_count": len(request.board_cards),
        },
    }
    return "Evaluate this game state and return the JSON decision.\n\n" + json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return loaded if isinstance(loaded, dict) else {}


def _normalize_probabilities(raw: Any, legal_actions: set[str], selected_action: str) -> dict[str, float]:
    values: dict[str, float] = {action: 0.0 for action in CANONICAL_ACTIONS}
    if isinstance(raw, dict):
        for action in CANONICAL_ACTIONS:
            if action in raw:
                values[action] = _bounded_float(raw[action], 0.0)
    if not any(values.values()):
        values[selected_action] = 1.0
    for action in CANONICAL_ACTIONS:
        if action not in legal_actions:
            values[action] = 0.0
    total = sum(values.values())
    if total <= 0:
        values[selected_action] = 1.0
        total = 1.0
    return {action: value / total for action, value in values.items()}


def _bounded_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, 0.0), 1.0)


def _nonnegative_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(number, 0.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _example_requests() -> dict[str, PredictionRequest]:
    return {
        "preflop_facing_raise": PredictionRequest(
            position="BTN",
            street="preflop",
            hole_cards=["Ah", "Kd"],
            board_cards=[],
            pot=2.5,
            to_call=1.0,
            stack=100.0,
            min_raise=4.5,
            player_count=6,
            betting_history=[{"player_position": "UTG", "action": "raise", "amount": 4.5}],
        ),
        "flop_no_bet": PredictionRequest(
            position="BB",
            street="flop",
            hole_cards=["7h", "6h"],
            board_cards=["2c", "9d", "Qs"],
            pot=6.0,
            to_call=0.0,
            stack=84.0,
            min_raise=2.0,
            player_count=4,
            betting_history=[{"player_position": "SB", "action": "check", "amount": 0.0}],
        ),
    }
