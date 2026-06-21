from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


@dataclass(frozen=True)
class ActionPlan:
    bet_size: float
    wait_time_ms: int
    sizing_method: str
    timing_method: str


def build_action_plan(
    request: PredictionRequest,
    action: str,
    confidence: float,
    processing_time_ms: float = 0.0,
) -> ActionPlan:
    bet_size, sizing_method = estimate_bet_size(request, action, confidence)
    wait_time_ms, timing_method = estimate_wait_time_ms(request, action, confidence, processing_time_ms)
    return ActionPlan(
        bet_size=bet_size,
        wait_time_ms=wait_time_ms,
        sizing_method=sizing_method,
        timing_method=timing_method,
    )


def estimate_bet_size(request: PredictionRequest, action: str, confidence: float) -> tuple[float, str]:
    stack = max(float(request.stack), 0.0)
    pot = max(float(request.pot), 0.0)
    to_call = max(float(request.to_call), 0.0)
    min_raise = max(float(request.min_raise), 0.0)
    if action in {"fold", "check"}:
        return 0.0, "no_chip_commitment"
    if action == "call":
        return round(min(to_call, stack), 2), "call_price"
    if action == "bet":
        if to_call > 0:
            return round(min(to_call, stack), 2), "bet_mapped_to_call_price"
        size = max(pot * _pot_fraction(request, confidence), min_raise)
        return round(min(size, stack), 2), "pot_fraction_bet"
    if action == "raise":
        floor = max(min_raise, to_call * 2.0, 0.01)
        pressure = _hand_pressure(request) * 0.45 + max(confidence, 0.0) * 0.55
        if pressure < 0.42:
            return round(min(floor, stack), 2), "pressure_raise"
        ceiling = max(floor, pot * 1.05)
        size = floor + (ceiling - floor) * min(pressure, 1.0)
        return round(min(size, stack), 2), "pressure_raise"
    return 0.0, "unsupported_action"


def estimate_wait_time_ms(
    request: PredictionRequest,
    action: str,
    confidence: float,
    processing_time_ms: float = 0.0,
) -> tuple[int, str]:
    street_weight = {"preflop": 0, "flop": 120, "turn": 180, "river": 260}.get(request.street, 120)
    action_weight = {"fold": 120, "check": 100, "call": 180, "bet": 300, "raise": 420}.get(action, 180)
    history_weight = min(len(request.betting_history), 8) * 55
    uncertainty_weight = int((1.0 - max(0.0, min(confidence, 1.0))) * 520)
    wait_time = 250 + street_weight + action_weight + history_weight + uncertainty_weight
    wait_time = max(wait_time, int(processing_time_ms) + 75)
    return min(wait_time, 3200), "complexity_calibrated"


def _pot_fraction(request: PredictionRequest, confidence: float) -> float:
    strength = _hand_pressure(request)
    return min(0.85, max(0.35, 0.35 + strength * 0.25 + confidence * 0.20))


def _hand_pressure(request: PredictionRequest) -> float:
    try:
        features: dict[str, Any] = request_to_features(request)
        return max(0.0, min(1.0, float(features.get("strength_proxy", 0.0))))
    except Exception:
        return 0.0
