from __future__ import annotations

from pathlib import Path
from typing import Any

from poker_agent.features import request_to_features
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest, PredictionResponse


class RuleBasedAgent:
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        strength = request_to_features(request)["strength_proxy"]
        if request.to_call <= 0 and strength < 0.45:
            probabilities = {"check": 0.72, "bet": 0.18, "fold": 0.04, "call": 0.04, "raise": 0.02}
        elif strength >= 0.75:
            probabilities = {"raise": 0.52, "call": 0.24, "bet": 0.18, "check": 0.04, "fold": 0.02}
        elif strength >= 0.48:
            probabilities = {"call": 0.52, "raise": 0.18, "check": 0.16, "fold": 0.10, "bet": 0.04}
        else:
            probabilities = {"fold": 0.62, "call": 0.20, "check": 0.12, "raise": 0.04, "bet": 0.02}
        action = max(probabilities, key=probabilities.get)
        return PredictionResponse(action=action, probabilities=probabilities)


class MLPolicyAgent:
    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def from_path(cls, path: Path) -> "MLPolicyAgent":
        return cls(load_policy(path))

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        action, probabilities = self.model.predict_from_features(request_to_features(request))
        return PredictionResponse(action=action, probabilities=probabilities)


class PromptLLMAgent:
    """Adapter for an external/local LLM poker decision model.

    This class intentionally avoids hard-coding a provider. Pass any callable
    that accepts a prompt string and returns text containing an action.
    """

    def __init__(self, complete):
        self.complete = complete

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        prompt = (
            "Choose one poker action: fold, call, check, bet, raise, all_in.\n"
            f"State: position={request.position}, street={request.street}, "
            f"hole={request.hole_cards}, board={request.board_cards}, "
            f"pot={request.pot}, to_call={request.to_call}, stack={request.stack}.\n"
            "Return only the action."
        )
        text = str(self.complete(prompt)).strip().lower()
        for action in ("all_in", "raise", "bet", "call", "check", "fold"):
            if action in text:
                return PredictionResponse(action=action, probabilities={action: 1.0})
        return RuleBasedAgent().predict(request)
