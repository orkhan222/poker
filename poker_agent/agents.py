from __future__ import annotations

from pathlib import Path
from typing import Any

from poker_agent.action_space import action_amounts, constrain_probabilities
from poker_agent.api_contract import model_version_from_metadata
from poker_agent.features import request_to_features
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest, PredictionResponse


def _response_from_probabilities(
    request: PredictionRequest,
    probabilities: dict[str, float],
    *,
    model_status: str,
    model_version: str,
    warnings: list[str] | None = None,
) -> PredictionResponse:
    action, legal_probabilities, action_warnings = constrain_probabilities(probabilities, request.action_space)
    sizing = action_amounts(action, request.action_space)
    return PredictionResponse(
        action=action,
        probabilities=legal_probabilities,
        model_version=model_version,
        confidence=max(legal_probabilities.values(), default=0.0),
        model_status=model_status,
        warnings=[*(warnings or []), *action_warnings],
        bet_size=float(sizing["bet_size"]),
        raise_to=sizing["raise_to"],
        raise_by=sizing["raise_by"],
        sizing_method=str(sizing["sizing_method"]),
        legal_actions=request.action_space.legal_actions,
        action_space=request.action_space.to_dict(),
        state_context=request.state_context(),
    )


class RuleBasedAgent:
    model_version = "rule_based:v1"

    def predict(self, request: PredictionRequest, warnings: list[str] | None = None) -> PredictionResponse:
        strength = request_to_features(request)["strength_proxy"]
        if request.to_call <= 0 and strength < 0.45:
            probabilities = {"check": 0.70, "bet": 0.17, "fold": 0.04, "call": 0.04, "raise": 0.02, "all_in": 0.03}
        elif strength >= 0.75:
            probabilities = {"raise": 0.50, "call": 0.23, "bet": 0.16, "all_in": 0.05, "check": 0.04, "fold": 0.02}
        elif strength >= 0.48:
            probabilities = {"call": 0.50, "raise": 0.17, "check": 0.15, "fold": 0.10, "bet": 0.04, "all_in": 0.04}
        else:
            probabilities = {"fold": 0.60, "call": 0.20, "check": 0.11, "raise": 0.04, "bet": 0.02, "all_in": 0.03}
        return _response_from_probabilities(
            request,
            probabilities,
            model_status="rule_based",
            model_version=self.model_version,
            warnings=warnings,
        )


class MissingCardFallbackAgent:
    """Conservative context policy for out-of-distribution missing-card requests."""

    model_version = "missing_card_fallback:v1"

    def predict(
        self,
        request: PredictionRequest,
        warnings: list[str] | None = None,
        model_version: str | None = None,
    ) -> PredictionResponse:
        pot_odds = request.to_call / (request.pot + request.to_call) if request.pot + request.to_call > 0 else 0.0
        no_price_to_continue = request.to_call <= 0
        low_price = 0.0 < pot_odds <= 0.18
        medium_price = 0.18 < pot_odds <= 0.32

        if no_price_to_continue:
            probabilities = {"check": 0.64, "bet": 0.14, "fold": 0.08, "call": 0.08, "raise": 0.03, "all_in": 0.03}
        elif low_price:
            probabilities = {"call": 0.44, "fold": 0.34, "raise": 0.09, "bet": 0.05, "check": 0.04, "all_in": 0.04}
        elif medium_price:
            probabilities = {"fold": 0.50, "call": 0.34, "raise": 0.07, "bet": 0.04, "check": 0.03, "all_in": 0.02}
        else:
            probabilities = {"fold": 0.70, "call": 0.18, "raise": 0.04, "bet": 0.03, "check": 0.03, "all_in": 0.02}

        return _response_from_probabilities(
            request,
            probabilities,
            model_status="missing_card_fallback",
            model_version=model_version or self.model_version,
            warnings=warnings,
        )


class MLPolicyAgent:
    def __init__(self, model: Any, model_path: Path | None = None):
        self.model = model
        metadata = getattr(model, "metadata", {}) or {}
        self.model_version = model_version_from_metadata(metadata, model_path)
        self.missing_card_fallback = MissingCardFallbackAgent()

    @classmethod
    def from_path(cls, path: Path) -> "MLPolicyAgent":
        return cls(load_policy(path), model_path=path)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        warnings: list[str] = []
        metadata = getattr(self.model, "metadata", {}) or {}
        trained_missing_mode = str(metadata.get("missing_hole_cards", "unknown"))
        if len(request.hole_cards) < 2 and trained_missing_mode == "drop":
            warnings.append(
                "Hole cards are missing, while the loaded model was trained with missing-card rows dropped. "
                "Using conservative context fallback instead of out-of-distribution model inference."
            )
            return self.missing_card_fallback.predict(request, warnings=warnings, model_version=self.model_version)

        action, probabilities = self.model.predict_from_features(request_to_features(request))
        return _response_from_probabilities(
            request,
            probabilities,
            model_status=str(metadata.get("policy", "model")),
            model_version=self.model_version,
            warnings=warnings,
        )

