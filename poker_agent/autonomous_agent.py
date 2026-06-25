from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from poker_agent.action_planning import build_action_plan
from poker_agent.llm_decision_context import legal_actions_for_request
from poker_agent.schemas import PredictionRequest, PredictionResponse, VALID_ACTIONS


class AgentLifecycleError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SessionStatus(str, Enum):
    ACTIVE = "active"
    SETTLED = "settled"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentObservation:
    hand_id: str
    sequence_number: int
    state: PredictionRequest
    event_id: str
    fingerprint: str
    legal_actions: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentObservation":
        hand_id = str(payload.get("hand_id") or "").strip()
        if not hand_id:
            raise AgentLifecycleError("missing_hand_id", "hand_id is required")

        try:
            sequence_number = int(payload.get("sequence_number"))
        except (TypeError, ValueError) as exc:
            raise AgentLifecycleError(
                "invalid_sequence_number",
                "sequence_number must be a non-negative integer",
            ) from exc
        if sequence_number < 0:
            raise AgentLifecycleError(
                "invalid_sequence_number",
                "sequence_number must be a non-negative integer",
            )

        raw_state = payload.get("state")
        if not isinstance(raw_state, dict):
            raise AgentLifecycleError("missing_state", "state must be an object")
        state = PredictionRequest.from_dict(raw_state)

        inferred_actions = legal_actions_for_request(state)
        raw_legal_actions = payload.get("legal_actions")
        if raw_legal_actions is None:
            legal_actions = inferred_actions
        else:
            if not isinstance(raw_legal_actions, list):
                raise AgentLifecycleError("invalid_legal_actions", "legal_actions must be an array")
            requested = tuple(dict.fromkeys(str(action).lower() for action in raw_legal_actions))
            invalid = sorted(set(requested) - set(VALID_ACTIONS))
            if invalid:
                raise AgentLifecycleError(
                    "invalid_legal_actions",
                    f"Unsupported legal actions: {', '.join(invalid)}",
                )
            legal_actions = tuple(action for action in requested if action in inferred_actions)
            if not legal_actions:
                raise AgentLifecycleError(
                    "empty_legal_actions",
                    "legal_actions does not contain an action valid for the current state",
                )

        fingerprint = _stable_identifier(
            {
                "hand_id": hand_id,
                "sequence_number": sequence_number,
                "state": raw_state,
                "legal_actions": legal_actions,
            }
        )
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            event_id = fingerprint

        return cls(
            hand_id=hand_id,
            sequence_number=sequence_number,
            state=state,
            event_id=event_id,
            fingerprint=fingerprint,
            legal_actions=legal_actions,
        )


@dataclass(frozen=True)
class AutonomousDecision:
    hand_id: str
    sequence_number: int
    event_id: str
    decision_id: str
    action: str
    probabilities: dict[str, float]
    confidence: float
    bet_size: float
    wait_time_ms: int
    sizing_method: str
    timing_method: str
    model_status: str
    legal_actions: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, idempotent_replay: bool = False) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "sequence_number": self.sequence_number,
            "event_id": self.event_id,
            "decision_id": self.decision_id,
            "action": self.action,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "bet_size": self.bet_size,
            "wait_time_ms": self.wait_time_ms,
            "sizing_method": self.sizing_method,
            "timing_method": self.timing_method,
            "model_status": self.model_status,
            "legal_actions": list(self.legal_actions),
            "warnings": list(self.warnings),
            "idempotent_replay": idempotent_replay,
        }


@dataclass
class AgentSession:
    hand_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    last_sequence_number: int = -1
    decisions_by_event: dict[str, AutonomousDecision] = field(default_factory=dict)
    fingerprints_by_event: dict[str, str] = field(default_factory=dict)
    decision_history: list[AutonomousDecision] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "status": self.status.value,
            "last_sequence_number": self.last_sequence_number,
            "decision_count": len(self.decision_history),
            "decisions": [decision.to_dict() for decision in self.decision_history],
            "result": self.result,
        }


class Policy(Protocol):
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        ...


class PokerEnvironment(Protocol):
    def reset(self) -> dict[str, Any]:
        ...

    def apply_action(self, decision: AutonomousDecision) -> dict[str, Any] | None:
        ...

    def is_terminal(self) -> bool:
        ...

    def result(self) -> dict[str, Any]:
        ...


class AutonomousPokerAgent:
    """Stateful policy controller for deterministic simulation and service orchestration."""

    def __init__(self, policy: Policy):
        self._policy = policy
        self._sessions: dict[str, AgentSession] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._lock = threading.RLock()

    def decide(self, payload: dict[str, Any]) -> tuple[AutonomousDecision, bool]:
        observation = AgentObservation.from_dict(payload)
        with self._lock:
            session = self._sessions.setdefault(
                observation.hand_id,
                AgentSession(hand_id=observation.hand_id),
            )
            session_lock = self._session_locks.setdefault(observation.hand_id, threading.RLock())
        with session_lock:
            if session.status is not SessionStatus.ACTIVE:
                raise AgentLifecycleError(
                    "session_not_active",
                    f"Hand {observation.hand_id} is {session.status.value}",
                )

            replay = session.decisions_by_event.get(observation.event_id)
            if replay is not None:
                if session.fingerprints_by_event[observation.event_id] != observation.fingerprint:
                    raise AgentLifecycleError(
                        "event_id_conflict",
                        f"event_id {observation.event_id} was already used for another observation",
                    )
                return replay, True

            if observation.sequence_number <= session.last_sequence_number:
                raise AgentLifecycleError(
                    "stale_observation",
                    (
                        f"sequence_number must be greater than {session.last_sequence_number} "
                        f"for hand {observation.hand_id}"
                    ),
                )

            raw_prediction = self._policy.predict(observation.state)
            decision = self._constrain_prediction(observation, raw_prediction)
            session.last_sequence_number = observation.sequence_number
            session.decisions_by_event[observation.event_id] = decision
            session.fingerprints_by_event[observation.event_id] = observation.fingerprint
            session.decision_history.append(decision)
            return decision, False

    def settle(self, hand_id: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_hand_id = str(hand_id).strip()
        with self._lock:
            session = self._sessions.get(normalized_hand_id)
            session_lock = self._session_locks.get(normalized_hand_id)
            if session is None:
                raise AgentLifecycleError("unknown_session", f"Unknown hand_id: {normalized_hand_id}")
        assert session_lock is not None
        with session_lock:
            if session.status is SessionStatus.SETTLED:
                return session.to_dict()
            session.status = SessionStatus.SETTLED
            session.result = dict(result or {})
            return session.to_dict()

    def session(self, hand_id: str) -> dict[str, Any]:
        with self._lock:
            normalized_hand_id = str(hand_id).strip()
            session = self._sessions.get(normalized_hand_id)
            session_lock = self._session_locks.get(normalized_hand_id)
            if session is None:
                raise AgentLifecycleError("unknown_session", f"Unknown hand_id: {hand_id}")
        assert session_lock is not None
        with session_lock:
            return session.to_dict()

    def run_episode(
        self,
        environment: PokerEnvironment,
        *,
        max_decisions: int = 512,
    ) -> dict[str, Any]:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")

        observation_payload = environment.reset()
        decisions: list[dict[str, Any]] = []
        while not environment.is_terminal():
            if len(decisions) >= max_decisions:
                raise AgentLifecycleError(
                    "episode_limit_exceeded",
                    f"Episode exceeded {max_decisions} decisions",
                )
            decision, replayed = self.decide(observation_payload)
            decisions.append(decision.to_dict(idempotent_replay=replayed))
            next_observation = environment.apply_action(decision)
            if environment.is_terminal():
                break
            if not isinstance(next_observation, dict):
                raise AgentLifecycleError(
                    "missing_environment_observation",
                    "Environment did not return the next observation",
                )
            observation_payload = next_observation

        result = dict(environment.result())
        hand_id = str(observation_payload.get("hand_id") or (decisions[0]["hand_id"] if decisions else ""))
        if not hand_id:
            raise AgentLifecycleError(
                "missing_hand_id",
                "Environment reset payload must include hand_id",
            )
        with self._lock:
            self._sessions.setdefault(hand_id, AgentSession(hand_id=hand_id))
            self._session_locks.setdefault(hand_id, threading.RLock())
        session = self.settle(hand_id, result)
        return {
            "status": "completed",
            "hand_id": hand_id,
            "decision_count": len(decisions),
            "decisions": decisions,
            "result": result,
            "session": session,
        }

    @staticmethod
    def capabilities() -> dict[str, Any]:
        return {
            "status": "IMPLEMENTED",
            "agent_type": "controlled_stateful_policy_agent",
            "supports": [
                "structured game-state observations",
                "stateful hand sessions",
                "legal-action enforcement",
                "idempotent event handling",
                "deterministic simulation episodes",
                "terminal hand settlement",
            ],
            "does_not_include": [
                "screen scraping",
                "mouse or keyboard control",
                "direct integration with a real-money poker client",
                "online policy updates without an approval gate",
            ],
            "production_boundary": (
                "The controller is simulation-ready and API-ready. A table-specific environment "
                "adapter and independent safety approval are required before external execution."
            ),
        }

    def _constrain_prediction(
        self,
        observation: AgentObservation,
        prediction: PredictionResponse,
    ) -> AutonomousDecision:
        normalized = {
            action: max(0.0, float(prediction.probabilities.get(action, 0.0)))
            for action in observation.legal_actions
        }
        total = sum(normalized.values())
        warnings = list(prediction.warnings)
        if total <= 0:
            normalized = {
                action: 1.0 / len(observation.legal_actions)
                for action in observation.legal_actions
            }
            warnings.append("Policy assigned no mass to legal actions; uniform legal fallback applied.")
        else:
            normalized = {action: value / total for action, value in normalized.items()}

        action = max(normalized, key=normalized.get)
        if prediction.action != action:
            warnings.append(
                f"Policy action '{prediction.action}' was replaced by legal action '{action}'."
            )
        confidence = normalized[action]
        plan = build_action_plan(observation.state, action, confidence)
        decision_id = _stable_identifier(
            {
                "hand_id": observation.hand_id,
                "sequence_number": observation.sequence_number,
                "event_id": observation.event_id,
                "action": action,
                "probabilities": normalized,
            }
        )
        return AutonomousDecision(
            hand_id=observation.hand_id,
            sequence_number=observation.sequence_number,
            event_id=observation.event_id,
            decision_id=decision_id,
            action=action,
            probabilities=normalized,
            confidence=confidence,
            bet_size=plan.bet_size,
            wait_time_ms=plan.wait_time_ms,
            sizing_method=plan.sizing_method,
            timing_method=plan.timing_method,
            model_status=prediction.model_status,
            legal_actions=observation.legal_actions,
            warnings=tuple(warnings),
        )


def _stable_identifier(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
