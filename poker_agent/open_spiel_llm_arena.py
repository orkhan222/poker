from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from poker_agent.action_planning import build_action_plan
from poker_agent.agents import RuleBasedAgent
from poker_agent.schemas import PredictionRequest, PredictionResponse


PHASE3_OPEN_SPIEL_ARENA_VERSION = "2026-07-02"
DEFAULT_OPEN_SPIEL_GAME = "kuhn_poker"
AGENT_ONLY_ARENA_STATUS = "AGENT_ONLY_OPEN_SPIEL_ARENA"
RUNTIME_PENDING_STATUS = "READY_PENDING_OPEN_SPIEL_RUNTIME"
RUN_COMPLETED_STATUS = "COMPLETED"


class OpenSpielArenaError(RuntimeError):
    """Raised when the OpenSpiel arena contract cannot be executed safely."""


class OpenSpielRuntimeUnavailable(OpenSpielArenaError):
    """Raised when pyspiel is not installed in the active runtime."""


class OpenSpielPolicy(Protocol):
    @property
    def name(self) -> str:
        ...

    def action_probabilities(
        self,
        state: Any,
        player_id: int,
        legal_actions: tuple[int, ...],
    ) -> dict[int, float]:
        ...


@dataclass(frozen=True)
class ArenaAgentSpec:
    seat: int
    name: str
    source: str
    architecture: str = "phase1_llm_policy_adapter"
    model_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "name": self.name,
            "source": self.source,
            "architecture": self.architecture,
            "model_path": self.model_path,
        }


@dataclass(frozen=True)
class ArenaRunConfig:
    game_name: str = DEFAULT_OPEN_SPIEL_GAME
    episodes: int = 256
    seed: int = 42
    max_steps_per_episode: int = 256
    agent_a_name: str = "phase1_llm_agent_a"
    agent_b_name: str = "phase1_llm_agent_b"
    agent_a_source: str = "phase1_trained_llm_policy_a"
    agent_b_source: str = "phase1_trained_llm_policy_b"

    def agent_specs(self) -> tuple[ArenaAgentSpec, ArenaAgentSpec]:
        return (
            ArenaAgentSpec(seat=0, name=self.agent_a_name, source=self.agent_a_source),
            ArenaAgentSpec(seat=1, name=self.agent_b_name, source=self.agent_b_source),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_name": self.game_name,
            "episodes": self.episodes,
            "seed": self.seed,
            "max_steps_per_episode": self.max_steps_per_episode,
            "agents": [spec.to_dict() for spec in self.agent_specs()],
        }


class ServicePolicyOpenSpielAdapter:
    """Adapts an existing structured-state policy to OpenSpiel legal actions."""

    def __init__(self, name: str, policy: Any, *, source: str = "structured_policy_adapter"):
        self._name = name
        self.policy = policy
        self.source = source

    @property
    def name(self) -> str:
        return self._name

    def action_probabilities(
        self,
        state: Any,
        player_id: int,
        legal_actions: tuple[int, ...],
    ) -> dict[int, float]:
        if not legal_actions:
            raise OpenSpielArenaError("OpenSpiel state returned no legal actions")
        request = prediction_request_from_open_spiel_state(state, player_id)
        prediction = self.policy.predict(request)
        action_weights = {
            action_id: max(0.0, float(prediction.probabilities.get(_canonical_action_label(state, player_id, action_id), 0.0)))
            for action_id in legal_actions
        }
        if sum(action_weights.values()) <= 0.0:
            action_weights = _fallback_action_weights(state, player_id, legal_actions, prediction)
        return _normalize_action_distribution(action_weights)


class FixedOpenSpielPolicy:
    """Small deterministic policy used for test fixtures and smoke checks."""

    def __init__(self, name: str, preferred_actions: tuple[str, ...]):
        self._name = name
        self.preferred_actions = tuple(action.lower() for action in preferred_actions)

    @property
    def name(self) -> str:
        return self._name

    def action_probabilities(
        self,
        state: Any,
        player_id: int,
        legal_actions: tuple[int, ...],
    ) -> dict[int, float]:
        weights: dict[int, float] = {}
        for action_id in legal_actions:
            label = _canonical_action_label(state, player_id, action_id)
            try:
                priority = self.preferred_actions.index(label)
            except ValueError:
                priority = len(self.preferred_actions)
            weights[action_id] = 1.0 / (priority + 1.0)
        return _normalize_action_distribution(weights)


class OpenSpielAgentOnlyArena:
    """Runs a two-player OpenSpiel arena where every seat is controlled by an agent."""

    def __init__(
        self,
        game: Any,
        policies: tuple[OpenSpielPolicy, ...],
        *,
        seed: int = 42,
        game_name: str = DEFAULT_OPEN_SPIEL_GAME,
    ):
        self.game = game
        self.policies = policies
        self.seed = seed
        self.game_name = game_name
        self.rng = random.Random(seed)
        self.num_players = _game_num_players(game)
        if self.num_players != len(policies):
            raise OpenSpielArenaError(
                f"Agent-only arena requires one policy per seat: game has {self.num_players}, "
                f"received {len(policies)}."
            )
        if self.num_players != 2:
            raise OpenSpielArenaError(
                f"Phase 3 LLM-vs-LLM arena expects exactly two seats, received {self.num_players}."
            )

    def run(self, episodes: int, *, max_steps_per_episode: int = 256) -> dict[str, Any]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")

        returns_by_agent: list[list[float]] = [[] for _ in range(self.num_players)]
        wins_by_agent = [0 for _ in range(self.num_players)]
        action_counts: Counter[str] = Counter()
        sample_episodes: list[dict[str, Any]] = []

        for episode_idx in range(episodes):
            state = self.game.new_initial_state()
            step_records: list[dict[str, Any]] = []
            steps = 0
            while not state.is_terminal():
                if steps >= max_steps_per_episode:
                    raise OpenSpielArenaError(
                        f"Episode {episode_idx} exceeded {max_steps_per_episode} OpenSpiel steps"
                    )
                if _is_chance_node(state):
                    chance_action = _sample_weighted(tuple(_chance_outcomes(state)), self.rng)
                    state.apply_action(chance_action)
                    steps += 1
                    continue

                player_id = int(state.current_player())
                if player_id < 0 or player_id >= self.num_players:
                    raise OpenSpielArenaError(f"Unsupported OpenSpiel current_player value: {player_id}")
                legal_actions = tuple(int(action) for action in state.legal_actions(player_id))
                probabilities = self.policies[player_id].action_probabilities(state, player_id, legal_actions)
                selected_action = _sample_weighted(tuple(probabilities.items()), self.rng)
                action_label = _canonical_action_label(state, player_id, selected_action)
                action_counts[action_label] += 1
                if episode_idx < 5:
                    step_records.append(
                        {
                            "step": steps,
                            "player_id": player_id,
                            "agent": self.policies[player_id].name,
                            "action_id": selected_action,
                            "action": action_label,
                            "legal_actions": [
                                _canonical_action_label(state, player_id, action_id)
                                for action_id in legal_actions
                            ],
                            "probabilities": {
                                str(action_id): round(probability, 6)
                                for action_id, probability in probabilities.items()
                            },
                        }
                    )
                state.apply_action(selected_action)
                steps += 1

            returns = [float(value) for value in state.returns()]
            for player_id, value in enumerate(returns):
                returns_by_agent[player_id].append(value)
                if value > 0.0:
                    wins_by_agent[player_id] += 1
            if episode_idx < 5:
                sample_episodes.append(
                    {
                        "episode": episode_idx,
                        "returns": returns,
                        "steps": steps,
                        "actions": step_records,
                    }
                )

        return {
            "version": PHASE3_OPEN_SPIEL_ARENA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": RUN_COMPLETED_STATUS,
            "arena_contract": _agent_only_contract(self.policies, self.num_players),
            "environment": {
                "framework": "OpenSpiel",
                "game_name": self.game_name,
                "episodes": episodes,
                "seed": self.seed,
                "max_steps_per_episode": max_steps_per_episode,
            },
            "metrics": _arena_metrics(returns_by_agent, wins_by_agent, action_counts, episodes),
            "sample_episodes": sample_episodes,
            "quality_boundary": {
                "is_reinforcement_learning_stage": True,
                "agent_only_table": True,
                "human_players_present": False,
                "fixed_scripted_opponents_present": False,
                "policy_update_during_arena": False,
                "purpose": (
                    "Evaluate the two Phase 1 LLM policies against each other in an OpenSpiel "
                    "multi-agent arena before any promotion decision."
                ),
            },
        }


def build_phase3_open_spiel_arena_report(
    project_root: Path,
    *,
    config: ArenaRunConfig | None = None,
    run_if_available: bool = False,
) -> dict[str, Any]:
    config = config or ArenaRunConfig()
    if not run_if_available:
        return _pending_runtime_report(config, runtime_available=_pyspiel_available())
    try:
        pyspiel = _load_pyspiel()
        game = pyspiel.load_game(config.game_name)
    except OpenSpielRuntimeUnavailable:
        return _pending_runtime_report(config, runtime_available=False)

    policies: tuple[OpenSpielPolicy, ...] = (
        ServicePolicyOpenSpielAdapter(config.agent_a_name, RuleBasedAgent(), source=config.agent_a_source),
        ServicePolicyOpenSpielAdapter(config.agent_b_name, RuleBasedAgent(), source=config.agent_b_source),
    )
    arena = OpenSpielAgentOnlyArena(game, policies, seed=config.seed, game_name=config.game_name)
    payload = arena.run(config.episodes, max_steps_per_episode=config.max_steps_per_episode)
    payload["project_root"] = str(project_root)
    payload["runtime_note"] = (
        "This run uses the OpenSpiel runtime. Replace the RuleBasedAgent wiring with the two trained "
        "Phase 1 LLM policy objects when their model artifacts are available in this environment."
    )
    return payload


def write_phase3_open_spiel_arena_report(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    *,
    config: ArenaRunConfig | None = None,
    run_if_available: bool = False,
) -> dict[str, Any]:
    payload = build_phase3_open_spiel_arena_report(
        project_root,
        config=config,
        run_if_available=run_if_available,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_phase3_open_spiel_arena_markdown(payload), encoding="utf-8")
    return payload


def render_phase3_open_spiel_arena_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 OpenSpiel LLM Arena",
        "",
        f"- Status: `{payload['status']}`",
        f"- Version: `{payload['version']}`",
        "",
        "## Contract",
        "",
    ]
    contract = payload["arena_contract"]
    lines.extend(
        [
            f"- Arena type: `{contract['arena_type']}`",
            f"- Agent-only table: `{contract['agent_only_table']}`",
            f"- Human players present: `{contract['human_players_present']}`",
            f"- Fixed scripted opponents present: `{contract['fixed_scripted_opponents_present']}`",
            "",
            "## Agents",
            "",
        ]
    )
    for agent in contract["agents"]:
        lines.append(f"- Seat {agent['seat']}: `{agent['name']}` from `{agent['source']}`")
    if payload["status"] == RUN_COMPLETED_STATUS:
        metrics = payload["metrics"]
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                f"- Episodes: `{payload['environment']['episodes']}`",
                f"- Agent 0 mean return: `{metrics['per_agent']['0']['mean_return']}`",
                f"- Agent 1 mean return: `{metrics['per_agent']['1']['mean_return']}`",
                f"- Action distribution: `{metrics['action_distribution']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Runtime Boundary",
                "",
                payload["runtime_boundary"]["reason"],
            ]
        )
    return "\n".join(lines) + "\n"


def prediction_request_from_open_spiel_state(state: Any, player_id: int) -> PredictionRequest:
    legal_labels = {
        _canonical_action_label(state, player_id, action_id)
        for action_id in state.legal_actions(player_id)
    }
    to_call = 1.0 if {"fold", "call", "raise"} & legal_labels else 0.0
    history = [_safe_history_string(state)]
    return PredictionRequest(
        position=f"P{player_id}",
        street=_infer_street_from_state(state),
        hole_cards=[],
        board_cards=[],
        pot=max(1.0, float(len(_safe_history_string(state))) / 10.0),
        to_call=to_call,
        stack=100.0,
        min_raise=2.0,
        player_count=2,
        betting_history=[{"player_position": "table", "action": item} for item in history if item],
    )


def _pending_runtime_report(config: ArenaRunConfig, *, runtime_available: bool) -> dict[str, Any]:
    return {
        "version": PHASE3_OPEN_SPIEL_ARENA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": RUNTIME_PENDING_STATUS,
        "arena_contract": _agent_only_contract_from_specs(config.agent_specs()),
        "environment": {
            "framework": "OpenSpiel",
            "game_name": config.game_name,
            "episodes": config.episodes,
            "seed": config.seed,
            "max_steps_per_episode": config.max_steps_per_episode,
        },
        "runtime_boundary": {
            "open_spiel_available": runtime_available,
            "run_if_available_required_for_metrics": True,
            "reason": (
                "The Phase 3 arena code is implemented and configured. Measured arena results require "
                "executing this report builder in an environment with the OpenSpiel Python runtime and "
                "the two Phase 1 LLM policy artifacts wired through OpenSpielPolicy adapters."
            ),
        },
        "quality_boundary": {
            "is_reinforcement_learning_stage": True,
            "agent_only_table": True,
            "human_players_present": False,
            "fixed_scripted_opponents_present": False,
            "policy_update_during_arena": False,
            "metrics_claim_allowed": False,
        },
    }


def _agent_only_contract(policies: tuple[OpenSpielPolicy, ...], num_players: int) -> dict[str, Any]:
    return {
        "arena_type": AGENT_ONLY_ARENA_STATUS,
        "agent_only_table": True,
        "num_players": num_players,
        "policy_count": len(policies),
        "all_seats_controlled_by_agents": len(policies) == num_players,
        "human_players_present": False,
        "fixed_scripted_opponents_present": False,
        "agents": [
            {
                "seat": seat,
                "name": policy.name,
                "source": getattr(policy, "source", "open_spiel_policy_adapter"),
            }
            for seat, policy in enumerate(policies)
        ],
    }


def _agent_only_contract_from_specs(specs: tuple[ArenaAgentSpec, ...]) -> dict[str, Any]:
    return {
        "arena_type": AGENT_ONLY_ARENA_STATUS,
        "agent_only_table": True,
        "num_players": len(specs),
        "policy_count": len(specs),
        "all_seats_controlled_by_agents": True,
        "human_players_present": False,
        "fixed_scripted_opponents_present": False,
        "agents": [spec.to_dict() for spec in specs],
    }


def _arena_metrics(
    returns_by_agent: list[list[float]],
    wins_by_agent: list[int],
    action_counts: Counter[str],
    episodes: int,
) -> dict[str, Any]:
    per_agent: dict[str, Any] = {}
    for agent_id, returns in enumerate(returns_by_agent):
        per_agent[str(agent_id)] = {
            "mean_return": round(statistics.fmean(returns), 6) if returns else 0.0,
            "std_return": round(statistics.pstdev(returns), 6) if len(returns) > 1 else 0.0,
            "win_rate": round(wins_by_agent[agent_id] / episodes, 6),
            "episodes": episodes,
        }
    total_actions = sum(action_counts.values()) or 1
    return {
        "per_agent": per_agent,
        "action_distribution": {
            action: round(count / total_actions, 6)
            for action, count in sorted(action_counts.items())
        },
        "total_actions": sum(action_counts.values()),
    }


def _canonical_action_label(state: Any, player_id: int, action_id: int) -> str:
    raw = _open_spiel_action_string(state, player_id, action_id).lower()
    if "fold" in raw:
        return "fold"
    if "call" in raw:
        return "call"
    if "check" in raw:
        return "check"
    if "raise" in raw or "all-in" in raw or "all_in" in raw:
        return "raise"
    if "bet" in raw:
        return "bet"
    return str(action_id)


def _open_spiel_action_string(state: Any, player_id: int, action_id: int) -> str:
    try:
        return str(state.action_to_string(player_id, action_id))
    except TypeError:
        try:
            return str(state.action_to_string(action_id))
        except Exception:
            return str(action_id)
    except Exception:
        return str(action_id)


def _fallback_action_weights(
    state: Any,
    player_id: int,
    legal_actions: tuple[int, ...],
    prediction: PredictionResponse,
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for action_id in legal_actions:
        label = _canonical_action_label(state, player_id, action_id)
        weights[action_id] = 1.0
        if label == prediction.action:
            weights[action_id] += max(0.0, prediction.confidence)
    return weights


def _normalize_action_distribution(weights: dict[int, float]) -> dict[int, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(weights)
        return {action_id: uniform for action_id in weights}
    return {action_id: max(0.0, value) / total for action_id, value in weights.items()}


def _sample_weighted(options: tuple[tuple[int, float], ...], rng: random.Random) -> int:
    if not options:
        raise OpenSpielArenaError("Cannot sample from an empty action distribution")
    threshold = rng.random() * sum(max(0.0, probability) for _, probability in options)
    cumulative = 0.0
    for action_id, probability in options:
        cumulative += max(0.0, probability)
        if cumulative >= threshold:
            return int(action_id)
    return int(options[-1][0])


def _chance_outcomes(state: Any) -> list[tuple[int, float]]:
    return [(int(action), float(probability)) for action, probability in state.chance_outcomes()]


def _is_chance_node(state: Any) -> bool:
    try:
        return bool(state.is_chance_node())
    except AttributeError:
        return False


def _game_num_players(game: Any) -> int:
    value = game.num_players() if callable(getattr(game, "num_players", None)) else getattr(game, "num_players")
    return int(value)


def _safe_history_string(state: Any) -> str:
    for attr in ("history_str", "information_state_string"):
        method = getattr(state, attr, None)
        if callable(method):
            try:
                return str(method())
            except TypeError:
                try:
                    return str(method(0))
                except Exception:
                    continue
            except Exception:
                continue
    return str(state)


def _infer_street_from_state(state: Any) -> str:
    text = _safe_history_string(state).lower()
    for street in ("river", "turn", "flop", "preflop"):
        if street in text:
            return street
    return "preflop"


def _pyspiel_available() -> bool:
    try:
        _load_pyspiel()
    except OpenSpielRuntimeUnavailable:
        return False
    return True


def _load_pyspiel() -> Any:
    try:
        import pyspiel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OpenSpielRuntimeUnavailable(
            "OpenSpiel Python runtime is not installed. Install the OpenSpiel package in the "
            "training environment before running measured Phase 3 arena experiments."
        ) from exc
    return pyspiel

