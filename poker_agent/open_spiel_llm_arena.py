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
from poker_agent.agents import MLPolicyAgent
from poker_agent.schemas import PredictionRequest, PredictionResponse


PHASE3_OPEN_SPIEL_ARENA_VERSION = "2026-07-02"
DEFAULT_OPEN_SPIEL_GAME = "kuhn_poker"
AGENT_ONLY_ARENA_STATUS = "AGENT_ONLY_OPEN_SPIEL_ARENA"
RUNTIME_PENDING_STATUS = "READY_PENDING_OPEN_SPIEL_RUNTIME"
RUN_COMPLETED_STATUS = "COMPLETED"
METRICS_BLOCKED_STATUS = "BLOCKED_UNTIL_OPEN_SPIEL_RUNTIME_AND_PHASE1_ADAPTERS"
RL_TRAINING_PROOF_PENDING_STATUS = "TRAINING_PROOF_NOT_COMPLETED"
RL_TRAINING_PROOF_COMPLETED_STATUS = "TRAINING_PROOF_COMPLETED"
MINIMUM_RL_STABILITY_SEEDS = 5
MINIMUM_RL_LONG_RUN_EPISODES = 5_000


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
    agent_a_model_path: str | None = None
    agent_b_model_path: str | None = None
    phase1_adapters_ready: bool = False
    independent_seed_count: int = 1
    policy_update_training_completed: bool = False

    def agent_specs(self) -> tuple[ArenaAgentSpec, ArenaAgentSpec]:
        return (
            ArenaAgentSpec(
                seat=0,
                name=self.agent_a_name,
                source=self.agent_a_source,
                model_path=self.agent_a_model_path,
            ),
            ArenaAgentSpec(
                seat=1,
                name=self.agent_b_name,
                source=self.agent_b_source,
                model_path=self.agent_b_model_path,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_name": self.game_name,
            "episodes": self.episodes,
            "seed": self.seed,
            "max_steps_per_episode": self.max_steps_per_episode,
            "phase1_adapters_ready": self.phase1_adapters_ready,
            "independent_seed_count": self.independent_seed_count,
            "policy_update_training_completed": self.policy_update_training_completed,
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
        phase1_policy_adapters_ready: bool = True,
        real_open_spiel_runtime_available: bool = False,
        independent_seed_count: int = 1,
        policy_update_training_completed: bool = False,
    ):
        self.game = game
        self.policies = policies
        self.seed = seed
        self.game_name = game_name
        self.rng = random.Random(seed)
        self.phase1_policy_adapters_ready = phase1_policy_adapters_ready
        self.real_open_spiel_runtime_available = real_open_spiel_runtime_available
        self.independent_seed_count = independent_seed_count
        self.policy_update_training_completed = policy_update_training_completed
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

        payload = {
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
                "open_spiel_runtime_executed": True,
                "phase1_policy_adapters_ready": self.phase1_policy_adapters_ready,
                "metrics_claim_allowed": self.phase1_policy_adapters_ready,
                "purpose": (
                    "Evaluate the two Phase 1 LLM policies against each other in an OpenSpiel "
                    "multi-agent arena before any promotion decision."
                ),
            },
        }
        payload["rl_training_proof_boundary"] = _rl_training_proof_boundary(
            real_open_spiel_runtime_available=self.real_open_spiel_runtime_available,
            phase1_trained_policy_artifacts_attached=self.phase1_policy_adapters_ready,
            agent_only_table_verified=payload["arena_contract"]["agent_only_table"],
            episodes=episodes,
            independent_seed_count=self.independent_seed_count,
            policy_update_training_completed=self.policy_update_training_completed,
        )
        payload["proof_cases"] = build_phase3_open_spiel_proof_cases(payload)
        payload["invariants"] = validate_phase3_open_spiel_arena_report(payload)
        payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
        return payload


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
    except OpenSpielRuntimeUnavailable:
        return _pending_runtime_report(config, runtime_available=False)

    if not config.phase1_adapters_ready:
        return _pending_runtime_report(
            config,
            runtime_available=True,
            reason=(
                "OpenSpiel is available, but measured metrics remain blocked until both Phase 1 "
                "trained policy adapters are explicitly provided."
            ),
        )

    game = pyspiel.load_game(config.game_name)
    policies = _load_phase1_policy_adapters(project_root, config)
    arena = OpenSpielAgentOnlyArena(
        game,
        policies,
        seed=config.seed,
        game_name=config.game_name,
        phase1_policy_adapters_ready=True,
        real_open_spiel_runtime_available=True,
        independent_seed_count=config.independent_seed_count,
        policy_update_training_completed=config.policy_update_training_completed,
    )
    payload = arena.run(config.episodes, max_steps_per_episode=config.max_steps_per_episode)
    payload["project_root"] = str(project_root)
    payload["runtime_note"] = (
        "This run uses the OpenSpiel runtime and explicitly supplied Phase 1 policy adapters."
    )
    payload["invariants"] = validate_phase3_open_spiel_arena_report(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
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
    proof = payload.get("rl_training_proof_boundary") or {}
    proof_cases = payload.get("proof_cases") or []
    lines.extend(
        [
            "",
            "## RL Training Proof Boundary",
            "",
            f"- Status: `{proof.get('status')}`",
            f"- Real OpenSpiel runtime available: `{proof.get('real_open_spiel_runtime_available')}`",
            f"- Phase 1 trained policy artifacts attached: `{proof.get('phase1_trained_policy_artifacts_attached')}`",
            f"- Seed stability evaluated: `{proof.get('seed_stability_evaluated')}`",
            f"- Long run completed: `{proof.get('long_run_completed')}`",
            f"- Policy-update training completed: `{proof.get('policy_update_training_completed')}`",
            f"- Measured win-rate claim allowed: `{proof.get('measured_win_rate_claim_allowed')}`",
            f"- Current delivery blocker: `{proof.get('current_delivery_blocker')}`",
            f"- Model-quality risk: `{proof.get('model_quality_risk')}`",
            "",
            "## Proof Cases",
            "",
        ]
    )
    for case in proof_cases:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, result `{case['result']}`"
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


def _rl_training_proof_boundary(
    *,
    real_open_spiel_runtime_available: bool,
    phase1_trained_policy_artifacts_attached: bool,
    agent_only_table_verified: bool,
    episodes: int,
    independent_seed_count: int,
    policy_update_training_completed: bool,
) -> dict[str, Any]:
    seed_stability_evaluated = int(independent_seed_count) >= MINIMUM_RL_STABILITY_SEEDS
    long_run_completed = int(episodes) >= MINIMUM_RL_LONG_RUN_EPISODES
    measured_win_rate_claim_allowed = all(
        (
            real_open_spiel_runtime_available,
            phase1_trained_policy_artifacts_attached,
            agent_only_table_verified,
            seed_stability_evaluated,
            long_run_completed,
            policy_update_training_completed,
        )
    )
    missing_requirements = []
    if not real_open_spiel_runtime_available:
        missing_requirements.append("real_open_spiel_runtime")
    if not phase1_trained_policy_artifacts_attached:
        missing_requirements.append("two_phase1_trained_policy_artifacts")
    if not agent_only_table_verified:
        missing_requirements.append("agent_only_table")
    if not seed_stability_evaluated:
        missing_requirements.append("seed_stability")
    if not long_run_completed:
        missing_requirements.append("long_run_training_volume")
    if not policy_update_training_completed:
        missing_requirements.append("policy_update_training")
    return {
        "status": RL_TRAINING_PROOF_COMPLETED_STATUS
        if measured_win_rate_claim_allowed
        else RL_TRAINING_PROOF_PENDING_STATUS,
        "real_open_spiel_runtime_required": True,
        "real_open_spiel_runtime_available": real_open_spiel_runtime_available,
        "phase1_trained_policy_artifacts_required": True,
        "phase1_trained_policy_artifacts_attached": phase1_trained_policy_artifacts_attached,
        "agent_only_table_required": True,
        "agent_only_table_verified": agent_only_table_verified,
        "seed_stability_required": True,
        "minimum_independent_seeds": MINIMUM_RL_STABILITY_SEEDS,
        "independent_seed_count": int(independent_seed_count),
        "seed_stability_evaluated": seed_stability_evaluated,
        "long_run_required": True,
        "minimum_long_run_episodes": MINIMUM_RL_LONG_RUN_EPISODES,
        "episodes": int(episodes),
        "long_run_completed": long_run_completed,
        "policy_update_training_required": True,
        "policy_update_training_completed": policy_update_training_completed,
        "measured_win_rate_claim_allowed": measured_win_rate_claim_allowed,
        "current_delivery_blocker": False,
        "model_quality_risk": not measured_win_rate_claim_allowed,
        "missing_requirements": missing_requirements,
        "allowed_current_claim": (
            "The Phase 3 agent-only OpenSpiel arena code is ready for a measured run when the "
            "runtime, Phase 1 adapters, seed-stability run, and long-run training profile are available."
        ),
        "blocked_claim": (
            "Do not claim RL win-rate or production strategy quality from Phase 3 until real OpenSpiel "
            "runtime execution, two trained Phase 1 policy artifacts, seed stability, long-run volume, "
            "and policy-update training are all complete."
        ),
    }


def build_phase3_open_spiel_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    def cloned() -> dict[str, Any]:
        return json.loads(json.dumps(payload))

    cases: list[dict[str, Any]] = []

    def record(name: str, candidate: dict[str, Any], expected_status: str) -> None:
        candidate.pop("proof_cases", None)
        candidate["invariants"] = validate_phase3_open_spiel_arena_report(candidate)
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
        ("blocks_win_rate_claim_without_real_open_spiel_runtime", "real_open_spiel_runtime_available"),
        ("blocks_win_rate_claim_without_two_phase1_artifacts", "phase1_trained_policy_artifacts_attached"),
        ("blocks_win_rate_claim_without_seed_stability", "seed_stability_evaluated"),
        ("blocks_win_rate_claim_without_long_run", "long_run_completed"),
        ("blocks_win_rate_claim_without_policy_update_training", "policy_update_training_completed"),
        ("blocks_win_rate_claim_without_agent_only_table", "agent_only_table_verified"),
    ):
        candidate = cloned()
        proof = candidate["rl_training_proof_boundary"]
        proof[field] = False
        proof["measured_win_rate_claim_allowed"] = True
        proof["status"] = RL_TRAINING_PROOF_COMPLETED_STATUS
        record(name, candidate, "FAIL")

    return cases


def validate_phase3_open_spiel_arena_report(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    status = payload.get("status")
    contract = payload.get("arena_contract") or {}
    quality = payload.get("quality_boundary") or {}
    proof = payload.get("rl_training_proof_boundary") or {}

    if status not in {RUNTIME_PENDING_STATUS, RUN_COMPLETED_STATUS}:
        violations.append("phase3_arena_status_must_be_pending_or_completed")
    if contract.get("arena_type") != AGENT_ONLY_ARENA_STATUS:
        violations.append("arena_type_must_be_agent_only_open_spiel")
    if contract.get("agent_only_table") is not True:
        violations.append("arena_must_be_agent_only")
    if contract.get("all_seats_controlled_by_agents") is not True:
        violations.append("all_open_spiel_seats_must_have_agent_policies")
    if contract.get("human_players_present") is not False:
        violations.append("human_players_must_not_be_present")
    if contract.get("fixed_scripted_opponents_present") is not False:
        violations.append("fixed_scripted_opponents_must_not_be_present")
    if int(contract.get("num_players") or 0) != 2:
        violations.append("phase3_llm_arena_must_have_two_players")
    if int(contract.get("policy_count") or 0) != int(contract.get("num_players") or 0):
        violations.append("policy_count_must_equal_player_count")
    if quality.get("is_reinforcement_learning_stage") is not True:
        violations.append("phase3_arena_must_be_marked_as_reinforcement_learning_stage")
    if quality.get("agent_only_table") is not True:
        violations.append("quality_boundary_must_preserve_agent_only_table")
    if quality.get("human_players_present") is not False:
        violations.append("quality_boundary_must_block_human_players")
    if quality.get("fixed_scripted_opponents_present") is not False:
        violations.append("quality_boundary_must_block_scripted_opponents")
    if proof.get("real_open_spiel_runtime_required") is not True:
        violations.append("rl_training_proof_must_require_real_open_spiel_runtime")
    if proof.get("phase1_trained_policy_artifacts_required") is not True:
        violations.append("rl_training_proof_must_require_two_phase1_adapters")
    if proof.get("agent_only_table_required") is not True:
        violations.append("rl_training_proof_must_require_agent_only_table")
    if proof.get("agent_only_table_verified") != contract.get("agent_only_table"):
        violations.append("rl_training_agent_only_proof_must_match_arena_contract")
    if proof.get("seed_stability_required") is not True:
        violations.append("rl_training_proof_must_require_seed_stability")
    if int(proof.get("minimum_independent_seeds") or 0) < MINIMUM_RL_STABILITY_SEEDS:
        violations.append("rl_training_proof_seed_threshold_too_low")
    if proof.get("long_run_required") is not True:
        violations.append("rl_training_proof_must_require_long_run")
    if int(proof.get("minimum_long_run_episodes") or 0) < MINIMUM_RL_LONG_RUN_EPISODES:
        violations.append("rl_training_proof_long_run_threshold_too_low")
    if proof.get("policy_update_training_required") is not True:
        violations.append("rl_training_proof_must_require_policy_update_training")
    if proof.get("current_delivery_blocker") is not False:
        violations.append("rl_training_proof_gap_must_not_block_current_delivery")
    required_training_gates = (
        proof.get("real_open_spiel_runtime_available") is True,
        proof.get("phase1_trained_policy_artifacts_attached") is True,
        proof.get("agent_only_table_verified") is True,
        proof.get("seed_stability_evaluated") is True,
        proof.get("long_run_completed") is True,
        proof.get("policy_update_training_completed") is True,
    )
    all_training_gates_pass = all(required_training_gates)
    if proof.get("measured_win_rate_claim_allowed") is True and not all_training_gates_pass:
        violations.append("rl_win_rate_claim_requires_runtime_adapters_seed_stability_long_run_and_training")
    if proof.get("status") == RL_TRAINING_PROOF_COMPLETED_STATUS and not all_training_gates_pass:
        violations.append("rl_training_proof_cannot_be_completed_without_all_gates")
    if not all_training_gates_pass and proof.get("model_quality_risk") is not True:
        violations.append("incomplete_rl_training_proof_must_remain_model_quality_risk")

    if status == RUNTIME_PENDING_STATUS:
        runtime_boundary = payload.get("runtime_boundary") or {}
        if "metrics" in payload:
            violations.append("pending_open_spiel_report_must_not_include_measured_metrics")
        if "sample_episodes" in payload:
            violations.append("pending_open_spiel_report_must_not_include_sample_episodes")
        if runtime_boundary.get("run_if_available_required_for_metrics") is not True:
            violations.append("pending_report_must_require_measured_runtime_run_for_metrics")
        if runtime_boundary.get("phase1_adapters_required_for_metrics") is not True:
            violations.append("pending_report_must_require_phase1_adapters_for_metrics")
        if quality.get("metrics_claim_allowed") is not False:
            violations.append("pending_report_must_block_metric_claims")
        if quality.get("metrics_blocked_until") != METRICS_BLOCKED_STATUS:
            violations.append("pending_report_must_explain_metrics_blocker")
        if proof.get("measured_win_rate_claim_allowed") is not False:
            violations.append("pending_report_must_block_rl_win_rate_claims")

    if status == RUN_COMPLETED_STATUS:
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            violations.append("completed_open_spiel_report_must_include_measured_metrics")
        if quality.get("open_spiel_runtime_executed") is not True:
            violations.append("completed_report_must_confirm_open_spiel_runtime_execution")
        if quality.get("phase1_policy_adapters_ready") is not True:
            violations.append("completed_report_must_confirm_phase1_policy_adapters")
        if quality.get("metrics_claim_allowed") is not True:
            violations.append("completed_report_must_allow_measured_metrics_only_after_runtime_and_adapters")

    return {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def _pending_runtime_report(
    config: ArenaRunConfig,
    *,
    runtime_available: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    payload = {
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
            "phase1_adapters_ready": config.phase1_adapters_ready,
            "phase1_adapters_required_for_metrics": True,
            "pyspiel_runtime_required_for_metrics": True,
            "reason": reason or (
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
            "metrics_blocked_until": METRICS_BLOCKED_STATUS,
        },
    }
    payload["rl_training_proof_boundary"] = _rl_training_proof_boundary(
        real_open_spiel_runtime_available=runtime_available,
        phase1_trained_policy_artifacts_attached=config.phase1_adapters_ready,
        agent_only_table_verified=payload["arena_contract"]["agent_only_table"],
        episodes=config.episodes,
        independent_seed_count=config.independent_seed_count,
        policy_update_training_completed=config.policy_update_training_completed,
    )
    payload["proof_cases"] = build_phase3_open_spiel_proof_cases(payload)
    payload["invariants"] = validate_phase3_open_spiel_arena_report(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def _load_phase1_policy_adapters(project_root: Path, config: ArenaRunConfig) -> tuple[OpenSpielPolicy, OpenSpielPolicy]:
    if not config.agent_a_model_path or not config.agent_b_model_path:
        raise OpenSpielArenaError(
            "Measured Phase 3 arena runs require both agent_a_model_path and agent_b_model_path."
        )
    agent_a_path = _resolve_project_path(project_root, config.agent_a_model_path)
    agent_b_path = _resolve_project_path(project_root, config.agent_b_model_path)
    if not agent_a_path.exists() or not agent_b_path.exists():
        raise OpenSpielArenaError(
            f"Missing Phase 1 policy adapter artifacts: {agent_a_path}, {agent_b_path}"
        )
    return (
        ServicePolicyOpenSpielAdapter(
            config.agent_a_name,
            MLPolicyAgent.from_path(agent_a_path),
            source=f"{config.agent_a_source}:{agent_a_path}",
        ),
        ServicePolicyOpenSpielAdapter(
            config.agent_b_name,
            MLPolicyAgent.from_path(agent_b_path),
            source=f"{config.agent_b_source}:{agent_b_path}",
        ),
    )


def _resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


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
