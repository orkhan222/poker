from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from poker_agent.action_space import CANONICAL_ACTIONS, normalize_action


RANKS = "23456789TJQKA"
SUITS = "cdhs"


@dataclass(frozen=True)
class PokerEngineConfig:
    name: str = "internal_single_decision_nlhe_simulator"
    game_type: str = "nl_holdem"
    table_size: int = 6
    starting_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    rake_percentage: float = 0.0
    rake_cap: float = 0.0
    max_opponents_per_hand: int = 5


@dataclass(frozen=True)
class RewardShapingConfig:
    chip_delta_weight: float = 1.0
    win_loss_weight: float = 0.15
    showdown_strength_weight: float = 0.05
    illegal_action_penalty: float = -1.0
    fold_penalty_when_free: float = -0.05
    aggression_bonus: float = 0.02
    survival_bonus: float = 0.01


@dataclass(frozen=True)
class SeedPolicy:
    base_seed: int = 20260713
    namespace: str = "poker_rl"

    def derive(self, *parts: Any) -> int:
        payload = "|".join([self.namespace, str(self.base_seed), *(str(part) for part in parts)])
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") & 0x7FFFFFFF

    def rng(self, *parts: Any) -> random.Random:
        return random.Random(self.derive(*parts))


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    family: str
    weight: float = 1.0
    aggression: float = 0.30
    call_threshold: float = 0.28
    bluff_frequency: float = 0.05


@dataclass
class OpponentPool:
    opponents: list[OpponentSpec] = field(default_factory=list)

    @classmethod
    def default(cls) -> "OpponentPool":
        return cls(
            [
                OpponentSpec("tight_value", "rule", weight=1.0, aggression=0.18, call_threshold=0.22, bluff_frequency=0.02),
                OpponentSpec("balanced_reg", "rule", weight=1.0, aggression=0.34, call_threshold=0.32, bluff_frequency=0.06),
                OpponentSpec("loose_aggressive", "rule", weight=0.8, aggression=0.55, call_threshold=0.42, bluff_frequency=0.14),
                OpponentSpec("calling_station", "rule", weight=0.7, aggression=0.14, call_threshold=0.55, bluff_frequency=0.01),
            ]
        )

    def sample(self, rng: random.Random, count: int) -> list[OpponentSpec]:
        if not self.opponents:
            raise ValueError("OpponentPool must contain at least one opponent")
        weights = [max(0.0, opponent.weight) for opponent in self.opponents]
        if sum(weights) <= 0:
            weights = [1.0 for _ in self.opponents]
        return [rng.choices(self.opponents, weights=weights, k=1)[0] for _ in range(count)]

    def to_dict(self) -> dict[str, Any]:
        return {"opponents": [asdict(opponent) for opponent in self.opponents]}


@dataclass
class PlayerRuntimeState:
    position: str
    stack: float
    cards: list[str]
    folded: bool = False
    committed: float = 0.0
    opponent: OpponentSpec | None = None


@dataclass
class RewardBreakdown:
    raw_chip_delta: float
    shaped_reward: float
    components: dict[str, float]


@dataclass
class RLEpisodeResult:
    episode_id: str
    seed: int
    hero_action: str
    hero_bet_size: float
    terminal: bool
    reward: RewardBreakdown
    final_stacks: dict[str, float]
    board_cards: list[str]
    opponents: list[str]
    winner_positions: list[str]


HeroPolicy = Callable[[dict[str, Any]], tuple[str, float]]


def describe_rl_environment(
    engine: PokerEngineConfig | None = None,
    reward_shaping: RewardShapingConfig | None = None,
    opponent_pool: OpponentPool | None = None,
    seed_policy: SeedPolicy | None = None,
) -> dict[str, Any]:
    engine = engine or PokerEngineConfig()
    reward_shaping = reward_shaping or RewardShapingConfig()
    opponent_pool = opponent_pool or OpponentPool.default()
    seed_policy = seed_policy or SeedPolicy()
    return {
        "poker_simulator_engine": asdict(engine),
        "self_play_league": {
            "unit": "single_hand_episode",
            "matchmaking": "seeded opponent-pool sampling per episode",
            "hero_position": "BTN",
            "opponent_positions": _positions(engine.table_size)[1:],
        },
        "opponent_pool": opponent_pool.to_dict(),
        "seed_policy": asdict(seed_policy),
        "reward_shaping": asdict(reward_shaping),
        "actions": list(CANONICAL_ACTIONS),
        "notes": [
            "This is an offline research/simulation environment, not a real-money automation adapter.",
            "The internal engine is deterministic and intentionally simple; swap engine.name for a validated poker engine before high-stakes evaluation.",
        ],
    }


class NoLimitHoldemSingleDecisionEngine:
    def __init__(
        self,
        *,
        config: PokerEngineConfig | None = None,
        reward_shaping: RewardShapingConfig | None = None,
        opponent_pool: OpponentPool | None = None,
    ) -> None:
        self.config = config or PokerEngineConfig()
        self.reward_shaping = reward_shaping or RewardShapingConfig()
        self.opponent_pool = opponent_pool or OpponentPool.default()
        self.players: list[PlayerRuntimeState] = []
        self.board_cards: list[str] = []
        self.pot = 0.0
        self.current_bet = 0.0
        self.seed = 0
        self.rng = random.Random(0)

    def reset(self, *, seed: int) -> dict[str, Any]:
        self.seed = seed
        self.rng = random.Random(seed)
        deck = _shuffled_deck(self.rng)
        positions = _positions(self.config.table_size)
        opponent_specs = self.opponent_pool.sample(self.rng, min(self.config.table_size - 1, self.config.max_opponents_per_hand))
        self.players = []
        for index, position in enumerate(positions):
            opponent = None if index == 0 else opponent_specs[index - 1]
            self.players.append(
                PlayerRuntimeState(
                    position=position,
                    stack=self.config.starting_stack,
                    cards=[deck.pop(), deck.pop()],
                    opponent=opponent,
                )
            )
        self.board_cards = [deck.pop() for _ in range(5)]
        self.pot = 0.0
        self.current_bet = 0.0
        self._post_forced_bets()
        return self._observation()

    def step(self, action: str, *, bet_size: float = 0.0) -> RLEpisodeResult:
        hero = self.players[0]
        start_stack = hero.stack
        legal_actions = self.legal_actions(hero)
        canonical_action = normalize_action(action)
        illegal = canonical_action not in legal_actions
        if illegal:
            canonical_action = "fold" if "fold" in legal_actions else legal_actions[0]

        hero_bet = self._apply_action(hero, canonical_action, bet_size)
        for player in self.players[1:]:
            if player.folded or player.stack <= 0:
                continue
            opponent_action, opponent_bet = self._opponent_action(player)
            self._apply_action(player, opponent_action, opponent_bet)

        active_players = [player for player in self.players if not player.folded]
        winners = self._settle(active_players)
        raw_delta = hero.stack - start_stack
        strength = _hand_strength(hero.cards, self.board_cards)
        reward = shape_reward(
            raw_chip_delta=raw_delta,
            won_hand=hero.position in winners,
            action=canonical_action,
            illegal_action=illegal,
            showdown_strength=strength,
            to_call=self.amount_to_call(hero),
            config=self.reward_shaping,
        )
        return RLEpisodeResult(
            episode_id=f"seed-{self.seed}",
            seed=self.seed,
            hero_action=canonical_action,
            hero_bet_size=hero_bet,
            terminal=True,
            reward=reward,
            final_stacks={player.position: round(player.stack, 6) for player in self.players},
            board_cards=list(self.board_cards),
            opponents=[player.opponent.name for player in self.players[1:] if player.opponent],
            winner_positions=winners,
        )

    def legal_actions(self, player: PlayerRuntimeState) -> tuple[str, ...]:
        if player.stack <= 0:
            return ("check",)
        to_call = self.amount_to_call(player)
        if to_call > 0:
            return ("fold", "call", "raise", "all_in")
        return ("check", "bet", "all_in")

    def amount_to_call(self, player: PlayerRuntimeState) -> float:
        return max(0.0, self.current_bet - player.committed)

    def _post_forced_bets(self) -> None:
        for index, amount in ((1, self.config.small_blind), (2, self.config.big_blind)):
            if index < len(self.players):
                player = self.players[index]
                paid = min(player.stack, amount)
                player.stack -= paid
                player.committed += paid
                self.pot += paid
                self.current_bet = max(self.current_bet, player.committed)
        for player in self.players:
            if self.config.ante > 0:
                paid = min(player.stack, self.config.ante)
                player.stack -= paid
                player.committed += paid
                self.pot += paid

    def _apply_action(self, player: PlayerRuntimeState, action: str, bet_size: float) -> float:
        action = normalize_action(action)
        if action == "fold":
            player.folded = True
            return 0.0
        if action == "check":
            return 0.0
        if action == "call":
            amount = min(player.stack, self.amount_to_call(player))
        elif action == "all_in":
            amount = player.stack
        elif action in {"bet", "raise"}:
            minimum = self.config.big_blind if self.current_bet <= 0 else self.current_bet + self.config.big_blind - player.committed
            amount = min(player.stack, max(float(bet_size or 0.0), minimum))
        else:
            player.folded = True
            return 0.0
        player.stack -= amount
        player.committed += amount
        self.pot += amount
        self.current_bet = max(self.current_bet, player.committed)
        return amount

    def _opponent_action(self, player: PlayerRuntimeState) -> tuple[str, float]:
        assert player.opponent is not None
        to_call = self.amount_to_call(player)
        strength = _hand_strength(player.cards, self.board_cards)
        bluff = self.rng.random() < player.opponent.bluff_frequency
        if to_call <= 0:
            if strength + player.opponent.aggression * 0.4 + (0.25 if bluff else 0.0) >= 0.55:
                return "bet", self.config.big_blind * (2.0 + player.opponent.aggression)
            return "check", 0.0
        if strength + (0.15 if bluff else 0.0) < player.opponent.call_threshold:
            return "fold", 0.0
        if strength + player.opponent.aggression >= 0.95:
            return "raise", self.current_bet + self.config.big_blind * 2.0
        return "call", to_call

    def _settle(self, active_players: list[PlayerRuntimeState]) -> list[str]:
        if not active_players:
            active_players = [self.players[0]]
        if len(active_players) == 1:
            winners = [active_players[0]]
        else:
            best_score = max(_hand_strength(player.cards, self.board_cards) for player in active_players)
            winners = [player for player in active_players if _hand_strength(player.cards, self.board_cards) == best_score]
        rake = min(self.pot * self.config.rake_percentage, self.config.rake_cap) if self.config.rake_percentage > 0 else 0.0
        payout = (self.pot - rake) / len(winners)
        for winner in winners:
            winner.stack += payout
        return [winner.position for winner in winners]

    def _observation(self) -> dict[str, Any]:
        hero = self.players[0]
        return {
            "position": hero.position,
            "hole_cards": list(hero.cards),
            "board_cards": [],
            "pot": self.pot,
            "current_bet": self.current_bet,
            "amount_to_call": self.amount_to_call(hero),
            "stack": hero.stack,
            "effective_stack": min((player.stack for player in self.players[1:] if not player.folded), default=hero.stack),
            "small_blind": self.config.small_blind,
            "big_blind": self.config.big_blind,
            "ante": self.config.ante,
            "legal_actions": list(self.legal_actions(hero)),
            "action_order": [player.position for player in self.players],
            "opponents": [player.opponent.name for player in self.players[1:] if player.opponent],
        }


def shape_reward(
    *,
    raw_chip_delta: float,
    won_hand: bool,
    action: str,
    illegal_action: bool,
    showdown_strength: float,
    to_call: float,
    config: RewardShapingConfig,
) -> RewardBreakdown:
    action = normalize_action(action)
    components = {
        "chip_delta": raw_chip_delta * config.chip_delta_weight,
        "win_loss": (1.0 if won_hand else -1.0) * config.win_loss_weight,
        "showdown_strength": showdown_strength * config.showdown_strength_weight,
        "illegal_action": config.illegal_action_penalty if illegal_action else 0.0,
        "free_fold": config.fold_penalty_when_free if action == "fold" and to_call <= 0 else 0.0,
        "aggression": config.aggression_bonus if action in {"bet", "raise", "all_in"} else 0.0,
        "survival": config.survival_bonus if raw_chip_delta >= 0 else 0.0,
    }
    return RewardBreakdown(
        raw_chip_delta=raw_chip_delta,
        shaped_reward=sum(components.values()),
        components=components,
    )


@dataclass
class SelfPlayLeague:
    engine_config: PokerEngineConfig = field(default_factory=PokerEngineConfig)
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    opponent_pool: OpponentPool = field(default_factory=OpponentPool.default)
    seed_policy: SeedPolicy = field(default_factory=SeedPolicy)

    def run_episode(
        self,
        hero_policy: HeroPolicy,
        *,
        generation: int,
        episode_index: int,
    ) -> RLEpisodeResult:
        seed = self.seed_policy.derive("generation", generation, "episode", episode_index)
        engine = NoLimitHoldemSingleDecisionEngine(
            config=self.engine_config,
            reward_shaping=self.reward_shaping,
            opponent_pool=self.opponent_pool,
        )
        observation = engine.reset(seed=seed)
        action, bet_size = hero_policy(observation)
        return engine.step(action, bet_size=bet_size)

    def run_match(
        self,
        hero_policy: HeroPolicy,
        *,
        generation: int = 0,
        episodes: int = 100,
    ) -> dict[str, Any]:
        results = [
            self.run_episode(hero_policy, generation=generation, episode_index=index)
            for index in range(episodes)
        ]
        rewards = [result.reward.shaped_reward for result in results]
        wins = [1.0 if "BTN" in result.winner_positions else 0.0 for result in results]
        return {
            "generation": generation,
            "episodes": episodes,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "win_rate": sum(wins) / len(wins) if wins else 0.0,
            "avg_chip_delta": sum(result.reward.raw_chip_delta for result in results) / len(results) if results else 0.0,
            "opponent_pool": self.opponent_pool.to_dict(),
            "engine": asdict(self.engine_config),
            "reward_shaping": asdict(self.reward_shaping),
        }


def seed_policy_hero(observation: dict[str, Any]) -> tuple[str, float]:
    legal_actions = set(observation.get("legal_actions") or [])
    to_call = float(observation.get("amount_to_call", 0.0))
    big_blind = float(observation.get("big_blind", 1.0))
    if to_call <= 0 and "bet" in legal_actions:
        return "bet", 2.5 * big_blind
    if "call" in legal_actions and to_call <= 3.0 * big_blind:
        return "call", to_call
    if "fold" in legal_actions:
        return "fold", 0.0
    return next(iter(legal_actions or {"check"})), 0.0


def _positions(table_size: int) -> list[str]:
    defaults = ["BTN", "SB", "BB", "UTG", "MP", "CO", "HJ", "LJ", "UTG1"]
    return defaults[: max(2, min(table_size, len(defaults)))]


def _shuffled_deck(rng: random.Random) -> list[str]:
    deck = [rank + suit for suit in SUITS for rank in RANKS]
    rng.shuffle(deck)
    return deck


def _hand_strength(hole_cards: list[str], board_cards: list[str]) -> float:
    ranks = [_rank_value(card) for card in [*hole_cards, *board_cards]]
    ranks = [rank for rank in ranks if rank > 0]
    if not ranks:
        return 0.0
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    pair_bonus = 0.18 if any(count >= 2 for count in counts.values()) else 0.0
    trips_bonus = 0.16 if any(count >= 3 for count in counts.values()) else 0.0
    high_card = max(ranks) / 14.0
    top_two = sorted(ranks, reverse=True)[:2]
    kicker = (sum(top_two) / 28.0) if top_two else 0.0
    return min(1.0, 0.62 * high_card + 0.22 * kicker + pair_bonus + trips_bonus)


def _rank_value(card: str) -> int:
    if not card:
        return 0
    rank = card[0].upper()
    return RANKS.index(rank) + 2 if rank in RANKS else 0
