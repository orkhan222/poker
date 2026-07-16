from __future__ import annotations

from poker_agent.rl_environment import (
    NoLimitHoldemSingleDecisionEngine,
    OpponentPool,
    PokerEngineConfig,
    RewardShapingConfig,
    SeedPolicy,
    SelfPlayLeague,
    describe_rl_environment,
    seed_policy_hero,
    shape_reward,
)


def test_rl_environment_contract_declares_engine_league_pool_seed_and_reward() -> None:
    contract = describe_rl_environment()

    assert contract["poker_simulator_engine"]["game_type"] == "nl_holdem"
    assert contract["self_play_league"]["unit"] == "single_hand_episode"
    assert len(contract["opponent_pool"]["opponents"]) >= 4
    assert contract["seed_policy"]["base_seed"] == 20260713
    assert "chip_delta_weight" in contract["reward_shaping"]
    assert contract["actions"] == ["fold", "check", "call", "bet", "raise", "all_in"]


def test_seed_policy_and_engine_reset_are_deterministic() -> None:
    seed_policy = SeedPolicy(base_seed=123)
    seed = seed_policy.derive("generation", 2, "episode", 7)
    left = NoLimitHoldemSingleDecisionEngine()
    right = NoLimitHoldemSingleDecisionEngine()

    assert seed == seed_policy.derive("generation", 2, "episode", 7)
    assert left.reset(seed=seed) == right.reset(seed=seed)


def test_reward_shaping_records_components() -> None:
    reward = shape_reward(
        raw_chip_delta=3.0,
        won_hand=True,
        action="raise",
        illegal_action=False,
        showdown_strength=0.75,
        to_call=1.0,
        config=RewardShapingConfig(),
    )

    assert reward.raw_chip_delta == 3.0
    assert reward.components["chip_delta"] == 3.0
    assert reward.components["win_loss"] > 0
    assert reward.components["aggression"] > 0
    assert reward.shaped_reward > reward.raw_chip_delta


def test_self_play_league_runs_seeded_episode_and_match() -> None:
    league = SelfPlayLeague(
        engine_config=PokerEngineConfig(table_size=4),
        opponent_pool=OpponentPool.default(),
        seed_policy=SeedPolicy(base_seed=999),
    )

    episode = league.run_episode(seed_policy_hero, generation=0, episode_index=0)
    repeat = league.run_episode(seed_policy_hero, generation=0, episode_index=0)
    report = league.run_match(seed_policy_hero, generation=0, episodes=5)

    assert episode.seed == repeat.seed
    assert episode.final_stacks == repeat.final_stacks
    assert episode.terminal is True
    assert len(episode.opponents) == 3
    assert report["episodes"] == 5
    assert "avg_reward" in report
    assert "win_rate" in report
