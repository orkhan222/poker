from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.rl_environment import (
    PokerEngineConfig,
    RewardShapingConfig,
    SeedPolicy,
    SelfPlayLeague,
    describe_rl_environment,
    seed_policy_hero,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or smoke-test the offline poker RL environment")
    parser.add_argument("--episodes", type=int, default=0, help="Run this many self-play smoke episodes.")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--table-size", type=int, default=6)
    parser.add_argument("--starting-stack", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = PokerEngineConfig(table_size=args.table_size, starting_stack=args.starting_stack)
    reward = RewardShapingConfig()
    seed_policy = SeedPolicy(base_seed=args.seed)
    payload = describe_rl_environment(engine=engine, reward_shaping=reward, seed_policy=seed_policy)
    if args.episodes > 0:
        league = SelfPlayLeague(engine_config=engine, reward_shaping=reward, seed_policy=seed_policy)
        payload["self_play_smoke"] = league.run_match(seed_policy_hero, episodes=args.episodes)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
