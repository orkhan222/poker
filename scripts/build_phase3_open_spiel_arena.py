from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.open_spiel_llm_arena import ArenaRunConfig, write_phase3_open_spiel_arena_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 3 OpenSpiel LLM-vs-LLM arena report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "phase3_open_spiel_arena.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "phase3_open_spiel_arena.md", type=Path)
    parser.add_argument("--game-name", default="kuhn_poker")
    parser.add_argument("--episodes", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-steps-per-episode", default=256, type=int)
    parser.add_argument("--agent-a-name", default="phase1_llm_agent_a")
    parser.add_argument("--agent-b-name", default="phase1_llm_agent_b")
    parser.add_argument("--agent-a-source", default="phase1_trained_llm_policy_a")
    parser.add_argument("--agent-b-source", default="phase1_trained_llm_policy_b")
    parser.add_argument(
        "--run-if-available",
        action="store_true",
        help="Run measured OpenSpiel episodes if pyspiel is installed; otherwise write a pending-runtime report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ArenaRunConfig(
        game_name=args.game_name,
        episodes=args.episodes,
        seed=args.seed,
        max_steps_per_episode=args.max_steps_per_episode,
        agent_a_name=args.agent_a_name,
        agent_b_name=args.agent_b_name,
        agent_a_source=args.agent_a_source,
        agent_b_source=args.agent_b_source,
    )
    payload = write_phase3_open_spiel_arena_report(
        args.project_root,
        args.out,
        args.markdown_out,
        config=config,
        run_if_available=args.run_if_available,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "agent_only_table": payload["arena_contract"]["agent_only_table"],
                "game_name": payload["environment"]["game_name"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
