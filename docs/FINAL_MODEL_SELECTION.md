# Final Model Selection

- Selected model: `QwenPoker checkpoint_40960`
- Model version: `qwenpoker:checkpoint_40960`
- Environment: `Heads-up No-Limit Hold'em 100 BB | OpenSpiel FCHPA`
- Action space: `fold, check_call, half_pot, full_pot, all_in`
- Hands: `5000` total, `2500` per seat
- Seed: `20260714`
- Policy mode: `sampled_policy`

## Metrics

- Win rate: `64.48%`
- Returns: `+365.29 BB/100`
- 95% return CI entirely positive: `true`
- Profitable from both positions: `true`

## Opponent Suite

- `pool_sft`: `40%`
- `random`: `15%`
- `calling`: `30%`
- `aggressive`: `15%`

## Selection Gates

- `selected_checkpoint`: `PASS`
- `benchmark_hands`: `PASS`
- `seat_balance`: `PASS`
- `opponent_suite_weights`: `PASS`
- `win_rate`: `PASS`
- `bb_per_100`: `PASS`
- `returns_ci_95_positive`: `PASS`
- `both_positions_profitable`: `PASS`
