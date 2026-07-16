# Poker ML Project Scope Contract

- Schema: `project_scope.v1`
- Validation status: `PASS`
- Fingerprint: `f10b5ab4b12dd68cd3138c441e4fb6b7ac589ddf6bc747a0988b9a8f175f15ed`

## Goals

- `learn_from_historical_human_poker_hands`
- `improve_via_offline_self_play`
- `match_human_decision_patterns`
- `deploy_as_authorized_microservice`

## Phases

### Phase 1 - Two Baselines

- Tracks: llm_based_agent, end_to_end_policy_model
- Deliverables: working_llm_decision_agent, trained_supervised_policy_checkpoint, baseline_evaluation_metrics

### Phase 2 - Selection and Optimization

- Tracks: baseline_comparison, reproducible_experiments, model_selection
- Deliverables: best_performing_model, reproducible_pipeline

### Phase 3 - Evaluation

- Tracks: held_out_human_alignment, simulation_performance, seed_stability
- Deliverables: final_model, evaluation_report

### Phase 4 - Deployment

- Tracks: fastapi_service, predict_endpoint, dockerized_runtime, api_docs
- Deliverables: deployable_agent_service

## Senior Requirements

### Game Scope and Operating Boundary

- Key: `game_scope`

### Deployment API Contract

- Key: `deployment_api`

### Dataset Schema Extensions

- Key: `dataset_schema_extensions`

### Data Validation Rules

- Key: `data_validation`

### Labeling Contract

- Key: `labeling_contract`

### Action and State Space

- Key: `action_and_state_space`

### Baseline and Model Architecture

- Key: `baseline_and_architecture`

### RL Environment and Self-Play

- Key: `rl_self_play`

### Evaluation and Acceptance Criteria

- Key: `evaluation_acceptance`

### MLOps, Monitoring, and Security

- Key: `mlops_monitoring_security`

### Final Deliverables

- Key: `final_deliverables`

## Dataset Tables

- `hands.csv`: one_hand (21 fields)
- `players.csv`: one_player_in_hand (10 fields)
- `actions.csv`: one_player_action (20 fields)
- `stack_events.csv`: one_stack_change_event (10 fields)
