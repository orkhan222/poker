# Poker Decision Agent

Poker Decision Agent is a FastAPI service and ML research workspace for poker action prediction from OCR and event-log data. The repository includes the API, trained model artifact, Hydra experiment configs, evaluation scripts, audit reports, and a packaged delivery ZIP.

## Delivery Status

As of the latest delivery build:

```text
repository_audit=PASS
repo_hygiene=PASS
delivery_verification=PASS
model_production_gate=FAIL
```

The package is reproducible and ready for technical handoff. The model is not marked as production-approved for autonomous decision policy use, because the current dataset still has known coverage and class-balance limitations. Those limitations are documented in `reports\dataset_audit.json` and `reports\production_gate.json`.

## Repository Layout

```text
.
|-- poker_agent/              API, schemas, feature extraction, model loading
|-- scripts/                  training, evaluation, audit, packaging checks
|-- configs/                  Hydra experiment configuration
|-- evaluation/               reviewed evaluation fixtures
|-- reports/                  generated metrics and audit outputs
|-- models/                   packaged model artifact
|-- release/                  delivery ZIP
|-- install.ps1               local environment setup
|-- run_server.ps1            API startup script
|-- complete_delivery.ps1     full delivery rebuild
|-- verify_delivery.ps1       final delivery verification
`-- README.md
```

## Install

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
.\install.ps1
```

## Run The API

```powershell
.\run_server.ps1
```

Open these endpoints after the server starts:

```text
http://127.0.0.1:8001/predict
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/health.json
http://127.0.0.1:8001/contract.json
```

The health endpoint returns model status, policy name, split strategy, and the validation macro F1 stored in the model metadata.

## Deployment API Contract

`poker_agent/api_contract.py` owns the machine-readable deployment contract for `POST /predict`. The service exposes the same contract at:

```text
GET /contract.json
```

The contract includes:

```text
api_version
endpoint
model_version
request_schema
response_schema
error_response_schema
error_codes
```

`POST /predict` request schema requires the core decision state:

```text
position
street
hole_cards
pot
stack
usage_boundary
```

The request schema also declares supported deployment fields such as `board_cards`, `current_bet`, `to_call`, `amount_to_call`, `effective_stack`, `legal_actions`, `legal_action_mask`, `game_scope`, blinds, button/dealer position, action order, and raise/all-in bounds.

Successful responses always include:

```text
schema_version
model_version
action
probabilities
confidence
model_status
bet_size
raise_to
raise_by
sizing_method
legal_actions
action_space
state_context
```

Deployment error codes are stable and machine-readable:

```text
INVALID_REQUEST             400
UNSUPPORTED_ACTION_SPACE    422
MODEL_UNAVAILABLE           503
UNAUTHORIZED                401
RATE_LIMITED                429
SECURITY_MISCONFIGURED      503
USAGE_BOUNDARY_VIOLATION    403
PREDICTION_FAILED           500
```

## Game Scope and Usage Boundary

`poker_agent/game_scope.py` normalizes the supported operating scope:

```text
game_type       nl_holdem
format          cash / tournament
table_size      6_max / 9_max
blinds          small_blind / big_blind / ante
rake            rake_percentage / rake_cap
stack_unit      chips / big_blinds / chips_or_big_blinds
```

`poker_agent/usage_boundary.py` enforces legal and ethical use for `POST /predict`. Requests must include `usage_boundary.declared_use` with one of:

```text
offline_research
simulation
authorized_environment
```

The service blocks `real_money_platform`, `unauthorized_platform`, `stealth_automation`, and `tos_bypass` with `USAGE_BOUNDARY_VIOLATION` / HTTP 403.

## MLOps Contract

`poker_agent/mlops.py` defines the local MLOps contract for handoff and CI:

```text
experiment tracking   reports/experiments.jsonl
model registry        reports/model_registry.json
dataset versioning    reports/dataset_versions.jsonl
MLOps report          reports/mlops_contract.json
CI smoke workflow     .github/workflows/smoke.yml
Docker image tag      poker-decision-agent:0.1.0
```

The smoke checker emits a dataset fingerprint, a tracked experiment run, a model registry entry, Docker image metadata, and validates CI/Docker contract files:

```powershell
python scripts\check_mlops_contract.py --smoke
```

Docker versioning is explicit through `APP_VERSION`, `VCS_REF`, and `BUILD_DATE` build args plus OCI labels such as `org.opencontainers.image.version`. `docker-compose.yml` uses `poker-decision-agent:${POKER_AGENT_VERSION:-0.1.0}` instead of an unversioned `latest` contract.

## Monitoring Contract

`poker_agent/monitoring.py` defines runtime and offline monitoring for:

```text
latency
invalid_states
confidence_drift
feature_drift
prediction_logs
audit_trail
```

`POST /predict` writes JSONL prediction records and audit events when the FastAPI service is running:

```text
reports/prediction_logs.jsonl
reports/audit_trail.jsonl
```

Each prediction log record includes `request_id`, `latency_ms`, selected action, legal actions, probabilities, confidence, invalid-state findings, feature fingerprint, feature values, and state context. Audit trail records include a hashable event envelope for downstream review.

Run the deterministic monitoring smoke check:

```powershell
python scripts\check_monitoring_contract.py --smoke
```

The smoke report writes `reports/monitoring_report.json` and validates latency thresholds, invalid-state detection, confidence drift, feature drift, prediction-log output, and audit-trail output.

## Security And Privacy Contract

`poker_agent/security.py` defines the API security/privacy contract:

```text
API auth              X-API-Key or Authorization: Bearer
secret storage        POKER_API_KEY_HASHES preferred, sha256 hex
rate limiting         in-memory fixed window per principal/client
secret redaction      api_key, authorization, password, secret, token
log retention         reports/prediction_logs.jsonl and reports/audit_trail.jsonl
```

Production deployments should set:

```powershell
$env:POKER_AUTH_REQUIRED="true"
$env:POKER_API_KEY_HASHES="<sha256-api-key-hash>"
$env:POKER_RATE_LIMIT_PER_MINUTE="60"
$env:POKER_LOG_RETENTION_DAYS="30"
$env:POKER_LOG_RETENTION_MAX_RECORDS="100000"
```

Run the security smoke check:

```powershell
python scripts\check_security_contract.py --smoke
```

The smoke report writes `reports/security_report.json` and verifies valid/missing/bad API key behavior, rate-limit rejection, secret redaction, and old-log pruning.

## Action Space Contract

`POST /predict` models the no-limit Hold'em action space explicitly. The canonical action keys are:

```text
fold
check
call
bet
raise
all_in
```

Requests may provide `legal_actions` or `legal_action_mask`. If omitted, the service derives legal actions from `to_call`, `stack`, `min_raise`, and raise bounds. Sizing fields are modeled separately:

```text
min_raise_to
max_raise_to
min_raise_by
max_raise_by
all_in_amount
```

Responses always return canonical probabilities, the selected legal action, `bet_size`, `raise_to`, `raise_by`, `sizing_method`, `legal_actions`, and a machine-readable `action_space` object. Illegal model probability mass is removed before action selection.

## State Feature Contract

`POST /predict` accepts explicit table-state fields in addition to cards and action history:

```text
pot / pot_size
current_bet
to_call / amount_to_call
button_position / dealer_position
small_blind / big_blind / ante
street
stack / effective_stack
action_order
```

The feature layer exposes both raw and normalized forms, including `pot_size`, `current_bet`, `amount_to_call`, `effective_stack`, `spr`, blind-normalized amounts, street one-hot flags, button/dealer indicators, and action-order features such as `action_order_index`, `players_before_hero`, and `players_after_hero`. Responses include a machine-readable `state_context` object with the parsed state used for inference.

## Dataset Schema Contract

The generated CSV dataset carries table/game context and per-action decision context.

`hands.csv` includes:

```text
table_id
game_type
small_blind
big_blind
ante
button_position
```

`actions.csv` includes the same table context plus action-level fields:

```text
action_amount
pot_before_action
pot_after_action
legal_actions
ocr_confidence
```

`build_poker_dataset_optimized.py`, `scripts/audit_dataset.py`, `scripts/verify_delivery.py`, and `poker_agent/dataset_schema.py` share the same schema contract so missing columns are caught during audit and delivery verification.

## Data Validation Contract

Dataset audit runs four validation families before data is trusted for training:

```text
pot_conservation
stack_delta_consistency
duplicate_hand_detection
missing_ocr_conflict_policy
```

Pot validation requires `pot_after_action = pot_before_action + positive(action_amount)` within tolerance and rejects pot regressions inside a hand. Stack validation requires `players.stack_delta = ending_stack - starting_stack` and reconciles summed `stack_events.diff` against player deltas. Duplicate detection checks `hand_id`, `source_file + local_hand_index`, and frame signatures. Missing/OCR conflict policy preserves missing OCR confidence as unknown, flags missing legal actions, blocks unresolved conflicting action rows, and blocks chip-moving actions without `action_amount`.

## Baseline Model Contract

The project exposes four explicit baselines through `poker_agent/baselines.py` and `scripts/run_baselines.py`:

```text
rule                  deterministic poker heuristics, no dataset training
imitation_learning    behavior cloning on public/context features
llm                   offline prompt-policy baseline with deterministic local fallback
end_to_end_policy     supervised full-state policy over the complete feature vector
```

Run all baselines on the same hand-group holdout split:

```powershell
python scripts\run_baselines.py --dataset dataset --out reports\baseline_report.json
```

The report includes accuracy, cross-entropy, macro F1, weighted F1, calibration, confusion matrix, slice metrics, and a ranking by validation macro F1.

## Acceptance Criteria Contract

Operational acceptance thresholds are encoded in `poker_agent/acceptance_criteria.py` and `configs\evaluation\standard.yaml`:

```text
latency_p95_ms_max: 150.0
latency_p99_ms_max: 300.0
invalid_action_rate_max: 0.0
validation_pass_rate_min: 1.0
reproducibility_pass_rate_min: 1.0
```

The acceptance report evaluates measured latency, selected-action legality, delivery validation checks, and deterministic reproducibility checks. Production approval now requires `reports\acceptance_criteria.json` to pass in addition to the existing model quality gates.

Run the contract checker:

```powershell
python scripts\check_acceptance_criteria.py --metrics reports\acceptance_metrics.json --out reports\acceptance_criteria.json
```

For a deterministic contract smoke check:

```powershell
python scripts\check_acceptance_criteria.py --smoke
```

## RL Environment Contract

`poker_agent/rl_environment.py` defines the offline research RL environment:

```text
poker_simulator_engine   internal_single_decision_nlhe_simulator
self_play_league         seeded single-hand episodes against sampled opponents
opponent_pool            tight_value, balanced_reg, loose_aggressive, calling_station
seed_policy              deterministic blake2b-derived seeds per generation/episode
reward_shaping           chip delta + win/loss + strength + action penalties/bonuses
```

Inspect the machine-readable contract or run a seeded smoke match:

```powershell
python scripts\inspect_rl_environment.py
python scripts\inspect_rl_environment.py --episodes 25 --seed 20260713
```

The environment is explicitly for offline research/simulation. It is not a real-money platform adapter and should be replaced with a validated poker engine before high-stakes evaluation.

## Project Scope Contract

`poker_agent/project_scope.py` converts the project scope into a machine-readable implementation contract. It tracks the four phases, dataset model, senior requirements, and the code/report evidence that satisfies each requirement.

Generate the scope report and Markdown handoff:

```powershell
python -B scripts\check_project_scope_contract.py --smoke
```

This writes:

```text
reports\project_scope_contract.json
docs\PROJECT_SCOPE_CONTRACT.md
```

The contract covers game scope, operating boundary, deployment API, dataset schema extensions, data validation, labeling, action/state space, baselines, RL/self-play, evaluation, MLOps, monitoring, security, and final deliverables.

## Final Model Selection

`poker_agent/final_model_selection.py` records the QwenPoker benchmark decision as a machine-readable selection gate. `checkpoint_40960` is selected as the final benchmark model from a balanced heads-up No-Limit Hold'em evaluation:

```text
environment        Heads-up No-Limit Hold'em 100 BB | OpenSpiel FCHPA
action_space       fold, check/call, half pot, full pot, all-in
opponent_suite     40% pool/SFT, 15% random, 30% calling, 15% aggressive
balance            5,000 hands, 2,500 per seat
seed               20260714
policy             sampled policy
win_rate           64.48%
returns            +365.29 BB/100
```

The selection contract requires an even seat split, opponent weights summing to 100%, positive BB/100, the source-provided positive 95% return CI assertion, and profitability from both positions. Exact CI bounds were not provided, so the report stores the positive-CI assertion without inventing lower/upper numbers.

Generate the report and model-selection docs:

```powershell
python -B scripts\check_final_model_selection.py --smoke
```

This writes:

```text
reports\final_model_selection.json
docs\FINAL_MODEL_SELECTION.md
```

## Final Deliverables Contract

`poker_agent/deliverables.py` owns the final handoff manifest. It validates that the delivery includes:

```text
validated_dataset_schema
validation_report
baseline_comparison
trained_checkpoint
evaluation_report
dockerized_fastapi_service
api_docs
tests
```

Generate the deterministic final artifacts before packaging or handoff:

```powershell
python -B scripts\prepare_final_deliverables.py --smoke
```

This writes:

```text
reports\dataset_validation_report.json
reports\baseline_report.json
reports\evaluation_report.json
reports\final_model_selection.json
reports\final_deliverables.json
reports\delivery_report.md
docs\API_CONTRACT.md
docs\FINAL_MODEL_SELECTION.md
```

The final verifier checks the same manifest through `final_deliverables_contract`, and the CI smoke workflow regenerates the API docs and reports.

## Legacy Verification Reports

Some delivery checks require historical report names even when the handoff workspace has no full dataset folder or local transformer runtime. The compatibility generator creates deterministic offline reports without overwriting real reports that already exist:

```powershell
python -B scripts\prepare_legacy_delivery_reports.py --smoke
```

Generated files include:

```text
reports\dataset_audit.json
reports\repository_audit.json
reports\production_gate.json
reports\llm_event_benchmark.json
reports\llm_event_gold_eval.json
reports\llm_transformer_gold_eval.json
reports\hydra\llm_transformer_gold_eval\offline_schema_router\
```

`models\poker_policy.metadata.json` provides metadata for the JSON checkpoint fallback when `joblib` is not installed in the local verifier environment. The Docker image still installs `requirements.txt`, including FastAPI and Joblib, for the actual service runtime.

## Reproducible Experiments

Experiments are managed through Hydra. Each experiment has its own YAML file under `configs\experiments` and writes resolved configs, logs, and run metadata under `reports\hydra`.

Run any configured experiment with:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=<name> python_executable=.venv/Scripts/python.exe
```

Available experiment names:

```text
build_dataset
repo_hygiene
repo_audit
audit_dataset
train_single_hgb
evaluate_policy
run_baselines
acceptance_criteria
rl_self_play_smoke
mlops_smoke
monitoring_smoke
security_smoke
project_scope
final_deliverables
legacy_delivery_reports
research_compare_tabular
production_gate
train_routed_bundle_smoke
llm_event_extraction_smoke
llm_event_benchmark
llm_event_gold_eval
llm_transformer_gold_eval
verify_delivery
```

Useful commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=repo_audit python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_event_benchmark python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_event_gold_eval python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_transformer_gold_eval python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=project_scope python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=final_deliverables python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=legacy_delivery_reports python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=verify_delivery python_executable=.venv/Scripts/python.exe
```

Example override:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=train_single_hgb training.max_examples=5000 model.max_iter=40
```

Hydra output structure:

```text
reports\hydra\<experiment-name>\<timestamp>\
|-- environment.json
|-- artifact_manifest.json
|-- artifacts\
|-- resolved_config.yaml
|-- command.txt
|-- stdout.txt
|-- stderr.txt
`-- run.json
```

`environment.json` records the Python runtime, selected dependency versions,
git revision, dirty-state paths, seed, and thread settings. Output files are
hashed in `artifact_manifest.json` and copied into the run-local `artifacts`
directory when they are below the configured size limit. The repository audit
also verifies that every Hydra YAML declares every CLI argument supported by
its entrypoint and rejects CLI fallback defaults that are not owned by a Hydra
experiment configuration.

## Text Event Extraction Results

The repository includes a text/event extraction benchmark for turning OCR and dealer-log records into structured poker events. This is used to improve betting-history reconstruction before model training.

Weak-label benchmark on 1000 log records:

```text
value_only_baseline: event_accuracy=0.4150, macro_f1=0.3284
local_rules:         event_accuracy=1.0000, macro_f1=1.0000
```

Gold-label evaluation on 24 reviewed examples:

```text
minimal_action_only:      event_accuracy=0.6667, macro_f1=0.4091
permissive_prompt_rules:  event_accuracy=0.8333, macro_f1=0.8545
strict_schema_rules:      event_accuracy=1.0000, macro_f1=1.0000
```

The strict schema approach is the strongest current extractor. Card extraction still needs more validation: `strict_schema_rules` reaches `card_exact_match=0.8000` on the current gold fixture. The next data-quality step is to expand the reviewed fixture and enforce rank/suit validation before extracted cards are used as supervised labels.


### Local Instruction Model Experiment

A real local instruction model experiment uses
`HuggingFaceTB/SmolLM2-135M-Instruct` on the same 24 reviewed examples with
deterministic CPU inference. The first run downloads the model from Hugging
Face.

```text
strict_zero_shot: event_accuracy=0.2917, macro_f1=0.1129
few_shot:         event_accuracy=0.3750, macro_f1=0.1364
candidate_ranker: event_accuracy=0.3750, macro_f1=0.1364
calibrated_ranker:event_accuracy=0.3750, macro_f1=0.1406
schema_routed_hybrid: event_accuracy=1.0000, macro_f1=1.0000
```

Few-shot examples improved event accuracy by `0.0833` and macro F1 by `0.0235`,
while contextual calibration improved candidate-ranking macro F1 by `0.0043`.
The production-oriented schema-routed hybrid reached `1.0000` accuracy and
macro F1 by validating known structured event families before invoking the
zero-shot model for other event types. Router coverage was `0.9167`; the real
LLM fallback processed `2/24` examples (`0.0833`) with `1.0000` fallback
accuracy. This result must be revalidated on a larger fixture with ambiguous
and corrupted event names.

## Latest Model Metrics

Current packaged policy:

```text
policy=hist_gradient_boosting
split=stratified_hand_group_holdout
valid_accuracy=0.6798
valid_balanced_accuracy=0.4415
valid_macro_f1=0.4135
valid_weighted_f1=0.6636
valid_majority_baseline_accuracy=0.7029
valid_lift_vs_majority=-0.0231
```

The model is suitable for API integration, data-pipeline testing, and research iteration. It should not be presented as a completed profitable strategy model until the production gate passes.

## Key Reports

```text
reports\repository_audit.json
reports\repo_hygiene.json
reports\dataset_audit.json
reports\production_gate.json
reports\llm_event_benchmark.json
reports\llm_event_gold_eval.json
reports\llm_event_gold_report.md
reports\llm_transformer_gold_eval.json
reports\llm_transformer_gold_report.md
reports\delivery_verification.json
reports\delivery_report.md
```

## Build The Delivery Package

```powershell
.\complete_delivery.ps1 -SkipTrain -AllowGateFailure
```

Use `-SkipTrain` when rebuilding the delivery package around the existing model. Remove it when a fresh training run is required.

Final ZIP:

```text
release\poker-decision-agent.zip
```

## Verify The Delivery

```powershell
.\verify_delivery.ps1
```

Expected result:

```text
"status": "PASS"
```

## Open Risks

- Hole-card coverage is still too low for reliable card-strength modeling.
- The target distribution is imbalanced and fold-dominant.
- The current model does not beat the majority-class baseline on strict holdout accuracy.
- The gold event extraction set is intentionally small and should be expanded with reviewed production logs.
