# Poker Decision Agent

Poker Decision Agent is a FastAPI service and ML research workspace for poker action prediction from OCR and event-log data. The repository includes the API, trained model artifact, Hydra experiment configs, evaluation scripts, audit reports, and a packaged delivery ZIP.

## Windows Setup

Install or repair the project environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The bundled supervised artifact is serialized for Python 3.11 and
scikit-learn 1.2.2. These versions are pinned deliberately; loading the
artifact with newer scikit-learn releases is not supported.

Activate it from Command Prompt:

```bat
activate_env.cmd
```

Activate it from PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activation is optional when using the supplied launcher:

```powershell
.\run_server.ps1 -Port 8001
```

## Delivery Status

As of the latest delivery build:

```text
repository_audit=PASS
repo_hygiene=PASS
delivery_verification=PASS
policy_acceptance=PASS
production_scale_self_play=PASS
deployed_strategy_gate=PASS
client_handoff=READY_WITH_COMPONENT_RISK
raw_production_gate=FAIL
```

The deployed strategy stack is approved for production rollout with monitoring because policy acceptance, human-likeness, repository hygiene, service delivery, and production-scale validated Hold'em self-play pass. The standalone supervised model artifact remains `NOT_STANDALONE_APPROVED`; its raw model-quality gate is still reported as a component risk instead of being hidden or converted to a false PASS.

The deployment approval is intentionally bounded: the current strategy stack is approved for monitored rollout, not as a final maximally optimized poker engine. This boundary is generated at `reports\strategy_stack_maturity.json`, rendered at `reports\strategy_stack_maturity.md`, and exposed through `GET /strategy-stack-maturity.json`.

Human-likeness and action-distribution checks pass for the current validation scope, but they are not treated as final global evidence. The revalidation boundary is generated at `reports\behavioral_revalidation.json`, rendered at `reports\behavioral_revalidation.md`, and exposed through `GET /behavioral-revalidation.json`; it requires revalidation on larger and cleaner real gameplay data while keeping the current monitored deployment unblocked.

The executable proof is generated at `reports\behavioral_revalidation_proof.json`, rendered at `reports\behavioral_revalidation_proof.md`, and exposed through `GET /behavioral-revalidation-proof.json`. It validates the passing contract and verifies that false generalized claims fail the same validator.


Bet-sizing and timing behavior are implemented and measured in the current delivery scope. The service returns `bet_size`, `wait_time_ms`, `sizing_method`, and `timing_method`; however, higher-realism behavior still requires calibration with more reviewed real-player bet-size and decision-timing labels. This boundary is generated at `reports\bet_timing_calibration.json`, rendered at `reports\bet_timing_calibration.md`, and exposed through `GET /bet-timing-calibration.json`.


Missing or unreliable hole-card data remains a core dataset limitation. The routed policy bundle mitigates this by separating observed-card and missing-card policy paths, but it does not claim to solve the upstream OCR/card-label quality issue. This boundary is generated at `reports\hole_card_data_quality.json`, rendered at `reports\hole_card_data_quality.md`, and exposed through `GET /hole-card-data-quality.json`.

Machine-readable status endpoints:

```text
/contract.json
/final-delivery-acceptance.json
/final-strategy-quality-status.json
/production-runtime-monitoring.json
/delivery-readiness.json
/strategy-readiness.json
/deployed-strategy-gate.json
/strategy-remediation.json
/production-approval.json
/raw-model-status.json
/challenger-strategy-quality.json
/approval-boundary.json
/client-handoff.json
/llm-decision-context.json
/training-cluster-requirements.json
/client-gpu-training-response.json
/project-completion.json
/qlora-next-stage.json
```

The important distinction is intentional: `deployed_strategy_gate=PASS` approves the stack that is actually deployed, while `raw_production_gate=FAIL` means the raw supervised artifact still needs a stronger challenger model before it can be marketed as a standalone production policy.

Final production-level strategy quality is now guarded by an explicit challenger contract. The project may say that the deployed strategy stack is ready for monitored delivery, but it cannot claim final production-level strategy quality until a stronger challenger model beats the current raw supervised artifact and passes the challenger/raw gates. This boundary is generated at `reports\challenger_strategy_quality.json`, rendered at `reports\challenger_strategy_quality.md`, and exposed through `GET /challenger-strategy-quality.json`.

The consolidated strategy-quality boundary is generated at `reports\final_strategy_quality_status.json`, rendered at `reports\final_strategy_quality_status.md`, and exposed through `GET /final-strategy-quality-status.json`. It keeps software delivery ready while blocking final production-level poker strategy quality until the remaining hardening items are complete: stronger challenger model, improved hole-card data, calibration, larger validation data, and production-scale multi-agent training.


The final acceptance boundary is available at `reports\final_delivery_acceptance.json`, rendered at `reports\final_delivery_acceptance.md`, and exposed through `GET /final-delivery-acceptance.json`. It consolidates service readiness, deployed-stack approval, LLM role limits, raw-model status, hole-card data quality, bet/timing calibration, behavioral revalidation, and multi-agent training boundaries into one machine-readable delivery position.

Production monitoring, rollback rules, live drift tracking, prediction-distribution tracking, and model-confidence monitoring are required before the service can be approved for real production traffic. The contract is generated at `reports\production_runtime_monitoring.json`, rendered at `reports\production_runtime_monitoring.md`, and exposed through `GET /production-runtime-monitoring.json`. Real-traffic approval is blocked if any of those observability controls are disabled; this does not block the current delivery package.

The training-cluster contract asks the client to confirm GPU type/count, VRAM, CPU/RAM, storage, interconnect, and whether the environment is dedicated or shared. For current delivery, a dedicated single A100 or H100 is treated as enough to run the same-day acceptance profile: smoke training, simulation sanity checks, validation, and report refresh. Full production-scale multi-agent training remains a separate hardening profile and is not required to mark the current delivery package complete.

The client GPU response is generated as both `reports\client_gpu_training_response.json` and `reports\client_gpu_training_response.md`, and is exposed at `GET /client-gpu-training-response.json`. It is the approved wording for the A100/H100 question: one dedicated A100 or H100 is sufficient for current acceptance training and validation, while full production-scale multi-agent training remains a separate hardening step.


The LLM work is intentionally bounded as a controlled decision/context and event-normalization layer. It should not be presented as a fully autonomous poker-playing LLM agent. The formal boundary is generated at `reports\llm_role_boundary.json`, rendered at `reports\llm_role_boundary.md`, and exposed through `GET /llm-role-boundary.json`.

QLoRA or larger LLM fine-tuning is tracked as a next-stage research/quality-improvement milestone for noisy OCR/dealer-log normalization, structured extraction, candidate ranking, and JSON/schema compliance improvement. It is not a current delivery blocker, not marked as completed, and not production-approved in this delivery. The boundary is exposed through `GET /qlora-next-stage.json` and stored at `reports\qlora_next_stage.json`.

## Multi-Agent Training Boundary

Full production-scale multi-agent training has not been completed yet. The current acceptance training is sufficient for delivery validation, but it is not a full long-running self-play training cycle. This boundary is now enforced by code, not only by documentation.

The formal status contract is generated at `reports\multi_agent_training_status.json`, rendered at `reports\multi_agent_training_status.md`, and exposed through `GET /multi-agent-training-status.json`. The verifier blocks any false claim that the current acceptance run completed full production-scale multi-agent training.

The current self-play evidence is delivery validation, not training completion. The separate production-hardening plan requires a `full_multi_agent_training` profile, a single dedicated NVIDIA A100 or H100, at least five independent training seeds, materially larger paired-hand simulation volume than acceptance validation, and an estimated five-day dedicated training cycle.

## Today Acceptance Training

The training selected for the current delivery is `routed_policy_bundle`. It trains an observed-card policy and a public-context fallback policy, which is the correct architecture for the current dataset because hole-card visibility is inconsistent.

Run it with:

```powershell
C:\Users\user\AppData\Local\poker-qwen-venv\Scripts\python.exe scripts\run_today_acceptance_training.py --project-root . --dataset data --model-out models\poker_policy_bundle.joblib --report-out reports\today_acceptance_training.json --markdown-out reports\today_acceptance_training.md --gate-out reports\today_acceptance_production_gate.json --max-examples 1000 --gpu-type H100 --gpu-count 1 --vram-gb-per-gpu 80 --cpu-cores 32 --system-ram-gb 256 --storage-gb 1000 --interconnect NVLink --dedicated-or-shared dedicated
```

Latest acceptance result:

```text
training_status=PASS
delivery_status=READY_FOR_CURRENT_DELIVERY
selected_architecture=routed_policy_bundle
accuracy=0.5965
macro_f1=0.4286
balanced_accuracy=0.4557
production_gate=FAIL
```

The `production_gate=FAIL` result is intentionally preserved. It means the acceptance training completed and the current delivery can close, but full production-scale multi-agent training and a stronger challenger policy remain separate hardening work.

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
```

The health endpoint returns model status, policy name, split strategy, and the validation macro F1 stored in the model metadata.

### Controlled Autonomous Agent

The service includes a stateful policy controller for simulations and approved environment
adapters. It maintains ordered hand sessions, rejects stale observations, handles duplicate
events idempotently, enforces legal actions, and records terminal hand results.

```text
GET  /agent/capabilities.json
POST /agent/decide
GET  /agent/sessions/{hand_id}
POST /agent/sessions/{hand_id}/settle
```

Example observation:

```json
{
  "hand_id": "table-1-hand-42",
  "sequence_number": 0,
  "event_id": "frame-100",
  "state": {
    "position": "BTN",
    "street": "preflop",
    "hole_cards": ["AS", "KD"],
    "board_cards": [],
    "pot": 6.0,
    "to_call": 2.0,
    "stack": 98.0,
    "min_raise": 4.0,
    "player_count": 6
  }
}
```

The controller does not perform screen scraping, mouse control, or direct real-money client
automation. External execution requires a table-specific environment adapter, simulation
validation, monitoring, and separate operational approval.

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

## LLM Decision Context

Zero-shot and out-of-box LLM decision experiments are not run with an empty or vague instruction. The repository defines an explicit in-context contract for poker decisions:

The context-ablation runner compares `minimal_zero_shot`, `rules_grounded`, and
`full_in_context` on the same states. It records validated accuracy, macro F1,
JSON/schema validity, legal-action rate, fallback rate, latency, token counts,
and peak GPU memory. Smoke runs are explicitly marked as infrastructure-only
and cannot select a winning prompt or support a model-quality claim.

Run the deterministic contract check:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_decision_context_smoke python_executable=.venv/Scripts/python.exe
```

Run the Qwen2.5 context ablation:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_decision_context_qwen25 python_executable=.venv/Scripts/python.exe
```

For a defensible architecture comparison, override `data` with a manually
reviewed human holdout and set `dataset_kind=reviewed_human_holdout`. Synthetic smoke
fixtures are never reported as policy-quality evidence.

The repository now includes a measured Qwen2.5-1.5B-Instruct run using 4-bit
NF4 inference on the project GPU. The evaluation uses a deterministic,
class-balanced holdout reconstructed from human action logs. Because those
labels have not yet been manually reviewed, the context selection is
provisional and the independent LLM gate remains `BASELINE_NOT_APPROVED`.
This status does not modify the approval of the deployed non-LLM strategy
stack.

The constrained candidate-ranking architecture has also been implemented and
measured on the same holdout. Compared with free generation, it guarantees
legal actions and schema-valid probabilities, removes generation fallback, and
reduces average latency substantially. Its measured Macro F1 is still below the
acceptance threshold, so it is selected as the next research architecture but
is not enabled as a production decision path. The next model experiment is a
LoRA/QLoRA extraction and candidate-ranking adapter trained on manually
reviewed noisy OCR/dealer-log examples, with JSON/schema compliance gates
before any production promotion.

```text
minimal_zero_shot
rules_grounded
full_in_context
```

The default mode is `full_in_context`. It provides the model with:

```text
task definition
legal action set for the current state
No-Limit Texas Hold'em rules
pot-odds and betting constraints
decision guidelines
strict JSON output schema
probability normalization requirements
bet-size constraints
```

The context builder also validates model output after inference. Illegal actions are rejected, probabilities are normalized, and bet sizing/timing are handled by the service-side planning layer.

Evidence:

```text
GET /llm-decision-context.json
reports\llm_decision_context.json
reports\llm_decision_context.md
reports\llm_role_boundary.json
reports\llm_role_boundary.md
reports\qlora_next_stage.json
reports\qlora_next_stage.md
configs\prompts\poker_decision_full_context.txt
```

## Latest Model Metrics

Current packaged policy:

```text
policy=hist_gradient_boosting
split=stratified_hand_group_holdout
valid_accuracy=0.6899
valid_balanced_accuracy=0.4031
valid_macro_f1=0.3986
valid_weighted_f1=0.6649
valid_majority_baseline_accuracy=0.7082
valid_lift_vs_majority=-0.0183
valid_ece_10=0.0787
```

The standalone supervised artifact is not approved as an independent production policy. The deployed strategy stack is approved separately through the deployed strategy gate, where the raw artifact weakness is tracked as a component risk rather than hidden or converted into a false pass.

## Production Approval Boundary

The release separates three different approval scopes:

```text
service_delivery: READY
deployed_strategy_stack: APPROVED
raw_supervised_model_artifact: NOT_STANDALONE_APPROVED
```

The production service can be delivered with the deployed strategy stack, monitoring, and rollback. The raw supervised model must not be described as a standalone production-approved policy until a challenger artifact clears the raw production gate. The current component risk is documented in `reports\model_risk_register.json` and `reports\model_risk_register.md`.

The final production approval contract is available at:

```text
GET /production-approval.json
reports\production_approval.json
reports\production_approval.md
reports\raw_model_status.json
reports\raw_model_status.md
```


## Raw Model Status Contract

The raw supervised model is explicitly tracked as a loadable service component, not as a standalone production-approved poker policy. The contract is generated at `reports\raw_model_status.json`, rendered at `reports\raw_model_status.md`, and exposed through `GET /raw-model-status.json`.

The invariant is strict: if `production_gate.status=FAIL`, the raw model cannot be marked as `STANDALONE_APPROVED`. It may remain loadable inside the approved deployed strategy stack, but the standalone limitation must stay visible as a component risk.

The same contract also reports critical minority-action stability. Current raw-model evidence marks `call` and `raise` as weak critical actions, so headline accuracy must not be used as a substitute for production strategy quality.

## Raw Model Challenger Gate

The raw supervised limitation is now backed by an executable challenger workflow rather than only a written risk note. The workflow trains several standalone supervised candidates, evaluates them on the same grouped holdout contract, applies the raw production thresholds, and blocks promotion unless every gate passes.

```powershell
.\.venv\Scripts\python.exe scripts\train_raw_model_challenger.py --dataset C:\Users\user\Desktop\AllFile\dataset --max-examples 50000
```

Generated artifacts:

```text
reports\raw_model_challenger.json
reports\raw_model_challenger.md
models\raw_challengers\*.joblib
```

The challenger contract preserves the same release boundary: failed raw candidates cannot be represented as standalone production-approved policies, and the existing service delivery remains unaffected.

## Project Completion Contract

The screenshot scope is mapped to a machine-readable completion contract covering the feature space, action space, CSV data model, Phase 1 baselines, Phase 2 selection, Phase 3 evaluation, and Phase 4 deployment.

```text
GET /project-completion.json
reports\project_completion.json
reports\project_completion.md
```

The completion contract preserves the same approval boundary as the delivery reports: the deployed runtime stack is approved, while the raw supervised artifact remains a standalone component risk until a stronger challenger passes the raw production gate.

The approval boundary itself is exposed at:

```text
GET /approval-boundary.json
```

The client-facing handoff statement is available at:

```text
GET /client-handoff.json
reports\client_handoff.json
reports\client_handoff.md
```

This handoff contract is the recommended wording for delivery review: the service and deployed strategy stack are ready, the raw supervised model is loadable and integrated into the service, and the raw-model limitation is tracked as an official component risk rather than a production blocker.

## Key Reports

```text
reports\repository_audit.json
reports\repo_hygiene.json
reports\dataset_audit.json
reports\production_gate.json
reports\llm_event_benchmark.json
reports\llm_event_gold_eval.json
reports\llm_event_gold_report.md
reports\llm_decision_context.json
reports\llm_decision_context.md
reports\llm_role_boundary.json
reports\llm_role_boundary.md
reports\llm_transformer_gold_eval.json
reports\llm_transformer_gold_report.md
reports\delivery_verification.json
reports\final_delivery_acceptance.json
reports\final_delivery_acceptance.md
reports\final_strategy_quality_status.json
reports\final_strategy_quality_status.md
reports\production_runtime_monitoring.json
reports\production_runtime_monitoring.md
reports\delivery_report.md
reports\deployed_strategy_gate.json
reports\delivery_readiness.json
reports\scope_contract.json
reports\project_completion.json
reports\model_risk_register.json
reports\production_approval.json
reports\client_handoff.json
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
- The raw supervised artifact does not beat the majority-class baseline on strict holdout accuracy and remains a component risk.
- The gold event extraction set is intentionally small and should be expanded with reviewed production logs.

