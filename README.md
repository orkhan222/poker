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

Machine-readable status endpoints:

```text
/contract.json
/delivery-readiness.json
/strategy-readiness.json
/deployed-strategy-gate.json
/strategy-remediation.json
/production-approval.json
/approval-boundary.json
/client-handoff.json
/llm-decision-context.json
/project-completion.json
```

The important distinction is intentional: `deployed_strategy_gate=PASS` approves the stack that is actually deployed, while `raw_production_gate=FAIL` means the raw supervised artifact still needs a stronger challenger model before it can be marketed as a standalone production policy.

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
LoRA/QLoRA action-ranking adapter trained on manually reviewed state-action
pairs.

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
```

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
reports\llm_transformer_gold_eval.json
reports\llm_transformer_gold_report.md
reports\delivery_verification.json
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
