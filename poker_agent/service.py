from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from poker_agent.agents import MLPolicyAgent, RuleBasedAgent
from poker_agent.api_contract import api_contract
from poker_agent.actions_dataset_export_contract import build_actions_dataset_export_contract
from poker_agent.actions_context_quality import build_actions_context_quality
from poker_agent.behavioral_revalidation import build_behavioral_revalidation
from poker_agent.behavioral_revalidation_proof import build_behavioral_revalidation_proof
from poker_agent.bet_timing_calibration import build_bet_timing_calibration
from poker_agent.hole_card_data_quality import build_hole_card_data_quality
from poker_agent.approval_boundary import build_approval_boundary
from poker_agent.autonomous_agent import AgentLifecycleError, AutonomousPokerAgent
from poker_agent.client_handoff import build_client_handoff
from poker_agent.client_gpu_training_response import build_client_gpu_training_response
from poker_agent.challenger_strategy_quality import build_challenger_strategy_quality
from poker_agent.data_leakage_contract import build_data_leakage_contract
from poker_agent.delivery_readiness import summarize_delivery_readiness
from poker_agent.evaluation_metric_contract import build_evaluation_metric_contract
from poker_agent.final_delivery_acceptance import build_final_delivery_acceptance
from poker_agent.final_strategy_quality_status import build_final_strategy_quality_status
from poker_agent.human_likeness_claim_gate import build_human_likeness_claim_gate
from poker_agent.human_likeness_evidence import build_human_likeness_evidence
from poker_agent.llm_decision_context import build_decision_context_report
from poker_agent.llm_policy_experimental import build_experimental_llm_policy_contract
from poker_agent.llm_role_boundary import build_llm_role_boundary
from poker_agent.model_risk_register import build_model_risk_register
from poker_agent.multi_agent_training_status import build_multi_agent_training_status
from poker_agent.normalized_action_contract import build_normalized_action_contract
from poker_agent.open_spiel_claim_contract import build_open_spiel_claim_contract
from poker_agent.open_spiel_llm_arena import build_phase3_open_spiel_arena_report
from poker_agent.phase2_selection_comparison import build_phase2_selection_comparison
from poker_agent.production_approval import build_production_approval
from poker_agent.project_completion import build_project_completion
from poker_agent.production_runtime_monitoring import build_production_runtime_monitoring, runtime_monitoring_state
from poker_agent.qlora_next_stage import build_qlora_next_stage
from poker_agent.raw_model_status import build_raw_model_status
from poker_agent.scenario_sanity import build_scenario_sanity
from poker_agent.schemas import PredictionRequest
from poker_agent.scope_contract import build_scope_contract
from poker_agent.stack_event_context_quality import build_stack_event_context_quality
from poker_agent.strategy_readiness import load_combined_strategy_readiness
from poker_agent.strategy_stack_maturity import build_strategy_stack_maturity
from poker_agent.test_execution_contract import build_test_execution_contract
from poker_agent.training_cluster import DEFAULT_RUN_PROFILE, build_training_cluster_requirements


ActionName = Literal["fold", "call", "check", "bet", "raise", "all_in"]
StreetName = Literal["preflop", "flop", "turn", "river"]


class BettingHistoryBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    player_position: str = Field(default="UTG", description="Acting player position.")
    action: ActionName = Field(default="raise", description="Canonical action before the hero decision.")
    amount: float = Field(default=4.5, ge=0.0, description="Observed action amount, if available.")
    street: StreetName = Field(default="preflop", description="Street where the action happened.")
    wait_time_ms: float | None = Field(default=None, ge=0.0, description="Observed player decision latency.")


class TimingContextBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    opponent_wait_before_turn_ms: float = Field(default=0.0, ge=0.0)
    opponent_wait_after_hero_action_ms: float = Field(default=0.0, ge=0.0)


class PredictRequestBody(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "position": "BTN",
                "street": "preflop",
                "hole_cards": ["Ah", "Kd"],
                "board_cards": [],
                "pot": 2.5,
                "to_call": 1.0,
                "stack": 100.0,
                "min_raise": 2.0,
                "player_count": 6,
            }
        },
    )

    position: str = Field(
        default="BTN",
        validation_alias=AliasChoices("position", "player_position"),
        description="Hero table position.",
    )
    street: StreetName = Field(default="preflop", description="Current betting street.")
    hole_cards: list[str] = Field(default_factory=list, description="Hero hole cards, for example Ah Kd.")
    board_cards: list[str] = Field(default_factory=list, description="Community cards visible before the decision.")
    pot: float = Field(default=0.0, ge=0.0, description="Current pot size before the hero action.")
    to_call: float = Field(default=0.0, ge=0.0, description="Amount required to call.")
    stack: float = Field(default=0.0, ge=0.0, description="Hero stack before the decision.")
    min_raise: float = Field(default=0.0, ge=0.0, description="Minimum legal raise size.")
    player_count: int = Field(default=6, ge=2, le=10, description="Number of players dealt into the hand.")
    betting_history: list[BettingHistoryBody] = Field(
        default_factory=list,
        validation_alias=AliasChoices("betting_history", "action_history"),
        description="Actions observable before the target hero action.",
    )
    opponent_wait_before_turn_ms: float = Field(default=0.0, ge=0.0)
    opponent_wait_after_hero_action_ms: float = Field(default=0.0, ge=0.0)
    timing_context: TimingContextBody | None = Field(default=None)

    def to_payload(self) -> dict[str, Any]:
        payload = self.model_dump()
        if self.timing_context is not None:
            payload["timing_context"] = self.timing_context.model_dump()
        return payload


class ActionProbabilitiesBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fold: float = Field(default=0.0, ge=0.0, le=1.0)
    call: float = Field(default=0.0, ge=0.0, le=1.0)
    check: float = Field(default=0.0, ge=0.0, le=1.0)
    bet: float = Field(default=0.0, ge=0.0, le=1.0)
    raise_: float = Field(default=0.0, ge=0.0, le=1.0, alias="raise")
    all_in: float = Field(default=0.0, ge=0.0, le=1.0)


class PredictResponseBody(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "action": "raise",
                "probabilities": {
                    "fold": 0.02,
                    "call": 0.24,
                    "check": 0.04,
                    "bet": 0.18,
                    "raise": 0.52,
                    "all_in": 0.0,
                },
                "confidence": 0.52,
                "bet_size": 4.5,
                "wait_time_ms": 1264,
                "sizing_method": "pressure_raise",
                "timing_method": "table_tempo_calibrated",
                "model_status": "routed_policy_bundle",
            }
        },
    )

    action: ActionName
    probabilities: ActionProbabilitiesBody
    confidence: float = Field(ge=0.0, le=1.0)
    bet_size: float = Field(default=0.0, ge=0.0)
    wait_time_ms: int = Field(default=250, ge=0)
    sizing_method: str
    timing_method: str
    model_status: str
    warnings: list[str] = Field(default_factory=list)


app = FastAPI(
    title="Poker Decision Agent API",
    description="API for real-time poker action prediction using the bundled trained policy model.",
    version="1.0.0",
    docs_url=None,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "full",
    },
    openapi_tags=[
        {
            "name": "Prediction",
            "description": "Poker action prediction endpoints.",
        },
        {
            "name": "System",
            "description": "Service status and model health endpoints.",
        },
    ],
)
PUBLIC_OPENAPI_PATHS = {"/predict"}
_agent = None
_autonomous_agent = None
_agent_load_error: str | None = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy.joblib"
OPTIONAL_BUNDLE_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy_bundle.joblib"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy.json"
PRODUCTION_GATE_REPORT_PATH = PROJECT_ROOT / "reports" / "production_gate.json"
DEPLOYED_STRATEGY_GATE_REPORT_PATH = PROJECT_ROOT / "reports" / "deployed_strategy_gate.json"
STRATEGY_REMEDIATION_REPORT_PATH = PROJECT_ROOT / "reports" / "strategy_remediation.json"
LLM_DECISION_CONTEXT_SMOKE_REPORT_PATH = PROJECT_ROOT / "reports" / "llm_decision_context_smoke.json"
LLM_DECISION_QWEN_REPORT_PATH = PROJECT_ROOT / "reports" / "llm_decision_context_qwen25.json"
LLM_DECISION_GATE_REPORT_PATH = PROJECT_ROOT / "reports" / "llm_decision_gate.json"
LLM_CANDIDATE_RANKER_REPORT_PATH = PROJECT_ROOT / "reports" / "llm_decision_candidate_ranker_qwen25.json"
LLM_ARCHITECTURE_COMPARISON_PATH = PROJECT_ROOT / "reports" / "llm_architecture_comparison.json"
LLM_ROLE_BOUNDARY_PATH = PROJECT_ROOT / "reports" / "llm_role_boundary.json"
LLM_POLICY_EXPERIMENTAL_PATH = PROJECT_ROOT / "reports" / "llm_policy_experimental.json"
QLORA_NEXT_STAGE_PATH = PROJECT_ROOT / "reports" / "qlora_next_stage.json"
TODAY_ACCEPTANCE_TRAINING_REPORT_PATH = PROJECT_ROOT / "reports" / "today_acceptance_training.json"
CLIENT_GPU_TRAINING_RESPONSE_PATH = PROJECT_ROOT / "reports" / "client_gpu_training_response.json"
MULTI_AGENT_TRAINING_STATUS_PATH = PROJECT_ROOT / "reports" / "multi_agent_training_status.json"
PHASE3_OPEN_SPIEL_ARENA_PATH = PROJECT_ROOT / "reports" / "phase3_open_spiel_arena.json"
OPEN_SPIEL_CLAIM_CONTRACT_PATH = PROJECT_ROOT / "reports" / "open_spiel_claim_contract.json"
RAW_MODEL_STATUS_PATH = PROJECT_ROOT / "reports" / "raw_model_status.json"
RAW_MODEL_CHALLENGER_PATH = PROJECT_ROOT / "reports" / "raw_model_challenger.json"
CHALLENGER_STRATEGY_QUALITY_PATH = PROJECT_ROOT / "reports" / "challenger_strategy_quality.json"
STRATEGY_STACK_MATURITY_PATH = PROJECT_ROOT / "reports" / "strategy_stack_maturity.json"
BEHAVIORAL_REVALIDATION_PATH = PROJECT_ROOT / "reports" / "behavioral_revalidation.json"
BEHAVIORAL_REVALIDATION_PROOF_PATH = PROJECT_ROOT / "reports" / "behavioral_revalidation_proof.json"
HOLE_CARD_DATA_QUALITY_PATH = PROJECT_ROOT / "reports" / "hole_card_data_quality.json"
DATA_LEAKAGE_CONTRACT_PATH = PROJECT_ROOT / "reports" / "data_leakage_contract.json"
NORMALIZED_ACTION_CONTRACT_PATH = PROJECT_ROOT / "reports" / "normalized_action_contract.json"
ACTION_CONTEXT_QUALITY_PATH = PROJECT_ROOT / "reports" / "actions_context_quality.json"
ACTIONS_DATASET_EXPORT_CONTRACT_PATH = PROJECT_ROOT / "reports" / "actions_dataset_export_contract.json"
STACK_EVENT_CONTEXT_QUALITY_PATH = PROJECT_ROOT / "reports" / "stack_event_context_quality.json"
BET_TIMING_CALIBRATION_PATH = PROJECT_ROOT / "reports" / "bet_timing_calibration.json"
FINAL_DELIVERY_ACCEPTANCE_PATH = PROJECT_ROOT / "reports" / "final_delivery_acceptance.json"
FINAL_STRATEGY_QUALITY_STATUS_PATH = PROJECT_ROOT / "reports" / "final_strategy_quality_status.json"
PRODUCTION_RUNTIME_MONITORING_PATH = PROJECT_ROOT / "reports" / "production_runtime_monitoring.json"
EVALUATION_METRIC_CONTRACT_PATH = PROJECT_ROOT / "reports" / "evaluation_metric_contract.json"
TEST_EXECUTION_CONTRACT_PATH = PROJECT_ROOT / "reports" / "test_execution_contract.json"
HUMAN_LIKENESS_EVIDENCE_PATH = PROJECT_ROOT / "reports" / "human_likeness_evidence.json"
HUMAN_LIKENESS_CLAIM_GATE_PATH = PROJECT_ROOT / "reports" / "human_likeness_claim_gate.json"
PHASE2_SELECTION_COMPARISON_PATH = PROJECT_ROOT / "reports" / "phase2_selection_comparison.json"
SCENARIO_SANITY_PATH = PROJECT_ROOT / "reports" / "scenario_sanity.json"


def public_openapi_schema() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    public_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) in PUBLIC_OPENAPI_PATHS
    ]
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=public_routes,
        tags=[
            {
                "name": "Prediction",
                "description": "Poker action prediction endpoints.",
            },
        ],
    )

    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.get("responses", {}).pop("422", None)

    components = schema.get("components") or {}
    schemas = components.get("schemas") or {}
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    if not schemas:
        components.pop("schemas", None)
    if not components:
        schema.pop("components", None)

    app.openapi_schema = schema
    return schema


app.openapi = public_openapi_schema


CLIENT_SWAGGER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker Decision Agent API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  <style>
    body.client-facing-swagger {
      margin: 0;
      background: #ffffff;
    }
    body.client-facing-swagger .swagger-ui {
      padding-top: 0;
    }
    .swagger-ui .wrapper {
      width: min(1480px, calc(100vw - 64px));
      max-width: none;
      margin: 0 auto;
      padding: 10px 0;
    }
    .swagger-ui .info {
      margin: 8px 0 16px;
    }
    .swagger-ui .info .title {
      color: #26354f;
      font-size: 34px;
      line-height: 1.15;
    }
    .swagger-ui .info p {
      margin: 8px 0 0;
    }
    .swagger-ui .information-container.wrapper {
      padding-bottom: 2px;
    }
    .swagger-ui .scheme-container {
      display: none;
    }
    .swagger-ui .opblock-tag-section {
      margin-top: 2px;
    }
    .swagger-ui .opblock-tag {
      border-bottom-color: #d8dee8;
      padding: 0 0 8px;
      margin: 0 0 8px;
    }
    .swagger-ui .opblock-tag small {
      padding-left: 8px;
    }
    .swagger-ui .opblock {
      border-radius: 4px;
      box-shadow: none;
    }
    .swagger-ui .opblock .opblock-summary {
      padding: 9px 16px;
    }
    .swagger-ui .opblock-description-wrapper {
      padding: 14px 22px;
    }
    .swagger-ui .parameters-container {
      display: none !important;
    }
    .swagger-ui table.parameters,
    .swagger-ui .parameters-col_description,
    .swagger-ui .parameters-col_name {
      display: none !important;
    }
    .swagger-ui .try-out,
    .swagger-ui .try-out__btn,
    .swagger-ui .opblock-section-header:has(.try-out) {
      display: none !important;
    }
    .swagger-ui .opblock-section-header {
      box-shadow: none;
      border-top: 1px solid #d8dee8;
      border-bottom: 1px solid #d8dee8;
    }
    .swagger-ui .responses-wrapper,
    .swagger-ui .request-body {
      padding: 0 22px 16px;
    }
    .swagger-ui .highlight-code {
      max-height: none;
    }
    .swagger-ui .highlight-code > .microlight,
    .swagger-ui pre {
      max-height: 340px;
      overflow: auto;
      border-radius: 4px;
      font-size: 13px;
      line-height: 1.45;
    }
    .swagger-ui .try-out {
      padding-right: 0;
    }
    .client-docs-helper {
      width: min(1480px, calc(100vw - 64px));
      margin: 8px auto 6px;
      border: 1px solid #b8d7c7;
      border-radius: 4px;
      background: #f3fbf7;
      color: #1f2f25;
      padding: 10px 14px;
      font-family: sans-serif;
    }
    .client-docs-helper strong {
      display: block;
      margin-bottom: 4px;
      font-size: 15px;
    }
    .client-docs-helper span {
      display: block;
      font-size: 13px;
      line-height: 1.45;
    }
    @media (max-width: 760px) {
      .swagger-ui .wrapper {
        width: calc(100vw - 24px);
        padding: 8px 0;
      }
      .swagger-ui .info .title {
        font-size: 28px;
      }
      .client-docs-helper {
        width: calc(100vw - 24px);
      }
    }
  </style>
</head>
<body class="client-facing-swagger compact-public-docs">
  <div class="client-docs-helper">
    <strong>Public API surface</strong>
    <span>Use <code>POST /predict</code>. Input is a JSON request body; there are no query parameters. The operation below opens in read-only example mode so request and response examples are visible without entering edit mode.</span>
  </div>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    function expandPredictOperation() {
      var blocks = document.querySelectorAll(".swagger-ui .opblock");
      blocks.forEach(function (block) {
        var summary = block.querySelector(".opblock-summary");
        if (!summary || !summary.textContent || summary.textContent.indexOf("/predict") === -1) {
          return;
        }
        if (!block.classList.contains("is-open")) {
          var control = summary.querySelector(".opblock-summary-control") || summary.querySelector("button") || summary;
          control.click();
        }
      });
    }

    function hideEmptyParameterSections() {
      document.querySelectorAll(".swagger-ui .parameters-container").forEach(function (node) {
        node.style.display = "none";
      });
      document.querySelectorAll(".swagger-ui table.parameters").forEach(function (node) {
        node.style.display = "none";
      });
      document.querySelectorAll(".swagger-ui .opblock-section-header").forEach(function (header) {
        var text = (header.textContent || "").replace(/\\s+/g, " ").trim().toLowerCase();
        if (text.indexOf("parameters") === 0) {
          header.style.display = "none";
        }
      });
    }

    function keepPredictExpanded() {
      expandPredictOperation();
      hideEmptyParameterSections();
      var target = document.getElementById("swagger-ui");
      if (!target || !window.MutationObserver) {
        return;
      }
      var observer = new MutationObserver(function () {
        window.requestAnimationFrame(function () {
          expandPredictOperation();
          hideEmptyParameterSections();
        });
      });
      observer.observe(target, { childList: true, subtree: true, attributes: true });
      [100, 300, 700, 1200].forEach(function (delayMs) {
        window.setTimeout(function () {
          expandPredictOperation();
          hideEmptyParameterSections();
        }, delayMs);
      });
    }

    window.onload = function () {
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        docExpansion: "full",
        defaultModelsExpandDepth: -1,
        defaultModelExpandDepth: 1,
        displayRequestDuration: false,
        tryItOutEnabled: false,
        supportedSubmitMethods: [],
        presets: [
          SwaggerUIBundle.presets.apis
        ],
        onComplete: keepPredictExpanded
      });
      keepPredictExpanded();
    };
  </script>
</body>
</html>
"""


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def public_docs_page() -> str:
    return CLIENT_SWAGGER_HTML


APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker Decision Agent</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #0f1412;
      color: #eef4ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 20% 10%, rgba(59, 130, 246, 0.18), transparent 30rem),
        radial-gradient(circle at 86% 12%, rgba(34, 197, 94, 0.16), transparent 28rem),
        linear-gradient(135deg, #0f1412 0%, #17201c 50%, #101516 100%);
    }
    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 36px 0;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 56px);
      font-weight: 800;
      letter-spacing: 0;
    }
    .subtitle {
      max-width: 720px;
      margin: 12px 0 0;
      color: #d6e3db;
      font-size: 17px;
      line-height: 1.45;
    }
    .status {
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.06);
      color: #9fe6b3;
      white-space: nowrap;
      font-weight: 750;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 440px);
      gap: 18px;
    }
    section {
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      background: rgba(16, 20, 18, 0.82);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
    }
    .form-panel { padding: 22px; }
    .panel-title {
      margin: 0 0 16px;
      color: #f5fff8;
      font-size: 18px;
      font-weight: 850;
    }
    .result-panel {
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    label {
      display: grid;
      gap: 7px;
      color: #b6c7bd;
      font-size: 13px;
      font-weight: 650;
    }
    input, select {
      width: 100%;
      height: 42px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      padding: 0 12px;
      background: #0f1714;
      color: #f4fff7;
      font: inherit;
      outline: none;
    }
    input:focus, select:focus {
      border-color: #6ee08c;
      box-shadow: 0 0 0 3px rgba(110, 224, 140, 0.16);
    }
    button {
      width: 100%;
      height: 46px;
      margin-top: 18px;
      border: 0;
      border-radius: 6px;
      background: #55c46f;
      color: #07110a;
      font-weight: 800;
      font-size: 15px;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    button:hover { background: #62d67c; }
    .action {
      min-height: 110px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #e9fff0;
      color: #102016;
      font-size: clamp(38px, 6vw, 72px);
      font-weight: 900;
      text-transform: uppercase;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      min-height: 72px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.04);
    }
    .metric span {
      display: block;
      color: #9fb0a6;
      font-size: 12px;
      font-weight: 750;
    }
    .metric strong {
      display: block;
      margin-top: 7px;
      color: #f4fff7;
      font-size: 18px;
    }
    .bars {
      display: grid;
      gap: 10px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 76px 1fr 58px;
      align-items: center;
      gap: 10px;
      color: #d5e3d9;
      font-size: 14px;
    }
    .track {
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.09);
    }
    .fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, #55c46f, #f0c14b);
      transition: width 180ms ease;
    }
    pre {
      overflow: auto;
      min-height: 120px;
      margin: 0;
      border-radius: 8px;
      padding: 14px;
      background: #0a0f0d;
      color: #cfe7d5;
      font-size: 12px;
      line-height: 1.5;
    }
    @media (max-width: 820px) {
      header, .layout { grid-template-columns: 1fr; }
      header { align-items: start; }
      .grid { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Poker Decision Agent</h1>
        <p class="subtitle">Real-time poker action prediction from game state inputs, backed by a trained policy model and exposed through a FastAPI service.</p>
      </div>
      <div class="status">Live API</div>
    </header>

    <div class="layout">
      <section class="form-panel">
        <h2 class="panel-title">Game State</h2>
        <form id="predict-form">
          <div class="grid">
            <label>Position
              <select name="position">
                <option>BTN</option>
                <option>SB</option>
                <option>BB</option>
                <option>UTG</option>
                <option>MP</option>
                <option>CO</option>
                <option>Player1_Bottom</option>
              </select>
            </label>
            <label>Street
              <select name="street">
                <option>preflop</option>
                <option>flop</option>
                <option>turn</option>
                <option>river</option>
              </select>
            </label>
            <label>Hole cards
              <input name="hole_cards" value="Ah Kd" aria-label="Hole cards, for example Ah Kd">
            </label>
            <label>Board cards
              <input name="board_cards" value="" aria-label="Board cards, for example 2c 7d Qs">
            </label>
            <label>Pot
              <input name="pot" type="number" step="0.1" value="2.5">
            </label>
            <label>To call
              <input name="to_call" type="number" step="0.1" value="1.0">
            </label>
            <label>Stack
              <input name="stack" type="number" step="0.1" value="100">
            </label>
            <label>Min raise
              <input name="min_raise" type="number" step="0.1" value="2.0">
            </label>
            <label>Players
              <input name="player_count" type="number" step="1" value="6">
            </label>
          </div>
          <button id="submit-button" type="submit">Predict action</button>
        </form>
      </section>

      <section class="result-panel">
        <div id="action" class="action">Ready</div>
        <div class="summary">
          <div class="metric"><span>Confidence</span><strong id="confidence">-</strong></div>
          <div class="metric"><span>Street</span><strong id="street-summary">-</strong></div>
          <div class="metric"><span>Position</span><strong id="position-summary">-</strong></div>
        </div>
        <div id="bars" class="bars"></div>
        <pre id="json-output">{}</pre>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById("predict-form");
    const button = document.getElementById("submit-button");
    const actionBox = document.getElementById("action");
    const confidence = document.getElementById("confidence");
    const streetSummary = document.getElementById("street-summary");
    const positionSummary = document.getElementById("position-summary");
    const bars = document.getElementById("bars");
    const output = document.getElementById("json-output");

    function cards(value) {
      return value.split(/[ ,]+/).map((card) => card.trim()).filter(Boolean);
    }

    function numberValue(data, name) {
      return Number(data.get(name) || 0);
    }

    function render(result, payload) {
      actionBox.textContent = result.action || "N/A";
      const probabilities = Object.values(result.probabilities || {});
      const topProbability = probabilities.length ? Math.max(...probabilities) : 0;
      confidence.textContent = `${(topProbability * 100).toFixed(1)}%`;
      streetSummary.textContent = payload.street || "-";
      positionSummary.textContent = payload.position || "-";
      output.textContent = JSON.stringify(result, null, 2);
      bars.innerHTML = "";
      Object.entries(result.probabilities || {})
        .sort((a, b) => b[1] - a[1])
        .forEach(([name, value]) => {
          const row = document.createElement("div");
          row.className = "bar-row";
          row.innerHTML = `
            <strong>${name}</strong>
            <div class="track"><div class="fill" style="width:${Math.round(value * 100)}%"></div></div>
            <span>${(value * 100).toFixed(1)}%</span>
          `;
          bars.appendChild(row);
        });
    }

    async function predict(event) {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Predicting...";
      const data = new FormData(form);
      const payload = {
        position: data.get("position"),
        street: data.get("street"),
        hole_cards: cards(data.get("hole_cards") || ""),
        board_cards: cards(data.get("board_cards") || ""),
        pot: numberValue(data, "pot"),
        to_call: numberValue(data, "to_call"),
        stack: numberValue(data, "stack"),
        min_raise: numberValue(data, "min_raise"),
        player_count: Number(data.get("player_count") || 6)
      };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        render(await response.json(), payload);
      } catch (error) {
        actionBox.textContent = "Error";
        output.textContent = String(error);
      } finally {
        button.disabled = false;
        button.textContent = "Predict action";
      }
    }

    form.addEventListener("submit", predict);
    form.dispatchEvent(new Event("submit"));
  </script>
</body>
</html>
"""


def health_payload() -> dict[str, str]:
    model_path = resolve_model_path()
    agent = get_agent()
    model_loaded = isinstance(agent, MLPolicyAgent)
    payload = {
        "status": "ok",
        "model": str(model_path),
        "model_status": (
            "loaded"
            if model_loaded
            else "fallback_rule_based_model_load_failed"
            if model_path.exists() and _agent_load_error
            else "fallback_rule_based"
        ),
    }
    if _agent_load_error:
        payload["model_load_error"] = _agent_load_error[:300]
    try:
        model = getattr(agent, "model", None)
        metadata = getattr(model, "metadata", {}) or {}
        if metadata:
            payload["policy"] = str(metadata.get("policy", getattr(model, "model_kind", "unknown")))
            payload["split"] = str((metadata.get("split") or {}).get("split_type", "unknown"))
            valid_metrics = metadata.get("valid_metrics") or {}
            if "macro_f1" in valid_metrics:
                payload["valid_macro_f1"] = f"{float(valid_metrics['macro_f1']):.4f}"
    except Exception:
        payload["metadata_status"] = "unavailable"
    return payload


def health_html(payload: dict[str, str]) -> str:
    status = payload["status"].upper()
    model_status = payload["model_status"].replace("_", " ")
    model_path = payload["model"]
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker Decision Agent Status</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #0f1412;
      color: #eef4ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        radial-gradient(circle at 20% 12%, rgba(59, 130, 246, 0.18), transparent 30rem),
        radial-gradient(circle at 85% 18%, rgba(34, 197, 94, 0.16), transparent 28rem),
        linear-gradient(135deg, #0f1412 0%, #17201c 50%, #101516 100%);
    }}
    main {{
      width: min(760px, 100%);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 28px;
      background: rgba(16, 20, 18, 0.88);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 5vw, 44px);
      letter-spacing: 0;
    }}
    .badge {{
      border-radius: 999px;
      padding: 8px 12px;
      background: #e9fff0;
      color: #102016;
      font-weight: 850;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .item {{
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.04);
    }}
    .item span {{
      display: block;
      color: #9fb0a6;
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .item strong {{
      display: block;
      margin-top: 8px;
      overflow-wrap: anywhere;
      color: #f4fff7;
      font-size: 17px;
    }}
    .model {{
      grid-column: 1 / -1;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 22px;
    }}
    a {{
      border-radius: 6px;
      padding: 10px 13px;
      background: #55c46f;
      color: #07110a;
      text-decoration: none;
      font-weight: 850;
    }}
    a.secondary {{
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(255, 255, 255, 0.06);
      color: #eef4ef;
    }}
    @media (max-width: 640px) {{
      header, .grid {{ grid-template-columns: 1fr; }}
      header {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Service Status</h1>
      </div>
      <div class="badge">{status}</div>
    </header>
    <section class="grid">
      <div class="item">
        <span>API</span>
        <strong>{payload["status"]}</strong>
      </div>
      <div class="item">
        <span>Model status</span>
        <strong>{model_status}</strong>
      </div>
      <div class="item model">
        <span>Model path</span>
        <strong>{model_path}</strong>
      </div>
    </section>
    <nav>
      <a href="/predict">Open application</a>
      <a class="secondary" href="/docs">API docs</a>
      <a class="secondary" href="/health.json">Raw JSON</a>
      <a class="secondary" href="/scope-contract.json">Scope contract</a>
    </nav>
  </main>
</body>
</html>
"""


def get_agent():
    global _agent, _agent_load_error
    if _agent is not None:
        return _agent
    model_path = resolve_model_path()
    if model_path.exists():
        try:
            _agent = MLPolicyAgent.from_path(model_path)
            _agent_load_error = None
            return _agent
        except Exception as exc:
            _agent_load_error = f"{type(exc).__name__}: {exc}"
    _agent = RuleBasedAgent()
    return _agent


def get_autonomous_agent() -> AutonomousPokerAgent:
    global _autonomous_agent
    if _autonomous_agent is None:
        _autonomous_agent = AutonomousPokerAgent(get_agent())
    return _autonomous_agent


def resolve_model_path() -> Path:
    configured = os.getenv("POKER_POLICY_PATH")
    if configured:
        return Path(configured)
    if OPTIONAL_BUNDLE_MODEL_PATH.exists():
        return OPTIONAL_BUNDLE_MODEL_PATH
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return FALLBACK_MODEL_PATH


@app.get("/health", include_in_schema=False)
def health(request: Request) -> Any:
    payload = health_payload()
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return HTMLResponse(health_html(payload))
    return payload


@app.get(
    "/health.json",
    tags=["System"],
    summary="Service status",
    description="Returns API status and confirms whether the bundled policy model is loaded.",
)
def health_json() -> dict[str, str]:
    return health_payload()


@app.get("/final-delivery-acceptance.json", tags=["System"], summary="Final delivery acceptance boundary")
def final_delivery_acceptance_json() -> dict[str, Any]:
    if FINAL_DELIVERY_ACCEPTANCE_PATH.exists():
        return json.loads(FINAL_DELIVERY_ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    return build_final_delivery_acceptance(PROJECT_ROOT)


@app.get(
    "/final-strategy-quality-status.json",
    tags=["System"],
    summary="Final production-level strategy quality boundary",
)
def final_strategy_quality_status_json() -> dict[str, Any]:
    if FINAL_STRATEGY_QUALITY_STATUS_PATH.exists():
        return json.loads(FINAL_STRATEGY_QUALITY_STATUS_PATH.read_text(encoding="utf-8"))
    return build_final_strategy_quality_status(PROJECT_ROOT)


@app.get("/production-runtime-monitoring.json", tags=["System"], summary="Production monitoring, rollback, and drift tracking")
def production_runtime_monitoring_json() -> dict[str, Any]:
    runtime_snapshot = runtime_monitoring_state.snapshot()
    if PRODUCTION_RUNTIME_MONITORING_PATH.exists():
        payload = json.loads(PRODUCTION_RUNTIME_MONITORING_PATH.read_text(encoding="utf-8"))
        payload["runtime_snapshot"] = runtime_snapshot
        return payload
    return build_production_runtime_monitoring(PROJECT_ROOT, runtime_snapshot=runtime_snapshot)


@app.get("/contract.json", tags=["System"], summary="API response contract")
def contract_json() -> dict[str, Any]:
    return api_contract()


@app.get("/scope-contract.json", tags=["System"], summary="DOCX/PDF scope contract")
def scope_contract_json() -> dict[str, Any]:
    return build_scope_contract(PROJECT_ROOT)


@app.get("/delivery-readiness.json", tags=["System"], summary="Delivery readiness")
def delivery_readiness_json() -> dict[str, Any]:
    return summarize_delivery_readiness(PROJECT_ROOT)


@app.get("/model-risk-register.json", tags=["System"], summary="Model risk register")
def model_risk_register_json() -> dict[str, Any]:
    return build_model_risk_register(PROJECT_ROOT)



@app.get("/raw-model-status.json", tags=["System"], summary="Raw supervised model status")
def raw_model_status_json() -> dict[str, Any]:
    if RAW_MODEL_STATUS_PATH.exists():
        return json.loads(RAW_MODEL_STATUS_PATH.read_text(encoding="utf-8"))
    return build_raw_model_status(PROJECT_ROOT)


@app.get("/raw-model-challenger.json", tags=["System"], summary="Raw supervised model challenger gate")
def raw_model_challenger_json() -> dict[str, Any]:
    if RAW_MODEL_CHALLENGER_PATH.exists():
        return json.loads(RAW_MODEL_CHALLENGER_PATH.read_text(encoding="utf-8"))
    return {
        "status": "MISSING",
        "standalone_status": "NOT_STANDALONE_APPROVED",
        "approved_as_standalone_policy": False,
        "report": str(RAW_MODEL_CHALLENGER_PATH),
    }


@app.get(
    "/challenger-strategy-quality.json",
    tags=["System"],
    summary="Challenger requirement before final strategy-quality claims",
)
def challenger_strategy_quality_json() -> dict[str, Any]:
    if CHALLENGER_STRATEGY_QUALITY_PATH.exists():
        return json.loads(CHALLENGER_STRATEGY_QUALITY_PATH.read_text(encoding="utf-8"))
    return build_challenger_strategy_quality(PROJECT_ROOT)


@app.get("/production-approval.json", tags=["System"], summary="Production approval contract")
def production_approval_json() -> dict[str, Any]:
    return build_production_approval(PROJECT_ROOT)


@app.get("/approval-boundary.json", tags=["System"], summary="Approval boundary")
def approval_boundary_json() -> dict[str, Any]:
    return build_approval_boundary(PROJECT_ROOT)


@app.get("/client-handoff.json", tags=["System"], summary="Client handoff statement")
def client_handoff_json() -> dict[str, Any]:
    return build_client_handoff(PROJECT_ROOT)


@app.get("/training-cluster-requirements.json", tags=["System"], summary="Training cluster requirements")
def training_cluster_requirements_json(
    run_profile: str = Query(
        default=DEFAULT_RUN_PROFILE,
        description="Training run profile: immediate_delivery or full_multi_agent_training.",
    ),
    gpu_type: str | None = Query(default=None, description="GPU model, for example A100 or H100."),
    gpu_count: int | None = Query(default=None, ge=1, description="Number of GPUs available for training."),
    vram_gb_per_gpu: float | None = Query(default=None, gt=0, description="VRAM per GPU in GB."),
    cpu_cores: int | None = Query(default=None, ge=1, description="Available CPU cores."),
    system_ram_gb: float | None = Query(default=None, gt=0, description="Available system RAM in GB."),
    storage_gb: float | None = Query(default=None, gt=0, description="Available local or attached storage in GB."),
    interconnect: str | None = Query(default=None, description="GPU interconnect, for example PCIe or NVLink."),
    dedicated_or_shared: str | None = Query(default=None, description="Whether the cluster is dedicated or shared."),
) -> dict[str, Any]:
    cluster = {
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "vram_gb_per_gpu": vram_gb_per_gpu,
        "cpu_cores": cpu_cores,
        "system_ram_gb": system_ram_gb,
        "storage_gb": storage_gb,
        "interconnect": interconnect,
        "dedicated_or_shared": dedicated_or_shared,
    }
    if not any(value is not None for value in cluster.values()):
        cluster = None
    return build_training_cluster_requirements(PROJECT_ROOT, cluster=cluster, run_profile=run_profile)


@app.get("/today-acceptance-training.json", tags=["System"], summary="Today acceptance training report")
def today_acceptance_training_json() -> dict[str, Any]:
    if not TODAY_ACCEPTANCE_TRAINING_REPORT_PATH.exists():
        return {
            "status": "MISSING",
            "report": str(TODAY_ACCEPTANCE_TRAINING_REPORT_PATH),
        }
    return json.loads(TODAY_ACCEPTANCE_TRAINING_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/client-gpu-training-response.json", tags=["System"], summary="Client GPU training response")
def client_gpu_training_response_json() -> dict[str, Any]:
    if CLIENT_GPU_TRAINING_RESPONSE_PATH.exists():
        return json.loads(CLIENT_GPU_TRAINING_RESPONSE_PATH.read_text(encoding="utf-8"))
    return build_client_gpu_training_response(PROJECT_ROOT)


@app.get(
    "/multi-agent-training-status.json",
    tags=["System"],
    summary="Multi-agent training completion boundary",
)
def multi_agent_training_status_json() -> dict[str, Any]:
    if MULTI_AGENT_TRAINING_STATUS_PATH.exists():
        return json.loads(MULTI_AGENT_TRAINING_STATUS_PATH.read_text(encoding="utf-8"))
    return build_multi_agent_training_status(PROJECT_ROOT)


@app.get(
    "/phase3-open-spiel-arena.json",
    tags=["System"],
    summary="Phase 3 OpenSpiel agent-only arena",
)
def phase3_open_spiel_arena_json() -> dict[str, Any]:
    if PHASE3_OPEN_SPIEL_ARENA_PATH.exists():
        return json.loads(PHASE3_OPEN_SPIEL_ARENA_PATH.read_text(encoding="utf-8"))
    return build_phase3_open_spiel_arena_report(PROJECT_ROOT)


@app.get(
    "/open-spiel-claim-contract.json",
    tags=["System"],
    summary="OpenSpiel/RL self-play claim boundary",
)
def open_spiel_claim_contract_json() -> dict[str, Any]:
    if OPEN_SPIEL_CLAIM_CONTRACT_PATH.exists():
        return json.loads(OPEN_SPIEL_CLAIM_CONTRACT_PATH.read_text(encoding="utf-8"))
    return build_open_spiel_claim_contract(PROJECT_ROOT)


@app.get(
    "/evaluation-metric-contract.json",
    tags=["System"],
    summary="Evaluation metric coverage contract",
)
def evaluation_metric_contract_json() -> dict[str, Any]:
    if EVALUATION_METRIC_CONTRACT_PATH.exists():
        return json.loads(EVALUATION_METRIC_CONTRACT_PATH.read_text(encoding="utf-8"))
    return build_evaluation_metric_contract(PROJECT_ROOT)


@app.get(
    "/test-execution-contract.json",
    tags=["System"],
    summary="Test execution transparency contract",
)
def test_execution_contract_json() -> dict[str, Any]:
    if TEST_EXECUTION_CONTRACT_PATH.exists():
        return json.loads(TEST_EXECUTION_CONTRACT_PATH.read_text(encoding="utf-8"))
    return build_test_execution_contract(PROJECT_ROOT)


@app.get("/llm-decision-context.json", tags=["System"], summary="LLM decision context contract")
def llm_decision_context_json() -> dict[str, Any]:
    return build_decision_context_report()


@app.get(
    "/llm-decision-context-smoke.json",
    tags=["System"],
    summary="LLM decision context ablation smoke report",
)
def llm_decision_context_smoke_json() -> dict[str, Any]:
    if not LLM_DECISION_CONTEXT_SMOKE_REPORT_PATH.exists():
        return {
            "status": "MISSING",
            "quality_claim_allowed": False,
            "report": str(LLM_DECISION_CONTEXT_SMOKE_REPORT_PATH),
        }
    return json.loads(LLM_DECISION_CONTEXT_SMOKE_REPORT_PATH.read_text(encoding="utf-8"))


@app.get(
    "/llm-decision-qwen25.json",
    tags=["System"],
    summary="Measured Qwen decision-context benchmark",
)
def llm_decision_qwen25_json() -> dict[str, Any]:
    if not LLM_DECISION_QWEN_REPORT_PATH.exists():
        return {"status": "MISSING", "report": str(LLM_DECISION_QWEN_REPORT_PATH)}
    return json.loads(LLM_DECISION_QWEN_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/llm-decision-gate.json", tags=["System"], summary="LLM decision model gate")
def llm_decision_gate_json() -> dict[str, Any]:
    if not LLM_DECISION_GATE_REPORT_PATH.exists():
        return {"status": "MISSING", "report": str(LLM_DECISION_GATE_REPORT_PATH)}
    return json.loads(LLM_DECISION_GATE_REPORT_PATH.read_text(encoding="utf-8"))


@app.get(
    "/llm-candidate-ranker.json",
    tags=["System"],
    summary="Measured Qwen candidate-ranking benchmark",
)
def llm_candidate_ranker_json() -> dict[str, Any]:
    if not LLM_CANDIDATE_RANKER_REPORT_PATH.exists():
        return {"status": "MISSING", "report": str(LLM_CANDIDATE_RANKER_REPORT_PATH)}
    return json.loads(LLM_CANDIDATE_RANKER_REPORT_PATH.read_text(encoding="utf-8"))


@app.get(
    "/llm-architecture-comparison.json",
    tags=["System"],
    summary="Measured LLM architecture comparison",
)
def llm_architecture_comparison_json() -> dict[str, Any]:
    if not LLM_ARCHITECTURE_COMPARISON_PATH.exists():
        return {"status": "MISSING", "report": str(LLM_ARCHITECTURE_COMPARISON_PATH)}
    return json.loads(LLM_ARCHITECTURE_COMPARISON_PATH.read_text(encoding="utf-8"))


@app.get(
    "/phase2-selection-comparison.json",
    tags=["System"],
    summary="Strict Phase 2 common-condition architecture selection contract",
)
def phase2_selection_comparison_json() -> dict[str, Any]:
    if PHASE2_SELECTION_COMPARISON_PATH.exists():
        return json.loads(PHASE2_SELECTION_COMPARISON_PATH.read_text(encoding="utf-8"))
    return build_phase2_selection_comparison(PROJECT_ROOT)


@app.get("/llm-role-boundary.json", tags=["System"], summary="LLM role boundary")
def llm_role_boundary_json() -> dict[str, Any]:
    if LLM_ROLE_BOUNDARY_PATH.exists():
        return json.loads(LLM_ROLE_BOUNDARY_PATH.read_text(encoding="utf-8"))
    return build_llm_role_boundary(PROJECT_ROOT)


@app.get(
    "/llm-policy-experimental.json",
    tags=["System"],
    summary="Experimental LLM policy adapter boundary",
)
def llm_policy_experimental_json() -> dict[str, Any]:
    if LLM_POLICY_EXPERIMENTAL_PATH.exists():
        return json.loads(LLM_POLICY_EXPERIMENTAL_PATH.read_text(encoding="utf-8"))
    return build_experimental_llm_policy_contract(PROJECT_ROOT)


@app.get("/qlora-next-stage.json", tags=["System"], summary="QLoRA next-stage improvement boundary")
def qlora_next_stage_json() -> dict[str, Any]:
    if QLORA_NEXT_STAGE_PATH.exists():
        return json.loads(QLORA_NEXT_STAGE_PATH.read_text(encoding="utf-8"))
    return build_qlora_next_stage(PROJECT_ROOT)


@app.get(
    "/agent/capabilities.json",
    tags=["System"],
    summary="Controlled session policy capabilities",
    include_in_schema=False,
)
def autonomous_agent_capabilities_json() -> dict[str, Any]:
    return get_autonomous_agent().capabilities()


@app.post(
    "/agent/decide",
    tags=["Prediction"],
    summary="Advance a controlled hand session",
    description=(
        "Accepts an ordered structured observation, enforces legal actions, and returns an "
        "idempotent policy decision for simulation or an approved environment adapter."
    ),
    include_in_schema=False,
)
def autonomous_agent_decide(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        decision, replayed = get_autonomous_agent().decide(payload)
        return decision.to_dict(idempotent_replay=replayed)
    except AgentLifecycleError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)}) from exc


@app.get(
    "/agent/sessions/{hand_id}",
    tags=["System"],
    summary="Controlled hand session state",
    include_in_schema=False,
)
def autonomous_agent_session(hand_id: str) -> dict[str, Any]:
    try:
        return get_autonomous_agent().session(hand_id)
    except AgentLifecycleError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": str(exc)}) from exc


@app.post(
    "/agent/sessions/{hand_id}/settle",
    tags=["Prediction"],
    summary="Settle a controlled hand session",
    include_in_schema=False,
)
def autonomous_agent_settle(hand_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        result = payload.get("result") if isinstance(payload, dict) else None
        if result is not None and not isinstance(result, dict):
            raise AgentLifecycleError("invalid_result", "result must be an object")
        return get_autonomous_agent().settle(hand_id, result)
    except AgentLifecycleError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)}) from exc


@app.get("/project-completion.json", tags=["System"], summary="Project completion contract")
def project_completion_json() -> dict[str, Any]:
    return build_project_completion(PROJECT_ROOT)


@app.get("/deployed-strategy-gate.json", tags=["System"], summary="Deployed strategy gate")
def deployed_strategy_gate_json() -> dict[str, Any]:
    if not DEPLOYED_STRATEGY_GATE_REPORT_PATH.exists():
        return {
            "status": "MISSING",
            "strategy_policy_status": "UNKNOWN",
            "report": str(DEPLOYED_STRATEGY_GATE_REPORT_PATH),
        }
    return json.loads(DEPLOYED_STRATEGY_GATE_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/strategy-remediation.json", tags=["System"], summary="Strategy remediation")
def strategy_remediation_json() -> dict[str, Any]:
    if not STRATEGY_REMEDIATION_REPORT_PATH.exists():
        return {
            "strategy_policy_status": "UNKNOWN",
            "report": str(STRATEGY_REMEDIATION_REPORT_PATH),
        }
    return json.loads(STRATEGY_REMEDIATION_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/scenario-sanity.json", tags=["System"], summary="Targeted poker scenario sanity validation")
def scenario_sanity_json() -> dict[str, Any]:
    if SCENARIO_SANITY_PATH.exists():
        return json.loads(SCENARIO_SANITY_PATH.read_text(encoding="utf-8"))
    return build_scenario_sanity(PROJECT_ROOT)




@app.get("/behavioral-revalidation.json", tags=["System"], summary="Behavioral revalidation boundary")
def behavioral_revalidation_json() -> dict[str, Any]:
    if BEHAVIORAL_REVALIDATION_PATH.exists():
        return json.loads(BEHAVIORAL_REVALIDATION_PATH.read_text(encoding="utf-8"))
    return build_behavioral_revalidation(PROJECT_ROOT)


@app.get("/human-likeness-evidence.json", tags=["System"], summary="Human-likeness evidence boundary")
def human_likeness_evidence_json() -> dict[str, Any]:
    if HUMAN_LIKENESS_EVIDENCE_PATH.exists():
        return json.loads(HUMAN_LIKENESS_EVIDENCE_PATH.read_text(encoding="utf-8"))
    return build_human_likeness_evidence(PROJECT_ROOT)


@app.get("/human-likeness-claim-gate.json", tags=["System"], summary="Human-likeness final-claim gate")
def human_likeness_claim_gate_json() -> dict[str, Any]:
    if HUMAN_LIKENESS_CLAIM_GATE_PATH.exists():
        return json.loads(HUMAN_LIKENESS_CLAIM_GATE_PATH.read_text(encoding="utf-8"))
    return build_human_likeness_claim_gate(PROJECT_ROOT)



@app.get("/behavioral-revalidation-proof.json", tags=["System"], summary="Behavioral revalidation executable proof")
def behavioral_revalidation_proof_json() -> dict[str, Any]:
    if BEHAVIORAL_REVALIDATION_PROOF_PATH.exists():
        return json.loads(BEHAVIORAL_REVALIDATION_PROOF_PATH.read_text(encoding="utf-8"))
    return build_behavioral_revalidation_proof(PROJECT_ROOT)


@app.get("/bet-timing-calibration.json", tags=["System"], summary="Bet-sizing and timing calibration boundary")
def bet_timing_calibration_json() -> dict[str, Any]:
    if BET_TIMING_CALIBRATION_PATH.exists():
        return json.loads(BET_TIMING_CALIBRATION_PATH.read_text(encoding="utf-8"))
    return build_bet_timing_calibration(PROJECT_ROOT)


@app.get("/hole-card-data-quality.json", tags=["System"], summary="Hole-card data-quality boundary")
def hole_card_data_quality_json() -> dict[str, Any]:
    if HOLE_CARD_DATA_QUALITY_PATH.exists():
        return json.loads(HOLE_CARD_DATA_QUALITY_PATH.read_text(encoding="utf-8"))
    return build_hole_card_data_quality(PROJECT_ROOT)


@app.get("/data-leakage-contract.json", tags=["System"], summary="Outcome-field data-leakage boundary")
def data_leakage_contract_json() -> dict[str, Any]:
    if DATA_LEAKAGE_CONTRACT_PATH.exists():
        payload = json.loads(DATA_LEAKAGE_CONTRACT_PATH.read_text(encoding="utf-8"))
        if "final_board_snapshot_contract" in payload:
            return payload
    return build_data_leakage_contract(PROJECT_ROOT)


@app.get("/normalized-action-contract.json", tags=["System"], summary="Normalized action label contract")
def normalized_action_contract_json() -> dict[str, Any]:
    if NORMALIZED_ACTION_CONTRACT_PATH.exists():
        return json.loads(NORMALIZED_ACTION_CONTRACT_PATH.read_text(encoding="utf-8"))
    return build_normalized_action_contract(PROJECT_ROOT)


@app.get("/actions-context-quality.json", tags=["System"], summary="actions.csv betting-context boundary")
def actions_context_quality_json() -> dict[str, Any]:
    if ACTION_CONTEXT_QUALITY_PATH.exists():
        return json.loads(ACTION_CONTEXT_QUALITY_PATH.read_text(encoding="utf-8"))
    return build_actions_context_quality(PROJECT_ROOT)


@app.get("/actions-dataset-export-contract.json", tags=["System"], summary="actions.csv future dataset export contract")
def actions_dataset_export_contract_json() -> dict[str, Any]:
    if ACTIONS_DATASET_EXPORT_CONTRACT_PATH.exists():
        return json.loads(ACTIONS_DATASET_EXPORT_CONTRACT_PATH.read_text(encoding="utf-8"))
    return build_actions_dataset_export_contract(PROJECT_ROOT)


@app.get("/stack-event-context-quality.json", tags=["System"], summary="stack_events.csv decision-context boundary")
def stack_event_context_quality_json() -> dict[str, Any]:
    if STACK_EVENT_CONTEXT_QUALITY_PATH.exists():
        return json.loads(STACK_EVENT_CONTEXT_QUALITY_PATH.read_text(encoding="utf-8"))
    return build_stack_event_context_quality(PROJECT_ROOT)


@app.get("/strategy-stack-maturity.json", tags=["System"], summary="Strategy stack maturity boundary")
def strategy_stack_maturity_json() -> dict[str, Any]:
    if STRATEGY_STACK_MATURITY_PATH.exists():
        return json.loads(STRATEGY_STACK_MATURITY_PATH.read_text(encoding="utf-8"))
    return build_strategy_stack_maturity(PROJECT_ROOT)


@app.get("/strategy-readiness.json", tags=["System"], summary="Strategy readiness")
def strategy_readiness_json() -> dict[str, Any]:
    return load_combined_strategy_readiness(PRODUCTION_GATE_REPORT_PATH, DEPLOYED_STRATEGY_GATE_REPORT_PATH)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home_page() -> str:
    return APP_HTML


@app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
def predict_page() -> str:
    return APP_HTML


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict poker action",
    description=(
        "Accepts a structured poker game state as a JSON request body and returns the selected "
        "action with probabilities, bet sizing, and timing. No query parameters are required."
    ),
    response_model=PredictResponseBody,
    response_model_by_alias=True,
    responses={
        200: {
            "description": "Poker action prediction with normalized probabilities, bet sizing, and timing.",
            "content": {
                "application/json": {
                    "example": {
                        "action": "raise",
                        "probabilities": {
                            "fold": 0.02,
                            "call": 0.24,
                            "check": 0.04,
                            "bet": 0.18,
                            "raise": 0.52,
                            "all_in": 0.0,
                        },
                        "confidence": 0.52,
                        "bet_size": 4.5,
                        "wait_time_ms": 1264,
                        "sizing_method": "pressure_raise",
                        "timing_method": "table_tempo_calibrated",
                        "model_status": "routed_policy_bundle",
                    }
                }
            },
        }
    },
)
def predict(payload: PredictRequestBody = Body(...)) -> dict[str, Any]:
    started = time.perf_counter()
    request_payload = payload.to_payload()
    try:
        request = PredictionRequest.from_dict(request_payload)
        result = get_agent().predict(request).to_dict()
        runtime_monitoring_state.observe_prediction(
            result,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_payload=request_payload,
        )
        return result
    except Exception as exc:
        runtime_monitoring_state.observe_error(
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_type=type(exc).__name__,
        )
        raise


