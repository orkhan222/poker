from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from poker_agent.agents import MLPolicyAgent, RuleBasedAgent
from poker_agent.api_contract import (
    API_VERSION,
    ERROR_CODES,
    api_error,
    deployment_api_contract,
    model_version_from_metadata,
    predict_request_schema,
    predict_response_schema,
)
from poker_agent.monitoring import append_prediction_monitoring, monotonic_ms
from poker_agent.schemas import PredictionRequest
from poker_agent.security import (
    InMemoryRateLimiter,
    LogRetentionPolicy,
    authenticate_headers,
    prune_jsonl_by_retention,
    security_config_from_env,
)
from poker_agent.usage_boundary import evaluate_usage_boundary


app = FastAPI(
    title="Poker Decision Agent API",
    description="API for real-time poker action prediction using the bundled trained policy model. See /contract.json for the deployment contract.",
    version="1.0.0",
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
_agent = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy.joblib"
OPTIONAL_BUNDLE_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy_bundle.joblib"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "poker_policy.json"
PREDICTION_LOG_PATH = PROJECT_ROOT / os.getenv("POKER_PREDICTION_LOG_PATH", "reports/prediction_logs.jsonl")
AUDIT_TRAIL_PATH = PROJECT_ROOT / os.getenv("POKER_AUDIT_TRAIL_PATH", "reports/audit_trail.jsonl")
SECURITY_CONFIG = security_config_from_env()
RATE_LIMITER = InMemoryRateLimiter(
    limit=SECURITY_CONFIG.rate_limit_burst or SECURITY_CONFIG.rate_limit_per_minute,
    window_seconds=SECURITY_CONFIG.rate_limit_window_seconds,
)
LOG_RETENTION_POLICY = LogRetentionPolicy(
    max_age_days=SECURITY_CONFIG.retention_days,
    max_records=SECURITY_CONFIG.retention_max_records,
    enabled=SECURITY_CONFIG.retention_enabled,
)


class ApiContractError(Exception):
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self.payload = payload
        super().__init__(payload["error"]["message"])


@app.exception_handler(ApiContractError)
def api_contract_error_handler(_: Request, exc: ApiContractError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


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
            <label>Current bet
              <input name="current_bet" type="number" step="0.1" value="1.0">
            </label>
            <label>To call
              <input name="to_call" type="number" step="0.1" value="1.0">
            </label>
            <label>Amount to call
              <input name="amount_to_call" type="number" step="0.1" value="1.0">
            </label>
            <label>Stack
              <input name="stack" type="number" step="0.1" value="100">
            </label>
            <label>Effective stack
              <input name="effective_stack" type="number" step="0.1" value="100">
            </label>
            <label>Small blind
              <input name="small_blind" type="number" step="0.1" value="0.5">
            </label>
            <label>Big blind
              <input name="big_blind" type="number" step="0.1" value="1.0">
            </label>
            <label>Ante
              <input name="ante" type="number" step="0.1" value="0">
            </label>
            <label>Button position
              <input name="button_position" value="BTN">
            </label>
            <label>Dealer position
              <input name="dealer_position" value="BTN">
            </label>
            <label>Action order
              <input name="action_order" value="UTG MP CO BTN SB BB" aria-label="Action order positions">
            </label>
            <label>Min raise
              <input name="min_raise" type="number" step="0.1" value="2.0">
            </label>
            <label>Max raise
              <input name="max_raise" type="number" step="0.1" value="100">
            </label>
            <label>Min raise to
              <input name="min_raise_to" type="number" step="0.1" value="3.0">
            </label>
            <label>Max raise to
              <input name="max_raise_to" type="number" step="0.1" value="100">
            </label>
            <label>Min raise by
              <input name="min_raise_by" type="number" step="0.1" value="2.0">
            </label>
            <label>Max raise by
              <input name="max_raise_by" type="number" step="0.1" value="99">
            </label>
            <label>All-in amount
              <input name="all_in_amount" type="number" step="0.1" value="100">
            </label>
            <label>Legal actions
              <input name="legal_actions" value="" aria-label="Legal actions, for example fold call raise all_in">
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
        current_bet: numberValue(data, "current_bet"),
        to_call: numberValue(data, "to_call"),
        amount_to_call: numberValue(data, "amount_to_call"),
        stack: numberValue(data, "stack"),
        effective_stack: numberValue(data, "effective_stack"),
        small_blind: numberValue(data, "small_blind"),
        big_blind: numberValue(data, "big_blind"),
        ante: numberValue(data, "ante"),
        button_position: data.get("button_position"),
        dealer_position: data.get("dealer_position"),
        action_order: cards(data.get("action_order") || ""),
        min_raise: numberValue(data, "min_raise"),
        max_raise: numberValue(data, "max_raise"),
        min_raise_to: numberValue(data, "min_raise_to"),
        max_raise_to: numberValue(data, "max_raise_to"),
        min_raise_by: numberValue(data, "min_raise_by"),
        max_raise_by: numberValue(data, "max_raise_by"),
        all_in_amount: numberValue(data, "all_in_amount"),
        legal_actions: cards(data.get("legal_actions") || ""),
        player_count: Number(data.get("player_count") || 6),
        game_scope: {
          game_type: "nl_holdem",
          format: "cash",
          table_size: Number(data.get("player_count") || 6) > 6 ? "9_max" : "6_max",
          small_blind: numberValue(data, "small_blind"),
          big_blind: numberValue(data, "big_blind"),
          ante: numberValue(data, "ante"),
          rake_percentage: 0,
          rake_cap: 0,
          stack_unit: "chips"
        },
        usage_boundary: {
          declared_use: "offline_research",
          real_money: false,
          terms_compliant: true
        }
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
    payload = {
        "status": "ok",
        "api_version": API_VERSION,
        "model": str(model_path),
        "model_version": model_version_from_metadata({}, model_path),
        "model_status": "loaded" if model_path.exists() else "fallback_rule_based",
        "auth_required": str(SECURITY_CONFIG.auth_required).lower(),
        "rate_limit_per_minute": str(SECURITY_CONFIG.rate_limit_per_minute),
        "log_retention_days": str(SECURITY_CONFIG.retention_days),
    }
    try:
        agent = get_agent()
        model = getattr(agent, "model", None)
        metadata = getattr(model, "metadata", {}) or {}
        payload["model_version"] = model_version_from_metadata(metadata, model_path)
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
    </nav>
  </main>
</body>
</html>
"""


def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    model_path = resolve_model_path()
    if model_path.exists():
        _agent = MLPolicyAgent.from_path(model_path)
    else:
        _agent = RuleBasedAgent()
    return _agent


def resolve_model_path() -> Path:
    configured = os.getenv("POKER_POLICY_PATH")
    if configured:
        return Path(configured)
    if OPTIONAL_BUNDLE_MODEL_PATH.exists():
        return OPTIONAL_BUNDLE_MODEL_PATH
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return FALLBACK_MODEL_PATH


def current_model_version() -> str:
    model_path = resolve_model_path()
    try:
        agent = get_agent()
        model = getattr(agent, "model", None)
        metadata = getattr(model, "metadata", {}) or {}
        return model_version_from_metadata(metadata, model_path)
    except Exception:
        return model_version_from_metadata({}, model_path)


def raise_api_error(code: str, message: str | None = None, details: dict[str, Any] | None = None) -> None:
    spec = ERROR_CODES.get(code, ERROR_CODES["PREDICTION_FAILED"])
    raise ApiContractError(status_code=int(spec["http_status"]), payload=api_error(code, message, details))


def emit_prediction_monitoring(
    *,
    request_id: str,
    raw_payload: dict[str, Any],
    started_ms: float,
    request: PredictionRequest | None = None,
    response: dict[str, Any] | None = None,
    status: str,
    error_code: str | None = None,
) -> None:
    try:
        append_prediction_monitoring(
            prediction_log_path=PREDICTION_LOG_PATH,
            audit_trail_path=AUDIT_TRAIL_PATH,
            request_id=request_id,
            raw_payload=raw_payload,
            request=request,
            response=response,
            latency_ms=monotonic_ms() - started_ms,
            status=status,
            error_code=error_code,
        )
        prune_jsonl_by_retention(PREDICTION_LOG_PATH, LOG_RETENTION_POLICY)
        prune_jsonl_by_retention(AUDIT_TRAIL_PATH, LOG_RETENTION_POLICY)
    except Exception:
        return


def request_headers(request: Request) -> dict[str, str]:
    return {key.lower(): value for key, value in request.headers.items()}


def client_identity(request: Request, principal: str) -> str:
    if principal and principal != "anonymous":
        return principal
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown-client"


def enforce_security(request: Request) -> dict[str, Any]:
    auth = authenticate_headers(request_headers(request), SECURITY_CONFIG)
    if not auth.allowed:
        raise_api_error(auth.error_code or "UNAUTHORIZED", auth.message)
    rate = RATE_LIMITER.check(client_identity(request, auth.principal))
    if not rate.allowed:
        raise_api_error(
            "RATE_LIMITED",
            "Request rate limit exceeded.",
            {
                "limit": rate.limit,
                "remaining": rate.remaining,
                "retry_after_seconds": rate.retry_after_seconds,
            },
        )
    return {
        "principal": auth.principal,
        "credential_hash_prefix": auth.credential_hash_prefix,
        "rate_limit": {
            "limit": rate.limit,
            "remaining": rate.remaining,
            "retry_after_seconds": rate.retry_after_seconds,
        },
    }


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


@app.get(
    "/contract.json",
    tags=["System"],
    summary="Deployment API contract",
    description="Returns the machine-readable /predict request schema, response schema, model version, and error codes.",
)
def contract_json() -> dict[str, Any]:
    return deployment_api_contract(model_version=current_model_version())


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
    description="Accepts a poker game state and returns the selected action with model version, confidence, probabilities, and sizing metadata.",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": predict_request_schema()}},
        },
        "responses": {
            "200": {"description": "Prediction response", "content": {"application/json": {"schema": predict_response_schema()}}},
            "400": {"description": "INVALID_REQUEST"},
            "403": {"description": "USAGE_BOUNDARY_VIOLATION"},
            "422": {"description": "UNSUPPORTED_ACTION_SPACE"},
            "500": {"description": "PREDICTION_FAILED"},
            "401": {"description": "UNAUTHORIZED"},
            "429": {"description": "RATE_LIMITED"},
            "503": {"description": "MODEL_UNAVAILABLE or SECURITY_MISCONFIGURED"},
        },
    },
)
def predict(http_request: Request, payload: Any = Body(...)) -> dict[str, Any]:
    started_ms = monotonic_ms()
    request_id = str(uuid4())
    raw_payload = payload if isinstance(payload, dict) else {}
    security_context: dict[str, Any] = {}
    try:
        security_context = enforce_security(http_request)
    except ApiContractError as exc:
        error_code = str((exc.payload.get("error") or {}).get("code") or "UNAUTHORIZED")
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            started_ms=started_ms,
            status="error",
            error_code=error_code,
        )
        raise
    if not isinstance(payload, dict):
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            started_ms=started_ms,
            status="error",
            error_code="INVALID_REQUEST",
        )
        raise_api_error("INVALID_REQUEST", "Request body must be a JSON object.")
    if not (payload.get("position") or payload.get("player_position")):
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            started_ms=started_ms,
            status="error",
            error_code="INVALID_REQUEST",
        )
        raise_api_error("INVALID_REQUEST", "Missing required field: position.", {"field": "position"})

    usage_boundary = evaluate_usage_boundary(payload)
    if not usage_boundary.allowed:
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            started_ms=started_ms,
            status="error",
            error_code="USAGE_BOUNDARY_VIOLATION",
        )
        raise_api_error("USAGE_BOUNDARY_VIOLATION", usage_boundary.message, usage_boundary.to_error_details())

    try:
        request = PredictionRequest.from_dict(payload)
    except (TypeError, ValueError) as exc:
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            started_ms=started_ms,
            status="error",
            error_code="INVALID_REQUEST",
        )
        raise_api_error("INVALID_REQUEST", str(exc))

    if not request.legal_actions:
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            request=request,
            started_ms=started_ms,
            status="error",
            error_code="UNSUPPORTED_ACTION_SPACE",
        )
        raise_api_error("UNSUPPORTED_ACTION_SPACE", "No legal actions are available for the supplied state.")

    try:
        response = get_agent().predict(request).to_dict()
        response["request_id"] = request_id
        response["usage_boundary"] = usage_boundary.response_context()
        response["security_context"] = {
            "principal": security_context.get("principal", "unknown"),
            "credential_hash_prefix": security_context.get("credential_hash_prefix"),
            "rate_limit": security_context.get("rate_limit", {}),
        }
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            request=request,
            response=response,
            started_ms=started_ms,
            status="ok",
        )
        return response
    except RuntimeError as exc:
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            request=request,
            started_ms=started_ms,
            status="error",
            error_code="MODEL_UNAVAILABLE",
        )
        raise_api_error("MODEL_UNAVAILABLE", str(exc))
    except Exception as exc:
        emit_prediction_monitoring(
            request_id=request_id,
            raw_payload=raw_payload,
            request=request,
            started_ms=started_ms,
            status="error",
            error_code="PREDICTION_FAILED",
        )
        raise_api_error("PREDICTION_FAILED", str(exc))
