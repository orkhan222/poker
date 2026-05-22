from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from poker_agent.agents import MLPolicyAgent, RuleBasedAgent
from poker_agent.schemas import PredictionRequest


app = FastAPI(title="Poker Agent Service", version="0.1.0")
_agent = None


DEMO_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker Agent Demo</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: #101412;
      color: #eef4ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 20% 10%, rgba(38, 120, 83, 0.35), transparent 32rem),
        linear-gradient(135deg, #101412 0%, #19211d 50%, #111716 100%);
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
    .status {
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.06);
      color: #9fe6b3;
      white-space: nowrap;
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
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Poker Agent Demo</h1>
        <p>Enter a game state and get the model decision instantly.</p>
      </div>
      <div class="status">API online</div>
    </header>

    <div class="layout">
      <section class="form-panel">
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
              <input name="hole_cards" value="Ah Kd" placeholder="Ah Kd">
            </label>
            <label>Board cards
              <input name="board_cards" value="" placeholder="2c 7d Qs">
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
        <div id="bars" class="bars"></div>
        <pre id="json-output">{}</pre>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById("predict-form");
    const button = document.getElementById("submit-button");
    const actionBox = document.getElementById("action");
    const bars = document.getElementById("bars");
    const output = document.getElementById("json-output");

    function cards(value) {
      return value.split(/[ ,]+/).map((card) => card.trim()).filter(Boolean);
    }

    function numberValue(data, name) {
      return Number(data.get(name) || 0);
    }

    function render(result) {
      actionBox.textContent = result.action || "N/A";
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
        render(await response.json());
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


def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    model_path = os.getenv("POKER_POLICY_PATH")
    if model_path and Path(model_path).exists():
        _agent = MLPolicyAgent.from_path(Path(model_path))
    else:
        _agent = RuleBasedAgent()
    return _agent


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def demo_home() -> str:
    return DEMO_HTML


@app.get("/predict", response_class=HTMLResponse)
def demo_predict() -> str:
    return DEMO_HTML


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    request = PredictionRequest.from_dict(payload)
    return get_agent().predict(request).to_dict()
