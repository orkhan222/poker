from __future__ import annotations

import json
import urllib.request


API_URL = "http://127.0.0.1:8001/predict"
CONTRACT_URL = "http://127.0.0.1:8001/contract.json"


def predict() -> None:
    with urllib.request.urlopen(CONTRACT_URL, timeout=10) as response:
        contract = json.loads(response.read().decode("utf-8"))
    if contract.get("endpoint") != "/predict":
        raise RuntimeError(f"Unexpected deployment contract endpoint: {contract.get('endpoint')}")

    game_state = {
        "position": "BTN",
        "street": "preflop",
        "hole_cards": ["Ah", "Kd"],
        "board_cards": [],
        "pot": 2.5,
        "current_bet": 1.0,
        "to_call": 1.0,
        "amount_to_call": 1.0,
        "stack": 100.0,
        "effective_stack": 100.0,
        "small_blind": 0.5,
        "big_blind": 1.0,
        "ante": 0.0,
        "button_position": "BTN",
        "dealer_position": "BTN",
        "action_order": ["UTG", "MP", "CO", "BTN", "SB", "BB"],
        "min_raise": 2.0,
        "max_raise": 100.0,
        "min_raise_to": 3.0,
        "max_raise_to": 100.0,
        "min_raise_by": 2.0,
        "max_raise_by": 99.0,
        "all_in_amount": 100.0,
        "legal_actions": ["fold", "call", "raise", "all_in"],
        "player_count": 6,
        "game_scope": {
            "game_type": "nl_holdem",
            "format": "cash",
            "table_size": "6_max",
            "small_blind": 0.5,
            "big_blind": 1.0,
            "ante": 0.0,
            "stack_unit": "bb",
        },
        "usage_boundary": {
            "declared_use": "offline_research",
            "real_money": False,
            "terms_compliant": True,
        },
    }

    body = json.dumps(game_state).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))

    required = {"schema_version", "model_version", "action", "confidence", "probabilities", "legal_actions"}
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"Predict response is missing deployment fields: {missing}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predict()

