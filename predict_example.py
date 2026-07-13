from __future__ import annotations

import json
import urllib.request


API_URL = "http://127.0.0.1:8000/predict"


def predict() -> None:
    game_state = {
        "position": "BTN",
        "street": "preflop",
        "hole_cards": ["Ah", "Kd"],
        "board_cards": [],
        "pot": 2.5,
        "to_call": 1.0,
        "stack": 100.0,
        "min_raise": 2.0,
        "player_count": 6,
        "game_scope": {
            "game_variant": "nl_holdem",
            "game_type": "cash",
            "table_format": "6_max",
            "small_blind": 0.5,
            "big_blind": 1.0,
            "ante": 0.0,
            "rake_percentage": 0.0,
            "rake_cap": 0.0,
            "stack_unit": "chips",
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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predict()

