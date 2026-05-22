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

