# Poker Decision Agent API Contract

- API version: `poker-decision-agent-api-v1`
- Endpoint: `POST /predict`
- Request schema: `predict_request.v1`
- Response schema: `predict_response.v1`
- Model version field: `poker_policy:handoff`

## Request

Required fields:

- `position`
- `street`
- `hole_cards`
- `pot`
- `stack`
- `usage_boundary`

Supported request properties:

- `action_order`
- `all_in_amount`
- `amount_to_call`
- `ante`
- `betting_history`
- `big_blind`
- `board_cards`
- `button_position`
- `current_bet`
- `dealer_position`
- `effective_stack`
- `game_scope`
- `hole_cards`
- `legal_action_mask`
- `legal_actions`
- `max_raise`
- `max_raise_by`
- `max_raise_to`
- `min_raise`
- `min_raise_by`
- `min_raise_to`
- `position`
- `pot`
- `pot_size`
- `small_blind`
- `stack`
- `street`
- `to_call`
- `usage_boundary`

## Response

Required response fields:

- `schema_version`
- `model_version`
- `action`
- `probabilities`
- `confidence`
- `model_status`
- `legal_actions`
- `action_space`
- `state_context`

Response properties:

- `schema_version`
- `request_id`
- `model_version`
- `action`
- `probabilities`
- `confidence`
- `model_status`
- `warnings`
- `bet_size`
- `raise_to`
- `raise_by`
- `sizing_method`
- `legal_actions`
- `action_space`
- `state_context`
- `usage_boundary`
- `security_context`

## Error Codes

- `INVALID_REQUEST`: HTTP 400, retryable=false
- `MODEL_UNAVAILABLE`: HTTP 503, retryable=true
- `PREDICTION_FAILED`: HTTP 500, retryable=true
- `RATE_LIMITED`: HTTP 429, retryable=true
- `SECURITY_MISCONFIGURED`: HTTP 503, retryable=true
- `UNAUTHORIZED`: HTTP 401, retryable=false
- `UNSUPPORTED_ACTION_SPACE`: HTTP 422, retryable=false
- `USAGE_BOUNDARY_VIOLATION`: HTTP 403, retryable=false

## Example Request

```json
{
  "action_order": [
    "UTG",
    "MP",
    "CO",
    "BTN",
    "SB",
    "BB"
  ],
  "amount_to_call": 1.0,
  "big_blind": 1.0,
  "board_cards": [],
  "button_position": "BTN",
  "current_bet": 1.0,
  "effective_stack": 100.0,
  "game_scope": {
    "ante": 0.0,
    "big_blind": 1.0,
    "format": "cash",
    "game_type": "nl_holdem",
    "rake_cap": 3.0,
    "rake_percentage": 0.05,
    "small_blind": 0.5,
    "stack_unit": "chips",
    "table_size": "6_max"
  },
  "hole_cards": [
    "Ah",
    "Kd"
  ],
  "legal_actions": [
    "fold",
    "call",
    "raise",
    "all_in"
  ],
  "position": "BTN",
  "pot": 2.5,
  "small_blind": 0.5,
  "stack": 100.0,
  "street": "preflop",
  "to_call": 1.0,
  "usage_boundary": {
    "declared_use": "offline_research",
    "real_money": false,
    "terms_compliant": true
  }
}
```
