from __future__ import annotations

from pathlib import Path
from typing import Any

from poker_agent.action_space import CANONICAL_ACTIONS
from poker_agent.game_scope import SUPPORTED_FORMATS, SUPPORTED_GAME_TYPES, SUPPORTED_STACK_UNITS, SUPPORTED_TABLE_SIZES
from poker_agent.usage_boundary import ALLOWED_USAGE, BLOCKED_USAGE

API_VERSION = "poker-decision-agent-api-v1"
PREDICT_REQUEST_SCHEMA_VERSION = "predict_request.v1"
PREDICT_RESPONSE_SCHEMA_VERSION = "predict_response.v1"
ERROR_RESPONSE_SCHEMA_VERSION = "error_response.v1"

PREDICT_ENDPOINT = "/predict"

ERROR_CODES: dict[str, dict[str, Any]] = {
    "INVALID_REQUEST": {
        "http_status": 400,
        "retryable": False,
        "message": "Request payload is malformed or missing required game-state fields.",
    },
    "UNSUPPORTED_ACTION_SPACE": {
        "http_status": 422,
        "retryable": False,
        "message": "Legal action metadata cannot be reconciled with no-limit Hold'em action rules.",
    },
    "MODEL_UNAVAILABLE": {
        "http_status": 503,
        "retryable": True,
        "message": "The configured model artifact is unavailable or cannot be loaded.",
    },
    "UNAUTHORIZED": {
        "http_status": 401,
        "retryable": False,
        "message": "Authentication failed or API key is missing.",
    },
    "RATE_LIMITED": {
        "http_status": 429,
        "retryable": True,
        "message": "Request rate limit exceeded.",
    },
    "SECURITY_MISCONFIGURED": {
        "http_status": 503,
        "retryable": True,
        "message": "Security is enabled but required secret configuration is missing.",
    },
    "USAGE_BOUNDARY_VIOLATION": {
        "http_status": 403,
        "retryable": False,
        "message": "The request violates the offline research, simulation, or authorized-environment usage boundary.",
    },
    "PREDICTION_FAILED": {
        "http_status": 500,
        "retryable": True,
        "message": "The prediction pipeline failed after request validation.",
    },
}


def predict_request_schema() -> dict[str, Any]:
    numeric = {"type": "number"}
    card_array = {"type": "array", "items": {"type": "string"}, "default": []}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PokerDecisionPredictRequest",
        "schema_version": PREDICT_REQUEST_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": True,
        "required": ["position", "street", "hole_cards", "pot", "stack", "usage_boundary"],
        "properties": {
            "position": {"type": "string", "description": "Hero position/seat identifier."},
            "street": {"type": "string", "enum": ["preflop", "flop", "turn", "river"]},
            "hole_cards": card_array,
            "board_cards": card_array,
            "pot": numeric,
            "pot_size": numeric,
            "current_bet": numeric,
            "to_call": numeric,
            "amount_to_call": numeric,
            "stack": numeric,
            "effective_stack": numeric,
            "small_blind": numeric,
            "big_blind": numeric,
            "ante": numeric,
            "button_position": {"type": "string"},
            "dealer_position": {"type": "string"},
            "action_order": {"type": "array", "items": {"type": "string"}},
            "min_raise": numeric,
            "max_raise": numeric,
            "min_raise_to": numeric,
            "max_raise_to": numeric,
            "min_raise_by": numeric,
            "max_raise_by": numeric,
            "all_in_amount": numeric,
            "legal_actions": {
                "type": "array",
                "items": {"type": "string", "enum": list(CANONICAL_ACTIONS)},
                "description": "Optional legal-action mask. If omitted, the service derives the mask from state.",
            },
            "legal_action_mask": {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
                "description": "Optional action-to-boolean mask using canonical action keys.",
            },
            "game_scope": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "game_type": {"type": "string", "enum": list(SUPPORTED_GAME_TYPES)},
                    "format": {"type": "string", "enum": list(SUPPORTED_FORMATS)},
                    "table_size": {"type": "string", "enum": list(SUPPORTED_TABLE_SIZES)},
                    "small_blind": numeric,
                    "big_blind": numeric,
                    "ante": numeric,
                    "rake_percentage": numeric,
                    "rake_cap": numeric,
                    "stack_unit": {"type": "string", "enum": list(SUPPORTED_STACK_UNITS)},
                },
            },
            "usage_boundary": {
                "type": "object",
                "additionalProperties": True,
                "required": ["declared_use"],
                "properties": {
                    "declared_use": {"type": "string", "enum": list(ALLOWED_USAGE)},
                    "environment": {"type": "string", "enum": list(ALLOWED_USAGE) + list(BLOCKED_USAGE)},
                    "authorized": {"type": "boolean"},
                    "real_money": {"type": "boolean", "default": False},
                    "terms_compliant": {"type": "boolean", "default": True},
                    "prohibited_use": {"type": "string", "enum": list(BLOCKED_USAGE)},
                },
                "description": "Required legal/ethical boundary. /predict is limited to offline_research, simulation, or authorized_environment.",
            },
            "betting_history": {"type": "array", "items": {"type": "object"}},
        },
        "examples": [
            {
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
                "button_position": "BTN",
                "action_order": ["UTG", "MP", "CO", "BTN", "SB", "BB"],
                "legal_actions": ["fold", "call", "raise", "all_in"],
                "game_scope": {
                    "game_type": "nl_holdem",
                    "format": "cash",
                    "table_size": "6_max",
                    "small_blind": 0.5,
                    "big_blind": 1.0,
                    "ante": 0.0,
                    "rake_percentage": 0.05,
                    "rake_cap": 3.0,
                    "stack_unit": "chips",
                },
                "usage_boundary": {
                    "declared_use": "offline_research",
                    "real_money": False,
                    "terms_compliant": True,
                },
            }
        ],
    }


def predict_response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PokerDecisionPredictResponse",
        "schema_version": PREDICT_RESPONSE_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "model_version",
            "action",
            "probabilities",
            "confidence",
            "model_status",
            "legal_actions",
            "action_space",
            "state_context",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": PREDICT_RESPONSE_SCHEMA_VERSION},
            "request_id": {"type": "string"},
            "model_version": {"type": "string"},
            "action": {"type": "string", "enum": list(CANONICAL_ACTIONS)},
            "probabilities": {
                "type": "object",
                "required": list(CANONICAL_ACTIONS),
                "additionalProperties": False,
                "properties": {action: {"type": "number", "minimum": 0.0, "maximum": 1.0} for action in CANONICAL_ACTIONS},
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "model_status": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "bet_size": {"type": "number"},
            "raise_to": {"type": ["number", "null"]},
            "raise_by": {"type": ["number", "null"]},
            "sizing_method": {"type": "string"},
            "legal_actions": {"type": "array", "items": {"type": "string", "enum": list(CANONICAL_ACTIONS)}},
            "action_space": {"type": "object"},
            "state_context": {"type": "object"},
            "usage_boundary": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "schema_version": {"type": "string"},
                    "declared_use": {"type": "string"},
                    "allowed": {"type": "boolean"},
                    "reason_code": {"type": "string"},
                },
            },
            "security_context": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "principal": {"type": "string"},
                    "credential_hash_prefix": {"type": ["string", "null"]},
                    "rate_limit": {"type": "object"},
                },
            },
        },
    }


def error_response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PokerDecisionErrorResponse",
        "schema_version": ERROR_RESPONSE_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "error"],
        "properties": {
            "schema_version": {"type": "string", "const": ERROR_RESPONSE_SCHEMA_VERSION},
            "error": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code", "message", "retryable", "details"],
                "properties": {
                    "code": {"type": "string", "enum": sorted(ERROR_CODES)},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                    "details": {"type": "object"},
                },
            },
        },
    }


def deployment_api_contract(*, model_version: str = "unknown") -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "endpoint": PREDICT_ENDPOINT,
        "model_version": model_version,
        "request_schema": predict_request_schema(),
        "response_schema": predict_response_schema(),
        "error_response_schema": error_response_schema(),
        "error_codes": ERROR_CODES,
    }


def model_version_from_metadata(metadata: dict[str, Any] | None, model_path: Path | str | None = None) -> str:
    metadata = metadata or {}
    for key in ("model_version", "artifact_version", "version"):
        value = metadata.get(key)
        if value:
            return str(value)
    policy = str(metadata.get("policy") or metadata.get("baseline") or "").strip()
    split = metadata.get("split") if isinstance(metadata.get("split"), dict) else {}
    split_type = str(split.get("split_type") or "").strip()
    if policy and split_type:
        return f"{policy}:{split_type}"
    if policy:
        return policy
    if model_path:
        return Path(model_path).stem
    return "rule_based:v1"


def api_error(code: str, message: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = ERROR_CODES.get(code)
    if spec is None:
        spec = ERROR_CODES["PREDICTION_FAILED"]
        code = "PREDICTION_FAILED"
    return {
        "schema_version": ERROR_RESPONSE_SCHEMA_VERSION,
        "error": {
            "code": code,
            "message": message or str(spec["message"]),
            "retryable": bool(spec["retryable"]),
            "details": details or {},
        },
    }
