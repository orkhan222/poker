from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "2026-06-24"


PREDICTION_REQUEST_FIELDS: dict[str, dict[str, Any]] = {
    "position": {"type": "string", "description": "Hero table position or normalized seat identifier."},
    "street": {"type": "string", "description": "Current betting street: preflop, flop, turn, or river."},
    "hole_cards": {"type": "array[string]", "description": "Hero private cards when observed."},
    "board_cards": {"type": "array[string]", "description": "Community cards visible at decision time."},
    "pot": {"type": "float", "description": "Current pot before the requested action."},
    "to_call": {"type": "float", "description": "Additional chips required to call."},
    "stack": {"type": "float", "description": "Hero stack available at decision time."},
    "min_raise": {"type": "float", "description": "Minimum legal raise increment or amount supplied by the table state."},
    "player_count": {"type": "integer", "description": "Number of players represented in the current state."},
    "betting_history": {
        "type": "array[object]",
        "description": "Only actions observable before the requested decision; events may include wait_time_ms and frame_delta.",
    },
    "timing_context": {
        "type": "object",
        "description": (
            "Optional observed opponent timing with opponent_wait_before_turn_ms and "
            "opponent_wait_after_hero_action_ms."
        ),
    },
}


PREDICTION_RESPONSE_FIELDS: dict[str, dict[str, Any]] = {
    "action": {"type": "string", "description": "Primary poker action selected for the submitted game state."},
    "probabilities": {"type": "object[string,float]", "description": "Normalized action probability distribution."},
    "confidence": {"type": "float", "description": "Confidence assigned to the selected action."},
    "bet_size": {"type": "float", "description": "Recommended chip amount for call, bet, or raise actions."},
    "wait_time_ms": {"type": "integer", "description": "Recommended decision delay in milliseconds."},
    "sizing_method": {"type": "string", "description": "Sizing policy used to calculate bet_size."},
    "timing_method": {"type": "string", "description": "Timing policy used to calculate wait_time_ms."},
    "model_status": {"type": "string", "description": "Inference path that produced the response."},
    "warnings": {"type": "array[string]", "description": "Optional degraded-path warnings."},
}


DELIVERY_STATUS_FIELDS: dict[str, dict[str, str]] = {
    "delivery_verification=PASS": {
        "meaning": "Source, reports, inference contract, hygiene, and ZIP package checks passed.",
        "implication": "The package is suitable for delivery review.",
    },
    "production_scale_self_play=PASS": {
        "meaning": "Validated Hold'em self-play ran at production review scale and passed.",
        "implication": "Self-play scale is no longer a strategy blocker.",
    },
    "deployed_strategy_gate=PASS": {
        "meaning": "The deployed strategy stack passed policy acceptance, human-likeness, production-scale self-play, service, and hygiene gates.",
        "implication": "The deployed stack can be approved with monitoring and rollback.",
    },
    "raw_production_gate=FAIL": {
        "meaning": "The standalone supervised artifact does not pass raw model-quality thresholds by itself.",
        "implication": "This remains a component risk, not a deployed-stack blocker when the deployed strategy gate passes.",
    },
}


def api_contract() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "prediction_request": {
            "contract_version": CONTRACT_VERSION,
            "request_fields": deepcopy(PREDICTION_REQUEST_FIELDS),
            "leakage_rule": "The request and betting history must contain only information observable before the target action.",
        },
        "prediction_response": {
            "contract_version": CONTRACT_VERSION,
            "response_fields": deepcopy(PREDICTION_RESPONSE_FIELDS),
        },
        "delivery_status": {
            "contract_version": CONTRACT_VERSION,
            "delivery_status_fields": deepcopy(DELIVERY_STATUS_FIELDS),
        },
        "strategy_readiness": {
            "endpoint": "/strategy-readiness.json",
            "status_values": ["APPROVED", "NOT_APPROVED", "UNKNOWN"],
        },
        "deployed_strategy_gate": {
            "endpoint": "/deployed-strategy-gate.json",
            "status_values": ["PASS", "FAIL", "MISSING"],
        },
        "delivery_readiness": {
            "endpoint": "/delivery-readiness.json",
            "overall_status_values": [
                "READY_FOR_PRODUCTION_POLICY",
                "READY_FOR_TECHNICAL_HANDOFF",
                "NOT_READY_FOR_HANDOFF",
            ],
        },
        "scope_contract": {
            "endpoint": "/scope-contract.json",
            "description": "Machine-readable mapping from the DOCX/PDF project scope to implemented evidence and remaining risks.",
            "source_documents": [
                "Poker ML Project.docx",
                "Poker_Agent_Development_EN_detailed.pdf",
            ],
            "overall_status_values": ["PASS", "PARTIAL", "FAIL"],
        },
        "model_risk_register": {
            "endpoint": "/model-risk-register.json",
            "description": "Tracks the approved deployed strategy stack separately from the raw supervised model component risk.",
            "status_values": ["PASS", "FAIL"],
            "risk_boundary": "Raw supervised model weakness is a component risk unless explicitly marked as a deployment blocker.",
        },
        "production_approval": {
            "endpoint": "/production-approval.json",
            "description": "Defines production claims that are allowed and explicitly disallowed for the delivered package.",
            "overall_status_values": ["APPROVED", "APPROVED_WITH_COMPONENT_RISK", "NOT_APPROVED"],
            "approval_boundary": "The deployed strategy stack can be approved while the raw supervised model remains not standalone approved.",
        },
        "approval_boundary": {
            "endpoint": "/approval-boundary.json",
            "description": "Single source of truth for service readiness, deployed-stack approval, raw-model standalone status, production blockers, and component risks.",
            "release_status_values": ["READY", "READY_WITH_COMPONENT_RISK", "NOT_READY"],
            "non_override_rule": "Deployed stack approval must not be represented as standalone raw-model approval.",
        },
        "client_handoff": {
            "endpoint": "/client-handoff.json",
            "description": "Client-facing delivery statement that separates production blockers from tracked component risks.",
            "handoff_status_values": ["READY", "READY_WITH_COMPONENT_RISK", "NOT_READY"],
            "delivery_boundary": "Service delivery and deployed strategy approval can be ready while raw-model standalone approval remains a tracked component risk.",
        },
        "llm_decision_context": {
            "endpoint": "/llm-decision-context.json",
            "description": "Defines the in-context learning contract for out-of-box LLM poker decision experiments.",
            "context_modes": ["minimal_zero_shot", "rules_grounded", "full_in_context"],
            "default_context_mode": "full_in_context",
            "controls": [
                "legal action filtering",
                "strict JSON-only output",
                "probability normalization",
                "bet-size and timing post-processing",
            ],
        },
        "llm_decision_context_ablation": {
            "smoke_endpoint": "/llm-decision-context-smoke.json",
            "qwen_endpoint": "/llm-decision-qwen25.json",
            "gate_endpoint": "/llm-decision-gate.json",
            "candidate_ranker_endpoint": "/llm-candidate-ranker.json",
            "architecture_comparison_endpoint": "/llm-architecture-comparison.json",
            "context_modes": ["minimal_zero_shot", "rules_grounded", "full_in_context"],
            "metrics": [
                "accuracy",
                "macro_f1",
                "json_valid_rate",
                "schema_valid_rate",
                "legal_action_rate",
                "fallback_rate",
                "latency",
                "token_count",
                "peak_memory",
            ],
            "claim_boundary": (
                "A winning context mode may be selected only for a real model evaluated on a "
                "manually reviewed human holdout."
            ),
            "selected_research_architecture": "candidate_ranker",
        },
        "project_completion": {
            "endpoint": "/project-completion.json",
            "description": "Maps the documented project scope to implemented evidence, metrics, deployment artifacts, and known component risks.",
            "overall_status_values": ["PASS", "PARTIAL"],
            "covered_sections": [
                "feature_space",
                "action_space",
                "dataset_model",
                "phase_1_two_baselines",
                "phase_2_selection_optimization",
                "phase_3_evaluation",
                "phase_4_deployment",
            ],
        },
        "approval_scope": {
            "software_delivery": "The API, package, and reproducibility checks are evaluated separately.",
            "deployed_strategy_stack": "Production policy approval is based on the deployed strategy gate.",
            "raw_strategy_model": "The standalone supervised artifact remains independently gated.",
        },
    }
