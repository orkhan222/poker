from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "2026-06-20"


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
        "client_handoff": {
            "endpoint": "/client-handoff.json",
            "description": "Client-facing delivery statement that separates production blockers from tracked component risks.",
            "handoff_status_values": ["READY", "READY_WITH_COMPONENT_RISK", "NOT_READY"],
            "delivery_boundary": "Service delivery and deployed strategy approval can be ready while raw-model standalone approval remains a tracked component risk.",
        },
        "approval_boundary": {
            "software_delivery": "The API, package, and reproducibility checks are evaluated separately.",
            "deployed_strategy_stack": "Production policy approval is based on the deployed strategy gate.",
            "raw_strategy_model": "The standalone supervised artifact remains independently gated.",
        },
    }
