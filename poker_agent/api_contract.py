from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "2026-06-25"


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
        "data_leakage_contract": {
            "endpoint": "/data-leakage-contract.json",
            "description": "Blocks post-hand outcome fields from training, prediction, and model-artifact feature sets.",
            "forbidden_outcome_fields": [
                "winner_positions",
                "stack_delta",
                "ending_stack",
                "dealer_winner",
                "dealer_pot",
                "pot_from_stacks",
            ],
            "boundary": "These fields may exist in raw CSVs for reporting, but are not valid decision-time features.",
        },
        "normalized_action_contract": {
            "endpoint": "/normalized-action-contract.json",
            "description": "Normalizes raw OCR/dealer action text into canonical action labels before training, evaluation, and policy comparison.",
            "source_field": "actions.csv::action",
            "normalized_field": "canonical_action",
            "canonical_actions": ["fold", "call", "check", "bet", "raise", "all_in"],
            "noisy_examples": {
                "ra1se": "raise",
                "cail": "call",
                "bett": "bet",
                "all-in": "all_in",
            },
            "boundary": "Raw OCR action strings must not be used directly as supervised labels.",
        },
        "actions_context_quality": {
            "endpoint": "/actions-context-quality.json",
            "description": "Documents missing explicit betting-context fields in actions.csv and verifies leakage-safe derived context features.",
            "risk_id": "actions_csv_betting_context_incomplete",
            "root_cause": "actions.csv has action/street, but not the full decision-time betting context needed for strong call/fold/raise learning.",
            "missing_or_required_explicit_fields": [
                "amount",
                "to_call",
                "pot_before_action",
                "min_raise",
                "legal_actions",
                "action_order",
            ],
            "derived_context_features": [
                "hand_action_order",
                "street_action_order",
                "facing_bet_or_raise",
                "call_price_ratio",
                "raise_pressure",
                "table_commitment_pressure",
            ],
            "current_delivery_blocker": False,
            "model_quality_risk": True,
            "target_row_values_are_labels_not_features": True,
            "boundary": "Derived context mitigates the current dataset gap but does not fully replace explicit decision-time betting labels.",
        },
        "stack_event_context_quality": {
            "endpoint": "/stack-event-context-quality.json",
            "description": "Documents how raw stack_events.csv changes are converted into decision-time pot, effective-stack, SPR, bet-size, and pressure features.",
            "risk_id": "raw_stack_events_require_decision_context_derivation",
            "root_cause": "stack_events.csv stores stack changes, not fully prepared decision-time policy features.",
            "implementation_module": "poker_agent.stack_context.build_stack_decision_context",
            "raw_event_boundary": "Raw stack events are retained as source data, not used directly as sufficient policy features.",
            "required_decision_context": ["pot", "effective_stack", "spr", "bet_size", "pressure"],
            "required_derived_features": [
                "pot",
                "stack",
                "to_call",
                "min_raise",
                "spr",
                "stack_to_pot",
                "table_commitment_pressure",
                "reconstructed_effective_stack",
                "reconstructed_spr_after_call",
                "reconstructed_current_street_bet_size",
                "reconstructed_call_pressure",
                "reconstructed_raise_pressure",
            ],
            "current_delivery_blocker": False,
            "model_quality_risk": True,
            "target_action_stack_delta_is_label_context_not_feature": True,
            "boundary": "Derived stack context mitigates the raw-event gap, but does not replace explicit instrumented pot/effective-stack/SPR labels.",
        },
        "bet_timing_calibration": {
            "endpoint": "/bet-timing-calibration.json",
            "description": "Documents that bet sizing and wait-time behavior are implemented, while timing label quality remains insufficient for final production human-likeness proof.",
            "timing_policy_type": "HEURISTIC_OR_TABLE_TEMPO_CALIBRATED",
            "timing_label_quality_status": "TIMING_LABEL_QUALITY_UNCERTAIN",
            "final_production_human_likeness_proof_allowed": False,
            "boundary": "wait_time_ms is produced and measured for delivery scope, but reviewed real human timing labels are still required for final high-realism timing claims.",
        },
        "llm_role_boundary": {
            "endpoint": "/llm-role-boundary.json",
            "description": "Separates the ambiguous LLM-based agent phrase into explicit implementation roles.",
            "term_status": "LLM_BASED_AGENT_IS_UMBRELLA_TERM",
            "controlled_layer_acceptance_status": "CONTROLLED_EVENT_CONTEXT_LAYER_APPROVED",
            "approved_delivery_scope": ["event_normalization", "decision_context"],
            "research_only_scope": ["candidate_ranking"],
            "excluded_delivery_scope": ["real_policy_agent", "fully_autonomous_poker_playing_llm_policy"],
            "fully_autonomous_poker_playing_llm_policy_status": "FULLY_AUTONOMOUS_LLM_POLICY_NOT_APPROVED",
            "fully_autonomous_poker_playing_llm_policy_approved": False,
            "fully_autonomous_policy_claim_allowed": False,
            "current_delivery_approval": "controlled_event_context_layer_only",
            "ambiguous_llm_agent_term_allowed": False,
            "role_disambiguation_required": True,
            "claim_validator": "poker_agent.llm_role_boundary.validate_llm_agent_claim",
            "unqualified_production_claim_allowed": False,
            "role_taxonomy": [
                "event_normalization",
                "decision_context",
                "candidate_ranking",
                "real_policy_agent",
            ],
            "role_types": {
                "event_normalization": "EVENT_NORMALIZER",
                "decision_context": "DECISION_CONTEXT_AGENT",
                "candidate_ranking": "CANDIDATE_RANKER",
                "real_policy_agent": "POLICY_AGENT",
            },
            "role_contracts": {
                "event_normalization": "Noisy OCR/dealer-log text -> validated event JSON; cannot emit poker policy actions.",
                "decision_context": "Structured game state + rules + legal actions -> constrained research decision suggestion; not production policy.",
                "candidate_ranking": "Candidate set -> candidate_id/confidence; no free-form policy generation.",
                "real_policy_agent": "Full game/session state -> final action/bet/timing; not implemented or approved in this delivery.",
            },
            "current_delivery_roles": [
                "event_normalization",
                "decision_context",
                "candidate_ranking_research_baseline",
            ],
            "not_current_delivery_role": "real_policy_agent",
            "autonomous_llm_policy_claim_allowed": False,
            "boundary": "LLM-based agent must be qualified by role; the delivered LLM work is not a fully autonomous poker-playing policy agent.",
        },
        "experimental_llm_policy": {
            "endpoint": "/llm-policy-experimental.json",
            "description": "Research-only LLM policy adapter with formal poker context, legal-action filtering, JSON validation, and deterministic fallback.",
            "status": "EXPERIMENTAL_LLM_POLICY_RESEARCH_ONLY",
            "role_type": "POLICY_AGENT",
            "approved_use": "offline_research_and_architecture_evaluation",
            "production_policy_approved": False,
            "autonomous_policy_claim_allowed": False,
            "served_by_predict_endpoint": False,
            "deployed_strategy_stack_affected": False,
            "current_delivery_blocker": False,
            "requires_stakeholder_approval_before_production": True,
            "guardrails": [
                "formal in-context learning",
                "legal-action filtering",
                "strict JSON output",
                "probability normalization",
                "confidence threshold",
                "deterministic fallback",
            ],
            "required_before_production": [
                "stakeholder architecture approval",
                "held-out action alignment",
                "macro F1 and balanced accuracy gate",
                "calibration ECE gate",
                "OpenSpiel agent-only self-play",
                "seed stability",
                "monitoring, rollback, and drift tracking",
            ],
            "boundary": "LLM policy code exists as an experimental adapter only; it is not the production-approved autonomous poker policy.",
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
        "phase2_selection_comparison": {
            "endpoint": "/phase2-selection-comparison.json",
            "description": (
                "Strict Phase 2 selection gate requiring LLM, supervised model, rule-based fallback, "
                "routed policy, and future RL agent to be compared on the same holdout and same simulation conditions."
            ),
            "required_candidates": [
                "llm_decision_agent",
                "supervised_model",
                "rule_based_fallback",
                "routed_policy_bundle",
                "future_rl_agent",
            ],
            "common_holdout_required": True,
            "common_simulation_required": True,
            "current_delivery_architecture": "routed_policy_bundle",
            "final_selection_claim_allowed_without_common_conditions": False,
            "claim_boundary": (
                "The routed policy bundle can remain the current delivery stack, but Phase 2 final "
                "architecture selection is blocked until every required candidate has the same holdout "
                "and same simulation evidence."
            ),
        },
        "autonomous_agent": {
            "capabilities_endpoint": "/agent/capabilities.json",
            "decision_endpoint": "/agent/decide",
            "session_endpoint": "/agent/sessions/{hand_id}",
            "settlement_endpoint": "/agent/sessions/{hand_id}/settle",
            "agent_type": "controlled_stateful_policy_agent",
            "lifecycle_controls": [
                "ordered hand-state observations",
                "legal-action enforcement",
                "idempotent event handling",
                "terminal hand settlement",
                "bounded simulation episodes",
            ],
            "execution_boundary": (
                "Simulation and API orchestration are implemented. Direct real-money client "
                "automation requires a separately approved environment adapter."
            ),
        },
        "phase3_open_spiel_arena": {
            "endpoint": "/phase3-open-spiel-arena.json",
            "arena_type": "agent_only_open_spiel_arena",
            "description": (
                "Tracks the Phase 3 LLM-vs-LLM OpenSpiel arena and separates executable arena "
                "readiness from real RL training proof."
            ),
            "rl_training_proof_required": True,
            "win_rate_claim_requires": [
                "real pyspiel runtime",
                "two trained Phase 1 policy artifacts",
                "agent-only table",
                "seed stability across at least five independent seeds",
                "long-run volume of at least 5000 episodes",
                "policy-update training completion",
            ],
            "current_claim_boundary": (
                "The arena code can be delivered as ready for measured execution. RL win-rate or "
                "production strategy-quality claims remain blocked until the full proof boundary passes."
            ),
        },
        "evaluation_metric_contract": {
            "endpoint": "/evaluation-metric-contract.json",
            "description": (
                "Defines the metric bundle required for strategy-quality claims. Accuracy alone is "
                "explicitly insufficient."
            ),
            "required_metric_families": [
                "action classification: accuracy, macro F1, balanced accuracy",
                "calibration: ECE, probability quality",
                "behavioral distribution: action-distribution divergence",
                "bet sizing: bet-size MAE or reviewed bet-size labels",
                "simulation return: win-rate and expected-value delta",
                "stability: seed-stability and long-run evidence",
            ],
            "claim_boundary": (
                "Final strategy-quality approval is blocked until the complete metric bundle passes. "
                "Current delivery is not blocked by this hardening boundary."
            ),
        },
        "test_execution_contract": {
            "endpoint": "/test-execution-contract.json",
            "description": (
                "Records validation execution status and keeps full-suite timeout transparency separate "
                "from delivery approval evidence."
            ),
            "approval_evidence": [
                "critical metric and delivery-boundary tests",
                "full delivery verifier",
                "ZIP contract",
            ],
            "not_approval_evidence": [
                "a timed-out full pytest run",
            ],
            "claim_boundary": (
                "A timed-out full pytest run must not be described as a passing full suite. Current "
                "delivery remains supported by critical tests and the full delivery verifier."
            ),
        },
        "human_likeness_evidence": {
            "endpoint": "/human-likeness-evidence.json",
            "description": (
                "Separates current-scope action-distribution similarity from full human-likeness proof."
            ),
            "required_behavior_dimensions": [
                "action distribution",
                "bet sizing",
                "timing",
                "position-based behavior",
                "street-level strategy",
            ],
            "claim_boundary": (
                "Action distribution can pass while full human-likeness remains unproven. Final "
                "human-likeness claims require bet-size, timing, position, and street-level validation."
            ),
        },
        "human_likeness_claim_gate": {
            "endpoint": "/human-likeness-claim-gate.json",
            "description": (
                "Final claim gate that blocks full human-likeness approval when only action-distribution "
                "similarity has passed."
            ),
            "claim": "FULL_HUMAN_LIKENESS",
            "decision_values": ["BLOCKED", "APPROVED"],
            "current_decision": "BLOCKED",
            "required_evidence_dimensions": [
                "action distribution",
                "bet sizing",
                "timing",
                "position-based behavior",
                "street-level strategy",
            ],
            "claim_boundary": (
                "The system may report current-scope action-distribution similarity, but it must not "
                "claim full human-likeness until all required behavior dimensions have reviewed evidence."
            ),
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
