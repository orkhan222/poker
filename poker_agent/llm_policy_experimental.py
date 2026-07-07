from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from poker_agent.action_planning import build_action_plan
from poker_agent.llm_decision_context import (
    CANONICAL_ACTIONS,
    build_decision_prompt,
    legal_actions_for_request,
    parse_decision_output,
)
from poker_agent.schemas import PredictionRequest, PredictionResponse


LLM_POLICY_EXPERIMENT_VERSION = "2026-07-07"
EXPERIMENTAL_POLICY_STATUS = "EXPERIMENTAL_LLM_POLICY_RESEARCH_ONLY"
NOT_PRODUCTION_APPROVED = "RESEARCH_ONLY_NOT_PRODUCTION_APPROVED"
POLICY_ROLE = "POLICY_AGENT"
CONTROLLED_WITH_GUARDRAILS = "CONTROLLED_LLM_POLICY_ADAPTER_WITH_GUARDRAILS"


class LLMPolicyBackend(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
    ) -> str:
        """Return a raw model completion for a structured chat prompt."""


@dataclass(frozen=True)
class ExperimentalLLMPolicyDecision:
    response: PredictionResponse
    prompt: dict[str, Any]
    raw_output: str
    fallback_used: bool
    validation_warnings: list[str]
    approval_status: str = NOT_PRODUCTION_APPROVED
    production_policy_approved: bool = False
    autonomous_policy_claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = self.response.to_dict()
        payload.update(
            {
                "llm_policy_status": EXPERIMENTAL_POLICY_STATUS,
                "approval_status": self.approval_status,
                "production_policy_approved": self.production_policy_approved,
                "autonomous_policy_claim_allowed": self.autonomous_policy_claim_allowed,
                "fallback_used": self.fallback_used,
                "validation_warnings": self.validation_warnings,
                "prompt_context_mode": self.prompt.get("context_mode"),
                "legal_actions": self.prompt.get("legal_actions", []),
            }
        )
        return payload


class ExperimentalLLMPolicyAdapter:
    """Research-only policy adapter with legal-action, JSON, and confidence guardrails."""

    def __init__(
        self,
        backend: LLMPolicyBackend,
        *,
        context_mode: str = "full_in_context",
        min_confidence: float = 0.35,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = 7,
    ) -> None:
        self.backend = backend
        self.context_mode = context_mode
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.max_tokens = max(32, int(max_tokens))
        self.temperature = max(float(temperature), 0.0)
        self.seed = seed

    def decide(self, request: PredictionRequest) -> ExperimentalLLMPolicyDecision:
        prompt = build_decision_prompt(request, self.context_mode)  # type: ignore[arg-type]
        prompt_payload = prompt.to_dict()
        raw_output = self.backend.complete(
            prompt.messages(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        parsed = parse_decision_output(raw_output, request)
        warnings = list(parsed.warnings)
        fallback_required = bool(warnings) or parsed.confidence < self.min_confidence

        if fallback_required:
            if parsed.confidence < self.min_confidence:
                warnings.append(
                    f"LLM confidence below threshold: {parsed.confidence:.4f} < {self.min_confidence:.4f}"
                )
            parsed = _fallback_response(request, warnings)

        return ExperimentalLLMPolicyDecision(
            response=parsed,
            prompt=prompt_payload,
            raw_output=raw_output,
            fallback_used=fallback_required,
            validation_warnings=warnings,
        )


class StaticLLMPolicyBackend:
    """Deterministic backend used for contract tests and offline smoke reports."""

    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int | None = None,
    ) -> str:
        return self.raw_output


def build_experimental_llm_policy_contract(project_root: Path | None = None) -> dict[str, Any]:
    sample_request = PredictionRequest(
        position="BTN",
        street="preflop",
        hole_cards=["Ah", "As"],
        board_cards=[],
        pot=6.5,
        to_call=3.0,
        stack=100.0,
        min_raise=6.0,
        player_count=6,
        betting_history=[
            {"player_position": "UTG", "action": "raise", "amount": 3.0, "street": "preflop"},
            {"player_position": "CO", "action": "call", "amount": 3.0, "street": "preflop"},
        ],
    )
    valid_backend = StaticLLMPolicyBackend(
        json.dumps(
            {
                "action": "raise",
                "probabilities": {
                    "fold": 0.0,
                    "check": 0.0,
                    "call": 0.2,
                    "bet": 0.0,
                    "raise": 0.8,
                },
                "confidence": 0.8,
                "bet_size": 9.0,
                "reason_code": "hand_strength",
            },
            sort_keys=True,
        )
    )
    invalid_backend = StaticLLMPolicyBackend(
        json.dumps(
            {
                "action": "check",
                "probabilities": {
                    "fold": 0.1,
                    "check": 0.6,
                    "call": 0.1,
                    "bet": 0.1,
                    "raise": 0.1,
                },
                "confidence": 0.25,
                "bet_size": 0,
                "reason_code": "uncertain",
            },
            sort_keys=True,
        )
    )
    valid_decision = ExperimentalLLMPolicyAdapter(valid_backend).decide(sample_request)
    fallback_decision = ExperimentalLLMPolicyAdapter(invalid_backend).decide(sample_request)

    payload: dict[str, Any] = {
        "version": LLM_POLICY_EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": EXPERIMENTAL_POLICY_STATUS,
        "adapter_type": CONTROLLED_WITH_GUARDRAILS,
        "role_type": POLICY_ROLE,
        "approved_use": "offline_research_and_architecture_evaluation",
        "production_policy_approved": False,
        "autonomous_policy_claim_allowed": False,
        "served_by_predict_endpoint": False,
        "deployed_strategy_stack_affected": False,
        "current_delivery_blocker": False,
        "requires_stakeholder_approval_before_production": True,
        "purpose": (
            "Provide a controlled LLM policy adapter for experiments where the model receives "
            "formal poker context, legal actions, and a strict JSON response contract."
        ),
        "must_not_be_presented_as": [
            "production-approved autonomous poker-playing LLM policy",
            "replacement for the deployed routed strategy stack",
            "final competitive poker engine",
        ],
        "guardrails": {
            "formal_in_context_learning_required": True,
            "legal_action_filtering_required": True,
            "strict_json_output_required": True,
            "probability_normalization_required": True,
            "confidence_threshold_required": True,
            "deterministic_fallback_required": True,
            "schema_bypass_allowed": False,
            "unconstrained_action_generation_allowed": False,
        },
        "required_gates_before_production": [
            "stakeholder_architecture_approval",
            "heldout_action_alignment",
            "macro_f1_and_balanced_accuracy_gate",
            "calibration_ece_gate",
            "bet_size_and_timing_revalidation",
            "open_spiel_agent_only_self_play",
            "seed_stability",
            "monitoring_rollback_and_drift_tracking",
        ],
        "sample_decisions": {
            "valid_llm_output": valid_decision.to_dict(),
            "invalid_or_low_confidence_output": fallback_decision.to_dict(),
        },
        "implementation": {
            "module": "poker_agent.llm_policy_experimental",
            "adapter": "ExperimentalLLMPolicyAdapter",
            "backend_protocol": "LLMPolicyBackend",
            "prompt_builder": "poker_agent.llm_decision_context.build_decision_prompt",
            "parser": "poker_agent.llm_decision_context.parse_decision_output",
            "endpoint": "/llm-policy-experimental.json",
        },
    }
    payload["proof_cases"] = build_experimental_llm_policy_proof_cases(payload)
    payload["invariants"] = validate_experimental_llm_policy_contract(payload)
    if not all(case["passed"] for case in payload["proof_cases"]):
        payload["invariants"]["status"] = "FAIL"
        payload["invariants"]["violations"].append("experimental_llm_policy_proof_cases_must_pass")
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_experimental_llm_policy_contract(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    guardrails = payload.get("guardrails") or {}
    samples = payload.get("sample_decisions") or {}

    if payload.get("status") != EXPERIMENTAL_POLICY_STATUS:
        violations.append("experimental_llm_policy_status_must_be_research_only")
    if payload.get("role_type") != POLICY_ROLE:
        violations.append("experimental_llm_policy_role_type_must_be_policy_agent")
    if payload.get("production_policy_approved") is not False:
        violations.append("experimental_llm_policy_must_not_be_production_approved")
    if payload.get("autonomous_policy_claim_allowed") is not False:
        violations.append("experimental_llm_policy_autonomous_claim_must_be_blocked")
    if payload.get("served_by_predict_endpoint") is not False:
        violations.append("experimental_llm_policy_must_not_be_served_by_predict")
    if payload.get("deployed_strategy_stack_affected") is not False:
        violations.append("experimental_llm_policy_must_not_affect_deployed_stack")
    if payload.get("current_delivery_blocker") is not False:
        violations.append("experimental_llm_policy_must_not_block_current_delivery")
    if payload.get("requires_stakeholder_approval_before_production") is not True:
        violations.append("experimental_llm_policy_requires_stakeholder_approval")
    for required_guardrail in (
        "formal_in_context_learning_required",
        "legal_action_filtering_required",
        "strict_json_output_required",
        "probability_normalization_required",
        "confidence_threshold_required",
        "deterministic_fallback_required",
    ):
        if guardrails.get(required_guardrail) is not True:
            violations.append(f"missing_guardrail:{required_guardrail}")
    if guardrails.get("schema_bypass_allowed") is not False:
        violations.append("schema_bypass_must_remain_blocked")
    if guardrails.get("unconstrained_action_generation_allowed") is not False:
        violations.append("unconstrained_action_generation_must_remain_blocked")
    for required_gate in (
        "heldout_action_alignment",
        "calibration_ece_gate",
        "open_spiel_agent_only_self_play",
        "seed_stability",
    ):
        if required_gate not in set(payload.get("required_gates_before_production") or []):
            violations.append(f"missing_required_production_gate:{required_gate}")
    valid_sample = samples.get("valid_llm_output") or {}
    fallback_sample = samples.get("invalid_or_low_confidence_output") or {}
    if valid_sample.get("production_policy_approved") is not False:
        violations.append("valid_sample_must_not_mark_policy_approved")
    if valid_sample.get("autonomous_policy_claim_allowed") is not False:
        violations.append("valid_sample_must_not_allow_autonomous_claim")
    if fallback_sample.get("fallback_used") is not True:
        violations.append("invalid_sample_must_use_fallback")
    if fallback_sample.get("production_policy_approved") is not False:
        violations.append("fallback_sample_must_not_mark_policy_approved")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def build_experimental_llm_policy_proof_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [_proof_case("base_contract_is_valid", payload, "PASS")]

    mutated = deepcopy(payload)
    mutated["production_policy_approved"] = True
    cases.append(_proof_case("blocks_production_approval_without_gates", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["served_by_predict_endpoint"] = True
    mutated["deployed_strategy_stack_affected"] = True
    cases.append(_proof_case("blocks_serving_by_public_predict_endpoint", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["autonomous_policy_claim_allowed"] = True
    mutated["guardrails"]["unconstrained_action_generation_allowed"] = True
    cases.append(_proof_case("blocks_autonomous_unconstrained_policy_claim", mutated, "FAIL"))

    mutated = deepcopy(payload)
    mutated["requires_stakeholder_approval_before_production"] = False
    mutated["required_gates_before_production"] = []
    cases.append(_proof_case("blocks_missing_approval_and_gate_contract", mutated, "FAIL"))

    return cases


def write_experimental_llm_policy_contract(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_experimental_llm_policy_contract(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_experimental_llm_policy_markdown(payload), encoding="utf-8")
    return payload


def render_experimental_llm_policy_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Experimental LLM Policy Adapter",
        "",
        f"- Status: `{payload['status']}`",
        f"- Role type: `{payload['role_type']}`",
        f"- Production policy approved: `{payload['production_policy_approved']}`",
        f"- Autonomous policy claim allowed: `{payload['autonomous_policy_claim_allowed']}`",
        f"- Served by `/predict`: `{payload['served_by_predict_endpoint']}`",
        f"- Deployed strategy stack affected: `{payload['deployed_strategy_stack_affected']}`",
        f"- Current delivery blocker: `{payload['current_delivery_blocker']}`",
        "",
        "## Purpose",
        "",
        payload["purpose"],
        "",
        "## Guardrails",
        "",
    ]
    for name, value in payload["guardrails"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Required Gates Before Production", ""])
    lines.extend(f"- {gate}" for gate in payload["required_gates_before_production"])
    lines.extend(["", "## Must Not Be Presented As", ""])
    lines.extend(f"- {item}" for item in payload["must_not_be_presented_as"])
    lines.extend(["", "## Proof Cases", ""])
    for case in payload.get("proof_cases") or []:
        lines.append(
            f"- `{case['name']}`: expected `{case['expected_status']}`, "
            f"observed `{case['observed_status']}`, passed `{case['passed']}`"
        )
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _fallback_response(request: PredictionRequest, warnings: list[str]) -> PredictionResponse:
    legal_actions = set(legal_actions_for_request(request))
    if request.to_call > 0:
        values = {"fold": 0.46, "call": 0.36, "raise": 0.18, "check": 0.0, "bet": 0.0}
        action = "call" if len(request.hole_cards) >= 2 else "fold"
    else:
        values = {"check": 0.64, "bet": 0.36, "fold": 0.0, "call": 0.0, "raise": 0.0}
        action = "check"
    values = {name: (value if name in legal_actions else 0.0) for name, value in values.items()}
    total = sum(values.values()) or 1.0
    probabilities = {name: value / total for name, value in values.items()}
    action = action if action in legal_actions else max(probabilities, key=probabilities.get)
    confidence = max(probabilities.values(), default=0.0)
    plan = build_action_plan(request, action, confidence)
    return PredictionResponse(
        action=action,
        probabilities={action_name: probabilities.get(action_name, 0.0) for action_name in CANONICAL_ACTIONS},
        confidence=confidence,
        bet_size=plan.bet_size,
        wait_time_ms=plan.wait_time_ms,
        sizing_method=plan.sizing_method,
        timing_method=plan.timing_method,
        model_status="experimental_llm_policy_guarded_fallback",
        warnings=warnings,
    )


def _proof_case(name: str, candidate: dict[str, Any], expected_status: str) -> dict[str, Any]:
    observed = validate_experimental_llm_policy_contract(candidate)
    return {
        "name": name,
        "expected_status": expected_status,
        "observed_status": observed["status"],
        "passed": observed["status"] == expected_status,
        "violations": observed["violations"],
    }
