from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from poker_agent.agents import RuleBasedAgent
from poker_agent.action_planning import build_action_plan
from poker_agent.llm_decision_context import (
    CANONICAL_ACTIONS,
    CONTEXT_MODE_SUMMARY,
    REASON_CODES,
    ContextMode,
    DecisionPrompt,
    build_decision_prompt,
    legal_actions_for_request,
    parse_decision_output,
)
from poker_agent.schemas import PredictionRequest


@dataclass(frozen=True)
class DecisionExample:
    example_id: str
    request: PredictionRequest
    expected_action: str


@dataclass(frozen=True)
class Generation:
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    peak_memory_mb: float = 0.0


@dataclass(frozen=True)
class OutputInspection:
    json_valid: bool
    schema_valid: bool
    legal_action: bool
    probability_valid: bool
    raw_action: str


class DecisionProvider(Protocol):
    name: str
    quality_claim_allowed: bool

    def generate(self, prompt: DecisionPrompt, request: PredictionRequest) -> Generation:
        ...


class RuleBaselineProvider:
    """Deterministic pipeline smoke provider; not an LLM quality benchmark."""

    name = "rule_baseline"
    quality_claim_allowed = False

    def __init__(self) -> None:
        self._agent = RuleBasedAgent()

    def generate(self, prompt: DecisionPrompt, request: PredictionRequest) -> Generation:
        del prompt
        started = time.perf_counter()
        response = self._agent.predict(request).to_dict()
        response["reason_code"] = "uncertain"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Generation(text=json.dumps(response, sort_keys=True), latency_ms=elapsed_ms)


class TransformersDecisionProvider:
    name = "transformers"
    quality_claim_allowed = True

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 192,
        torch_dtype: str = "auto",
        quantization: str = "none",
        max_gpu_memory_mb: int = 0,
        max_cpu_memory_gb: int = 12,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers before running the LLM benchmark") from exc

        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        dtype = _resolve_torch_dtype(torch, torch_dtype, resolved_device)
        self._torch = torch
        self._max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if quantization == "4bit_nf4":
            if resolved_device != "cuda":
                raise RuntimeError("4-bit NF4 inference requires a CUDA device")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("The installed transformers build has no BitsAndBytesConfig") from exc
            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": "auto",
                    "max_memory": {
                        0: f"{max_gpu_memory_mb or 2400}MiB",
                        "cpu": f"{max_cpu_memory_gb}GiB",
                    },
                }
            )
        elif quantization == "none":
            model_kwargs["device_map"] = {"": resolved_device}
        else:
            raise ValueError(f"Unsupported quantization mode: {quantization}")
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._model.eval()
        self._input_device = self._model.get_input_embeddings().weight.device
        self._cuda_metrics = self._input_device.type == "cuda"
        self.name = f"transformers:{model_id}:{quantization}"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(self, prompt: DecisionPrompt, request: PredictionRequest) -> Generation:
        del request
        input_text = _format_messages(self._tokenizer, prompt.messages())
        encoded = self._tokenizer(input_text, return_tensors="pt")
        encoded = {name: value.to(self._input_device) for name, value in encoded.items()}
        if self._cuda_metrics:
            self._torch.cuda.reset_peak_memory_stats()
            self._torch.cuda.synchronize()
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        if self._cuda_metrics:
            self._torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated = output[0][encoded["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        peak_memory_mb = (
            float(self._torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if self._cuda_metrics
            else 0.0
        )
        return Generation(
            text=text,
            latency_ms=elapsed_ms,
            prompt_tokens=int(encoded["input_ids"].shape[1]),
            completion_tokens=int(generated.shape[0]),
            peak_memory_mb=peak_memory_mb,
        )


class TransformersCandidateRanker(TransformersDecisionProvider):
    """Ranks legal actions by conditional likelihood and emits validated JSON."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
        quantization: str = "none",
        max_gpu_memory_mb: int = 0,
        max_cpu_memory_gb: int = 12,
        score_reduction: str = "mean",
        probability_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            model_id,
            device=device,
            max_new_tokens=1,
            torch_dtype=torch_dtype,
            quantization=quantization,
            max_gpu_memory_mb=max_gpu_memory_mb,
            max_cpu_memory_gb=max_cpu_memory_gb,
        )
        if score_reduction not in {"mean", "sum"}:
            raise ValueError("score_reduction must be mean or sum")
        if probability_temperature <= 0:
            raise ValueError("probability_temperature must be positive")
        self._score_reduction = score_reduction
        self._probability_temperature = probability_temperature
        self.name = f"transformers_candidate_ranker:{model_id}:{quantization}"

    def generate(self, prompt: DecisionPrompt, request: PredictionRequest) -> Generation:
        legal_actions = legal_actions_for_request(request)
        ranking_text = _format_messages(self._tokenizer, _ranking_messages(prompt))
        prompt_ids = self._tokenizer(
            ranking_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        candidate_ids = {
            action: self._tokenizer(
                action,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            for action in legal_actions
        }
        if self._cuda_metrics:
            self._torch.cuda.reset_peak_memory_stats()
            self._torch.cuda.synchronize()
        started = time.perf_counter()
        losses = self._score_candidates(prompt_ids, candidate_ids)
        if self._cuda_metrics:
            self._torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        probabilities = _candidate_probabilities(
            losses,
            temperature=self._probability_temperature,
        )
        action = max(probabilities, key=probabilities.get)
        all_probabilities = {name: probabilities.get(name, 0.0) for name in CANONICAL_ACTIONS}
        confidence = all_probabilities[action]
        plan = build_action_plan(request, action, confidence)
        payload = {
            "action": action,
            "probabilities": all_probabilities,
            "confidence": confidence,
            "bet_size": plan.bet_size,
            "reason_code": _reason_code(request),
            "candidate_losses": losses,
        }
        peak_memory_mb = (
            float(self._torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if self._cuda_metrics
            else 0.0
        )
        return Generation(
            text=json.dumps(payload, sort_keys=True),
            latency_ms=elapsed_ms,
            prompt_tokens=int(prompt_ids.shape[0]),
            completion_tokens=max(int(ids.shape[0]) for ids in candidate_ids.values()),
            peak_memory_mb=peak_memory_mb,
        )

    def _score_candidates(
        self,
        prompt_ids: Any,
        candidate_ids: dict[str, Any],
    ) -> dict[str, float]:
        prompt_length = int(prompt_ids.shape[0])
        losses: dict[str, float] = {}
        for action, ids in candidate_ids.items():
            sequence = self._torch.cat([prompt_ids, ids], dim=0)
            input_ids = sequence.unsqueeze(0).to(self._input_device)
            attention_mask = self._torch.ones_like(input_ids)
            with self._torch.inference_mode():
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            candidate_length = int(ids.shape[0])
            prediction_positions = self._torch.arange(
                prompt_length - 1,
                prompt_length + candidate_length - 1,
                device=logits.device,
            )
            selected_logits = logits[0, prediction_positions, :].float()
            targets = ids.to(logits.device)
            values = self._torch.nn.functional.cross_entropy(
                selected_logits,
                targets,
                reduction="none",
            )
            reduced = values.sum() if self._score_reduction == "sum" else values.mean()
            losses[action] = float(reduced.item())
            del logits, selected_logits, input_ids, attention_mask, sequence
        return losses


def load_decision_examples(path: Path, max_examples: int = 0) -> list[DecisionExample]:
    examples: list[DecisionExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            state = payload.get("game_state")
            if not isinstance(state, dict):
                raise ValueError(f"Line {line_number}: game_state must be an object")
            expected_action = str(payload.get("expected_action") or "").strip().lower()
            request = PredictionRequest.from_dict(state)
            if expected_action not in legal_actions_for_request(request):
                raise ValueError(
                    f"Line {line_number}: expected action {expected_action!r} is illegal for the state"
                )
            examples.append(
                DecisionExample(
                    example_id=str(payload.get("id") or f"line-{line_number}"),
                    request=request,
                    expected_action=expected_action,
                )
            )
            if max_examples and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"No decision examples found in {path}")
    return examples


def inspect_decision_output(raw_text: str, request: PredictionRequest) -> OutputInspection:
    payload = _first_json_object(raw_text)
    if payload is None:
        return OutputInspection(False, False, False, False, "")
    action = str(payload.get("action") or "").strip().lower()
    if action == "all_in":
        action = "raise"
    legal_action = action in legal_actions_for_request(request)
    probabilities = payload.get("probabilities")
    probability_valid = _probability_contract_valid(probabilities)
    confidence_valid = _bounded_number(payload.get("confidence"))
    bet_size_valid = _nonnegative_number(payload.get("bet_size"))
    reason_code = str(payload.get("reason_code") or "").strip().lower()
    schema_valid = (
        bool(action)
        and probability_valid
        and confidence_valid
        and bet_size_valid
        and reason_code in REASON_CODES
    )
    return OutputInspection(True, schema_valid, legal_action, probability_valid, action)


def benchmark_context_modes(
    examples: Sequence[DecisionExample],
    provider: DecisionProvider,
    *,
    context_modes: Sequence[ContextMode],
    seed: int = 42,
    dataset_kind: str = "smoke",
) -> dict[str, Any]:
    random.seed(seed)
    systems: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for mode in context_modes:
        expected: list[str] = []
        predicted: list[str] = []
        latencies: list[float] = []
        schema_valid = 0
        json_valid = 0
        legal = 0
        fallbacks = 0
        prompt_tokens = 0
        completion_tokens = 0
        peak_memory_mb = 0.0
        for example in examples:
            prompt = build_decision_prompt(example.request, mode)
            generation = provider.generate(prompt, example.request)
            inspection = inspect_decision_output(generation.text, example.request)
            validated = parse_decision_output(generation.text, example.request)
            fallback_used = bool(validated.warnings)
            expected.append(example.expected_action)
            predicted.append(validated.action)
            latencies.append(generation.latency_ms)
            schema_valid += int(inspection.schema_valid)
            json_valid += int(inspection.json_valid)
            legal += int(inspection.legal_action)
            fallbacks += int(fallback_used)
            prompt_tokens += generation.prompt_tokens
            completion_tokens += generation.completion_tokens
            peak_memory_mb = max(peak_memory_mb, generation.peak_memory_mb)
            prediction_rows.append(
                {
                    "id": example.example_id,
                    "context_mode": mode,
                    "expected_action": example.expected_action,
                    "raw_action": inspection.raw_action,
                    "validated_action": validated.action,
                    "correct": validated.action == example.expected_action,
                    "json_valid": inspection.json_valid,
                    "schema_valid": inspection.schema_valid,
                    "legal_action": inspection.legal_action,
                    "fallback_used": fallback_used,
                    "latency_ms": generation.latency_ms,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "raw_response": generation.text,
                }
            )
        count = len(examples)
        systems[mode] = {
            "context_description": CONTEXT_MODE_SUMMARY[mode],
            "examples": count,
            "accuracy": _accuracy(expected, predicted),
            "macro_f1": _macro_f1(expected, predicted),
            "json_valid_rate": json_valid / count,
            "schema_valid_rate": schema_valid / count,
            "legal_action_rate": legal / count,
            "fallback_rate": fallbacks / count,
            "average_latency_ms": sum(latencies) / count,
            "p95_latency_ms": _percentile(latencies, 0.95),
            "average_prompt_tokens": prompt_tokens / count,
            "average_completion_tokens": completion_tokens / count,
            "peak_memory_mb": peak_memory_mb,
        }
    comparison_allowed = provider.quality_claim_allowed and dataset_kind != "smoke"
    quality_claim_allowed = (
        provider.quality_claim_allowed and dataset_kind == "reviewed_human_holdout"
    )
    provisional_best_mode = (
        max(
            systems,
            key=lambda name: (
                systems[name]["macro_f1"],
                systems[name]["schema_valid_rate"],
                -systems[name]["average_latency_ms"],
            ),
        )
        if comparison_allowed
        else None
    )
    best_mode = (
        provisional_best_mode
        if quality_claim_allowed
        else None
    )
    return {
        "benchmark_version": "2026-06-25",
        "provider": provider.name,
        "dataset_kind": dataset_kind,
        "quality_claim_allowed": quality_claim_allowed,
        "comparison_allowed": comparison_allowed,
        "quality_claim_note": (
            "Metrics may be used for final model comparison."
            if quality_claim_allowed
            else (
                "Metrics are provisional because labels were reconstructed from human logs and have not been manually reviewed."
                if comparison_allowed
                else "This run validates infrastructure only; it must not be presented as LLM policy quality."
            )
        ),
        "seed": seed,
        "context_modes": list(context_modes),
        "best_mode": best_mode,
        "provisional_best_mode": provisional_best_mode,
        "systems": systems,
        "predictions": prediction_rows,
    }


def write_benchmark_outputs(
    result: dict[str, Any],
    *,
    out_path: Path,
    predictions_path: Path,
    report_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "predictions"}
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(render_benchmark_markdown(summary), encoding="utf-8")


def render_benchmark_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LLM Decision Context Ablation",
        "",
        f"- Provider: `{result['provider']}`",
        f"- Dataset kind: `{result['dataset_kind']}`",
        f"- Quality claim allowed: `{result['quality_claim_allowed']}`",
        f"- Best context mode: `{result['best_mode'] or 'not_applicable'}`",
        f"- Provisional best context mode: `{result.get('provisional_best_mode') or 'not_applicable'}`",
        "",
        result["quality_claim_note"],
        "",
        "| Context | Accuracy | Macro F1 | Schema valid | Legal action | Fallback | Avg latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["systems"].items():
        lines.append(
            f"| `{name}` | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['schema_valid_rate']:.4f} | {metrics['legal_action_rate']:.4f} | "
            f"{metrics['fallback_rate']:.4f} | {metrics['average_latency_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _probability_contract_valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or any(action not in raw for action in CANONICAL_ACTIONS):
        return False
    values: list[float] = []
    for action in CANONICAL_ACTIONS:
        try:
            value = float(raw[action])
        except (TypeError, ValueError):
            return False
        if not 0.0 <= value <= 1.0:
            return False
        values.append(value)
    return math.isclose(sum(values), 1.0, abs_tol=1e-3)


def _bounded_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= number <= 1.0


def _nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def _accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    return sum(left == right for left, right in zip(expected, predicted)) / len(expected)


def _macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(e == label and p == label for e, p in zip(expected, predicted))
        false_positive = sum(e != label and p == label for e, p in zip(expected, predicted))
        false_negative = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _resolve_torch_dtype(torch: Any, requested: str, device: str) -> Any:
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def _format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)


def _ranking_messages(prompt: DecisionPrompt) -> list[dict[str, str]]:
    system = prompt.system_context.split("\nRequired JSON schema:", 1)[0]
    excluded_prefixes = (
        "Return JSON only.",
        "Output constraints:",
        "- The action must",
        "- Probabilities must",
        "- Confidence must",
        "- Bet size must",
    )
    system_lines = [
        line
        for line in system.splitlines()
        if not any(line.startswith(prefix) for prefix in excluded_prefixes)
    ]
    system_lines.extend(
        [
            "",
            "Candidate-ranking contract:",
            "- Score only the candidate action supplied after the game state.",
            "- Higher likelihood must indicate a better action for this exact state.",
            "- Use the supplied rules and decision guidelines; do not invent hidden information.",
        ]
    )
    user = prompt.user_context.replace(
        "Evaluate this game state and return the JSON decision.",
        "Evaluate this game state.",
        1,
    )
    user += "\n\nCandidate action:"
    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": user},
    ]


def _candidate_probabilities(
    losses: dict[str, float],
    *,
    temperature: float,
) -> dict[str, float]:
    scores = {action: -loss / temperature for action, loss in losses.items()}
    maximum = max(scores.values())
    exponentials = {action: math.exp(score - maximum) for action, score in scores.items()}
    total = sum(exponentials.values()) or 1.0
    probabilities = {action: value / total for action, value in exponentials.items()}
    correction_action = max(probabilities, key=probabilities.get)
    probabilities[correction_action] += 1.0 - sum(probabilities.values())
    return probabilities


def _reason_code(request: PredictionRequest) -> str:
    if request.to_call > 0:
        return "pot_odds"
    if request.hole_cards:
        return "hand_strength"
    if request.position:
        return "position"
    return "uncertain"
