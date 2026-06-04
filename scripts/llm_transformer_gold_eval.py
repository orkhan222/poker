from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.features import normalize_action, parse_cards, safe_float, safe_int
from poker_agent.schemas import VALID_ACTIONS
from scripts.llm_event_benchmark import LABELS
from scripts.llm_event_extraction import ExtractedEvent
from scripts.llm_event_gold_eval import load_gold, score_system, strict_schema_rules

VALID_EVENT_TYPES = set(LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local instruction model on the gold event fixture")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompts", required=True, help="Comma-separated prompt files")
    parser.add_argument("--system-names", required=True, help="Comma-separated names matching prompt files")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--ranking-prompt", required=True, type=Path)
    parser.add_argument("--ranking-system-name", required=True)
    parser.add_argument("--calibrated-ranking-system-name", required=True)
    parser.add_argument("--candidate-labels", required=True)
    parser.add_argument("--candidate-score-reduction", required=True, choices=("mean", "sum"))
    parser.add_argument("--calibration-record", required=True)
    parser.add_argument("--hybrid-system-name", required=True)
    parser.add_argument("--hybrid-fallback-system", required=True)
    parser.add_argument("--hybrid-routed-event-names", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--max-examples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def first_json_object(text: str) -> dict[str, Any] | None:
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


def to_event(payload: dict[str, Any] | None, *, line_number: int, provider: str, raw: str) -> ExtractedEvent:
    if payload is None:
        return ExtractedEvent(
            source_file="gold",
            line_number=line_number,
            frame_id=0,
            provider=provider,
            extracted_type="unmatched",
            confidence=0.0,
            evidence=raw[:500],
        )
    event_type = str(payload.get("event_type") or payload.get("extracted_type") or "unmatched").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        event_type = "unmatched"
    action = normalize_action(str(payload.get("action") or "")) or None
    if action == "all_in":
        action = "raise"
    if action not in VALID_ACTIONS:
        action = None
    amount_raw = payload.get("amount")
    amount = None if amount_raw in (None, "", "null") else safe_float(amount_raw, default=0.0)
    cards = parse_cards(payload.get("cards"))
    confidence = safe_float(payload.get("confidence"), default=0.0)
    return ExtractedEvent(
        source_file="gold",
        line_number=line_number,
        frame_id=safe_int(payload.get("frame_id")),
        provider=provider,
        extracted_type=event_type,
        player_position=str(payload.get("player_position")) if payload.get("player_position") is not None else None,
        action=action,
        amount=amount,
        cards=cards,
        confidence=max(0.0, min(1.0, confidence)),
        evidence=raw[:500],
    )


def amount_from_payload(payload: dict[str, Any]) -> float | None:
    for key in ("amount", "diff", "stack"):
        value = payload.get(key)
        if value not in (None, ""):
            return safe_float(value, default=0.0)
    return None


def event_from_ranked_label(
    row: dict[str, Any],
    *,
    label: str,
    provider: str,
    confidence: float,
    evidence: str,
) -> ExtractedEvent:
    record = row["record"]
    payload = record.get("event_value") if isinstance(record.get("event_value"), dict) else {}
    action = normalize_action(str(payload.get("value") or ""))
    if action == "all_in":
        action = "raise"
    if action not in VALID_ACTIONS:
        action = None
    return ExtractedEvent(
        source_file="gold",
        line_number=row["line_number"],
        frame_id=safe_int(record.get("frame_id")),
        provider=provider,
        extracted_type=label,
        player_position=str(payload.get("player_position")) if payload.get("player_position") is not None else None,
        action=action if label == "player_action" else None,
        amount=amount_from_payload(payload) if label in {"player_action", "stack_update"} else None,
        cards=parse_cards(payload.get("cards")) if label == "card_update" else [],
        raw_event_name=str(record.get("event_name") or ""),
        confidence=confidence,
        evidence=evidence[:500],
    )


def load_model(model_id: str, device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers before running this experiment") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return torch, tokenizer, model


def format_input(tokenizer: Any, prompt: str, record: dict[str, Any]) -> str:
    user_text = "Input record:\n" + json.dumps(record, sort_keys=True) + "\nReturn only one JSON object."
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\n\n" + user_text


def format_ranking_input(tokenizer: Any, prompt: str, record: dict[str, Any]) -> str:
    user_text = "Input record:\n" + json.dumps(record, sort_keys=True) + "\nEvent type:"
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\n\n" + user_text + "\n"


def run_prompt(
    rows: list[dict[str, Any]],
    *,
    name: str,
    prompt: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
) -> tuple[list[ExtractedEvent], list[dict[str, Any]], float, float]:
    predictions: list[ExtractedEvent] = []
    raw_rows: list[dict[str, Any]] = []
    parsed_count = 0
    started = time.perf_counter()
    for row in rows:
        input_text = format_input(tokenizer, prompt, row["record"])
        encoded = tokenizer(input_text, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        payload = first_json_object(raw)
        parsed_count += int(payload is not None)
        event = to_event(payload, line_number=row["line_number"], provider=name, raw=raw)
        predictions.append(event)
        raw_rows.append({"id": row["id"], "expected": row["expected"], "raw_response": raw, "parsed": asdict(event)})
    elapsed = time.perf_counter() - started
    valid_json_rate = parsed_count / len(rows) if rows else 0.0
    return predictions, raw_rows, elapsed, valid_json_rate


def candidate_loss(
    *,
    prompt_ids: Any,
    candidate: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    reduction: str,
) -> float:
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    active = shift_labels.ne(-100)
    token_losses = torch.nn.functional.cross_entropy(
        shift_logits[active],
        shift_labels[active],
        reduction="none",
    )
    if reduction == "sum":
        return float(token_losses.sum().item())
    return float(token_losses.mean().item())


def run_candidate_ranker(
    rows: list[dict[str, Any]],
    *,
    name: str,
    prompt: str,
    candidate_labels: list[str],
    reduction: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    calibration_record: dict[str, Any] | None = None,
) -> tuple[list[ExtractedEvent], list[dict[str, Any]], float]:
    invalid = sorted(set(candidate_labels) - VALID_EVENT_TYPES)
    if invalid:
        raise ValueError(f"Unsupported candidate labels: {invalid}")
    predictions: list[ExtractedEvent] = []
    raw_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    prior_losses: dict[str, float] = {}
    if calibration_record is not None:
        calibration_text = format_ranking_input(tokenizer, prompt, calibration_record)
        calibration_ids = tokenizer(
            calibration_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        prior_losses = {
            label: candidate_loss(
                prompt_ids=calibration_ids,
                candidate=label,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                reduction=reduction,
            )
            for label in candidate_labels
        }
    for row in rows:
        input_text = format_ranking_input(tokenizer, prompt, row["record"])
        prompt_ids = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        losses = {
            label: candidate_loss(
                prompt_ids=prompt_ids,
                candidate=label,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                reduction=reduction,
            )
            for label in candidate_labels
        }
        calibrated_losses = {
            label: loss - prior_losses.get(label, 0.0)
            for label, loss in losses.items()
        }
        selected = min(calibrated_losses, key=calibrated_losses.get)
        scores = {label: math.exp(-loss) for label, loss in calibrated_losses.items()}
        score_total = sum(scores.values()) or 1.0
        confidence = scores[selected] / score_total
        evidence = json.dumps(
            {
                "selected": selected,
                "losses": losses,
                "prior_losses": prior_losses,
                "calibrated_losses": calibrated_losses,
            },
            sort_keys=True,
        )
        event = event_from_ranked_label(
            row,
            label=selected,
            provider=name,
            confidence=confidence,
            evidence=evidence,
        )
        predictions.append(event)
        raw_rows.append(
            {
                "id": row["id"],
                "expected": row["expected"],
                "candidate_losses": losses,
                "candidate_prior_losses": prior_losses,
                "candidate_calibrated_losses": calibrated_losses,
                "selected": selected,
                "parsed": asdict(event),
            }
        )
    return predictions, raw_rows, time.perf_counter() - started


def run_schema_routed_hybrid(
    rows: list[dict[str, Any]],
    *,
    name: str,
    fallback_name: str,
    fallback_predictions: list[ExtractedEvent],
    routed_event_names: set[str],
) -> tuple[list[ExtractedEvent], list[dict[str, Any]], dict[str, Any]]:
    predictions: list[ExtractedEvent] = []
    raw_rows: list[dict[str, Any]] = []
    routed_count = 0
    fallback_correct = 0
    fallback_count = 0
    for row, fallback in zip(rows, fallback_predictions):
        event_name = str(row["record"].get("event_name") or "")
        if event_name in routed_event_names:
            routed = strict_schema_rules(row)
            event = ExtractedEvent(**{**asdict(routed), "provider": name})
            source = "schema_router"
            routed_count += 1
        else:
            event = ExtractedEvent(**{**asdict(fallback), "provider": name})
            source = fallback_name
            fallback_count += 1
            fallback_correct += int(event.extracted_type == row["expected"]["event_type"])
        predictions.append(event)
        raw_rows.append(
            {
                "id": row["id"],
                "expected": row["expected"],
                "route": source,
                "parsed": asdict(event),
            }
        )
    total = len(rows)
    routing = {
        "routed_event_names": sorted(routed_event_names),
        "router_count": routed_count,
        "router_coverage": routed_count / total if total else 0.0,
        "llm_fallback_system": fallback_name,
        "llm_fallback_count": fallback_count,
        "llm_fallback_rate": fallback_count / total if total else 0.0,
        "llm_fallback_accuracy": fallback_correct / fallback_count if fallback_count else 0.0,
    }
    return predictions, raw_rows, routing


def build_report(results: dict[str, Any]) -> str:
    rows = [
        "| System | Event accuracy | Macro F1 | Action exact | Card exact | Amount exact | LLM fallback | Seconds/example |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in results["systems"].items():
        rows.append(
            f"| {name} | {metrics['event_type']['accuracy']:.4f} | {metrics['event_type']['macro_f1']:.4f} | "
            f"{metrics['action_exact_match']:.4f} | {metrics['card_exact_match']:.4f} | {metrics['amount_exact_match']:.4f} | "
            f"{metrics.get('llm_fallback_rate', 1.0):.4f} | {metrics['seconds_per_example']:.4f} |"
        )
    best_name, best_metrics = max(
        results["systems"].items(),
        key=lambda item: item[1]["event_type"]["macro_f1"],
    )
    error_rows = [
        f"- `{name}` predicted counts: `{json.dumps(metrics['event_type']['predicted_counts'], sort_keys=True)}`"
        for name, metrics in results["systems"].items()
    ]
    return "\n".join([
        "# Local Instruction Model Event Extraction Evaluation",
        "",
        "## Objective",
        "",
        "Measure whether a compact local instruction model can convert OCR/dealer records into schema-valid poker events, compare unconstrained generation with candidate-likelihood ranking, and evaluate a production-oriented schema-routed LLM hybrid.",
        "",
        "## Dataset",
        "",
        f"Gold fixture: `{results['gold']}`",
        f"Model: `{results['model_id']}`",
        f"Examples: `{results['examples']}`",
        f"Label counts: `{json.dumps(results['label_counts'], sort_keys=True)}`",
        f"Seed: `{results['seed']}`",
        "",
        "## Results",
        "",
        *rows,
        "",
        "## Method",
        "",
        "The same reviewed fixture is evaluated with strict zero-shot generation, few-shot generation, constrained candidate ranking, contextually calibrated candidate ranking, and a schema-routed hybrid. Generation is deterministic (`do_sample=False`). The hybrid validates known structured event families with the deterministic schema extractor and sends other event families to the configured LLM fallback.",
        "",
        "## Findings",
        "",
        f"Best macro F1: `{best_name}` at `{best_metrics['event_type']['macro_f1']:.4f}`.",
        *error_rows,
        "",
        "Free-form generation is sensitive to prompt priors and can collapse to one class. The hybrid improves reliability by limiting LLM inference to records outside the known structured event contract. Hybrid metrics are reported separately from pure-model metrics and include explicit fallback coverage.",
        "",
        "## Limitations",
        "",
        "This is a small CPU-oriented model and a small reviewed fixture. Hybrid performance depends on event-name stability and should be re-evaluated on a larger fixture containing ambiguous and corrupted event names. Results measure structured extraction behavior, not general poker reasoning quality.",
        "",
    ])


def main() -> None:
    args = parse_args()
    prompt_paths = [Path(item) for item in split_csv(args.prompts)]
    names = split_csv(args.system_names)
    candidate_labels = split_csv(args.candidate_labels)
    hybrid_routed_event_names = set(split_csv(args.hybrid_routed_event_names))
    calibration_record = json.loads(args.calibration_record)
    if not isinstance(calibration_record, dict):
        raise ValueError("calibration-record must be a JSON object")
    if len(prompt_paths) != len(names):
        raise ValueError("prompts and system-names must have the same length")
    random.seed(args.seed)
    torch, tokenizer, model = load_model(args.model_id, args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = load_gold(args.gold)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    systems: dict[str, Any] = {}
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    prediction_events: dict[str, list[ExtractedEvent]] = {}
    for name, prompt_path in zip(names, prompt_paths):
        predictions, raw_rows, elapsed, valid_json_rate = run_prompt(
            rows,
            name=name,
            prompt=prompt_path.read_text(encoding="utf-8"),
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        metrics = score_system(rows, predictions)
        metrics["elapsed_seconds"] = elapsed
        metrics["seconds_per_example"] = elapsed / len(rows) if rows else 0.0
        metrics["structured_output_rate"] = valid_json_rate
        systems[name] = metrics
        all_predictions[name] = raw_rows
        prediction_events[name] = predictions

    ranking_predictions, ranking_rows, ranking_elapsed = run_candidate_ranker(
        rows,
        name=args.ranking_system_name,
        prompt=args.ranking_prompt.read_text(encoding="utf-8"),
        candidate_labels=candidate_labels,
        reduction=args.candidate_score_reduction,
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
    )
    ranking_metrics = score_system(rows, ranking_predictions)
    ranking_metrics["elapsed_seconds"] = ranking_elapsed
    ranking_metrics["seconds_per_example"] = ranking_elapsed / len(rows) if rows else 0.0
    ranking_metrics["structured_output_rate"] = 1.0
    systems[args.ranking_system_name] = ranking_metrics
    all_predictions[args.ranking_system_name] = ranking_rows
    prediction_events[args.ranking_system_name] = ranking_predictions

    calibrated_predictions, calibrated_rows, calibrated_elapsed = run_candidate_ranker(
        rows,
        name=args.calibrated_ranking_system_name,
        prompt=args.ranking_prompt.read_text(encoding="utf-8"),
        candidate_labels=candidate_labels,
        reduction=args.candidate_score_reduction,
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        calibration_record=calibration_record,
    )
    calibrated_metrics = score_system(rows, calibrated_predictions)
    calibrated_metrics["elapsed_seconds"] = calibrated_elapsed
    calibrated_metrics["seconds_per_example"] = calibrated_elapsed / len(rows) if rows else 0.0
    calibrated_metrics["structured_output_rate"] = 1.0
    systems[args.calibrated_ranking_system_name] = calibrated_metrics
    all_predictions[args.calibrated_ranking_system_name] = calibrated_rows
    prediction_events[args.calibrated_ranking_system_name] = calibrated_predictions

    if args.hybrid_fallback_system not in prediction_events:
        raise ValueError(f"Unknown hybrid fallback system: {args.hybrid_fallback_system}")
    hybrid_predictions, hybrid_rows, hybrid_routing = run_schema_routed_hybrid(
        rows,
        name=args.hybrid_system_name,
        fallback_name=args.hybrid_fallback_system,
        fallback_predictions=prediction_events[args.hybrid_fallback_system],
        routed_event_names=hybrid_routed_event_names,
    )
    hybrid_metrics = score_system(rows, hybrid_predictions)
    hybrid_metrics.update(hybrid_routing)
    fallback_seconds_per_example = systems[args.hybrid_fallback_system]["seconds_per_example"]
    hybrid_metrics["estimated_llm_seconds"] = fallback_seconds_per_example * hybrid_routing["llm_fallback_count"]
    hybrid_metrics["seconds_per_example"] = fallback_seconds_per_example * hybrid_routing["llm_fallback_rate"]
    hybrid_metrics["elapsed_seconds"] = hybrid_metrics["estimated_llm_seconds"]
    hybrid_metrics["structured_output_rate"] = 1.0
    systems[args.hybrid_system_name] = hybrid_metrics
    all_predictions[args.hybrid_system_name] = hybrid_rows

    results = {
        "gold": str(args.gold),
        "model_id": args.model_id,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": args.device,
        "examples": len(rows),
        "label_counts": {
            label: sum(1 for row in rows if row["expected"]["event_type"] == label)
            for label in candidate_labels
        },
        "seed": args.seed,
        "deterministic": True,
        "candidate_score_reduction": args.candidate_score_reduction,
        "calibration_record": calibration_record,
        "systems": systems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    args.predictions_out.write_text(json.dumps(all_predictions, indent=2, sort_keys=True), encoding="utf-8")
    args.report_out.write_text(build_report(results), encoding="utf-8")
    for name, metrics in systems.items():
        print(f"{name}_macro_f1={metrics['event_type']['macro_f1']:.6f}")
        print(f"{name}_accuracy={metrics['event_type']['accuracy']:.6f}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
