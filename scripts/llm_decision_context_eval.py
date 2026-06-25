from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_decision_benchmark import (
    RuleBaselineProvider,
    TransformersCandidateRanker,
    TransformersDecisionProvider,
    benchmark_context_modes,
    load_decision_examples,
    write_benchmark_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLM poker decision context variants")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--provider",
        required=True,
        choices=("rule_baseline", "transformers", "candidate_ranker"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--quantization", default="none", choices=("none", "4bit_nf4"))
    parser.add_argument("--max-gpu-memory-mb", default=0, type=int)
    parser.add_argument("--max-cpu-memory-gb", default=12, type=int)
    parser.add_argument("--candidate-score-reduction", default="mean", choices=("mean", "sum"))
    parser.add_argument("--probability-temperature", default=1.0, type=float)
    parser.add_argument("--max-new-tokens", default=192, type=int)
    parser.add_argument("--max-examples", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--dataset-kind",
        default="smoke",
        choices=("smoke", "reconstructed_human_holdout", "reviewed_human_holdout"),
    )
    parser.add_argument(
        "--context-modes",
        default="minimal_zero_shot,rules_grounded,full_in_context",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [item.strip() for item in args.context_modes.split(",") if item.strip()]
    valid_modes = {"minimal_zero_shot", "rules_grounded", "full_in_context"}
    if not modes or any(mode not in valid_modes for mode in modes):
        raise ValueError(f"context-modes must be selected from {sorted(valid_modes)}")
    if args.provider == "rule_baseline":
        provider = RuleBaselineProvider()
    elif args.provider == "candidate_ranker":
        provider = TransformersCandidateRanker(
            args.model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            max_gpu_memory_mb=args.max_gpu_memory_mb,
            max_cpu_memory_gb=args.max_cpu_memory_gb,
            score_reduction=args.candidate_score_reduction,
            probability_temperature=args.probability_temperature,
        )
    else:
        provider = TransformersDecisionProvider(
            args.model_id,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            max_gpu_memory_mb=args.max_gpu_memory_mb,
            max_cpu_memory_gb=args.max_cpu_memory_gb,
        )
    examples = load_decision_examples(args.data, max_examples=args.max_examples)
    result = benchmark_context_modes(
        examples,
        provider,
        context_modes=modes,
        seed=args.seed,
        dataset_kind=args.dataset_kind,
    )
    write_benchmark_outputs(
        result,
        out_path=args.out,
        predictions_path=args.predictions_out,
        report_path=args.report_out,
    )
    print(f"provider={result['provider']}")
    print(f"examples={len(examples)}")
    print(f"best_context_mode={result['best_mode']}")
    print(f"provisional_best_context_mode={result['provisional_best_mode']}")
    print(f"quality_claim_allowed={str(result['quality_claim_allowed']).lower()}")
    print(json.dumps(result["systems"], sort_keys=True))


if __name__ == "__main__":
    main()
