from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_agent.mlops import stable_digest, utc_now

FINAL_MODEL_SELECTION_SCHEMA_VERSION = "final_model_selection.v1"

QWENPOKER_FINAL_SELECTION: dict[str, Any] = {
    "schema_version": FINAL_MODEL_SELECTION_SCHEMA_VERSION,
    "selected_model": {
        "model_family": "QwenPoker",
        "checkpoint": "checkpoint_40960",
        "model_version": "qwenpoker:checkpoint_40960",
        "selection_stage": "final_selected",
        "policy_mode": "sampled_policy",
        "artifact_status": "external_checkpoint_not_bundled",
    },
    "benchmark": {
        "name": "qwenpoker_balanced_heads_up_nlhe_20260714",
        "environment": {
            "game": "Heads-up No-Limit Hold'em",
            "stack_depth_bb": 100,
            "engine": "OpenSpiel FCHPA",
        },
        "action_space": ["fold", "check_call", "half_pot", "full_pot", "all_in"],
        "opponent_suite": [
            {"name": "pool_sft", "weight": 0.40},
            {"name": "random", "weight": 0.15},
            {"name": "calling", "weight": 0.30},
            {"name": "aggressive", "weight": 0.15},
        ],
        "balance": {
            "total_hands": 5000,
            "hands_per_seat": 2500,
            "seat_count": 2,
            "seed": 20260714,
            "balanced_by_position": True,
        },
    },
    "metrics": {
        "win_rate": 0.6448,
        "win_rate_percent": 64.48,
        "bb_per_100": 365.29,
        "returns_ci_95": {
            "is_entirely_positive": True,
            "lower_bb_per_100": None,
            "upper_bb_per_100": None,
            "source_note": "Source result states the 95% confidence interval for returns was entirely positive; exact bounds were not provided.",
        },
        "position_profitability": {
            "both_positions_profitable": True,
            "positions": [
                {"name": "button_small_blind", "hands": 2500, "profitable": True},
                {"name": "big_blind", "hands": 2500, "profitable": True},
            ],
        },
    },
    "selection_reasons": [
        "highest accepted checkpoint in balanced benchmark set",
        "95_percent_return_ci_entirely_positive",
        "profitable_from_both_positions",
        "positive_bb_per_100_against_mixed_opponent_suite",
    ],
    "acceptance_gates": {
        "min_hands": 5000,
        "require_even_seat_balance": True,
        "min_win_rate": 0.50,
        "min_bb_per_100": 0.0,
        "require_positive_return_ci_95": True,
        "require_profitability_from_both_positions": True,
        "require_opponent_weight_sum": 1.0,
    },
}


def describe_final_model_selection() -> dict[str, Any]:
    payload = json.loads(json.dumps(QWENPOKER_FINAL_SELECTION))
    payload["contract_fingerprint"] = stable_digest(payload)
    return payload


def validate_final_model_selection(selection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    selection = selection or describe_final_model_selection()
    benchmark = selection.get("benchmark", {})
    balance = benchmark.get("balance", {})
    metrics = selection.get("metrics", {})
    selected_model = selection.get("selected_model", {})
    gates = selection.get("acceptance_gates", {})
    opponent_suite = benchmark.get("opponent_suite", [])
    returns_ci = metrics.get("returns_ci_95", {})
    position_profitability = metrics.get("position_profitability", {})

    total_hands = int(balance.get("total_hands", 0) or 0)
    hands_per_seat = int(balance.get("hands_per_seat", 0) or 0)
    seat_count = int(balance.get("seat_count", 0) or 0)
    opponent_weight_sum = sum(float(item.get("weight", 0.0) or 0.0) for item in opponent_suite)

    checks = [
        {
            "name": "selected_checkpoint",
            "passed": selected_model.get("checkpoint") == "checkpoint_40960",
            "detail": selected_model.get("checkpoint"),
        },
        {
            "name": "benchmark_hands",
            "passed": total_hands >= int(gates.get("min_hands", 5000)),
            "detail": {"total_hands": total_hands, "min_hands": gates.get("min_hands", 5000)},
        },
        {
            "name": "seat_balance",
            "passed": bool(balance.get("balanced_by_position")) and total_hands == hands_per_seat * seat_count,
            "detail": {"total_hands": total_hands, "hands_per_seat": hands_per_seat, "seat_count": seat_count},
        },
        {
            "name": "opponent_suite_weights",
            "passed": abs(opponent_weight_sum - float(gates.get("require_opponent_weight_sum", 1.0))) < 1e-9,
            "detail": {"weight_sum": opponent_weight_sum},
        },
        {
            "name": "win_rate",
            "passed": float(metrics.get("win_rate", 0.0) or 0.0) > float(gates.get("min_win_rate", 0.5)),
            "detail": {"win_rate": metrics.get("win_rate"), "threshold": gates.get("min_win_rate")},
        },
        {
            "name": "bb_per_100",
            "passed": float(metrics.get("bb_per_100", 0.0) or 0.0) > float(gates.get("min_bb_per_100", 0.0)),
            "detail": {"bb_per_100": metrics.get("bb_per_100"), "threshold": gates.get("min_bb_per_100")},
        },
        {
            "name": "returns_ci_95_positive",
            "passed": bool(returns_ci.get("is_entirely_positive")),
            "detail": returns_ci,
        },
        {
            "name": "both_positions_profitable",
            "passed": bool(position_profitability.get("both_positions_profitable"))
            and all(bool(item.get("profitable")) for item in position_profitability.get("positions", [])),
            "detail": position_profitability,
        },
    ]
    return checks


def final_model_selection_status(selection: dict[str, Any] | None = None) -> dict[str, Any]:
    selection = selection or describe_final_model_selection()
    checks = validate_final_model_selection(selection)
    return {
        "schema_version": FINAL_MODEL_SELECTION_SCHEMA_VERSION,
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "selected_model": selection["selected_model"],
        "benchmark": selection["benchmark"],
        "metrics": selection["metrics"],
        "selection_reasons": selection["selection_reasons"],
        "acceptance_gates": selection["acceptance_gates"],
        "contract_fingerprint": selection["contract_fingerprint"],
        "checks": checks,
    }


def qwenpoker_model_registry_entry(root: Path) -> dict[str, Any]:
    selection = describe_final_model_selection()
    return {
        "schema_version": "model_registry.v1",
        "model_name": "qwen_poker",
        "model_version": selection["selected_model"]["model_version"],
        "stage": "staging",
        "registered_at": utc_now(),
        "artifact": {
            "kind": "external_checkpoint",
            "path": "QwenPoker/checkpoint_40960",
            "exists": False,
            "bundled": False,
            "status": selection["selected_model"]["artifact_status"],
        },
        "run_id": selection["benchmark"]["name"],
        "dataset_version": "openspiel_fchpa_seed_20260714",
        "metrics": selection["metrics"],
        "api_contract_version": "poker-decision-agent-api-v1",
        "selection_report": "reports/final_model_selection.json",
        "docker_image": "poker-decision-agent:0.1.0",
    }


def upsert_qwenpoker_model_registry(root: Path, registry_path: Path | None = None) -> Path:
    registry_path = registry_path or root / "reports" / "model_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": "model_registry.v1", "models": {}}

    entry = qwenpoker_model_registry_entry(root)
    models = registry.setdefault("models", {})
    model_record = models.setdefault(entry["model_name"], {"versions": {}, "latest_version": None})
    model_record.setdefault("versions", {})[entry["model_version"]] = entry
    model_record["latest_version"] = entry["model_version"]
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return registry_path


def final_model_selection_markdown(status: dict[str, Any]) -> str:
    selected = status["selected_model"]
    benchmark = status["benchmark"]
    metrics = status["metrics"]
    lines = [
        "# Final Model Selection",
        "",
        f"- Selected model: `{selected['model_family']} {selected['checkpoint']}`",
        f"- Model version: `{selected['model_version']}`",
        f"- Environment: `{benchmark['environment']['game']} {benchmark['environment']['stack_depth_bb']} BB | {benchmark['environment']['engine']}`",
        f"- Action space: `{', '.join(benchmark['action_space'])}`",
        f"- Hands: `{benchmark['balance']['total_hands']}` total, `{benchmark['balance']['hands_per_seat']}` per seat",
        f"- Seed: `{benchmark['balance']['seed']}`",
        f"- Policy mode: `{selected['policy_mode']}`",
        "",
        "## Metrics",
        "",
        f"- Win rate: `{metrics['win_rate_percent']:.2f}%`",
        f"- Returns: `+{metrics['bb_per_100']:.2f} BB/100`",
        f"- 95% return CI entirely positive: `{str(metrics['returns_ci_95']['is_entirely_positive']).lower()}`",
        f"- Profitable from both positions: `{str(metrics['position_profitability']['both_positions_profitable']).lower()}`",
        "",
        "## Opponent Suite",
        "",
        *[f"- `{item['name']}`: `{item['weight']:.0%}`" for item in benchmark["opponent_suite"]],
        "",
        "## Selection Gates",
        "",
        *[f"- `{item['name']}`: `{'PASS' if item['passed'] else 'FAIL'}`" for item in status["checks"]],
        "",
    ]
    return "\n".join(lines)


def write_final_model_selection_reports(root: Path, out: Path | None = None, docs_out: Path | None = None) -> dict[str, Path]:
    out = out or root / "reports" / "final_model_selection.json"
    docs_out = docs_out or root / "docs" / "FINAL_MODEL_SELECTION.md"
    status = final_model_selection_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    docs_out.write_text(final_model_selection_markdown(status), encoding="utf-8")
    upsert_qwenpoker_model_registry(root)
    return {"json": out, "docs": docs_out}
