from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples, public_context_features
from poker_agent.model import SklearnPolicy, load_policy
from poker_agent.slices import evaluate_policy_slices
from poker_agent.validation import stratified_group_holdout_split


CHALLENGER_VERSION = "2026-06-27"
STANDALONE_APPROVED = "STANDALONE_APPROVED"
NOT_STANDALONE_APPROVED = "NOT_STANDALONE_APPROVED"

RAW_GATE_THRESHOLDS = {
    "min_accuracy_lift": 0.0,
    "min_macro_f1": 0.50,
    "min_balanced_accuracy": 0.50,
    "max_ece_10": 0.10,
    "min_observed_hole_macro_f1": 0.50,
    "min_facing_bet_macro_f1": 0.45,
}


@dataclass(frozen=True)
class ChallengerSpec:
    name: str
    model_kind: str
    class_weighting: str = "sqrt_balanced"
    max_class_weight: float = 8.0
    feature_mode: str = "full"
    max_iter: int = 180
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.01
    n_estimators: int = 300


DEFAULT_CHALLENGERS = [
    ChallengerSpec(
        name="hgb_sqrt_balanced_full",
        model_kind="hist_gradient_boosting",
        class_weighting="sqrt_balanced",
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=31,
    ),
    ChallengerSpec(
        name="hgb_balanced_full",
        model_kind="hist_gradient_boosting",
        class_weighting="balanced",
        max_class_weight=10.0,
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=39,
    ),
    ChallengerSpec(
        name="extra_trees_sqrt_balanced_full",
        model_kind="extra_trees",
        class_weighting="sqrt_balanced",
        max_class_weight=8.0,
        n_estimators=260,
    ),
    ChallengerSpec(
        name="random_forest_sqrt_balanced_full",
        model_kind="random_forest",
        class_weighting="sqrt_balanced",
        max_class_weight=8.0,
        n_estimators=220,
    ),
    ChallengerSpec(
        name="hgb_public_context_guardrail",
        model_kind="hist_gradient_boosting",
        class_weighting="sqrt_balanced",
        feature_mode="public_context",
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=31,
    ),
]


def train_raw_model_challengers(
    *,
    project_root: Path,
    dataset: Path,
    report_out: Path,
    markdown_out: Path | None = None,
    model_dir: Path | None = None,
    audit_report: Path | None = None,
    max_examples: int = 50000,
    valid_ratio: float = 0.15,
    seed: int = 42,
    challengers: list[ChallengerSpec] | None = None,
    promote_model_out: Path | None = None,
) -> dict[str, Any]:
    specs = challengers or DEFAULT_CHALLENGERS
    project_root = Path(project_root)
    model_dir = Path(model_dir or project_root / "models" / "raw_challengers")
    records = load_training_examples(
        Path(dataset),
        max_examples=max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        merge_all_in=True,
        include_hand_id=True,
    )
    if not records:
        raise ValueError(f"No training examples found in {dataset}")

    train_examples, valid_examples, split_info = stratified_group_holdout_split(
        records,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    audit_findings = _read_audit_findings(audit_report)
    blocker_findings = [item for item in audit_findings if item.get("severity") == "blocker"]

    candidate_reports: list[dict[str, Any]] = []
    for spec in specs:
        candidate_reports.append(
            _train_candidate(
                spec,
                train_examples=train_examples,
                valid_examples=valid_examples,
                model_dir=model_dir,
                seed=seed,
                audit_blockers=blocker_findings,
            )
        )

    best = _select_best_candidate(candidate_reports)
    standalone_approved = bool(best) and best.get("gate", {}).get("status") == "PASS"
    standalone_status = STANDALONE_APPROVED if standalone_approved else NOT_STANDALONE_APPROVED

    promoted_to = None
    if standalone_approved and promote_model_out is not None:
        source = Path(best["artifact_path"])
        promote_model_out.parent.mkdir(parents=True, exist_ok=True)
        promote_model_out.write_bytes(source.read_bytes())
        promoted_to = str(promote_model_out)

    payload: dict[str, Any] = {
        "version": CHALLENGER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "objective": (
            "Train and gate stronger standalone supervised policy candidates without allowing a false "
            "standalone production approval when raw gates fail."
        ),
        "dataset": str(dataset),
        "max_examples": max_examples,
        "split": split_info,
        "thresholds": RAW_GATE_THRESHOLDS,
        "audit": {
            "report": str(audit_report) if audit_report else None,
            "blocker_count": len(blocker_findings),
            "blockers": blocker_findings,
        },
        "standalone_status": standalone_status,
        "approved_as_standalone_policy": standalone_approved,
        "best_candidate": best,
        "candidates": candidate_reports,
        "promotion": {
            "promoted": promoted_to is not None,
            "artifact_path": promoted_to,
            "rule": "Promotion is allowed only after the selected challenger passes every raw production gate.",
        },
        "approval_boundary": {
            "existing_service_delivery_affected": False,
            "deployed_strategy_stack_affected": False,
            "raw_model_standalone_allowed": standalone_approved,
            "false_pass_guard": "enabled",
            "non_override_rule": (
                "A loadable raw supervised model cannot be presented as standalone production-approved "
                "unless this challenger gate and the raw production gate both pass."
            ),
        },
        "next_actions": _next_actions(best, blocker_findings),
    }
    payload["invariants"] = {
        "status": "PASS" if not validate_challenger_report(payload) else "FAIL",
        "violations": validate_challenger_report(payload),
    }
    assert_challenger_report(payload)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_challenger_markdown(payload), encoding="utf-8")
    return payload


def build_existing_model_challenger_report(
    *,
    project_root: Path,
    model_path: Path,
    dataset: Path,
    report_out: Path,
    markdown_out: Path | None = None,
    audit_report: Path | None = None,
    max_examples: int = 50000,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    records = load_training_examples(
        Path(dataset),
        max_examples=max_examples,
        require_hole_cards=False,
        missing_hole_cards="flag",
        merge_all_in=True,
        include_hand_id=True,
    )
    _, valid_examples, split_info = stratified_group_holdout_split(records, valid_ratio=valid_ratio, seed=seed)
    audit_findings = _read_audit_findings(audit_report)
    blocker_findings = [item for item in audit_findings if item.get("severity") == "blocker"]
    model = load_policy(Path(model_path))
    valid_metrics = evaluate_policy(model, valid_examples)
    slice_metrics = evaluate_policy_slices(model, valid_examples, min_examples=100)
    gate = evaluate_challenger_gate(valid_metrics, slice_metrics, blocker_findings)
    candidate = {
        "name": "existing_raw_supervised_model",
        "status": "EVALUATED",
        "artifact_path": str(model_path),
        "valid_metrics": _compact_metrics(valid_metrics),
        "valid_slice_metrics": _compact_slice_metrics(slice_metrics),
        "gate": gate,
        "ranking_score": _ranking_score({"gate": gate, "valid_metrics": _compact_metrics(valid_metrics)}),
    }
    standalone_approved = gate["status"] == "PASS"
    payload = {
        "version": CHALLENGER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "objective": "Evaluate the existing raw supervised artifact under the standalone challenger contract.",
        "dataset": str(dataset),
        "max_examples": max_examples,
        "split": split_info,
        "thresholds": RAW_GATE_THRESHOLDS,
        "audit": {
            "report": str(audit_report) if audit_report else None,
            "blocker_count": len(blocker_findings),
            "blockers": blocker_findings,
        },
        "standalone_status": STANDALONE_APPROVED if standalone_approved else NOT_STANDALONE_APPROVED,
        "approved_as_standalone_policy": standalone_approved,
        "best_candidate": candidate,
        "candidates": [candidate],
        "promotion": {"promoted": False, "artifact_path": None, "rule": "Evaluation-only mode never promotes."},
        "approval_boundary": {
            "existing_service_delivery_affected": False,
            "deployed_strategy_stack_affected": False,
            "raw_model_standalone_allowed": standalone_approved,
            "false_pass_guard": "enabled",
            "non_override_rule": "Standalone approval requires every raw quality gate to pass.",
        },
        "next_actions": _next_actions(candidate, blocker_findings),
    }
    payload["invariants"] = {
        "status": "PASS" if not validate_challenger_report(payload) else "FAIL",
        "violations": validate_challenger_report(payload),
    }
    assert_challenger_report(payload)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_challenger_markdown(payload), encoding="utf-8")
    return payload


def evaluate_challenger_gate(
    valid_metrics: dict[str, Any],
    slice_metrics: dict[str, dict[str, Any]],
    audit_blockers: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = thresholds or RAW_GATE_THRESHOLDS
    gates = [
        _gate(
            "accuracy_lift",
            float(valid_metrics.get("lift_vs_majority", -999.0)) >= limits["min_accuracy_lift"],
            valid_metrics.get("lift_vs_majority"),
            limits["min_accuracy_lift"],
            "Candidate must beat the majority-class baseline on grouped holdout.",
        ),
        _gate(
            "macro_f1",
            float(valid_metrics.get("macro_f1", 0.0)) >= limits["min_macro_f1"],
            valid_metrics.get("macro_f1"),
            limits["min_macro_f1"],
            "Minority actions must be learned rather than hidden by fold dominance.",
        ),
        _gate(
            "balanced_accuracy",
            float(valid_metrics.get("balanced_accuracy", 0.0)) >= limits["min_balanced_accuracy"],
            valid_metrics.get("balanced_accuracy"),
            limits["min_balanced_accuracy"],
            "Recall must be acceptable across action classes.",
        ),
        _gate(
            "calibration",
            float(valid_metrics.get("ece_10", 1.0)) <= limits["max_ece_10"],
            valid_metrics.get("ece_10"),
            limits["max_ece_10"],
            "Confidence must remain calibrated enough for downstream gating.",
        ),
    ]
    observed_hole = slice_metrics.get("observed_hole_cards") or {}
    gates.append(
        _gate(
            "observed_hole_cards_macro_f1",
            bool(observed_hole) and float(observed_hole.get("macro_f1", 0.0)) >= limits["min_observed_hole_macro_f1"],
            observed_hole.get("macro_f1"),
            limits["min_observed_hole_macro_f1"],
            "The candidate must perform on the slice where card signal exists.",
        )
    )
    facing_bet = slice_metrics.get("facing_bet") or {}
    gates.append(
        _gate(
            "facing_bet_macro_f1",
            bool(facing_bet) and float(facing_bet.get("macro_f1", 0.0)) >= limits["min_facing_bet_macro_f1"],
            facing_bet.get("macro_f1"),
            limits["min_facing_bet_macro_f1"],
            "Call/fold/raise behavior under pressure must be strong enough for standalone use.",
        )
    )
    gates.append(
        _gate(
            "dataset_audit_blockers",
            len(audit_blockers) == 0,
            len(audit_blockers),
            0,
            "No standalone production policy can pass while dataset audit blockers remain open.",
        )
    )
    return {
        "status": "PASS" if all(item["passed"] for item in gates) else "FAIL",
        "passed_gates": sum(1 for item in gates if item["passed"]),
        "total_gates": len(gates),
        "failed_gates": [item["name"] for item in gates if not item["passed"]],
        "gates": gates,
    }


def validate_challenger_report(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    approved = bool(payload.get("approved_as_standalone_policy"))
    status = payload.get("standalone_status")
    best = payload.get("best_candidate") or {}
    gate_status = (best.get("gate") or {}).get("status")
    audit = payload.get("audit") or {}
    boundary = payload.get("approval_boundary") or {}

    if approved and gate_status != "PASS":
        violations.append("standalone_approval_requires_challenger_gate_pass")
    if approved and int(audit.get("blocker_count") or 0) > 0:
        violations.append("standalone_approval_requires_zero_dataset_blockers")
    if gate_status != "PASS" and status == STANDALONE_APPROVED:
        violations.append("standalone_status_cannot_be_approved_when_gate_fails")
    if gate_status != "PASS" and boundary.get("raw_model_standalone_allowed"):
        violations.append("approval_boundary_cannot_allow_raw_standalone_when_gate_fails")
    if boundary.get("existing_service_delivery_affected") is not False:
        violations.append("raw_challenger_must_not_break_existing_service_delivery")
    return violations


def assert_challenger_report(payload: dict[str, Any]) -> None:
    violations = validate_challenger_report(payload)
    if violations:
        raise ValueError(f"Invalid raw model challenger contract: {violations}")


def render_challenger_markdown(payload: dict[str, Any]) -> str:
    best = payload.get("best_candidate") or {}
    metrics = best.get("valid_metrics") or {}
    gate = best.get("gate") or {}
    lines = [
        "# Raw Supervised Model Challenger",
        "",
        "## Status",
        "",
        f"- Standalone status: `{payload.get('standalone_status')}`",
        f"- Approved as standalone policy: `{payload.get('approved_as_standalone_policy')}`",
        f"- Best candidate: `{best.get('name')}`",
        f"- Candidate gate: `{gate.get('status')}`",
        f"- Failed gates: `{', '.join(gate.get('failed_gates') or []) or 'none'}`",
        f"- Dataset blocker count: `{(payload.get('audit') or {}).get('blocker_count')}`",
        "",
        "## Best Candidate Metrics",
        "",
        f"- Accuracy: `{metrics.get('accuracy')}`",
        f"- Macro F1: `{metrics.get('macro_f1')}`",
        f"- Balanced accuracy: `{metrics.get('balanced_accuracy')}`",
        f"- Majority baseline accuracy: `{metrics.get('majority_baseline_accuracy')}`",
        f"- Lift vs majority: `{metrics.get('lift_vs_majority')}`",
        f"- ECE@10: `{metrics.get('ece_10')}`",
        "",
        "## Boundary",
        "",
        f"- Existing service delivery affected: `{(payload.get('approval_boundary') or {}).get('existing_service_delivery_affected')}`",
        f"- Deployed strategy stack affected: `{(payload.get('approval_boundary') or {}).get('deployed_strategy_stack_affected')}`",
        f"- False pass guard: `{(payload.get('approval_boundary') or {}).get('false_pass_guard')}`",
        "",
        "## Next Actions",
        "",
    ]
    for item in payload.get("next_actions") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _train_candidate(
    spec: ChallengerSpec,
    *,
    train_examples: list[tuple[dict[str, float], str]],
    valid_examples: list[tuple[dict[str, float], str]],
    model_dir: Path,
    seed: int,
    audit_blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    train_rows = _apply_feature_mode(train_examples, spec.feature_mode)
    valid_rows = _apply_feature_mode(valid_examples, spec.feature_mode)
    try:
        _patch_optional_runtime_metadata()
        model = SklearnPolicy()
        model.fit(
            train_rows,
            model_kind=spec.model_kind,
            class_weighting=spec.class_weighting,
            max_class_weight=spec.max_class_weight,
            random_state=seed,
            max_iter=spec.max_iter,
            learning_rate=spec.learning_rate,
            max_leaf_nodes=spec.max_leaf_nodes,
            l2_regularization=spec.l2_regularization,
            n_estimators=spec.n_estimators,
        )
        train_metrics = evaluate_policy(model, train_rows)
        valid_metrics = evaluate_policy(model, valid_rows)
        slice_metrics = evaluate_policy_slices(model, valid_rows, min_examples=100)
        gate = evaluate_challenger_gate(valid_metrics, slice_metrics, audit_blockers)
        model.metadata = {
            "candidate": asdict(spec),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "valid_slice_metrics": slice_metrics,
            "gate": gate,
            "standalone_approval_rule": "Do not promote unless gate.status is PASS.",
        }
        artifact_path = model_dir / f"{spec.name}.joblib"
        model.save(artifact_path)
        report = {
            "name": spec.name,
            "status": "TRAINED",
            "spec": asdict(spec),
            "artifact_path": str(artifact_path),
            "train_metrics": _compact_metrics(train_metrics),
            "valid_metrics": _compact_metrics(valid_metrics),
            "valid_slice_metrics": _compact_slice_metrics(slice_metrics),
            "gate": gate,
        }
        report["ranking_score"] = _ranking_score(report)
        return report
    except Exception as exc:
        return {
            "name": spec.name,
            "status": "TRAINING_FAILED",
            "spec": asdict(spec),
            "error": f"{type(exc).__name__}: {exc}",
            "gate": {"status": "FAIL", "failed_gates": ["training_failed"], "passed_gates": 0, "total_gates": 1},
            "ranking_score": [-1, 0.0, 0.0, -999.0, 0.0],
        }


def _patch_optional_runtime_metadata() -> None:
    # Some Windows environments keep an incomplete namespace-only pyarrow package.
    # Pandas/scikit-learn only need the version attribute during import checks here.
    try:
        import pyarrow  # type: ignore
    except Exception:
        return
    if not hasattr(pyarrow, "__version__"):
        pyarrow.__version__ = "0.0.0"  # type: ignore[attr-defined]


def _apply_feature_mode(
    examples: list[tuple[dict[str, float], str]],
    feature_mode: str,
) -> list[tuple[dict[str, float], str]]:
    if feature_mode == "full":
        return examples
    if feature_mode == "public_context":
        return [(public_context_features(features), label) for features, label in examples]
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def _select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=_ranking_score)


def _ranking_score(candidate: dict[str, Any]) -> list[float]:
    gate = candidate.get("gate") or {}
    metrics = candidate.get("valid_metrics") or {}
    return [
        1.0 if gate.get("status") == "PASS" else 0.0,
        float(gate.get("passed_gates") or 0),
        float(metrics.get("macro_f1") or 0.0),
        float(metrics.get("balanced_accuracy") or 0.0),
        float(metrics.get("lift_vs_majority") or -999.0),
        float(metrics.get("accuracy") or 0.0),
    ]


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "examples",
        "accuracy",
        "cross_entropy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "ece_10",
        "brier_loss",
        "majority_baseline_accuracy",
        "lift_vs_majority",
        "class_counts",
        "predicted_class_counts",
        "per_class",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _compact_slice_metrics(slice_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: _compact_metrics(metrics) for name, metrics in slice_metrics.items()}


def _gate(name: str, passed: bool, observed: Any, threshold: Any, impact: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "impact": impact,
    }


def _read_audit_findings(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("findings") or [])


def _next_actions(best: dict[str, Any] | None, blockers: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if blockers:
        actions.append("Close dataset audit blockers, starting with hole-card coverage, before standalone approval.")
    if not best:
        actions.append("Run challenger training and produce a selected candidate report.")
        return actions
    gate = best.get("gate") or {}
    failed = gate.get("failed_gates") or []
    if "accuracy_lift" in failed:
        actions.append("Improve supervised features or labels until the candidate beats the majority baseline.")
    if "macro_f1" in failed or "balanced_accuracy" in failed:
        actions.append("Continue class-imbalance work with stronger resampling, focal loss, or calibrated routing.")
    if "observed_hole_cards_macro_f1" in failed:
        actions.append("Increase reviewed hole-card coverage and train a card-aware observed-hand specialist.")
    if "facing_bet_macro_f1" in failed:
        actions.append("Add better pot-odds, aggression, and previous-action features for facing-bet decisions.")
    if not actions:
        actions.append("Promote only after independent raw production gate reproduction passes.")
    return actions
