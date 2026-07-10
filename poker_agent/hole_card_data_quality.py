from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOLE_CARD_DATA_QUALITY_VERSION = "2026-06-28"
OPEN_DATA_QUALITY_LIMITATION = "OPEN_DATA_QUALITY_LIMITATION"
MITIGATED_BY_ROUTED_POLICY_BUNDLE = "MITIGATED_BY_ROUTED_POLICY_BUNDLE"
UPSTREAM_NOT_RESOLVED = "UPSTREAM_NOT_RESOLVED"
DEGRADED_STRENGTH_SIGNAL = "DEGRADED_BY_MISSING_HOLE_CARDS"
MISSING_RATE_RISK_THRESHOLD = 0.50
STRENGTH_PROXY_ZERO_RATE_THRESHOLD = 0.50
RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD = 0.80
INVALID_CARD_RATE_PROMOTION_THRESHOLD = 0.02
CARD_TOKEN_RE = re.compile(r"^(?:[2-9TJQKA][CDHS]|10[CDHS])$", re.IGNORECASE)
CARD_SPLIT_RE = re.compile(r"[\s,;|/]+")
MAX_CARD_AUDIT_EXAMPLES = 10
HOLE_CARD_RISK_CONTRACT = {
    "risk_id": "hole_card_data_risk",
    "root_cause": "ocr_hole_card_extraction_missing_or_unreliable",
    "primary_dataset_column": "players.cards",
    "source_table": "players.csv",
    "source_field": "players.csv::cards",
    "cards_storage_boundary": {
        "players_csv_stores_hole_cards": True,
        "storage_does_not_imply_reliability": True,
        "card_values_are_ocr_or_recognition_derived": True,
        "missing_or_unreliable_cards_are_expected_dataset_conditions": True,
    },
    "weakens_primary_poker_signal": True,
    "affected_signal": "private_card_strength_and_texture",
    "affected_features": [
        "strength_proxy",
        "hole_card_observed_ratio",
        "hole_cards_missing",
        "card_texture_features",
        "made_hand_score",
        "draw_pressure",
    ],
    "decision_impact": [
        "weaker preflop raise/call/fold separation",
        "less reliable made-hand and draw-strength estimation",
        "lower confidence on observed-card policy promotion",
    ],
    "hand_strength_feature_boundary": {
        "private_cards_are_primary_strategy_signal": True,
        "missing_or_invalid_hole_cards_limit_hand_strength_features": True,
        "hand_strength_features_must_be_slice_aware": True,
        "standalone_card_aware_policy_requires_reliable_two_card_coverage": True,
    },
    "feature_policy": {
        "missing_or_invalid_cards": "flag_and_route",
        "do_not_impute_unknown_cards_as_known_private_cards": True,
        "do_not_treat_missing_cards_as_reliable_zero_strength": True,
        "train_observed_card_and_public_context_slices_separately": True,
    },
    "current_delivery_blocker": False,
    "final_strategy_quality_claim_blocker": True,
}
DELIVERY_STRATEGY_QUALITY_BOUNDARY = {
    "risk_scope": "MODEL_QUALITY_RISK_NOT_SERVICE_DELIVERY_BLOCKER",
    "current_delivery_blocker": False,
    "service_delivery_claim_allowed": True,
    "deployed_routed_stack_delivery_allowed": True,
    "final_strategy_quality_claim_allowed": False,
    "final_strategy_quality_claim_blocked_by_hole_card_data_quality": True,
    "model_quality_risk": True,
    "component_risk": True,
    "requires_to_clear_final_strategy_claim": [
        "improved_ocr_or_card_parser",
        "larger_reviewed_hole_card_label_set",
        "reliable_two_card_coverage_gate",
        "observed_card_policy_slice_gate",
        "standalone_card_aware_policy_promotion_gate",
    ],
}


def scan_players_hole_cards(
    project_root: Path,
    *,
    max_examples: int = MAX_CARD_AUDIT_EXAMPLES,
) -> dict[str, Any]:
    candidates = [
        project_root / "data" / "players.csv",
        project_root / "dataset" / "players.csv",
    ]
    players_path = next((path for path in candidates if path.exists()), None)
    if players_path is None:
        return {
            "status": "MISSING_PLAYERS_CSV",
            "candidate_paths": [str(path) for path in candidates],
            "rows_scanned": 0,
        }

    rows_scanned = 0
    missing_rows = 0
    partial_rows = 0
    complete_rows = 0
    reliable_two_card_rows = 0
    invalid_card_rows = 0
    duplicate_card_rows = 0
    overcomplete_card_rows = 0
    card_count_distribution: Counter[str] = Counter()
    malformed_examples: list[dict[str, Any]] = []

    with players_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "cards" not in reader.fieldnames:
            return {
                "status": "MISSING_CARDS_COLUMN",
                "path": str(players_path),
                "columns": reader.fieldnames or [],
                "rows_scanned": 0,
            }

        for row_number, row in enumerate(reader, start=2):
            rows_scanned += 1
            raw_cards = row.get("cards")
            tokens = _split_card_tokens(raw_cards)
            valid_cards, invalid_tokens = _partition_card_tokens(tokens)
            valid_count = len(valid_cards)
            unique_valid_count = len(set(valid_cards))
            has_duplicate = unique_valid_count < valid_count
            has_invalid = bool(invalid_tokens)
            is_reliable_two_card_row = valid_count == 2 and unique_valid_count == 2 and not has_invalid

            card_count_distribution[str(valid_count)] += 1
            if valid_count == 0:
                missing_rows += 1
            elif valid_count == 1:
                partial_rows += 1
            elif valid_count == 2:
                complete_rows += 1
            else:
                overcomplete_card_rows += 1

            if is_reliable_two_card_row:
                reliable_two_card_rows += 1
            if has_invalid:
                invalid_card_rows += 1
            if has_duplicate:
                duplicate_card_rows += 1
            if (has_invalid or has_duplicate or valid_count > 2) and len(malformed_examples) < max_examples:
                malformed_examples.append(
                    {
                        "row_number": row_number,
                        "hand_id": row.get("hand_id"),
                        "position": row.get("position"),
                        "cards": raw_cards,
                        "valid_cards": valid_cards,
                        "invalid_tokens": invalid_tokens,
                        "duplicate_cards": has_duplicate,
                    }
                )

    denominator = max(rows_scanned, 1)
    missing_rate = missing_rows / denominator
    reliable_two_card_rate = reliable_two_card_rows / denominator
    invalid_card_rate = invalid_card_rows / denominator
    risk_status = (
        "HIGH_RISK"
        if missing_rate >= MISSING_RATE_RISK_THRESHOLD
        or reliable_two_card_rate < RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD
        or invalid_card_rate >= INVALID_CARD_RATE_PROMOTION_THRESHOLD
        else "ACCEPTABLE_FOR_PROMOTION"
    )

    return {
        "status": "PASS" if rows_scanned > 0 else "EMPTY_PLAYERS_CSV",
        "path": str(players_path),
        "rows_scanned": rows_scanned,
        "missing_rows": missing_rows,
        "partial_rows": partial_rows,
        "complete_rows": complete_rows,
        "reliable_two_card_rows": reliable_two_card_rows,
        "invalid_card_rows": invalid_card_rows,
        "duplicate_card_rows": duplicate_card_rows,
        "overcomplete_card_rows": overcomplete_card_rows,
        "missing_hole_card_rate": missing_rate,
        "partial_hole_card_rate": partial_rows / denominator,
        "complete_hole_card_rate": complete_rows / denominator,
        "reliable_two_card_rate": reliable_two_card_rate,
        "invalid_card_rate": invalid_card_rate,
        "duplicate_card_rate": duplicate_card_rows / denominator,
        "overcomplete_card_rate": overcomplete_card_rows / denominator,
        "card_count_distribution": dict(sorted(card_count_distribution.items())),
        "malformed_examples": malformed_examples,
        "risk_status": risk_status,
    }


def build_hole_card_data_quality(project_root: Path) -> dict[str, Any]:
    reports = project_root / "reports"
    audit = _read_optional_json(reports / "dataset_audit.json")
    production_gate = _read_optional_json(reports / "production_gate.json")
    today_training = _read_optional_json(reports / "today_acceptance_training.json")
    raw_challenger = _read_optional_json(reports / "raw_model_challenger.json")
    direct_card_audit = scan_players_hole_cards(project_root)

    players = audit.get("players") or {}
    features = audit.get("features") or {}
    zero_rates = features.get("critical_feature_zero_rates") or {}
    audit_findings = audit.get("findings") or production_gate.get("audit_findings") or []
    selected_architecture = today_training.get("selected_architecture", "UNKNOWN")
    routed_active = selected_architecture == "routed_policy_bundle"
    hole_card_finding = _find_hole_card_finding(audit_findings)
    strength_proxy_finding = _find_strength_proxy_finding(audit_findings)
    production_observed_gate = _gate_by_name(production_gate.get("gates") or [], "observed_hole_cards_macro_f1")
    raw_challenger_gate = ((raw_challenger.get("best_candidate") or {}).get("gate") or {})
    raw_challenger_slices = (
        (raw_challenger.get("best_candidate") or {}).get("valid_slice_metrics")
        or (raw_challenger.get("best_candidate") or {}).get("slice_metrics")
        or {}
    )

    payload: dict[str, Any] = {
        "version": HOLE_CARD_DATA_QUALITY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subject": "Hole-card data-quality limitation and routed-policy mitigation boundary",
        "client_statement": (
            "Missing or unreliable hole-card data remains a core dataset limitation. "
            "The routed policy bundle handles this better, but it does not fully solve the upstream data-quality issue."
        ),
        "risk_contract": HOLE_CARD_RISK_CONTRACT,
        "coverage_snapshot": {
            "coverage_source": (
                "direct_players_csv"
                if direct_card_audit.get("status") == "PASS"
                else "dataset_audit_report"
            ),
            "players_csv_cards_contract": {
                "source_table": "players.csv",
                "source_field": "players.csv::cards",
                "semantic": "player_hole_cards",
                "quality_source": "ocr_or_card_recognition",
                "may_be_missing_or_unreliable": True,
                "must_not_be_treated_as_reliable_by_presence_alone": True,
            },
            "players_rows": players.get("rows"),
            "missing_hole_card_rate": players.get("missing_hole_card_rate"),
            "partial_hole_card_rate": players.get("partial_hole_card_rate"),
            "complete_hole_card_rate": players.get("complete_hole_card_rate"),
            "card_count_distribution": players.get("card_count_distribution", {}),
            "direct_players_csv_audit": direct_card_audit,
            "audit_finding": hole_card_finding,
        },
        "strength_signal_impact": {
            "status": DEGRADED_STRENGTH_SIGNAL,
            "affected_features": HOLE_CARD_RISK_CONTRACT["affected_features"],
            "strength_proxy_zero_rate": zero_rates.get("strength_proxy"),
            "missing_hole_card_rate": players.get("missing_hole_card_rate"),
            "complete_hole_card_rate": players.get("complete_hole_card_rate"),
            "direct_reliable_two_card_rate": direct_card_audit.get("reliable_two_card_rate"),
            "direct_invalid_card_rate": direct_card_audit.get("invalid_card_rate"),
            "primary_signal_weakened_by_ocr_missingness": True,
            "hand_strength_features_limited_by_card_quality": True,
            "primary_hand_strength_signal_reliable_for_standalone_policy": False,
            "observed_hole_cards_macro_f1": production_observed_gate.get("observed"),
            "observed_hole_cards_threshold": production_observed_gate.get("threshold"),
            "observed_hole_cards_gate_passed": production_observed_gate.get("passed"),
            "challenger_observed_hole_cards_macro_f1": (
                raw_challenger_slices.get("observed_hole_cards") or {}
            ).get("macro_f1"),
            "strength_proxy_audit_finding": strength_proxy_finding,
            "expected_model_impact": (
                "Card-strength features cannot be treated as the primary reliable signal while most "
                "player rows have missing hole cards and strength_proxy is zero-dominant."
            ),
        },
        "mitigation_boundary": {
            "selected_architecture": selected_architecture,
            "mitigation_status": MITIGATED_BY_ROUTED_POLICY_BUNDLE if routed_active else "MITIGATION_NOT_ACTIVE",
            "routed_policy_bundle_handles_missingness": routed_active,
            "observed_card_policy_path": "uses private-card/card-texture features when two hole cards are observed",
            "public_context_policy_path": "removes private-card features when hole cards are missing or unreliable",
            "mitigation_scope": "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR",
            "requires_slice_specific_monitoring": True,
            "fully_solves_upstream_data_quality_issue": False,
        },
        "upstream_data_quality_boundary": {
            "limitation_status": OPEN_DATA_QUALITY_LIMITATION,
            "upstream_status": UPSTREAM_NOT_RESOLVED,
            "upstream_data_quality_issue_resolved": False,
            "requires_ocr_or_parser_improvement": True,
            "requires_larger_reviewed_card_labels": True,
            "production_blocker_for_current_deployment": False,
            "component_risk": True,
            "raw_standalone_policy_affected": True,
            "players_csv_cards_are_not_reliability_guarantee": True,
            "hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve": True,
        },
        "promotion_boundary": {
            "standalone_policy_promotion_allowed": False,
            "model_promotion_blocker": True,
            "current_deployment_blocker": False,
            "requires_reliable_two_card_rate": RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD,
            "requires_invalid_card_rate_below": INVALID_CARD_RATE_PROMOTION_THRESHOLD,
            "requires_reviewed_card_label_set": True,
            "reason": (
                "Hole cards are a primary poker-strength signal. Standalone strategy promotion remains "
                "blocked until players.csv has enough reliable two-card rows and low malformed-card rate."
            ),
        },
        "delivery_strategy_quality_boundary": DELIVERY_STRATEGY_QUALITY_BOUNDARY,
        "evidence": {
            "dataset_audit": "reports/dataset_audit.json",
            "production_gate": "reports/production_gate.json",
            "today_acceptance_training": "reports/today_acceptance_training.json",
            "raw_model_challenger": "reports/raw_model_challenger.json",
            "raw_challenger_failed_gates": raw_challenger_gate.get("failed_gates", []),
        },
        "required_upstream_fixes": [
            "Improve OCR/card extraction and dealer-log reconciliation for player hole cards.",
            "Create a larger reviewed card-label set with complete, partial, and missing-card slices.",
            "Report observed-card and missing-card model quality separately in every production gate.",
            "Promote a standalone challenger only after observed-card gates and dataset audit blockers pass.",
        ],
        "allowed_claims": [
            "The routed policy bundle mitigates missing-card data better than a single raw supervised model.",
            "The deployed service can route observed-card and missing-card requests through separate policy paths.",
        ],
        "not_allowed_claims": [
            "The upstream hole-card data-quality issue is fully solved.",
            "The routed policy bundle makes missing or unreliable hole-card data irrelevant.",
            "The raw supervised model is standalone production-approved despite the hole-card blocker.",
        ],
    }
    payload["delivery_strategy_claim_decision"] = evaluate_hole_card_delivery_strategy_boundary(payload)
    payload["invariants"] = validate_hole_card_data_quality(payload)
    payload["overall_status"] = "PASS" if payload["invariants"]["status"] == "PASS" else "FAIL"
    return payload


def validate_hole_card_data_quality(payload: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    risk = payload.get("risk_contract") or {}
    feature_policy = risk.get("feature_policy") or {}
    storage_boundary = risk.get("cards_storage_boundary") or {}
    strength_boundary = risk.get("hand_strength_feature_boundary") or {}
    coverage = payload.get("coverage_snapshot") or {}
    players_csv_contract = coverage.get("players_csv_cards_contract") or {}
    strength = payload.get("strength_signal_impact") or {}
    mitigation = payload.get("mitigation_boundary") or {}
    upstream = payload.get("upstream_data_quality_boundary") or {}
    direct_audit = coverage.get("direct_players_csv_audit") or {}
    promotion = payload.get("promotion_boundary") or {}
    delivery_strategy = payload.get("delivery_strategy_quality_boundary") or {}

    missing_rate = _as_float(coverage.get("missing_hole_card_rate"))
    complete_rate = _as_float(coverage.get("complete_hole_card_rate"))
    strength_zero_rate = _as_float(strength.get("strength_proxy_zero_rate"))
    direct_missing_rate = _as_float(direct_audit.get("missing_hole_card_rate"))
    direct_reliable_two_card_rate = _as_float(direct_audit.get("reliable_two_card_rate"))
    direct_invalid_card_rate = _as_float(direct_audit.get("invalid_card_rate"))
    required_affected_features = set(HOLE_CARD_RISK_CONTRACT["affected_features"])
    observed_affected_features = set(str(item) for item in strength.get("affected_features") or [])

    if risk.get("risk_id") != "hole_card_data_risk":
        violations.append("hole_card_risk_id_must_be_explicit")
    if risk.get("root_cause") != "ocr_hole_card_extraction_missing_or_unreliable":
        violations.append("hole_card_root_cause_must_remain_ocr_extraction_quality")
    if risk.get("primary_dataset_column") != "players.cards":
        violations.append("hole_card_primary_dataset_column_must_be_players_cards")
    if risk.get("source_table") != "players.csv":
        violations.append("hole_card_source_table_must_be_players_csv")
    if risk.get("source_field") != "players.csv::cards":
        violations.append("hole_card_source_field_must_be_players_csv_cards")
    if storage_boundary.get("players_csv_stores_hole_cards") is not True:
        violations.append("players_csv_cards_storage_must_be_explicit")
    if storage_boundary.get("storage_does_not_imply_reliability") is not True:
        violations.append("players_csv_cards_storage_must_not_imply_reliability")
    if storage_boundary.get("card_values_are_ocr_or_recognition_derived") is not True:
        violations.append("hole_card_values_must_remain_ocr_or_recognition_derived")
    if storage_boundary.get("missing_or_unreliable_cards_are_expected_dataset_conditions") is not True:
        violations.append("missing_or_unreliable_hole_cards_must_be_expected_dataset_conditions")
    if risk.get("weakens_primary_poker_signal") is not True:
        violations.append("hole_card_risk_must_weaken_primary_poker_signal")
    if strength_boundary.get("private_cards_are_primary_strategy_signal") is not True:
        violations.append("private_cards_must_be_marked_as_primary_strategy_signal")
    if strength_boundary.get("missing_or_invalid_hole_cards_limit_hand_strength_features") is not True:
        violations.append("missing_or_invalid_hole_cards_must_limit_hand_strength_features")
    if strength_boundary.get("hand_strength_features_must_be_slice_aware") is not True:
        violations.append("hand_strength_features_must_remain_slice_aware")
    if strength_boundary.get("standalone_card_aware_policy_requires_reliable_two_card_coverage") is not True:
        violations.append("standalone_card_aware_policy_must_require_reliable_two_card_coverage")
    if risk.get("current_delivery_blocker") is not False:
        violations.append("hole_card_risk_must_not_be_current_delivery_blocker")
    if risk.get("final_strategy_quality_claim_blocker") is not True:
        violations.append("hole_card_risk_must_block_final_strategy_quality_claim")
    if feature_policy.get("missing_or_invalid_cards") != "flag_and_route":
        violations.append("missing_or_invalid_hole_cards_must_be_flagged_and_routed")
    if feature_policy.get("do_not_impute_unknown_cards_as_known_private_cards") is not True:
        violations.append("unknown_hole_cards_must_not_be_imputed_as_known_private_cards")
    if feature_policy.get("do_not_treat_missing_cards_as_reliable_zero_strength") is not True:
        violations.append("missing_hole_cards_must_not_be_treated_as_reliable_zero_strength")
    if feature_policy.get("train_observed_card_and_public_context_slices_separately") is not True:
        violations.append("observed_and_missing_hole_card_slices_must_train_separately")
    if not required_affected_features.issubset(observed_affected_features):
        violations.append("hole_card_affected_features_must_include_primary_strength_signals")

    if missing_rate is None:
        violations.append("missing_hole_card_rate_is_required")
    if complete_rate is None:
        violations.append("complete_hole_card_rate_is_required")
    if players_csv_contract.get("source_table") != "players.csv":
        violations.append("players_csv_cards_contract_must_bind_source_table")
    if players_csv_contract.get("source_field") != "players.csv::cards":
        violations.append("players_csv_cards_contract_must_bind_source_field")
    if players_csv_contract.get("semantic") != "player_hole_cards":
        violations.append("players_csv_cards_contract_must_define_hole_card_semantic")
    if players_csv_contract.get("quality_source") != "ocr_or_card_recognition":
        violations.append("players_csv_cards_contract_must_mark_ocr_quality_source")
    if players_csv_contract.get("may_be_missing_or_unreliable") is not True:
        violations.append("players_csv_cards_contract_must_allow_missing_or_unreliable_values")
    if players_csv_contract.get("must_not_be_treated_as_reliable_by_presence_alone") is not True:
        violations.append("players_csv_cards_presence_must_not_imply_reliability")
    if missing_rate is not None and complete_rate is not None and missing_rate <= complete_rate:
        violations.append("current_audit_must_show_missing_hole_cards_as_the_dominant_condition")
    if not coverage.get("audit_finding"):
        violations.append("hole_card_audit_finding_must_remain_visible")

    high_missingness = missing_rate is not None and missing_rate >= MISSING_RATE_RISK_THRESHOLD
    high_strength_zero_rate = (
        strength_zero_rate is not None and strength_zero_rate >= STRENGTH_PROXY_ZERO_RATE_THRESHOLD
    )
    if high_missingness and strength.get("status") != DEGRADED_STRENGTH_SIGNAL:
        violations.append("high_hole_card_missingness_must_degrade_strength_signal")
    if high_strength_zero_rate and strength.get("status") != DEGRADED_STRENGTH_SIGNAL:
        violations.append("high_strength_proxy_zero_rate_must_degrade_strength_signal")
    if (high_missingness or high_strength_zero_rate) and (
        strength.get("primary_hand_strength_signal_reliable_for_standalone_policy") is not False
    ):
        violations.append("degraded_strength_signal_cannot_be_standalone_reliable")
    if strength.get("primary_signal_weakened_by_ocr_missingness") is not True:
        violations.append("hole_card_ocr_missingness_must_weaken_primary_signal")
    if strength.get("hand_strength_features_limited_by_card_quality") is not True:
        violations.append("hole_card_quality_must_limit_hand_strength_features")
    if high_strength_zero_rate and not strength.get("strength_proxy_audit_finding"):
        violations.append("strength_proxy_audit_finding_must_remain_visible")

    if direct_audit:
        if direct_audit.get("status") != "PASS":
            violations.append("direct_players_csv_audit_must_pass_when_present")
        if int(direct_audit.get("rows_scanned") or 0) <= 0:
            violations.append("direct_players_csv_audit_must_scan_rows")
        if direct_reliable_two_card_rate is None:
            violations.append("direct_reliable_two_card_rate_is_required")
        if direct_missing_rate is None:
            violations.append("direct_missing_hole_card_rate_is_required")
        if direct_invalid_card_rate is None:
            violations.append("direct_invalid_card_rate_is_required")
        direct_high_missingness = (
            direct_missing_rate is not None and direct_missing_rate >= MISSING_RATE_RISK_THRESHOLD
        )
        direct_low_reliability = (
            direct_reliable_two_card_rate is not None
            and direct_reliable_two_card_rate < RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD
        )
        direct_high_invalid_rate = (
            direct_invalid_card_rate is not None
            and direct_invalid_card_rate >= INVALID_CARD_RATE_PROMOTION_THRESHOLD
        )
        if (direct_high_missingness or direct_low_reliability or direct_high_invalid_rate) and (
            strength.get("status") != DEGRADED_STRENGTH_SIGNAL
        ):
            violations.append("direct_players_csv_risk_must_degrade_strength_signal")
        if direct_high_invalid_rate and not direct_audit.get("malformed_examples"):
            violations.append("direct_invalid_card_risk_must_include_examples")

    if mitigation.get("mitigation_status") != MITIGATED_BY_ROUTED_POLICY_BUNDLE:
        violations.append("routed_policy_bundle_mitigation_must_be_active")
    if mitigation.get("routed_policy_bundle_handles_missingness") is not True:
        violations.append("routed_policy_bundle_must_handle_missingness")
    if mitigation.get("fully_solves_upstream_data_quality_issue") is not False:
        violations.append("routed_policy_bundle_must_not_claim_to_fully_solve_upstream_data_quality")
    if mitigation.get("mitigation_scope") != "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR":
        violations.append("routed_policy_bundle_scope_must_remain_runtime_mitigation_not_data_repair")
    if mitigation.get("requires_slice_specific_monitoring") is not True:
        violations.append("hole_card_routes_must_require_slice_specific_monitoring")

    if upstream.get("limitation_status") != OPEN_DATA_QUALITY_LIMITATION:
        violations.append("hole_card_limitation_must_remain_open")
    if upstream.get("upstream_status") != UPSTREAM_NOT_RESOLVED:
        violations.append("upstream_hole_card_issue_must_remain_not_resolved")
    if upstream.get("upstream_data_quality_issue_resolved") is not False:
        violations.append("upstream_data_quality_issue_cannot_be_marked_resolved")
    if upstream.get("requires_ocr_or_parser_improvement") is not True:
        violations.append("ocr_or_parser_improvement_must_remain_required")
    if upstream.get("requires_larger_reviewed_card_labels") is not True:
        violations.append("larger_reviewed_card_labels_must_remain_required")
    if upstream.get("production_blocker_for_current_deployment") is not False:
        violations.append("hole_card_limitation_must_not_block_current_monitored_deployment")
    if upstream.get("component_risk") is not True:
        violations.append("hole_card_limitation_must_remain_a_component_risk")
    if upstream.get("players_csv_cards_are_not_reliability_guarantee") is not True:
        violations.append("players_csv_cards_must_not_be_reliability_guarantee")
    if upstream.get("hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve") is not True:
        violations.append("hand_strength_signal_limit_must_remain_until_data_repair")

    if promotion.get("standalone_policy_promotion_allowed") is not False:
        violations.append("hole_card_risk_must_block_standalone_policy_promotion")
    if promotion.get("model_promotion_blocker") is not True:
        violations.append("hole_card_risk_must_remain_model_promotion_blocker")
    if promotion.get("current_deployment_blocker") is not False:
        violations.append("hole_card_risk_must_not_block_current_deployment")
    if _as_float(promotion.get("requires_reliable_two_card_rate")) != RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD:
        violations.append("hole_card_promotion_reliable_two_card_threshold_must_be_explicit")
    if _as_float(promotion.get("requires_invalid_card_rate_below")) != INVALID_CARD_RATE_PROMOTION_THRESHOLD:
        violations.append("hole_card_promotion_invalid_card_threshold_must_be_explicit")
    if promotion.get("requires_reviewed_card_label_set") is not True:
        violations.append("hole_card_promotion_must_require_reviewed_card_labels")

    if delivery_strategy.get("risk_scope") != "MODEL_QUALITY_RISK_NOT_SERVICE_DELIVERY_BLOCKER":
        violations.append("hole_card_delivery_strategy_boundary_scope_must_be_explicit")
    if delivery_strategy.get("current_delivery_blocker") is not False:
        violations.append("hole_card_risk_must_not_be_delivery_blocker")
    if delivery_strategy.get("service_delivery_claim_allowed") is not True:
        violations.append("hole_card_risk_must_not_block_service_delivery_claim")
    if delivery_strategy.get("deployed_routed_stack_delivery_allowed") is not True:
        violations.append("hole_card_risk_must_not_block_deployed_routed_stack_delivery")
    if delivery_strategy.get("final_strategy_quality_claim_allowed") is not False:
        violations.append("hole_card_risk_must_block_final_strategy_quality_claim")
    if delivery_strategy.get("final_strategy_quality_claim_blocked_by_hole_card_data_quality") is not True:
        violations.append("hole_card_data_quality_must_block_final_strategy_quality_claim")
    if delivery_strategy.get("model_quality_risk") is not True:
        violations.append("hole_card_limitation_must_remain_model_quality_risk")
    if delivery_strategy.get("component_risk") is not True:
        violations.append("hole_card_limitation_must_remain_component_risk")
    required_clearance_items = {
        "improved_ocr_or_card_parser",
        "larger_reviewed_hole_card_label_set",
        "reliable_two_card_coverage_gate",
        "observed_card_policy_slice_gate",
        "standalone_card_aware_policy_promotion_gate",
    }
    if set(delivery_strategy.get("requires_to_clear_final_strategy_claim") or []) != required_clearance_items:
        violations.append("hole_card_final_strategy_claim_clearance_requirements_must_be_explicit")

    return {"status": "FAIL" if violations else "PASS", "violations": violations}


def is_open_hole_card_data_quality_risk(payload: dict[str, Any]) -> bool:
    risk = payload.get("risk_contract") or {}
    feature_policy = risk.get("feature_policy") or {}
    storage_boundary = risk.get("cards_storage_boundary") or {}
    strength_boundary = risk.get("hand_strength_feature_boundary") or {}
    coverage = payload.get("coverage_snapshot") or {}
    players_csv_contract = coverage.get("players_csv_cards_contract") or {}
    strength = payload.get("strength_signal_impact") or {}
    mitigation = payload.get("mitigation_boundary") or {}
    upstream = payload.get("upstream_data_quality_boundary") or {}
    promotion = payload.get("promotion_boundary") or {}
    delivery_strategy = payload.get("delivery_strategy_quality_boundary") or {}
    direct_audit = coverage.get("direct_players_csv_audit") or {}

    direct_reliable_two_card_rate = _as_float(direct_audit.get("reliable_two_card_rate"))
    direct_missing_rate = _as_float(direct_audit.get("missing_hole_card_rate"))
    direct_invalid_rate = _as_float(direct_audit.get("invalid_card_rate"))
    strength_zero_rate = _as_float(strength.get("strength_proxy_zero_rate"))

    if risk.get("risk_id") != "hole_card_data_risk":
        return False
    if risk.get("primary_dataset_column") != "players.cards":
        return False
    if risk.get("source_field") != "players.csv::cards":
        return False
    if storage_boundary.get("players_csv_stores_hole_cards") is not True:
        return False
    if storage_boundary.get("storage_does_not_imply_reliability") is not True:
        return False
    if storage_boundary.get("card_values_are_ocr_or_recognition_derived") is not True:
        return False
    if risk.get("weakens_primary_poker_signal") is not True:
        return False
    if strength_boundary.get("missing_or_invalid_hole_cards_limit_hand_strength_features") is not True:
        return False
    if risk.get("final_strategy_quality_claim_blocker") is not True:
        return False
    if players_csv_contract.get("source_field") != "players.csv::cards":
        return False
    if players_csv_contract.get("may_be_missing_or_unreliable") is not True:
        return False
    if players_csv_contract.get("must_not_be_treated_as_reliable_by_presence_alone") is not True:
        return False
    if feature_policy.get("missing_or_invalid_cards") != "flag_and_route":
        return False
    if feature_policy.get("do_not_impute_unknown_cards_as_known_private_cards") is not True:
        return False
    if feature_policy.get("do_not_treat_missing_cards_as_reliable_zero_strength") is not True:
        return False
    if strength.get("status") != DEGRADED_STRENGTH_SIGNAL:
        return False
    if strength.get("primary_hand_strength_signal_reliable_for_standalone_policy") is not False:
        return False
    if strength.get("primary_signal_weakened_by_ocr_missingness") is not True:
        return False
    if strength.get("hand_strength_features_limited_by_card_quality") is not True:
        return False
    if mitigation.get("mitigation_status") != MITIGATED_BY_ROUTED_POLICY_BUNDLE:
        return False
    if mitigation.get("mitigation_scope") != "RUNTIME_RISK_REDUCTION_NOT_DATA_REPAIR":
        return False
    if mitigation.get("fully_solves_upstream_data_quality_issue") is not False:
        return False
    if upstream.get("limitation_status") != OPEN_DATA_QUALITY_LIMITATION:
        return False
    if upstream.get("upstream_data_quality_issue_resolved") is not False:
        return False
    if upstream.get("production_blocker_for_current_deployment") is not False:
        return False
    if upstream.get("component_risk") is not True:
        return False
    if upstream.get("players_csv_cards_are_not_reliability_guarantee") is not True:
        return False
    if upstream.get("hand_strength_signal_remains_limited_until_ocr_and_reviewed_labels_improve") is not True:
        return False
    if promotion.get("standalone_policy_promotion_allowed") is not False:
        return False
    if promotion.get("model_promotion_blocker") is not True:
        return False
    if promotion.get("current_deployment_blocker") is not False:
        return False
    if delivery_strategy.get("risk_scope") != "MODEL_QUALITY_RISK_NOT_SERVICE_DELIVERY_BLOCKER":
        return False
    if delivery_strategy.get("current_delivery_blocker") is not False:
        return False
    if delivery_strategy.get("service_delivery_claim_allowed") is not True:
        return False
    if delivery_strategy.get("final_strategy_quality_claim_allowed") is not False:
        return False
    if delivery_strategy.get("final_strategy_quality_claim_blocked_by_hole_card_data_quality") is not True:
        return False
    if delivery_strategy.get("model_quality_risk") is not True:
        return False

    observed_data_risk = any(
        [
            direct_missing_rate is not None and direct_missing_rate >= MISSING_RATE_RISK_THRESHOLD,
            direct_reliable_two_card_rate is not None
            and direct_reliable_two_card_rate < RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD,
            direct_invalid_rate is not None and direct_invalid_rate >= INVALID_CARD_RATE_PROMOTION_THRESHOLD,
            strength_zero_rate is not None and strength_zero_rate >= STRENGTH_PROXY_ZERO_RATE_THRESHOLD,
        ]
    )
    return observed_data_risk


def evaluate_hole_card_delivery_strategy_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    risk = payload.get("risk_contract") or {}
    strength = payload.get("strength_signal_impact") or {}
    upstream = payload.get("upstream_data_quality_boundary") or {}
    boundary = payload.get("delivery_strategy_quality_boundary") or {}
    declared_open_hole_card_risk = (
        risk.get("risk_id") == "hole_card_data_risk"
        and risk.get("final_strategy_quality_claim_blocker") is True
        and strength.get("status") == DEGRADED_STRENGTH_SIGNAL
        and upstream.get("upstream_data_quality_issue_resolved") is False
    )
    open_hole_card_risk = is_open_hole_card_data_quality_risk(payload) or declared_open_hole_card_risk
    service_delivery_allowed = boundary.get("service_delivery_claim_allowed") is True
    routed_stack_delivery_allowed = boundary.get("deployed_routed_stack_delivery_allowed") is True
    delivery_blocker = boundary.get("current_delivery_blocker") is True
    final_claim_allowed = boundary.get("final_strategy_quality_claim_allowed") is True
    final_claim_blocked_by_hole_cards = (
        boundary.get("final_strategy_quality_claim_blocked_by_hole_card_data_quality") is True
    )

    violations: list[str] = []
    if delivery_blocker:
        violations.append("hole_card_risk_cannot_block_current_service_delivery")
    if not service_delivery_allowed:
        violations.append("service_delivery_must_remain_allowed_with_open_hole_card_risk")
    if not routed_stack_delivery_allowed:
        violations.append("deployed_routed_stack_delivery_must_remain_allowed")
    if open_hole_card_risk and final_claim_allowed:
        violations.append("open_hole_card_risk_must_block_final_strategy_quality_claim")
    if open_hole_card_risk and not final_claim_blocked_by_hole_cards:
        violations.append("final_strategy_quality_claim_must_name_hole_card_data_quality_blocker")
    if boundary.get("model_quality_risk") is not True:
        violations.append("hole_card_boundary_must_remain_model_quality_risk")
    if boundary.get("component_risk") is not True:
        violations.append("hole_card_boundary_must_remain_component_risk")

    return {
        "status": "PASS" if not violations else "FAIL",
        "service_delivery_ready": service_delivery_allowed and not delivery_blocker,
        "deployed_routed_stack_delivery_ready": routed_stack_delivery_allowed and not delivery_blocker,
        "open_hole_card_data_quality_risk": open_hole_card_risk,
        "final_strategy_quality_claim_allowed": final_claim_allowed and not open_hole_card_risk,
        "final_strategy_quality_claim_blocked": open_hole_card_risk and final_claim_blocked_by_hole_cards,
        "blocking_reason": "hole_card_data_quality_open" if open_hole_card_risk else None,
        "boundary": "DELIVERY_READY_FINAL_STRATEGY_CLAIM_BLOCKED_BY_HOLE_CARD_DATA_QUALITY",
        "violations": violations,
    }


def can_promote_standalone_policy_with_hole_cards(payload: dict[str, Any]) -> bool:
    if is_open_hole_card_data_quality_risk(payload):
        return False

    coverage = payload.get("coverage_snapshot") or {}
    strength = payload.get("strength_signal_impact") or {}
    upstream = payload.get("upstream_data_quality_boundary") or {}
    promotion = payload.get("promotion_boundary") or {}
    direct_audit = coverage.get("direct_players_csv_audit") or {}
    direct_reliable_two_card_rate = _as_float(direct_audit.get("reliable_two_card_rate"))
    direct_invalid_rate = _as_float(direct_audit.get("invalid_card_rate"))

    if direct_reliable_two_card_rate is None or direct_invalid_rate is None:
        return False
    return all(
        (
            direct_reliable_two_card_rate >= RELIABLE_TWO_CARD_RATE_PROMOTION_THRESHOLD,
            direct_invalid_rate < INVALID_CARD_RATE_PROMOTION_THRESHOLD,
            strength.get("primary_hand_strength_signal_reliable_for_standalone_policy") is True,
            upstream.get("upstream_data_quality_issue_resolved") is True,
            promotion.get("standalone_policy_promotion_allowed") is True,
        )
    )


def write_hole_card_data_quality(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_hole_card_data_quality(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_hole_card_data_quality_markdown(payload), encoding="utf-8")
    return payload


def render_hole_card_data_quality_markdown(payload: dict[str, Any]) -> str:
    risk = payload["risk_contract"]
    storage_boundary = risk["cards_storage_boundary"]
    strength_boundary = risk["hand_strength_feature_boundary"]
    coverage = payload["coverage_snapshot"]
    players_csv_contract = coverage["players_csv_cards_contract"]
    strength = payload["strength_signal_impact"]
    mitigation = payload["mitigation_boundary"]
    upstream = payload["upstream_data_quality_boundary"]
    direct_audit = coverage.get("direct_players_csv_audit") or {}
    promotion = payload["promotion_boundary"]
    delivery_strategy = payload["delivery_strategy_quality_boundary"]
    lines = [
        "# Hole-Card Data Quality Contract",
        "",
        payload["client_statement"],
        "",
        "## Risk Contract",
        "",
        f"- Risk ID: `{risk['risk_id']}`",
        f"- Root cause: `{risk['root_cause']}`",
        f"- Primary dataset column: `{risk['primary_dataset_column']}`",
        f"- Source field: `{risk['source_field']}`",
        f"- players.csv stores hole cards: `{storage_boundary['players_csv_stores_hole_cards']}`",
        f"- Storage implies reliability: `{not storage_boundary['storage_does_not_imply_reliability']}`",
        f"- Card values are OCR/recognition derived: `{storage_boundary['card_values_are_ocr_or_recognition_derived']}`",
        f"- Missing or unreliable cards are expected: `{storage_boundary['missing_or_unreliable_cards_are_expected_dataset_conditions']}`",
        f"- Weakens primary poker signal: `{risk['weakens_primary_poker_signal']}`",
        f"- Affected signal: `{risk['affected_signal']}`",
        f"- Private cards are primary strategy signal: `{strength_boundary['private_cards_are_primary_strategy_signal']}`",
        f"- Missing or invalid cards limit hand-strength features: `{strength_boundary['missing_or_invalid_hole_cards_limit_hand_strength_features']}`",
        f"- Hand-strength features must be slice-aware: `{strength_boundary['hand_strength_features_must_be_slice_aware']}`",
        f"- Current delivery blocker: `{risk['current_delivery_blocker']}`",
        f"- Final strategy-quality claim blocker: `{risk['final_strategy_quality_claim_blocker']}`",
        f"- Feature policy: `{risk['feature_policy']['missing_or_invalid_cards']}`",
        "",
        "Affected features:",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in risk["affected_features"])
    lines.extend(
        [
            "",
            "Decision impact:",
            "",
        ]
    )
    lines.extend(f"- {impact}" for impact in risk["decision_impact"])
    lines.extend(
        [
            "",
        "## Coverage Snapshot",
        "",
        f"- Missing hole-card rate: `{coverage.get('missing_hole_card_rate')}`",
        f"- Partial hole-card rate: `{coverage.get('partial_hole_card_rate')}`",
        f"- Complete hole-card rate: `{coverage.get('complete_hole_card_rate')}`",
        f"- Coverage source: `{coverage.get('coverage_source')}`",
        f"- players.csv cards source field: `{players_csv_contract['source_field']}`",
        f"- players.csv cards quality source: `{players_csv_contract['quality_source']}`",
        f"- players.csv cards may be missing/unreliable: `{players_csv_contract['may_be_missing_or_unreliable']}`",
        f"- Card presence alone is reliable evidence: `{not players_csv_contract['must_not_be_treated_as_reliable_by_presence_alone']}`",
        f"- Direct players.csv audit status: `{direct_audit.get('status')}`",
        f"- Direct players.csv rows scanned: `{direct_audit.get('rows_scanned')}`",
        f"- Direct reliable two-card rate: `{direct_audit.get('reliable_two_card_rate')}`",
        f"- Direct invalid-card rate: `{direct_audit.get('invalid_card_rate')}`",
        f"- Direct malformed examples retained: `{len(direct_audit.get('malformed_examples') or [])}`",
        "",
        "## Strength Signal Impact",
        "",
        f"- Strength signal status: `{strength['status']}`",
        f"- Strength proxy zero rate: `{strength.get('strength_proxy_zero_rate')}`",
        f"- Primary signal weakened by OCR missingness: `{strength.get('primary_signal_weakened_by_ocr_missingness')}`",
        f"- Hand-strength features limited by card quality: `{strength.get('hand_strength_features_limited_by_card_quality')}`",
        f"- Direct reliable two-card rate: `{strength.get('direct_reliable_two_card_rate')}`",
        f"- Direct invalid-card rate: `{strength.get('direct_invalid_card_rate')}`",
        "- Affected features: "
        + ", ".join(f"`{feature}`" for feature in strength.get("affected_features", [])),
        f"- Primary hand-strength signal reliable for standalone policy: `{strength['primary_hand_strength_signal_reliable_for_standalone_policy']}`",
        f"- Observed-hole-card macro F1: `{strength.get('observed_hole_cards_macro_f1')}`",
        f"- Observed-hole-card threshold: `{strength.get('observed_hole_cards_threshold')}`",
        f"- Challenger observed-hole-card macro F1: `{strength.get('challenger_observed_hole_cards_macro_f1')}`",
        f"- Expected impact: {strength['expected_model_impact']}",
        "",
        "## Mitigation Boundary",
        "",
        f"- Selected architecture: `{mitigation['selected_architecture']}`",
        f"- Mitigation status: `{mitigation['mitigation_status']}`",
        f"- Mitigation scope: `{mitigation['mitigation_scope']}`",
        f"- Requires slice-specific monitoring: `{mitigation['requires_slice_specific_monitoring']}`",
        f"- Fully solves upstream data-quality issue: `{mitigation['fully_solves_upstream_data_quality_issue']}`",
        "",
        "## Upstream Boundary",
        "",
        f"- Limitation status: `{upstream['limitation_status']}`",
        f"- Upstream status: `{upstream['upstream_status']}`",
        f"- Upstream data-quality issue resolved: `{upstream['upstream_data_quality_issue_resolved']}`",
        f"- Production blocker for current deployment: `{upstream['production_blocker_for_current_deployment']}`",
        f"- Component risk: `{upstream['component_risk']}`",
        "",
        "## Promotion Boundary",
        "",
        f"- Standalone policy promotion allowed: `{promotion['standalone_policy_promotion_allowed']}`",
        f"- Model promotion blocker: `{promotion['model_promotion_blocker']}`",
        f"- Current deployment blocker: `{promotion['current_deployment_blocker']}`",
        f"- Required reliable two-card rate: `{promotion['requires_reliable_two_card_rate']}`",
        f"- Required invalid-card rate below: `{promotion['requires_invalid_card_rate_below']}`",
        f"- Requires reviewed card-label set: `{promotion['requires_reviewed_card_label_set']}`",
        f"- Reason: {promotion['reason']}",
        "",
        "## Delivery vs Strategy-Quality Boundary",
        "",
        f"- Risk scope: `{delivery_strategy['risk_scope']}`",
        f"- Current delivery blocker: `{delivery_strategy['current_delivery_blocker']}`",
        f"- Service delivery claim allowed: `{delivery_strategy['service_delivery_claim_allowed']}`",
        f"- Deployed routed stack delivery allowed: `{delivery_strategy['deployed_routed_stack_delivery_allowed']}`",
        f"- Final strategy-quality claim allowed: `{delivery_strategy['final_strategy_quality_claim_allowed']}`",
        f"- Final strategy-quality claim blocked by hole-card data quality: `{delivery_strategy['final_strategy_quality_claim_blocked_by_hole_card_data_quality']}`",
        f"- Model-quality risk: `{delivery_strategy['model_quality_risk']}`",
        f"- Component risk: `{delivery_strategy['component_risk']}`",
        "",
        "Required to clear final strategy-quality claim:",
        "",
    ]
    )
    lines.extend(f"- `{item}`" for item in delivery_strategy["requires_to_clear_final_strategy_claim"])
    lines.extend(
        [
            "",
            "## Required Upstream Fixes",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["required_upstream_fixes"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["not_allowed_claims"])
    lines.extend(["", f"Invariant status: `{payload['invariants']['status']}`", ""])
    return "\n".join(lines)


def _split_card_tokens(raw_cards: Any) -> list[str]:
    if raw_cards is None:
        return []
    text = str(raw_cards).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]", "{}"}:
        return []
    text = (
        text.replace("[", " ")
        .replace("]", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace('"', " ")
        .replace("'", " ")
    )
    return [token for token in CARD_SPLIT_RE.split(text) if token]


def _partition_card_tokens(tokens: list[str]) -> tuple[list[str], list[str]]:
    valid_cards: list[str] = []
    invalid_tokens: list[str] = []
    for token in tokens:
        normalized = _normalize_card_token(token)
        if normalized is None:
            invalid_tokens.append(token)
        else:
            valid_cards.append(normalized)
    return valid_cards, invalid_tokens


def _normalize_card_token(token: str) -> str | None:
    normalized = (
        token.strip()
        .upper()
        .replace("♠", "S")
        .replace("♤", "S")
        .replace("♥", "H")
        .replace("♡", "H")
        .replace("♦", "D")
        .replace("♢", "D")
        .replace("♣", "C")
        .replace("♧", "C")
    )
    if normalized.startswith("10") and len(normalized) == 3:
        normalized = "T" + normalized[-1]
    if not CARD_TOKEN_RE.match(normalized):
        return None
    return normalized


def _find_hole_card_finding(findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for finding in findings:
        issue = str(finding.get("issue", "")).lower()
        if "hole-card" in issue or "hole card" in issue:
            return finding
    return None


def _find_strength_proxy_finding(findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for finding in findings:
        issue = str(finding.get("issue", "")).lower()
        if "strength_proxy" in issue or "strength proxy" in issue:
            return finding
    return None


def _gate_by_name(gates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for gate in gates:
        if gate.get("name") == name:
            return gate
    return {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
