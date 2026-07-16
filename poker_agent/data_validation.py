from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from poker_agent.action_space import normalize_action


DEFAULT_TOLERANCE = 0.01

MISSING_OCR_CONFLICT_POLICY = {
    "missing_ocr_confidence": "preserve_as_missing_and_report_rate",
    "missing_hole_cards": "flag_and_route_to_missing_card_fallback",
    "conflicting_action_rows": "do_not_merge_silently; require upstream review or highest-confidence resolver",
    "missing_chip_action_amount": "flag_for_review; do not infer amount for call/bet/raise/all_in labels",
    "missing_legal_actions": "derive_action_space_at_inference_but_report_dataset_gap",
}


def validate_dataset(
    dataset_dir: Path,
    *,
    max_rows: int = 0,
    tolerance: float = DEFAULT_TOLERANCE,
    max_samples: int = 20,
) -> dict[str, Any]:
    paths = {
        "hands": dataset_dir / "hands.csv",
        "players": dataset_dir / "players.csv",
        "actions": dataset_dir / "actions.csv",
        "stack_events": dataset_dir / "stack_events.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {
            "status": "FAIL",
            "policy": MISSING_OCR_CONFLICT_POLICY,
            "missing_files": missing,
            "issues": [
                _issue(
                    "blocker",
                    "Dataset validation cannot run because required CSV files are missing.",
                    f"Missing files: {missing}",
                )
            ],
        }

    hands = _read_rows(paths["hands"], max_rows)
    players = _read_rows(paths["players"], max_rows)
    actions = _read_rows(paths["actions"], max_rows)
    stack_events = _read_rows(paths["stack_events"], max_rows)

    checks = {
        "pot_conservation": _check_pot_conservation(actions, tolerance=tolerance, max_samples=max_samples),
        "stack_delta_consistency": _check_stack_delta_consistency(
            players,
            stack_events,
            tolerance=tolerance,
            max_samples=max_samples,
        ),
        "duplicate_hand_detection": _check_duplicate_hands(hands, max_samples=max_samples),
        "missing_ocr_conflict_policy": _check_missing_ocr_conflicts(actions, max_samples=max_samples),
    }
    issues: list[dict[str, Any]] = []
    for check in checks.values():
        issues.extend(check.get("issues", []))
    return {
        "status": "PASS" if not any(issue["severity"] in {"blocker", "high"} for issue in issues) else "FAIL",
        "tolerance": tolerance,
        "policy": MISSING_OCR_CONFLICT_POLICY,
        "checks": checks,
        "issues": issues,
    }


def _check_pot_conservation(rows: list[dict[str, str]], *, tolerance: float, max_samples: int) -> dict[str, Any]:
    total = 0
    comparable = 0
    missing_context = 0
    violations = 0
    monotonic_violations = 0
    samples: list[dict[str, Any]] = []
    last_after_by_hand: dict[str, float] = {}

    for row in rows:
        total += 1
        before = _optional_float(row.get("pot_before_action"))
        after = _optional_float(row.get("pot_after_action"))
        amount = _optional_float(row.get("action_amount"))
        if before is None or after is None:
            missing_context += 1
            continue
        comparable += 1
        expected_after = before + max(amount or 0.0, 0.0)
        if abs(after - expected_after) > tolerance:
            violations += 1
            _append_sample(
                samples,
                {
                    "hand_id": row.get("hand_id"),
                    "frame_id": row.get("frame_id"),
                    "action": row.get("action"),
                    "pot_before_action": before,
                    "action_amount": amount,
                    "pot_after_action": after,
                    "expected_pot_after_action": round(expected_after, 6),
                },
                max_samples,
            )
        hand_id = str(row.get("hand_id") or "")
        if hand_id:
            previous_after = last_after_by_hand.get(hand_id)
            if previous_after is not None and before + tolerance < previous_after:
                monotonic_violations += 1
            last_after_by_hand[hand_id] = after

    issues: list[dict[str, Any]] = []
    if violations or monotonic_violations:
        issues.append(
            _issue(
                "blocker",
                "Pot conservation failed for action rows.",
                "pot_after_action must equal pot_before_action plus positive action_amount, and hand pot must not regress.",
            )
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "rows": total,
        "comparable_rows": comparable,
        "missing_context_rows": missing_context,
        "violation_rows": violations,
        "monotonic_violation_rows": monotonic_violations,
        "samples": samples,
        "issues": issues,
    }


def _check_stack_delta_consistency(
    player_rows: list[dict[str, str]],
    stack_rows: list[dict[str, str]],
    *,
    tolerance: float,
    max_samples: int,
) -> dict[str, Any]:
    player_violations = 0
    event_violations = 0
    comparable_players = 0
    comparable_events = 0
    samples: list[dict[str, Any]] = []
    player_delta_by_key: dict[tuple[str, str], float] = {}

    for row in player_rows:
        start = _optional_float(row.get("starting_stack"))
        end = _optional_float(row.get("ending_stack"))
        delta = _optional_float(row.get("stack_delta"))
        key = (str(row.get("hand_id") or ""), str(row.get("position") or ""))
        if delta is not None:
            player_delta_by_key[key] = delta
        if start is None or end is None or delta is None:
            continue
        comparable_players += 1
        expected_delta = end - start
        if abs(delta - expected_delta) > tolerance:
            player_violations += 1
            _append_sample(
                samples,
                {
                    "hand_id": key[0],
                    "position": key[1],
                    "starting_stack": start,
                    "ending_stack": end,
                    "stack_delta": delta,
                    "expected_stack_delta": round(expected_delta, 6),
                },
                max_samples,
            )

    event_deltas: dict[tuple[str, str], float] = defaultdict(float)
    stack_after_violations = 0
    for row in stack_rows:
        key = (str(row.get("hand_id") or ""), str(row.get("player_position") or ""))
        diff = _optional_float(row.get("diff"))
        if diff is not None:
            event_deltas[key] += diff
        stack = _optional_float(row.get("stack"))
        stack_after = _optional_float(row.get("stack_after_event"))
        if stack is not None and stack_after is not None and abs(stack - stack_after) > tolerance:
            stack_after_violations += 1

    for key, event_delta in event_deltas.items():
        player_delta = player_delta_by_key.get(key)
        if player_delta is None:
            continue
        comparable_events += 1
        if abs(event_delta - player_delta) > tolerance:
            event_violations += 1
            _append_sample(
                samples,
                {
                    "hand_id": key[0],
                    "position": key[1],
                    "sum_stack_event_diff": round(event_delta, 6),
                    "player_stack_delta": player_delta,
                },
                max_samples,
            )

    issues: list[dict[str, Any]] = []
    if player_violations or event_violations or stack_after_violations:
        issues.append(
            _issue(
                "blocker",
                "Stack delta consistency failed.",
                "players.stack_delta must match ending-starting stack and stack event diffs must reconcile to player deltas.",
            )
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "comparable_player_rows": comparable_players,
        "player_delta_violation_rows": player_violations,
        "comparable_event_groups": comparable_events,
        "event_delta_violation_groups": event_violations,
        "stack_after_event_violation_rows": stack_after_violations,
        "samples": samples,
        "issues": issues,
    }


def _check_duplicate_hands(rows: list[dict[str, str]], *, max_samples: int) -> dict[str, Any]:
    hand_ids = Counter(str(row.get("hand_id") or "") for row in rows if row.get("hand_id"))
    source_local = Counter(
        (
            str(row.get("source_file") or ""),
            str(row.get("local_hand_index") or ""),
        )
        for row in rows
        if row.get("source_file") and row.get("local_hand_index") not in {None, ""}
    )
    frame_signature = Counter(
        (
            str(row.get("source_file") or ""),
            str(row.get("start_frame") or ""),
            str(row.get("end_frame") or ""),
            str(row.get("board_cards") or ""),
        )
        for row in rows
        if row.get("source_file") and row.get("start_frame") not in {None, ""}
    )

    duplicates = {
        "hand_id": _duplicate_samples(hand_ids, max_samples),
        "source_local_hand_index": _duplicate_samples(source_local, max_samples),
        "frame_signature": _duplicate_samples(frame_signature, max_samples),
    }
    duplicate_count = sum(len(values) for values in duplicates.values())
    issues: list[dict[str, Any]] = []
    if duplicate_count:
        issues.append(
            _issue(
                "blocker",
                "Duplicate hand detection failed.",
                "hand_id, source/local hand index, and frame signature must identify one hand only.",
            )
        )
    return {
        "status": "PASS" if not issues else "FAIL",
        "rows": len(rows),
        "duplicate_groups": duplicates,
        "issues": issues,
    }


def _check_missing_ocr_conflicts(rows: list[dict[str, str]], *, max_samples: int) -> dict[str, Any]:
    missing_ocr_confidence = 0
    missing_legal_actions = 0
    missing_chip_action_amount = 0
    grouped: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    samples: list[dict[str, Any]] = []

    for row in rows:
        action = normalize_action(row.get("action"))
        if not row.get("ocr_confidence"):
            missing_ocr_confidence += 1
        if not row.get("legal_actions"):
            missing_legal_actions += 1
        if action in {"call", "bet", "raise", "all_in"} and not row.get("action_amount"):
            missing_chip_action_amount += 1
            _append_sample(
                samples,
                {
                    "hand_id": row.get("hand_id"),
                    "frame_id": row.get("frame_id"),
                    "player_position": row.get("player_position"),
                    "action": row.get("action"),
                    "issue": "missing_chip_action_amount",
                },
                max_samples,
            )
        grouped[
            (
                str(row.get("hand_id") or ""),
                str(row.get("frame_id") or ""),
                str(row.get("player_position") or ""),
            )
        ].add((action, str(row.get("action_amount") or "")))

    conflict_groups = {
        key: sorted(values)
        for key, values in grouped.items()
        if key[0] and key[1] and key[2] and len(values) > 1
    }
    for key, values in list(conflict_groups.items())[:max_samples]:
        _append_sample(
            samples,
            {
                "hand_id": key[0],
                "frame_id": key[1],
                "player_position": key[2],
                "conflicting_actions": values,
            },
            max_samples,
        )

    issues: list[dict[str, Any]] = []
    if conflict_groups:
        issues.append(
            _issue(
                "high",
                "Conflicting OCR action rows were detected.",
                "Rows with the same hand/frame/player must not disagree on action or amount without explicit resolution.",
            )
        )
    if missing_chip_action_amount:
        issues.append(
            _issue(
                "high",
                "Chip-moving actions are missing action_amount.",
                "call, bet, raise, and all_in rows must carry action_amount or be routed to review.",
            )
        )
    if missing_legal_actions or missing_ocr_confidence:
        issues.append(
            _issue(
                "medium",
                "OCR metadata is incomplete.",
                "Missing legal_actions can be derived at inference, and missing ocr_confidence is preserved as unknown.",
            )
        )

    total = len(rows)
    return {
        "status": "PASS" if not any(issue["severity"] in {"blocker", "high"} for issue in issues) else "FAIL",
        "rows": total,
        "policy": MISSING_OCR_CONFLICT_POLICY,
        "missing_ocr_confidence_rows": missing_ocr_confidence,
        "missing_ocr_confidence_rate": missing_ocr_confidence / total if total else 0.0,
        "missing_legal_actions_rows": missing_legal_actions,
        "missing_legal_actions_rate": missing_legal_actions / total if total else 0.0,
        "missing_chip_action_amount_rows": missing_chip_action_amount,
        "conflict_groups": len(conflict_groups),
        "samples": samples,
        "issues": issues,
    }


def _read_rows(path: Path, max_rows: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader, start=1):
            if max_rows and index > max_rows:
                break
            rows.append(row)
        return rows


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def _duplicate_samples(counter: Counter[Any], max_samples: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for key, count in counter.items():
        if count <= 1:
            continue
        _append_sample(samples, {"key": key, "count": count}, max_samples)
    return samples


def _append_sample(samples: list[dict[str, Any]], sample: dict[str, Any], max_samples: int) -> None:
    if len(samples) < max_samples:
        samples.append(sample)


def _issue(severity: str, issue: str, recommendation: str) -> dict[str, str]:
    return {
        "severity": severity,
        "issue": issue,
        "expected_impact": "High data quality risk for supervised poker policy training.",
        "recommendation": recommendation,
    }
