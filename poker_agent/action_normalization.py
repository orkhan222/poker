from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


CANONICAL_ACTIONS = ("fold", "call", "check", "bet", "raise", "all_in")
NON_DECISION_ACTIONS = (
    "ante",
    "post_sb",
    "post_bb",
    "small_blind",
    "big_blind",
    "joined",
    "sit_out",
    "won",
    "muck",
)

ACTION_ALIASES = {
    "f0ld": "fold",
    "folds": "fold",
    "foid": "fold",
    "ca11": "call",
    "ca1l": "call",
    "cal1": "call",
    "cail": "call",
    "calls": "call",
    "called": "call",
    "chek": "check",
    "ch3ck": "check",
    "checks": "check",
    "bett": "bet",
    "bettt": "bet",
    "bets": "bet",
    "b3t": "bet",
    "ra1se": "raise",
    "ra1sed": "raise",
    "r4ise": "raise",
    "ralse": "raise",
    "raises": "raise",
    "raised": "raise",
    "all-in": "all_in",
    "all in": "all_in",
    "allin": "all_in",
    "a11in": "all_in",
    "a11 in": "all_in",
    "all_in": "all_in",
    "jam": "all_in",
    "jams": "all_in",
    "shove": "all_in",
    "shoves": "all_in",
}

OCR_REPLACEMENTS = str.maketrans(
    {
        "|": "l",
        "!": "i",
        "$": "s",
        "€": "e",
    }
)


@dataclass(frozen=True)
class ActionNormalizationResult:
    raw_action: str
    canonical_action: str
    status: str
    method: str
    confidence: float
    is_decision_action: bool


def normalize_action(raw_action: Any) -> str:
    return normalize_action_result(raw_action).canonical_action


def assert_canonical_decision_action(action: Any, *, context: str) -> str:
    canonical = "" if action is None else str(action)
    if canonical not in CANONICAL_ACTIONS:
        raise ValueError(
            f"Non-canonical action label in {context}: {canonical!r}. "
            "Raw OCR/dealer labels must be normalized to fold/call/check/bet/raise/all_in before use."
        )
    return canonical


def normalize_action_record(
    record: Mapping[str, Any],
    *,
    action_field: str = "action",
    normalized_field: str = "canonical_action",
    raw_field: str = "raw_action",
) -> dict[str, Any]:
    """Return a row copy with explicit canonical action metadata."""
    result = normalize_action_result(record.get(action_field))
    enriched = dict(record)
    if raw_field and raw_field not in enriched:
        enriched[raw_field] = result.raw_action
    enriched[normalized_field] = result.canonical_action
    enriched["action_normalization_status"] = result.status
    enriched["action_normalization_method"] = result.method
    enriched["action_normalization_confidence"] = result.confidence
    enriched["is_decision_action"] = result.is_decision_action
    return enriched


def normalize_action_result(raw_action: Any) -> ActionNormalizationResult:
    raw = "" if raw_action is None else str(raw_action)
    text = _clean_text(raw)
    if not text:
        return _result(raw, "unknown", "unknown", "empty", 0.0)

    direct = _direct_match(text)
    if direct is not None:
        return _result(raw, direct, "canonical", "direct_or_alias", 1.0)

    phrase = _phrase_match(text)
    if phrase is not None:
        return _result(raw, phrase, "canonical", "phrase", 0.96)

    token_match = _token_match(text)
    if token_match is not None:
        canonical, method, confidence = token_match
        return _result(raw, canonical, "canonical", method, confidence)

    non_decision = _non_decision_match(text)
    if non_decision is not None:
        return _result(raw, non_decision, "non_decision", "non_decision_alias", 1.0)

    return _result(raw, "unknown", "unknown", "unmatched", 0.0)


def _clean_text(raw: str) -> str:
    text = raw.strip().lower().translate(OCR_REPLACEMENTS)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _direct_match(text: str) -> str | None:
    collapsed = text.replace(" ", "")
    spaced = text.replace(" all in ", " all_in ")
    for candidate in (text, collapsed, spaced):
        if candidate in ACTION_ALIASES:
            return ACTION_ALIASES[candidate]
        if candidate in CANONICAL_ACTIONS:
            return candidate
    return None


def _phrase_match(text: str) -> str | None:
    if re.search(r"\ball\s+in\b|\ballin\b", text):
        return "all_in"
    for canonical in ("fold", "check", "call", "raise", "bet"):
        if re.search(rf"\b{canonical}\w*\b", text):
            return canonical
    return None


def _token_match(text: str) -> tuple[str, str, float] | None:
    tokens = [token for token in text.split() if not token.isnumeric()]
    candidates: list[tuple[str, str, float]] = []
    for token in tokens:
        if token in ACTION_ALIASES:
            candidates.append((ACTION_ALIASES[token], "token_alias", 0.98))
            continue
        normalized = _normalize_common_ocr_digits(token)
        if normalized in ACTION_ALIASES:
            candidates.append((ACTION_ALIASES[normalized], "ocr_digit_alias", 0.95))
            continue
        if normalized in CANONICAL_ACTIONS:
            candidates.append((normalized, "ocr_digit_canonical", 0.95))
            continue
        for canonical in CANONICAL_ACTIONS:
            score = SequenceMatcher(None, normalized, canonical.replace("_", "")).ratio()
            if score >= 0.72:
                candidates.append((canonical, "fuzzy", round(score, 4)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0]


def _non_decision_match(text: str) -> str | None:
    collapsed = text.replace(" ", "_")
    for action in NON_DECISION_ACTIONS:
        if action in text or action in collapsed:
            return action
    return None


def _normalize_common_ocr_digits(token: str) -> str:
    if any(char.isdigit() for char in token):
        token = token.replace("0", "o").replace("1", "i").replace("3", "e").replace("5", "s")
    return token


def _result(
    raw_action: str,
    canonical_action: str,
    status: str,
    method: str,
    confidence: float,
) -> ActionNormalizationResult:
    return ActionNormalizationResult(
        raw_action=raw_action,
        canonical_action=canonical_action,
        status=status,
        method=method,
        confidence=confidence,
        is_decision_action=canonical_action in CANONICAL_ACTIONS,
    )
