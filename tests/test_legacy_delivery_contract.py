from __future__ import annotations

import json
from pathlib import Path

from poker_agent.legacy_reports import (
    LEGACY_REPORTS_CONTRACT_VERSION,
    build_legacy_delivery_reports,
    describe_legacy_reports_contract,
    validate_legacy_delivery_reports,
)
from poker_agent.model import load_policy
from scripts.check_repo_hygiene import path_findings

ROOT = Path(__file__).resolve().parents[1]


def test_joblib_model_path_can_fallback_to_json_checkpoint_metadata() -> None:
    policy = load_policy(ROOT / "models" / "poker_policy.joblib")
    metadata = getattr(policy, "metadata", {}) or {}

    assert metadata["split"]["split_type"] == "stratified_hand_group_holdout"
    assert "macro_f1" in metadata["valid_metrics"]


def test_legacy_reports_contract_generates_required_compatibility_artifacts() -> None:
    contract = describe_legacy_reports_contract()
    outputs = build_legacy_delivery_reports(ROOT, overwrite=True)
    validation = validate_legacy_delivery_reports(ROOT)

    assert contract["schema_version"] == LEGACY_REPORTS_CONTRACT_VERSION
    assert validation["status"] == "PASS"
    assert outputs["llm_transformer_gold_eval"].exists()
    transformer = json.loads(outputs["llm_transformer_gold_eval"].read_text(encoding="utf-8"))
    assert transformer["systems"]["schema_routed_smol_hybrid"]["event_type"]["macro_f1"] >= 0.90
    assert transformer["systems"]["schema_routed_smol_hybrid"]["llm_fallback_count"] > 0


def test_repo_hygiene_ignores_runtime_pycache_directories() -> None:
    cache_dir = ROOT / "tests" / "__pycache__"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "hygiene_probe.pyc").write_bytes(b"0")

    findings = path_findings(ROOT)

    assert not any("__pycache__" in item["path"] for item in findings)
