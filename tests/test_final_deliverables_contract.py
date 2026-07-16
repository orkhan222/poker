from __future__ import annotations

from pathlib import Path

from poker_agent.deliverables import (
    FINAL_DELIVERABLES_CONTRACT_VERSION,
    build_smoke_reports,
    describe_final_deliverables_contract,
    validate_final_deliverables,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_deliverables_contract_declares_all_required_outputs() -> None:
    contract = describe_final_deliverables_contract()

    assert contract["schema_version"] == FINAL_DELIVERABLES_CONTRACT_VERSION
    keys = {item["key"] for item in contract["deliverables"]}
    assert {
        "validated_dataset_schema",
        "validation_report",
        "baseline_comparison",
        "trained_checkpoint",
        "evaluation_report",
        "dockerized_fastapi_service",
        "api_docs",
        "tests",
    }.issubset(keys)


def test_smoke_generation_writes_machine_readable_final_artifacts() -> None:
    outputs = build_smoke_reports(ROOT)
    manifest = validate_final_deliverables(ROOT)

    assert manifest["status"] == "PASS"
    assert manifest["schema_version"] == FINAL_DELIVERABLES_CONTRACT_VERSION
    assert manifest["fingerprint"]
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert (ROOT / "reports" / "final_model_selection.json").exists()
    assert (ROOT / "docs" / "FINAL_MODEL_SELECTION.md").exists()


def test_generated_api_docs_cover_endpoint_schema_and_errors() -> None:
    build_smoke_reports(ROOT)
    docs = (ROOT / "docs" / "API_CONTRACT.md").read_text(encoding="utf-8")

    assert "POST /predict" in docs
    assert "predict_request.v1" in docs
    assert "predict_response.v1" in docs
    assert "INVALID_REQUEST" in docs
    assert "RATE_LIMITED" in docs


def test_generated_evaluation_report_references_qwenpoker_final_selection() -> None:
    build_smoke_reports(ROOT)
    report = (ROOT / "reports" / "evaluation_report.json").read_text(encoding="utf-8")

    assert "checkpoint_40960" in report
    assert "qwenpoker:checkpoint_40960" in report
