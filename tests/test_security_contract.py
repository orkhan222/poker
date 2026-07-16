from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poker_agent.security import (
    InMemoryRateLimiter,
    LogRetentionPolicy,
    SecurityConfig,
    authenticate_headers,
    describe_security_contract,
    hash_secret,
    prune_jsonl_by_retention,
    redact_mapping,
    security_config_from_env,
    validate_security_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_api_auth_accepts_valid_key_and_rejects_missing_or_bad_key() -> None:
    secret = "unit-test-key"
    config = SecurityConfig(auth_required=True, api_key_hashes=(hash_secret(secret),))

    good = authenticate_headers({"x-api-key": secret}, config)
    missing = authenticate_headers({}, config)
    bad = authenticate_headers({"authorization": "Bearer wrong"}, config)

    assert good.allowed is True
    assert good.principal.startswith("api_key:")
    assert missing.allowed is False and missing.error_code == "UNAUTHORIZED"
    assert bad.allowed is False and bad.error_code == "UNAUTHORIZED"


def test_security_config_loads_hashed_and_plaintext_env_without_exposing_secret() -> None:
    secret = "env-test-key"
    config = security_config_from_env({"POKER_API_KEYS": secret, "POKER_AUTH_REQUIRED": "true"})

    assert config.auth_required is True
    assert hash_secret(secret) in config.api_key_hashes
    assert secret not in json.dumps(config.__dict__)


def test_rate_limiter_blocks_after_limit_until_window_resets() -> None:
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)

    assert limiter.check("principal", now=100.0).allowed is True
    assert limiter.check("principal", now=101.0).allowed is True
    blocked = limiter.check("principal", now=102.0)
    assert blocked.allowed is False
    assert blocked.retry_after_seconds > 0
    assert limiter.check("principal", now=200.0).allowed is True


def test_secret_redaction_removes_sensitive_values_recursively() -> None:
    secret = "secret-value"
    redacted = redact_mapping({"api_key": secret, "nested": {"authorization": f"Bearer {secret}"}})

    assert secret not in json.dumps(redacted)
    assert "redacted" in redacted["api_key"]


def test_log_retention_removes_old_records_and_limits_record_count() -> None:
    with tempfile.TemporaryDirectory() as raw_temp:
        path = Path(raw_temp) / "prediction_logs.jsonl"
        old = {"created_at": (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(), "event": "old"}
        recent_a = {"created_at": datetime.now(timezone.utc).isoformat(), "event": "recent_a"}
        recent_b = {"created_at": datetime.now(timezone.utc).isoformat(), "event": "recent_b"}
        path.write_text("\n".join(json.dumps(item) for item in (old, recent_a, recent_b)) + "\n", encoding="utf-8")

        report = prune_jsonl_by_retention(
            path,
            LogRetentionPolicy(max_age_days=30, max_records=1, enabled=True),
            now=datetime.now(timezone.utc),
        )
        retained = path.read_text(encoding="utf-8")

    assert report["removed"] == 2
    assert "old" not in retained
    assert "recent_b" in retained


def test_repo_security_contract_files_are_present() -> None:
    contract = describe_security_contract()
    failed = [item for item in validate_security_contract(ROOT) if not item["passed"]]

    assert contract["schema_version"] == "security_contract.v1"
    assert contract["api_auth"]["methods"] == ["X-API-Key", "Authorization: Bearer"]
    assert not failed
