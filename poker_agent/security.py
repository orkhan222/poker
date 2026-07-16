from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from poker_agent.mlops import stable_digest, utc_now

SECURITY_CONTRACT_VERSION = "security_contract.v1"
AUTH_SCHEMA_VERSION = "api_auth.v1"
RATE_LIMIT_SCHEMA_VERSION = "rate_limit.v1"
SECRET_MANAGEMENT_SCHEMA_VERSION = "secret_management.v1"
LOG_RETENTION_SCHEMA_VERSION = "log_retention.v1"

DEFAULT_SECURITY_PATHS = {
    "security_report": "reports/security_report.json",
    "config": "configs/security/local.yaml",
}

SECRET_FIELD_HINTS = ("api_key", "authorization", "password", "secret", "token")


@dataclass(frozen=True)
class SecurityConfig:
    auth_required: bool = False
    api_key_hashes: tuple[str, ...] = ()
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 60
    rate_limit_window_seconds: int = 60
    retention_days: int = 30
    retention_max_records: int = 100_000
    retention_enabled: bool = True


@dataclass(frozen=True)
class AuthResult:
    allowed: bool
    principal: str
    credential_hash_prefix: str | None = None
    error_code: str | None = None
    message: str = ""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int = 0


@dataclass(frozen=True)
class LogRetentionPolicy:
    max_age_days: int = 30
    max_records: int = 100_000
    enabled: bool = True


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def redact_secret(value: str | None, *, visible: int = 4) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= visible:
        return "*" * len(text)
    return f"{text[:visible]}...redacted"


def redact_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = key.lower()
        if any(hint in lowered for hint in SECRET_FIELD_HINTS):
            redacted[key] = redact_secret(str(value))
        elif isinstance(value, dict):
            redacted[key] = redact_mapping(value)
        else:
            redacted[key] = value
    return redacted


def normalize_hashes(raw_hashes: str | None) -> tuple[str, ...]:
    if not raw_hashes:
        return ()
    hashes = []
    for item in raw_hashes.replace(";", ",").split(","):
        value = item.strip().lower()
        if value:
            hashes.append(value)
    return tuple(sorted(set(hashes)))


def security_config_from_env(env: dict[str, str] | None = None) -> SecurityConfig:
    env = env or os.environ
    plaintext_keys = tuple(key.strip() for key in env.get("POKER_API_KEYS", "").split(",") if key.strip())
    hash_values = set(normalize_hashes(env.get("POKER_API_KEY_HASHES") or env.get("POKER_API_KEY_SHA256")))
    hash_values.update(hash_secret(key) for key in plaintext_keys)
    auth_required = str(env.get("POKER_AUTH_REQUIRED", "")).lower() in {"1", "true", "yes"}
    if hash_values:
        auth_required = True if env.get("POKER_AUTH_REQUIRED") is None else auth_required
    return SecurityConfig(
        auth_required=auth_required,
        api_key_hashes=tuple(sorted(hash_values)),
        rate_limit_per_minute=int(env.get("POKER_RATE_LIMIT_PER_MINUTE", "60")),
        rate_limit_burst=int(env.get("POKER_RATE_LIMIT_BURST", env.get("POKER_RATE_LIMIT_PER_MINUTE", "60"))),
        rate_limit_window_seconds=int(env.get("POKER_RATE_LIMIT_WINDOW_SECONDS", "60")),
        retention_days=int(env.get("POKER_LOG_RETENTION_DAYS", "30")),
        retention_max_records=int(env.get("POKER_LOG_RETENTION_MAX_RECORDS", "100000")),
        retention_enabled=str(env.get("POKER_LOG_RETENTION_ENABLED", "true")).lower() not in {"0", "false", "no"},
    )


def bearer_or_api_key(headers: dict[str, str]) -> str | None:
    normalized = {key.lower(): value for key, value in headers.items()}
    explicit = normalized.get("x-api-key")
    if explicit:
        return explicit.strip()
    authorization = normalized.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def authenticate_headers(headers: dict[str, str], config: SecurityConfig) -> AuthResult:
    supplied = bearer_or_api_key(headers)
    if not config.auth_required and not supplied:
        return AuthResult(allowed=True, principal="anonymous")
    if config.auth_required and not config.api_key_hashes:
        return AuthResult(
            allowed=False,
            principal="unknown",
            error_code="SECURITY_MISCONFIGURED",
            message="API authentication is required but no API key hashes are configured.",
        )
    if not supplied:
        return AuthResult(
            allowed=False,
            principal="unknown",
            error_code="UNAUTHORIZED",
            message="Missing API key. Use X-API-Key or Authorization: Bearer.",
        )
    supplied_hash = hash_secret(supplied)
    for expected_hash in config.api_key_hashes:
        if hmac.compare_digest(supplied_hash, expected_hash):
            return AuthResult(
                allowed=True,
                principal=f"api_key:{supplied_hash[:12]}",
                credential_hash_prefix=supplied_hash[:12],
            )
    return AuthResult(
        allowed=False,
        principal="unknown",
        credential_hash_prefix=supplied_hash[:12],
        error_code="UNAUTHORIZED",
        message="Invalid API key.",
    )


class InMemoryRateLimiter:
    def __init__(self, *, limit: int = 60, window_seconds: int = 60):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, identity: str, *, now: float | None = None) -> RateLimitResult:
        now = time.time() if now is None else now
        lower_bound = now - self.window_seconds
        with self._lock:
            hits = [stamp for stamp in self._hits.get(identity, []) if stamp > lower_bound]
            if len(hits) >= self.limit:
                retry_after = max(1, int(round(self.window_seconds - (now - hits[0]))))
                self._hits[identity] = hits
                return RateLimitResult(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )
            hits.append(now)
            self._hits[identity] = hits
            return RateLimitResult(
                allowed=True,
                limit=self.limit,
                remaining=max(0, self.limit - len(hits)),
            )


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def prune_jsonl_by_retention(
    path: Path,
    policy: LogRetentionPolicy,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not policy.enabled:
        return {"schema_version": LOG_RETENTION_SCHEMA_VERSION, "enabled": False, "path": str(path)}
    if not path.exists():
        return {
            "schema_version": LOG_RETENTION_SCHEMA_VERSION,
            "enabled": True,
            "path": str(path),
            "kept": 0,
            "removed": 0,
        }
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(0, policy.max_age_days))
    kept: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            continue
        created_at = parse_timestamp(payload.get("created_at"))
        if created_at is not None and created_at < cutoff:
            removed += 1
            continue
        kept.append(json.dumps(payload, sort_keys=True))
    if policy.max_records > 0 and len(kept) > policy.max_records:
        removed += len(kept) - policy.max_records
        kept = kept[-policy.max_records :]
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return {
        "schema_version": LOG_RETENTION_SCHEMA_VERSION,
        "enabled": True,
        "path": str(path),
        "kept": len(kept),
        "removed": removed,
        "max_age_days": policy.max_age_days,
        "max_records": policy.max_records,
    }


def auth_audit_payload(result: AuthResult) -> dict[str, Any]:
    return {
        "principal": result.principal,
        "credential_hash_prefix": result.credential_hash_prefix,
        "allowed": result.allowed,
        "error_code": result.error_code,
    }


def describe_security_contract() -> dict[str, Any]:
    return {
        "schema_version": SECURITY_CONTRACT_VERSION,
        "api_auth": {
            "schema_version": AUTH_SCHEMA_VERSION,
            "methods": ["X-API-Key", "Authorization: Bearer"],
            "secret_storage": "environment_sha256_hashes",
            "required_env": ["POKER_API_KEY_HASHES or POKER_API_KEYS", "POKER_AUTH_REQUIRED"],
            "error_codes": ["UNAUTHORIZED", "SECURITY_MISCONFIGURED"],
        },
        "rate_limiting": {
            "schema_version": RATE_LIMIT_SCHEMA_VERSION,
            "strategy": "in_memory_fixed_window_per_principal",
            "env": ["POKER_RATE_LIMIT_PER_MINUTE", "POKER_RATE_LIMIT_WINDOW_SECONDS"],
            "error_code": "RATE_LIMITED",
        },
        "secret_management": {
            "schema_version": SECRET_MANAGEMENT_SCHEMA_VERSION,
            "rules": [
                "Never commit raw API keys.",
                "Prefer POKER_API_KEY_HASHES with sha256 hex values.",
                "Redact authorization, api_key, token, password, and secret fields before reporting.",
            ],
        },
        "log_retention": {
            "schema_version": LOG_RETENTION_SCHEMA_VERSION,
            "paths": ["reports/prediction_logs.jsonl", "reports/audit_trail.jsonl"],
            "default_max_age_days": 30,
            "default_max_records": 100000,
        },
        "paths": DEFAULT_SECURITY_PATHS,
    }


def validate_security_contract(root: Path) -> list[dict[str, Any]]:
    expected_files = [
        "poker_agent/security.py",
        "scripts/check_security_contract.py",
        "configs/security/local.yaml",
        "configs/experiments/security_smoke.yaml",
        "tests/test_security_contract.py",
    ]
    checks = [
        {"name": f"file:{relative}", "passed": (root / relative).exists(), "detail": relative}
        for relative in expected_files
    ]
    service_text = (root / "poker_agent" / "service.py").read_text(encoding="utf-8")
    for token in ("authenticate_headers", "InMemoryRateLimiter", "prune_jsonl_by_retention"):
        checks.append({"name": f"service:{token}", "passed": token in service_text, "detail": token})
    return checks


def security_smoke_fingerprint() -> str:
    return stable_digest(describe_security_contract())
