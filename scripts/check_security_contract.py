from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.security import (
    InMemoryRateLimiter,
    LogRetentionPolicy,
    SecurityConfig,
    authenticate_headers,
    describe_security_contract,
    hash_secret,
    prune_jsonl_by_retention,
    redact_mapping,
    security_smoke_fingerprint,
    validate_security_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate API auth, rate limiting, secret management, and log retention")
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--out", default=Path("reports/security_report.json"), type=Path)
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--retention-max-records", type=int, default=100000)
    parser.add_argument("--rate-limit-per-minute", type=int, default=60)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def write_retention_fixture(path: Path) -> None:
    old = {"created_at": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(), "event": "old"}
    recent = {"created_at": datetime.now(timezone.utc).isoformat(), "event": "recent"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old) + "\n" + json.dumps(recent) + "\n", encoding="utf-8")


def smoke_checks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    secret = "local-smoke-api-key"
    config = SecurityConfig(
        auth_required=True,
        api_key_hashes=(hash_secret(secret),),
        rate_limit_per_minute=args.rate_limit_per_minute,
        rate_limit_burst=2,
        rate_limit_window_seconds=60,
        retention_days=args.retention_days,
        retention_max_records=args.retention_max_records,
    )
    good = authenticate_headers({"x-api-key": secret}, config)
    missing = authenticate_headers({}, config)
    bad = authenticate_headers({"authorization": "Bearer wrong"}, config)

    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)
    first = limiter.check("principal", now=100.0)
    second = limiter.check("principal", now=101.0)
    third = limiter.check("principal", now=102.0)

    redacted = redact_mapping(
        {
            "api_key": secret,
            "nested": {"authorization": f"Bearer {secret}", "safe": "visible"},
        }
    )

    with tempfile.TemporaryDirectory() as raw_temp:
        log_path = Path(raw_temp) / "prediction_logs.jsonl"
        write_retention_fixture(log_path)
        retention = prune_jsonl_by_retention(
            log_path,
            LogRetentionPolicy(max_age_days=30, max_records=10, enabled=True),
            now=datetime.now(timezone.utc),
        )
        retained_text = log_path.read_text(encoding="utf-8")

    checks = [
        check("api_auth:valid_key", good.allowed and good.principal.startswith("api_key:"), good.__dict__),
        check("api_auth:missing_key_rejected", not missing.allowed and missing.error_code == "UNAUTHORIZED", missing.__dict__),
        check("api_auth:bad_key_rejected", not bad.allowed and bad.error_code == "UNAUTHORIZED", bad.__dict__),
        check("rate_limit:first_allowed", first.allowed and second.allowed, second.__dict__),
        check("rate_limit:third_rejected", not third.allowed and third.retry_after_seconds > 0, third.__dict__),
        check("secret_management:redacted", secret not in json.dumps(redacted), redacted),
        check("log_retention:old_removed", retention["removed"] == 1 and "old" not in retained_text, retention),
    ]
    artifacts = {
        "auth": {
            "valid": good.__dict__,
            "missing": missing.__dict__,
            "bad": bad.__dict__,
        },
        "rate_limit": {
            "first": first.__dict__,
            "second": second.__dict__,
            "third": third.__dict__,
        },
        "redacted_example": redacted,
        "retention": retention,
        "security_fingerprint": security_smoke_fingerprint(),
    }
    return checks, artifacts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    checks = validate_security_contract(root)
    artifacts: dict[str, Any] = {}
    if args.smoke:
        smoke, artifacts = smoke_checks(args)
        checks.extend(smoke)

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    payload = {
        "status": status,
        "contract": describe_security_contract(),
        "checks": checks,
        "artifacts": artifacts,
    }
    out = resolve(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
