from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.model import load_policy
from poker_agent.today_training import (
    build_today_training_plan,
    build_today_training_report,
    write_today_training_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run today's routed policy bundle acceptance training")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--dataset", default=ROOT / "data", type=Path)
    parser.add_argument("--model-out", default=ROOT / "models" / "poker_policy_bundle.joblib", type=Path)
    parser.add_argument("--report-out", default=ROOT / "reports" / "today_acceptance_training.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "today_acceptance_training.md", type=Path)
    parser.add_argument("--gate-out", default=ROOT / "reports" / "today_acceptance_production_gate.json", type=Path)
    parser.add_argument("--max-examples", default=1000, type=int)
    parser.add_argument("--gpu-type", default=None)
    parser.add_argument("--gpu-count", default=None, type=int)
    parser.add_argument("--vram-gb-per-gpu", default=None, type=float)
    parser.add_argument("--cpu-cores", default=None, type=int)
    parser.add_argument("--system-ram-gb", default=None, type=float)
    parser.add_argument("--storage-gb", default=None, type=float)
    parser.add_argument("--interconnect", default=None)
    parser.add_argument("--dedicated-or-shared", default=None, choices=("dedicated", "shared"))
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> None:
    args = parse_args()
    cluster = {
        "gpu_type": args.gpu_type,
        "gpu_count": args.gpu_count,
        "vram_gb_per_gpu": args.vram_gb_per_gpu,
        "cpu_cores": args.cpu_cores,
        "system_ram_gb": args.system_ram_gb,
        "storage_gb": args.storage_gb,
        "interconnect": args.interconnect,
        "dedicated_or_shared": args.dedicated_or_shared,
    }
    plan = build_today_training_plan(
        args.project_root,
        dataset=args.dataset,
        model_out=args.model_out,
        max_examples=args.max_examples,
        cluster=cluster,
    )

    if args.skip_training:
        training_result = {
            "command": plan["training_command"],
            "returncode": 0 if args.model_out.exists() else 2,
            "stdout_tail": "training skipped; existing model artifact reused" if args.model_out.exists() else "",
            "stderr_tail": "model artifact missing" if not args.model_out.exists() else "",
        }
    else:
        command = [sys.executable] + plan["training_command"][1:]
        training_result = run_command(command, args.project_root)

    metadata: dict[str, Any] = {}
    if args.model_out.exists():
        model = load_policy(args.model_out)
        metadata = getattr(model, "metadata", {}) or {}

    gate_result: dict[str, Any] = {"status": "NOT_RUN"}
    if args.model_out.exists():
        gate_command = [
            sys.executable,
            "scripts/production_gate.py",
            "--model",
            str(args.model_out),
            "--out",
            str(args.gate_out),
        ]
        gate_process = run_command(gate_command, args.project_root)
        if args.gate_out.exists():
            gate_result = json.loads(args.gate_out.read_text(encoding="utf-8"))
        gate_result["process"] = gate_process

    report = build_today_training_report(
        plan,
        training_result=training_result,
        model_metadata=metadata,
        gate_result=gate_result,
    )
    write_today_training_report(report, args.report_out, args.markdown_out)
    print(
        json.dumps(
            {
                "delivery_status": report["delivery_status"],
                "training_status": report["training_status"],
                "production_gate_status": report["production_gate_status"],
                "model_out": report["model_out"],
                "report_out": str(args.report_out),
            },
            sort_keys=True,
        )
    )
    if report["training_status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()