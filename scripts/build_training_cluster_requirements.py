from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.training_cluster import DEFAULT_RUN_PROFILE, RUN_PROFILES, write_training_cluster_requirements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-agent training cluster requirements")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--run-profile", default=DEFAULT_RUN_PROFILE, choices=tuple(RUN_PROFILES))
    parser.add_argument("--out", default=ROOT / "reports" / "training_cluster_requirements.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "training_cluster_requirements.md", type=Path)
    parser.add_argument("--gpu-type", default=None)
    parser.add_argument("--gpu-count", default=None, type=int)
    parser.add_argument("--vram-gb-per-gpu", default=None, type=float)
    parser.add_argument("--cpu-cores", default=None, type=int)
    parser.add_argument("--system-ram-gb", default=None, type=float)
    parser.add_argument("--storage-gb", default=None, type=float)
    parser.add_argument("--interconnect", default=None)
    parser.add_argument("--dedicated-or-shared", default=None, choices=("dedicated", "shared"))
    return parser.parse_args()


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
    payload = write_training_cluster_requirements(
        args.project_root,
        args.out,
        markdown_out=args.markdown_out,
        cluster=cluster,
        run_profile=args.run_profile,
    )
    print(
        json.dumps(
            {
                "status": payload["estimate"]["status"],
                "run_profile": payload["run_profile"],
                "estimated_hours": payload["estimate"]["estimated_hours"],
                "estimated_days": payload["estimate"]["estimated_days"],
                "confidence": payload["estimate"]["confidence"],
                "requested_fields": len(payload["requested_fields"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()