import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=["baseline", "arms"], required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap-gpu-seconds", type=int, default=72000)
    parser.add_argument("--port-base", type=int, default=18200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    template = json.loads((args.root / "configs/qwen3-template.json").read_text())
    args.out.mkdir(parents=True)
    for shard in range(8):
        stem = f"panel-shard{shard}"
        plan = {
            "root": str(args.root),
            "name": f"ioc27-{args.phase}-s{shard}",
            "gpu": shard,
            "port": args.port_base + shard,
            "model_path": template["model_path"],
            "model": template["model"],
            "seconds": args.seconds,
            "image": template["image"],
            "context_length": template["runtime_contract"]["context_length"],
            "cap_gpu_seconds": args.cap_gpu_seconds,
            "config": str(args.root / f"configs/independent-panel/{stem}.json"),
            "out": str(args.root / f"raw/{stem}"),
            "workers": args.workers,
            "baseline_only": args.phase == "baseline",
        }
        (args.out / f"{stem}.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
