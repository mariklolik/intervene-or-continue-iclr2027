import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--panel", default="independent-panel")
    parser.add_argument("--prefix", default="ioc27")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--raw", default="raw")
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
    gpus = [int(value) for value in args.gpus.split(",") if value != ""]
    for shard, gpu in enumerate(gpus):
        stem = f"panel-shard{shard}"
        plan = {
            "root": str(args.root),
            "name": f"{args.prefix}-{args.phase}-s{shard}",
            "gpu": gpu,
            "port": args.port_base + shard,
            "model_path": template["model_path"],
            "model": template["model"],
            "seconds": args.seconds,
            "image": template["image"],
            "context_length": template["runtime_contract"]["context_length"],
            "cap_gpu_seconds": args.cap_gpu_seconds,
            "config": str(args.root / f"configs/{args.panel}/{stem}.json"),
            "out": str(args.root / f"{args.raw}/{stem}"),
            "workers": args.workers,
            "baseline_only": args.phase.startswith("baseline"),
        }
        (args.out / f"{stem}.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
