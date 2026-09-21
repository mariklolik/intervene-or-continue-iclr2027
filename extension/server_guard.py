import argparse
import fcntl
import json
import os
import subprocess
import signal
import sys
import time
from pathlib import Path


def reserve(path: Path, name: str, seconds: int, gpus: int, cap: int = 36000) -> None:
    if seconds <= 0 or gpus <= 0:
        raise ValueError("Positive allocation required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        leases = {row["name"]: row["gpu_seconds"] for row in rows if row["event"] == "reserve"}
        for row in rows:
            if row["event"] == "closed":
                leases[row["name"]] = row["gpu_seconds"]
        requested = seconds * gpus
        if name in leases or sum(leases.values()) + requested > cap:
            raise ValueError("Duplicate lease or H100 GPU-hour cap exceeded")
        stream.write(json.dumps({"event": "reserve", "name": name, "time": time.time(), "gpu_seconds": requested, "cap": cap}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--cap-gpu-seconds", type=int, default=36000)
    parser.add_argument("--mem-fraction", type=float, default=0.75)
    args = parser.parse_args()
    if args.seconds < 60:
        raise ValueError("Shutdown reserve requires at least 60 seconds")
    root = Path(args.root)
    ledger = root / "gpu_budget.jsonl"
    reserve(ledger, args.name, args.seconds, 1, args.cap_gpu_seconds)
    started = time.time()
    command = ["sudo", "-n", "docker", "run", "--rm", "--pull=never", "--name", args.name, "--gpus", f"device={args.gpu}", "--ipc=host", "--network=host", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-v", "/home/mekashirskiy/.cache/huggingface/hub:/home/mekashirskiy/.cache/huggingface/hub:ro", "-v", f"{root}/extension:/experiment:ro", "--entrypoint", "python3", args.image, "/experiment/container_deadline.py", str(started + args.seconds - 15), "python3", "-m", "sglang.launch_server", "--model-path", args.model_path, "--served-model-name", args.model, "--host", "127.0.0.1", "--port", str(args.port), "--context-length", str(args.context_length), "--mem-fraction-static", str(args.mem_fraction), "--max-running-requests", "64", "--tp-size", "1", "--disable-cuda-graph", "--enable-deterministic-inference"]
    log_path = root / "logs" / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, lambda signum, frame: sys.exit(128 + signum))
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None and time.time() - started < args.seconds - 30:
                if (root / f"{args.name}.stop").exists():
                    break
                time.sleep(1)
        finally:
            try:
                subprocess.run(["sudo", "-n", "docker", "stop", "--time", "10", args.name], timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    subprocess.run(["sudo", "-n", "docker", "rm", "-f", args.name], timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.TimeoutExpired:
                    pass
                if process.poll() is None:
                    process.wait(timeout=5)
    elapsed = time.time() - started
    inspection = subprocess.run(["sudo", "-n", "docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5)
    verified = inspection.returncode == 0 and args.name not in inspection.stdout.splitlines()
    with ledger.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({"event": "closed" if verified else "cleanup_unverified", "name": args.name, "time": time.time(), "gpu_seconds": elapsed if verified else args.seconds, "exit_code": process.returncode, "log": str(log_path)}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"name": args.name, "gpu_seconds": elapsed, "exit_code": process.returncode}), flush=True)


if __name__ == "__main__":
    main()
