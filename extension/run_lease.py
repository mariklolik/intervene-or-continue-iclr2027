import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def await_model(endpoint: str, model: str, deadline: float, server: subprocess.Popen) -> None:
    while time.time() < deadline and server.poll() is None:
        try:
            with urllib.request.urlopen(endpoint + "/models", timeout=min(2, max(0.01, deadline - time.time()))) as response:
                if model in {row["id"] for row in json.load(response)["data"]}:
                    return
        except Exception:
            pass
        time.sleep(min(0.5, max(0, deadline - time.time())))
    raise RuntimeError("Exact served model did not become ready within the lease")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lease")
    args = parser.parse_args()
    lease = json.loads(Path(args.lease).read_text())
    root = Path(lease["root"])
    sys.path[:0] = [str(root / "extension"), str(root)]
    from sampling_preflight import verify_sampling
    endpoint = f"http://127.0.0.1:{lease['port']}/v1"
    started = time.time()
    command = [sys.executable, str(root / "extension/server_guard.py")]
    for key in ("root", "name", "gpu", "port", "model_path", "model", "seconds", "image"):
        command.extend(["--" + key.replace("_", "-"), str(lease[key])])
    command.extend(["--context-length", str(lease.get("context_length", 16384))])
    command.extend(["--cap-gpu-seconds", str(lease.get("cap_gpu_seconds", 36000))])
    server = subprocess.Popen(command)
    experiment = None
    log_path = root / "logs" / f"{lease['name']}.episodes.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        await_model(endpoint, lease["model"], min(started + 150, started + lease["seconds"] - 60), server)
        verify_sampling(endpoint, lease["model"], min(started + 210, started + lease["seconds"] - 60), root / "logs" / f"{lease['name']}.log", root / "runtime" / f"{lease['name']}.sampling.json", expected_context=lease.get("context_length", 16384))
        remaining = int(started + lease["seconds"] - time.time() - 60)
        if remaining < 1:
            raise RuntimeError("No experiment time remains in the lease")
        environment = {**os.environ, "ALFWORLD_DATA": str(root / "data/alfworld"), "PYTHONPATH": str(root)}
        command = [sys.executable, str(root / "extension/experiment.py"), "--config", lease["config"], "--endpoint", endpoint, "--out", lease["out"], "--workers", str(lease["workers"]), "--max-seconds", str(remaining)]
        if lease.get("baseline_only"):
            command.append("--baseline-only")
        with log_path.open("w") as log:
            experiment = subprocess.Popen(command, env=environment, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = experiment.wait(timeout=remaining + 10)
            except subprocess.TimeoutExpired:
                code = 124
        print(json.dumps({"name": lease["name"], "experiment_exit": code, "wall_seconds": time.time() - started, "log": str(log_path)}), flush=True)
    finally:
        (root / f"{lease['name']}.stop").touch()
        if experiment is not None and experiment.poll() is None:
            os.killpg(experiment.pid, signal.SIGTERM)
            try:
                experiment.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(experiment.pid, signal.SIGKILL)
                experiment.wait(timeout=2)
        server.wait(timeout=40)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
