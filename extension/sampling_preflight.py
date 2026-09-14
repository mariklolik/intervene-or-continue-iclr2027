import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from local_client import LocalClient


def verify_sampling(endpoint: str, model: str, deadline: float, log: Path, output: Path, expected_context: int | None = None) -> None:
    source = log.read_bytes()
    text = source.decode(errors="replace")
    flags = bool(re.search(r"['\"]enable_deterministic_inference['\"]:\s*True", text) and re.search(r"['\"]sampling_backend['\"]:\s*['\"]pytorch['\"]", text))
    receipt = {"time": time.time(), "model": model, "server_log": str(log), "server_log_sha256_at_probe": hashlib.sha256(source).hexdigest(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "required_runtime_flags": flags, "events": [], "passed": False}
    contexts = [int(value) for value in re.findall(r"['\"]context_length['\"]:\s*(\d+)(?=[,}\s]|$)", text)]
    context_match = expected_context is None or bool(contexts) and all(value == expected_context for value in contexts)
    receipt.update(expected_context=expected_context, logged_context_lengths=contexts, context_log_match=context_match, context_probe=None)
    if flags and context_match:
        def call(seed: int):
            client = LocalClient(endpoint, model, seed, deadline, temperature=1.0, max_tokens=64, system_prompt="Follow the request directly.")
            result = client.call("Return twelve unrelated common English nouns separated by commas. Choose a varied selection.", tag="seed-probe")
            return result, client.events

        first = call(103)
        with ThreadPoolExecutor(max_workers=3) as pool:
            others = list(pool.map(call, [103, 107, 109]))
        results = [first, *others]
        receipt["events"] = [event for _, events in results for event in events]
        receipt["same_seed_same_text"] = first[0].text == others[0][0].text
        receipt["different_seed_diversity"] = len({result.text for result, _ in results})
        receipt["passed"] = bool(receipt["same_seed_same_text"] and all(result.ok and result.text and result.models == [model] for result, _ in results))
        receipt["limitation"] = "Concrete seeded request and one mixed batch verified; not a universal bitwise-invariance guarantee"
        if receipt["passed"] and expected_context == 32768:
            prompt = " x" * 17000
            client = LocalClient(endpoint, model, 113, deadline, temperature=1.0, max_tokens=1, system_prompt="Follow the request directly.")
            result = client.call(prompt, tag="context-probe")
            tokens = result.usage.get("input_tokens")
            passed = bool(result.ok and result.models == [model] and type(tokens) is int and 16384 < tokens and tokens + 1 <= expected_context)
            receipt["context_probe"] = {"passed": passed, "requested_max_tokens": 1, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "usage": result.usage, "events": client.events, "models": result.models, "ok": result.ok, "errors": result.errors}
            receipt["passed"] = passed
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise RuntimeError(f"Sampling preflight failed; receipt: {output}")
