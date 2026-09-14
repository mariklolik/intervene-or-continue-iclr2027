import hashlib
import json
import time
import urllib.request
from urllib.parse import urlparse

from src.agent import SYSTEM_PROMPT
from src.claude_client import CallResult


class LocalClient:
    def __init__(self, endpoint: str, model: str, seed: int, deadline: float, temperature: float = 0.7, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 96):
        if urlparse(endpoint).hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("The experiment requires a local inference endpoint")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.seed = seed
        self.deadline = deadline
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.n_requests = 0
        self.events = []

    def call(self, prompt: str, tag: str = "") -> CallResult:
        started = time.time()
        remaining = self.deadline - started
        if remaining <= 0:
            return CallResult(ok=False, text="", errors=["budget_deadline"])
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.n_requests}:{tag}".encode()).digest()[:4], "big") % 2147483647
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(self.endpoint + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        self.n_requests += 1
        try:
            with urllib.request.urlopen(request, timeout=min(120, remaining)) as response:
                data = json.load(response)
            text = data["choices"][0]["message"].get("content") or ""
            elapsed = time.time() - started
            usage = data.get("usage", {})
            result = CallResult(ok=True, text=text, raw={"model": data.get("model"), "seed": seed, "finish_reason": data["choices"][0].get("finish_reason")}, usage={"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}, models=[data.get("model", self.model)], duration_ms=round(elapsed * 1000), duration_api_ms=round(elapsed * 1000), wall_s=elapsed, session_id=f"local:{self.model}:{seed}")
        except Exception as error:
            result = CallResult(ok=False, text="", errors=[f"{type(error).__name__}: {error}"], wall_s=time.time() - started)
        self.events.append({"started": started, "tag": tag, "seed": seed, "ok": result.ok, "reply": result.text, "errors": result.errors, "wall_s": result.wall_s, "usage_known": result.ok, **result.usage_vec(), **result.raw})
        return result
