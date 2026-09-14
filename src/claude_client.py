"""Headless driver for the acting agent via the first-party Claude Code CLI.

Hard project constraint: the acting model is reached ONLY through the bundled
Claude Code CLI on the Claude Max subscription. No metered API, no OpenRouter,
no LLM proxy, no provider fallback. This module is the single choke point that
enforces that, and it fails loudly rather than falling back.

Design notes
------------
* Stateless prompting. Every call is an isolated CLI invocation carrying the
  full observable transcript. We never use `--resume`/`--continue`, because the
  counterfactual fork needs byte-exact control of the dialogue prefix; hidden
  server-side session state would leak between arms.
* `--tools ""` gives the model no tools: it can only emit text, which is what
  our strict single-action output contract requires.
* `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` removes the Claude Code internal
  haiku side-traffic, so `modelUsage` contains `claude-opus-5` only (verified).
* There is NO temperature/top-p/seed flag in the CLI (verified against
  `claude --help`, v2.1.126). Decoding is whatever the service defaults to and
  cannot be forced greedy. Consequence: arm A0 must be re-executed per prefix.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

CLAUDE_BIN = os.environ.get(
    "CLAUDE_BIN",
    "/opt/prod/ai_scientists/Holosophus/.venv/lib/python3.14/site-packages/"
    "claude_agent_sdk/_bundled/claude",
)
ACTING_MODEL = "claude-opus-5"

FORBIDDEN_ENV = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)


class ProviderContractViolation(RuntimeError):
    pass


# Substrings that identify a Claude Max subscription rate limit in the CLI JSON.
RATE_LIMIT_MARKERS = (
    '"api_error_status":429',
    '"api_error_status": 429',
    "hit your limit",
    "rate_limit_error",
    "usage limit reached",
)

_RESET_RE = re.compile(r"resets\s+([^\"'}\\]+)", re.I)


class RateLimitError(RuntimeError):
    """Raised the moment the subscription limit is hit.

    Retrying a 429 on a Claude Max window is pointless (the window resets on a
    clock, not on backoff) and it burns the rest of the run's tasks. The driver
    catches this, persists partial progress, and exits cleanly.
    """

    def __init__(self, detail: str, reset_hint: str | None = None):
        super().__init__(detail[:400])
        self.detail = detail
        self.reset_hint = reset_hint


def detect_rate_limit(payload: str) -> str | None:
    """Return the matched marker if `payload` is a rate-limit response."""
    if not payload:
        return None
    low = payload.lower()
    for m in RATE_LIMIT_MARKERS:
        if m in low:
            return m
    return None


def parse_reset_hint(payload: str) -> str | None:
    m = _RESET_RE.search(payload or "")
    return m.group(1).strip() if m else None


@dataclass
class CallResult:
    ok: bool
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    session_id: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    attempts: int = 1
    errors: list[str] = field(default_factory=list)
    wall_s: float = 0.0

    def usage_vec(self) -> dict[str, int]:
        u = self.usage or {}
        return {
            "input_tokens": int(u.get("input_tokens", 0) or 0),
            "output_tokens": int(u.get("output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(u.get("cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(u.get("cache_read_input_tokens", 0) or 0),
        }


class ClaudeCLI:
    """One process per call; no shared conversation state."""

    def __init__(
        self,
        model: str = ACTING_MODEL,
        system_prompt: str | None = None,
        timeout_s: int = 240,
        max_attempts: int = 3,
        backoff_s: float = 8.0,
        binary: str = CLAUDE_BIN,
        stop_on_rate_limit: bool = True,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.binary = binary
        self.stop_on_rate_limit = stop_on_rate_limit
        self.event_log: list[dict] = []
        self.rate_limit_events: list[dict] = []
        self.n_requests = 0
        if not os.path.isfile(self.binary):
            raise ProviderContractViolation(f"Claude CLI not found at {self.binary}")

    def _raise_if_rate_limited(self, payload: str, tag: str, attempt: int) -> None:
        marker = detect_rate_limit(payload)
        if marker is None:
            return
        hint = parse_reset_hint(payload)
        ev = {
            "ts": time.time(),
            "tag": tag,
            "attempt": attempt,
            "marker": marker,
            "reset_hint": hint,
            "payload": payload[:800],
        }
        self.rate_limit_events.append(ev)
        self._record(tag, "rate_limit_429", attempt, payload[:800])
        if self.stop_on_rate_limit:
            raise RateLimitError(payload[:800], hint)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Never let a metered key silently take over the subscription path.
        for k in FORBIDDEN_ENV:
            env.pop(k, None)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env

    def _argv(self) -> list[str]:
        argv = [
            self.binary,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--tools",
            "",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
        ]
        if self.system_prompt is not None:
            argv += ["--system-prompt", self.system_prompt]
        return argv

    def call(self, prompt: str, tag: str = "") -> CallResult:
        errors: list[str] = []
        t_start = time.time()
        for attempt in range(1, self.max_attempts + 1):
            try:
                proc = subprocess.run(
                    self._argv(),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env=self._env(),
                    cwd="/tmp",
                )
            except subprocess.TimeoutExpired:
                errors.append(f"attempt{attempt}: timeout after {self.timeout_s}s")
                self._record(tag, "timeout", attempt, errors[-1])
                time.sleep(self.backoff_s * attempt)
                continue

            self.n_requests += 1

            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "")[:600]
                # A subscription 429 arrives here: rc=1 with a JSON body that
                # carries api_error_status 429. Check before spending retries.
                self._raise_if_rate_limited(proc.stdout or proc.stderr or "", tag, attempt)
                errors.append(f"attempt{attempt}: rc={proc.returncode} {msg}")
                self._record(tag, "nonzero_rc", attempt, errors[-1])
                time.sleep(self.backoff_s * attempt)
                continue

            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                errors.append(f"attempt{attempt}: non-JSON stdout {proc.stdout[:300]}")
                self._record(tag, "bad_json", attempt, errors[-1])
                time.sleep(self.backoff_s * attempt)
                continue

            models = sorted((data.get("modelUsage") or {}).keys())
            if data.get("is_error"):
                self._raise_if_rate_limited(proc.stdout, tag, attempt)
                errors.append(f"attempt{attempt}: is_error subtype={data.get('subtype')} "
                              f"result={str(data.get('result'))[:300]}")
                self._record(tag, "api_error", attempt, errors[-1])
                time.sleep(self.backoff_s * attempt)
                continue

            if self.model not in models:
                # Model did not resolve as requested -> provider contract broken.
                raise ProviderContractViolation(
                    f"expected {self.model} in modelUsage, got {models}"
                )

            return CallResult(
                ok=True,
                text=str(data.get("result", "")),
                raw={k: data.get(k) for k in
                     ("subtype", "num_turns", "duration_ms", "duration_api_ms",
                      "session_id", "total_cost_usd", "permission_denials")},
                duration_ms=int(data.get("duration_ms") or 0),
                duration_api_ms=int(data.get("duration_api_ms") or 0),
                num_turns=int(data.get("num_turns") or 0),
                session_id=str(data.get("session_id") or ""),
                usage=data.get("usage") or {},
                models=models,
                attempts=attempt,
                errors=errors,
                wall_s=round(time.time() - t_start, 3),
            )

        return CallResult(
            ok=False,
            text="",
            attempts=self.max_attempts,
            errors=errors,
            wall_s=round(time.time() - t_start, 3),
        )

    def _record(self, tag: str, kind: str, attempt: int, detail: str) -> None:
        self.event_log.append(
            {
                "ts": time.time(),
                "tag": tag,
                "kind": kind,
                "attempt": attempt,
                "detail": detail[:1000],
            }
        )


def verify_provider(binary: str = CLAUDE_BIN) -> dict:
    """Smoke-check that claude-opus-5 resolves and no side-model is billed."""
    cli = ClaudeCLI(system_prompt="You are a terse assistant.")
    r = cli.call("Reply with exactly: ok", tag="provider_check")
    ver = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.strip()
    return {
        "cli_version": ver,
        "ok": r.ok,
        "text": r.text,
        "models_billed": r.models,
        "only_acting_model": r.models == [ACTING_MODEL],
        "usage": r.usage_vec(),
        "duration_ms": r.duration_ms,
    }


if __name__ == "__main__":
    print(json.dumps(verify_provider(), indent=2))
