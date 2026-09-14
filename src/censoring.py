"""Canonical censoring rule for the whole project.

Motivation
----------
The Step-2 pilot was interrupted mid-run by a Claude Max 5-hour subscription
rate limit (HTTP 429, "You've hit your limit - resets 5:30pm (UTC)"). Episodes
that were killed by that limit are NOT agent failures: the environment never
reached a terminal state and the agent was never given the chance to act. If
they were counted as failures every downstream number (success rate, training
label Y, tau_a, policy value) would be biased downward by an artefact of the
billing system.

This module is the SINGLE place where "is this record scientifically usable?"
is decided, so that the pilot analysis, the baseline trajectory stage, the
counterfactual replay matrix, the predictors and the policy evaluation all
agree by construction.

Rule
----
A record is VALID iff BOTH hold:

  1. It terminated for a genuine environment reason, i.e. one of
       - ``success``          : the environment reported the task won;
       - ``step_limit``       : the official episode limit was reached;
       - ``env_terminal``     : the environment declared done (e.g. ScienceWorld
                                score == -100, or an ALFWorld terminal state)
                                without a Claude-side error;
  2. and it suffered NO rate-limit / CLI-call / runtime error, and it actually
     consumed at least one Claude request.

Everything else is CENSORED and must be excluded from every rate, label,
effect estimate and conclusion -- while being PRESERVED as audit evidence with
an explicit reason. Censored records are never silently dropped and never
counted as agent failures.

Censor reasons (most specific first)
------------------------------------
``rate_limit_429``      : an HTTP 429 / "hit your limit" response from the
                          first-party Claude path terminated or degraded the run.
``cli_call_failed``     : the CLI call failed for a non-429 reason (timeout,
                          non-zero rc, unparseable JSON, other api_error).
``runtime_exception``   : the driver caught a Python exception for this task.
``zero_claude_requests``: the episode consumed no acting-model request at all,
                          so no agent behaviour was observed.
``explicit_marker``     : the rollout driver already marked the record censored.
``no_terminal_reason``  : the record neither succeeded, nor hit the step limit,
                          nor reported an environment-terminal state -- it was
                          truncated for an unexplained reason.

Prefix / checkpoint records
---------------------------
Not every record is an episode. A *prefix* record (``record_kind="prefix"``)
describes a stored trajectory state that the counterfactual matrix will fork
from. It has no terminal reason and consumes no Claude request of its own, so
the episode criteria above are meaningless for it. Its single validity
criterion is the replay-fidelity gate in ``src/replay_gate.py``, which writes
an explicit ``censored``/``censor_reason`` marker
(``replay_fidelity_failed``, ``rollback_target_mismatch``,
``replay_hash_unavailable``, ``replay_reconstruction_error``). This module
honours that marker and otherwise treats prefix records as valid, so there is
still exactly ONE place that answers "is this record usable?".
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

CENSORING_RULE_VERSION = "1.0.0"

# Substrings that identify a subscription rate-limit event in the CLI payload.
# Kept deliberately narrow: a generic "resets" or "limit" would also match
# ordinary environment text such as "step limit".
RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    '"api_error_status":429',
    '"api_error_status": 429',
    "api_error_status=429",
    "hit your limit",
    "rate_limit_error",
    "rate limit exceeded",
    "usage limit reached",
    "429 too many requests",
)

# Terminal reasons that count as a genuine end of the episode.
GENUINE_TERMINAL_REASONS = ("success", "step_limit", "env_terminal")

# Record kinds that describe a STATE rather than an episode; see the module
# docstring. Their validity is decided solely by the replay-fidelity gate.
STATE_RECORD_KINDS = ("prefix", "checkpoint")

_RESET_RE = re.compile(r"resets\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?\s*\([^)]+\))", re.I)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _flatten(value: Any, out: list[str]) -> None:
    """Collect raw leaf strings.

    Deliberately NOT json.dumps: the CLI error payload is itself embedded JSON,
    so dumping it again would escape the inner quotes and make patterns such as
    ``"api_error_status":429`` unmatchable.
    """
    if value is None:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            out.append(str(k))
            _flatten(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _flatten(v, out)
    else:
        out.append(str(value))


def _blob(record: Mapping[str, Any]) -> str:
    """Flatten the error-bearing fields of a record into one lowercase string.

    We only look at fields that can carry provider errors, never at raw model
    text: an agent that happens to type "429" into an action must not censor
    its own episode.
    """
    parts: list[str] = []
    for key in ("failure", "error", "censor_reason", "censored_reason", "cli_errors"):
        _flatten(record.get(key), parts)
    for s in record.get("steps") or []:
        if isinstance(s, Mapping):
            _flatten(s.get("call_errors"), parts)
            if s.get("call_ok") is False:
                parts.append("call_ok_false")
    _flatten(record.get("cli_event_log"), parts)
    return " ".join(parts).lower()


def has_rate_limit(record: Mapping[str, Any]) -> bool:
    """True iff a 429 / subscription-limit response is recorded anywhere."""
    if record.get("rate_limit_hit") is True:
        return True
    blob = _blob(record)
    return any(p in blob for p in RATE_LIMIT_PATTERNS)


def parse_reset_hint(text: str) -> str | None:
    """Extract the human-readable reset time from a rate-limit message."""
    if not text:
        return None
    m = _RESET_RE.search(text)
    return m.group(1).strip() if m else None


def n_claude_requests(record: Mapping[str, Any]) -> int:
    n = record.get("n_claude_requests")
    if n is None:
        steps = record.get("steps") or []
        n = sum(int(s.get("parse_retries", 0)) + 1 for s in steps if isinstance(s, Mapping))
    return int(n or 0)


def terminal_reason(record: Mapping[str, Any]) -> str:
    """Classify how the episode ended, independent of censoring."""
    if record.get("error"):
        return "runtime_exception"
    failure = record.get("failure")
    if failure:
        if has_rate_limit(record):
            return "rate_limit"
        return str((failure or {}).get("reason", "cli_call_failed")) if isinstance(
            failure, Mapping
        ) else "cli_call_failed"
    if record.get("success"):
        return "success"
    if record.get("step_limit_reached"):
        return "step_limit"
    limit = record.get("step_limit")
    n_steps = record.get("n_steps")
    if limit is not None and n_steps is not None and int(n_steps) >= int(limit):
        return "step_limit"
    if record.get("env_done") or record.get("done"):
        return "env_terminal"
    # ScienceWorld encodes unrecoverable failure as score == -100; the episode
    # is genuinely over even though the step limit was not reached.
    try:
        if float(record.get("final_score", 0.0)) <= -100.0:
            return "env_terminal"
    except (TypeError, ValueError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def censor_reason(record: Mapping[str, Any]) -> str | None:
    """Return the censoring reason, or None when the record is valid."""
    if record.get("censored") is True and record.get("censor_reason"):
        # A driver-emitted marker is authoritative, but we still normalise a
        # 429 to the canonical reason so counts are comparable across stages.
        return "rate_limit_429" if has_rate_limit(record) else str(record["censor_reason"])

    if has_rate_limit(record):
        return "rate_limit_429"
    if record.get("error"):
        return "runtime_exception"
    if record.get("failure"):
        return "cli_call_failed"
    if record.get("censored") is True:
        return "explicit_marker"
    if record.get("record_kind") in STATE_RECORD_KINDS:
        # A verified prefix/checkpoint is valid by construction: the episode
        # criteria (terminal reason, >=1 Claude request) do not apply to a
        # state record, and its own criterion -- the replay gate -- already
        # passed, otherwise an explicit marker would have been set above.
        return None
    if n_claude_requests(record) <= 0:
        return "zero_claude_requests"
    if terminal_reason(record) not in GENUINE_TERMINAL_REASONS:
        return "no_terminal_reason"
    return None


def is_censored(record: Mapping[str, Any]) -> bool:
    """True iff the record must be excluded from all scientific statistics."""
    return censor_reason(record) is not None


def is_valid(record: Mapping[str, Any]) -> bool:
    return not is_censored(record)


def partition(records: Iterable[Mapping[str, Any]]):
    """Split records into (valid, censored) preserving input order."""
    valid: list[Mapping[str, Any]] = []
    censored: list[Mapping[str, Any]] = []
    for r in records:
        (censored if is_censored(r) else valid).append(r)
    return valid, censored


def annotate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of the record with canonical censoring fields attached."""
    out = dict(record)
    reason = censor_reason(record)
    out["censored"] = reason is not None
    out["censor_reason"] = reason
    out["terminal_reason"] = terminal_reason(record)
    out["censoring_rule_version"] = CENSORING_RULE_VERSION
    if reason == "rate_limit_429":
        out["rate_limit_reset_hint"] = parse_reset_hint(_blob(record))
    return out


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit summary: valid/censored counts plus a reason histogram."""
    records = list(records)
    valid, censored = partition(records)
    reasons: dict[str, int] = {}
    for r in censored:
        k = censor_reason(r) or "unknown"
        reasons[k] = reasons.get(k, 0) + 1
    terminals: dict[str, int] = {}
    for r in valid:
        k = terminal_reason(r)
        terminals[k] = terminals.get(k, 0) + 1
    return {
        "censoring_rule_version": CENSORING_RULE_VERSION,
        "n_total": len(records),
        "n_valid": len(valid),
        "n_censored": len(censored),
        "censor_reasons": reasons,
        "valid_terminal_reasons": terminals,
        "valid_indices": [r.get("task_index") for r in valid],
        "censored_indices": [r.get("task_index") for r in censored],
    }


def load_pilot_records(path: str) -> list[dict[str, Any]]:
    """Load records from either the pilot payload dict or a bare list."""
    with open(path) as f:
        data = json.load(f)
    return list(data["records"] if isinstance(data, dict) else data)


if __name__ == "__main__":  # pragma: no cover - manual audit helper
    import sys

    for p in sys.argv[1:]:
        print(p)
        print(json.dumps(summarize(load_pilot_records(p)), indent=2))
