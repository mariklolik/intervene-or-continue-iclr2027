"""Rate-limit-resilient, resumable rollout driver.

Why this exists
---------------
The first Step-2 pilot lost 12 of 20 tasks to a single Claude Max 5-hour
subscription limit: the old loop retried the 429 three times per step, gave up,
recorded a zero-request pseudo-failure, and then marched through the *remaining*
tasks doing exactly the same thing. Every one of those tasks was burned even
though no agent behaviour was ever observed.

This driver fixes that with four properties:

1. **Stop on first 429.** `ClaudeCLI(stop_on_rate_limit=True)` raises
   `RateLimitError` on the first rate-limited response. The driver stops the
   whole invocation immediately and records the parsed reset hint. It never
   consumes a further task.
2. **Step-level suspend/resume.** A task interrupted by a 429 or by the
   wall-clock budget is *suspended*, not failed: its action list, observable
   transcript and resource carry-over are persisted. The next invocation
   reconstructs the environment by verified action replay (state-hash checked)
   and continues from the same step with a byte-identical observable prefix.
   Prompting is stateless (one CLI process per step, full transcript re-sent),
   so resumption is exact and costs no extra Claude requests.
3. **Atomic per-task persistence.** Completed tasks are appended to a JSONL and
   fsynced before the next task starts. Partial state is written via
   write-temp + fsync + `os.replace`.
4. **Wall-clock budget.** `--max-wall-seconds N` makes the process exit cleanly
   inside a foreground shell timeout, printing one machine-readable status line.

Task identity is a stable content key (env + task index + hash of the task
spec), never a list position, so resume is correct even if sampling order or
the task list length changes.

Usage
-----
  python -m src.rollout_driver --env alfworld \
      --specs-from data/raw/pilot_alfworld_full.json \
      --out data/raw/pilot_v2_alfworld --resume --max-wall-seconds 540
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from src.agent import SYSTEM_PROMPT, TranscriptEntry, run_episode
from src.censoring import annotate, is_censored
from src.claude_client import ClaudeCLI, RateLimitError

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_key(env: str, task_index: int, task_spec: dict[str, Any]) -> str:
    """Stable content-addressed identity for a task unit."""
    payload = json.dumps(task_spec, sort_keys=True, default=str)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{env}:{int(task_index)}:{h}"


@dataclass
class TaskUnit:
    env: str
    task_index: int
    task_spec: dict[str, Any]

    @property
    def key(self) -> str:
        return task_key(self.env, self.task_index, self.task_spec)


# ---------------------------------------------------------------------------
# durable IO
# ---------------------------------------------------------------------------
def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    """Append one record and fsync, so a crash cannot lose completed work."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
    return out


# ---------------------------------------------------------------------------
# resource carry-over across suspended segments
# ---------------------------------------------------------------------------
def empty_carry() -> dict[str, Any]:
    return {
        "n_claude_requests": 0,
        "total_api_ms": 0,
        "wall_s": 0.0,
        "n_invalid_outputs": 0,
        "n_inadmissible_actions": 0,
        "n_segments": 0,
        "totals": {k: 0 for k in USAGE_KEYS},
        "steps": [],
    }


def merge_carry(carry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(carry, default=str))
    out["n_claude_requests"] += int(result.get("n_claude_requests", 0) or 0)
    out["total_api_ms"] += int(result.get("total_api_ms", 0) or 0)
    out["wall_s"] = round(out["wall_s"] + float(result.get("wall_s", 0.0) or 0.0), 2)
    out["n_invalid_outputs"] += int(result.get("n_invalid_outputs", 0) or 0)
    out["n_inadmissible_actions"] += int(result.get("n_inadmissible_actions", 0) or 0)
    out["n_segments"] += 1
    for k in USAGE_KEYS:
        out["totals"][k] += int((result.get("totals") or {}).get(k, 0) or 0)
    out["steps"].extend(result.get("steps") or [])
    return out


def finalize_record(unit: TaskUnit, result: dict[str, Any],
                    carry: dict[str, Any],
                    keep_transcript: bool = False) -> dict[str, Any]:
    """Merge the last segment into the carry and emit the canonical record.

    `keep_transcript` is on for baseline rollouts: Step 4 forks four arms from
    a stored prefix and must reproduce the observable dialogue prefix
    byte-exactly, so the transcript is evidence, not debug output. It stays off
    elsewhere to keep records small.
    """
    merged = merge_carry(carry, result)
    rec = dict(result)
    transcript = rec.pop("transcript", None)
    if keep_transcript:
        prior = (carry.get("transcript") or []) if carry else []
        # A resumed episode replays its prefix, so the final segment's
        # transcript already contains the prior one; prefer the longer.
        rec["transcript"] = (
            transcript if len(transcript or []) >= len(prior) else prior
        )
    rec["task_key"] = unit.key
    rec["task_index"] = unit.task_index
    rec["env"] = unit.env
    rec["task_spec"] = unit.task_spec
    rec["n_claude_requests"] = merged["n_claude_requests"]
    rec["total_api_ms"] = merged["total_api_ms"]
    rec["wall_s"] = merged["wall_s"]
    rec["n_invalid_outputs"] = merged["n_invalid_outputs"]
    rec["n_inadmissible_actions"] = merged["n_inadmissible_actions"]
    rec["totals"] = merged["totals"]
    rec["steps"] = merged["steps"]
    rec["n_segments"] = merged["n_segments"]
    rec["resumed"] = merged["n_segments"] > 1
    rec["ts"] = _now()
    return annotate(rec)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
@dataclass
class DriverStatus:
    completed_this_invocation: int = 0
    valid_this_invocation: int = 0
    censored_this_invocation: int = 0
    suspended_this_invocation: int = 0
    discarded_rate_limit_partials: int = 0
    requests_used: int = 0
    wall_s: float = 0.0
    rate_limit_hit: bool = False
    rate_limit_reset_hint: str | None = None
    budget_exhausted: bool = False
    work_remaining: bool = True
    n_remaining: int = 0
    total_valid_on_disk: int = 0
    total_records_on_disk: int = 0
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"driver_status": True, **self.__dict__}, default=str)


class RolloutDriver:
    def __init__(
        self,
        out_prefix: str,
        cli_factory: Callable[[], Any] | None = None,
        episode_fn: Callable[..., dict[str, Any]] | None = None,
        verbose: bool = True,
        keep_transcript: bool = False,
        fork_source: Callable[[TaskUnit], dict[str, Any] | None] | None = None,
    ):
        # `fork_source` turns the driver into a counterfactual-arm runner
        # without duplicating any of its resume/censor/fsync logic. For a unit
        # with no partial on disk it supplies the stored prefix to start FROM:
        #   {"checkpoint": {...}, "transcript": [...], "expected_state_hash": str}
        # Forking really is a resume from a verified checkpoint, so this reuses
        # exactly the mechanism the baseline stage already proved. Returning
        # None means "this unit is not forkable", which the caller must have
        # decided via the replay gate; the driver then records an explicit
        # censored row rather than silently starting a FRESH episode, which
        # would fabricate a trajectory that was never forked from x_t.
        self.fork_source = fork_source
        self.keep_transcript = keep_transcript
        self.out_prefix = out_prefix
        self.records_path = f"{out_prefix}_records.jsonl"
        self.partial_path = f"{out_prefix}_partial.json"
        self.events_path = f"{out_prefix}_events.jsonl"
        self.cli_factory = cli_factory or (lambda: ClaudeCLI(
            system_prompt=SYSTEM_PROMPT, stop_on_rate_limit=True, max_attempts=2
        ))
        self.episode_fn = episode_fn or run_episode
        self.verbose = verbose

    # -- state ------------------------------------------------------------
    def load_records(self) -> dict[str, dict[str, Any]]:
        """key -> latest record on disk."""
        out: dict[str, dict[str, Any]] = {}
        for r in read_jsonl(self.records_path):
            k = r.get("task_key")
            if k:
                out[k] = r
        return out

    def load_partials(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self.partial_path):
            return {}
        try:
            with open(self.partial_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _discard_rate_limit_partial(
        self,
        unit: TaskUnit,
        p: dict[str, Any],
        partials: dict[str, dict[str, Any]],
    ) -> None:
        """Retire a 429-suspended attempt as censored audit evidence.

        The record is written to the normal records stream so that it is
        preserved forever and so that request accounting -- which sums
        ``n_claude_requests`` over ALL records on disk -- still reflects the
        requests this attempt really spent against the subscription. Because
        the record is censored, ``pending()`` keeps the cell in the work queue
        and every statistic excludes it.
        """
        cp = p.get("checkpoint") or {}
        spent = int((p.get("carry") or {}).get("n_claude_requests") or 0)
        rec = annotate({
            "env": unit.env,
            "task_index": unit.task_index,
            "task_key": unit.key,
            "task_spec": unit.task_spec,
            "success": False,
            "censored": True,
            "censor_reason": "suspended_rate_limit",
            "n_claude_requests": spent,
            "discarded_partial": True,
            "n_steps": int(cp.get("step_index") or 0),
            "suspended_at_step": int(cp.get("step_index") or 0),
            "suspended_state_hash": cp.get("state_hash") or "",
            "suspended_updated": p.get("updated"),
            "ts": _now(),
        })
        append_jsonl(self.records_path, rec)
        partials.pop(unit.key, None)
        write_json_atomic(self.partial_path, partials)
        self.event("suspended_rate_limit_discarded", task_key=unit.key,
                   suspended_at_step=rec["suspended_at_step"],
                   n_claude_requests=spent)

    def event(self, kind: str, **kw: Any) -> None:
        ev = {"ts": _now(), "kind": kind, **kw}
        append_jsonl(self.events_path, ev)
        append_jsonl("logs/progress.jsonl", {"stage": "rollout", **ev})

    def pending(self, units: Iterable[TaskUnit], resume: bool = True) -> list[TaskUnit]:
        """Units still needing work: never completed, or completed but censored."""
        done = self.load_records() if resume else {}
        out = []
        for u in units:
            rec = done.get(u.key)
            if rec is not None and not is_censored(rec):
                continue
            out.append(u)
        return out

    # -- main loop --------------------------------------------------------
    def run(
        self,
        units: list[TaskUnit],
        max_wall_seconds: float = 540.0,
        resume: bool = True,
        min_task_seconds: float = 20.0,
    ) -> DriverStatus:
        t0 = time.time()
        hard_deadline = t0 + max_wall_seconds
        st = DriverStatus()
        todo = self.pending(units, resume=resume)
        partials = self.load_partials()
        self.event("invocation_start", out=self.out_prefix, n_units=len(units),
                   n_pending=len(todo), max_wall_seconds=max_wall_seconds)

        cli = self.cli_factory()
        processed_keys: set[str] = set()
        # Units started from `fork_source` rather than resumed from a partial.
        # A state mismatch means different things in the two cases and must not
        # be reported under one label.
        forked_keys: set[str] = set()

        for unit in todo:
            remaining = hard_deadline - time.time()
            if remaining < min_task_seconds:
                st.budget_exhausted = True
                self.event("budget_stop", remaining_s=round(remaining, 1),
                           n_left=len(todo) - len(processed_keys))
                break

            processed_keys.add(unit.key)
            p = partials.get(unit.key)
            if p is not None and p.get("reason") == "rate_limit":
                # A suspension caused by a subscription 429 is NOT a valid
                # outcome and must never be stitched back together: resuming it
                # would splice the two halves of one episode across a
                # rate-limit boundary, on top of requests already spent in the
                # previous window. Discard it, keep it as audit evidence, and
                # let the unit be retried cleanly from its prefix fork under
                # the usual replay-fidelity gate. Suspensions caused by the
                # wall-clock budget stay resumable: those are pure chunking
                # inside one window and carry no cross-window contamination.
                self._discard_rate_limit_partial(unit, p, partials)
                st.discarded_rate_limit_partials += 1
                st.censored_this_invocation += 1
                p = None
            carry = p["carry"] if p else empty_carry()
            if p:
                # Belt-and-braces for `keep_transcript`: a resumed segment
                # normally re-emits the whole dialogue (prior_transcript is fed
                # back in), but if it dies before producing one, the suspended
                # transcript is still the best evidence we have of the prefix.
                carry = {**carry, "transcript": p.get("transcript") or []}
            restore_from = p["checkpoint"] if p else None
            prior = (
                [TranscriptEntry(**e) for e in p["transcript"]] if p else None
            )
            expected_hash = p["checkpoint"]["state_hash"] if p else None
            if p is None and self.fork_source is not None:
                fs = self.fork_source(unit)
                if fs is None:
                    rec = annotate({
                        "env": unit.env, "task_index": unit.task_index,
                        "task_key": unit.key, "task_spec": unit.task_spec,
                        "success": False, "censored": True,
                        "censor_reason": "prefix_not_forkable",
                        "n_claude_requests": 0, "ts": _now(),
                    })
                    append_jsonl(self.records_path, rec)
                    st.completed_this_invocation += 1
                    st.censored_this_invocation += 1
                    self.event("prefix_not_forkable", task_key=unit.key)
                    continue
                restore_from = fs["checkpoint"]
                prior = [TranscriptEntry(**e) for e in fs.get("transcript") or []]
                expected_hash = fs.get("expected_state_hash")
                forked_keys.add(unit.key)
            if self.verbose:
                mark = f" (resume @step {p['checkpoint']['step_index']})" if p else ""
                print(f"[{unit.env}] task {unit.task_index}{mark} "
                      f"{unit.task_spec.get('task_type', '')}", flush=True)

            try:
                res = self.episode_fn(
                    unit.env,
                    unit.task_spec,
                    cli,
                    restore_from=restore_from,
                    prior_transcript=prior,
                    deadline=hard_deadline,
                    expected_state_hash=expected_hash,
                    verbose=self.verbose,
                )
            except RateLimitError as exc:
                # Raised outside the episode loop (e.g. during env construction).
                st.rate_limit_hit = True
                st.rate_limit_reset_hint = exc.reset_hint
                self.event("rate_limit", task_key=unit.key, reset_hint=exc.reset_hint)
                break
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                rec = annotate({
                    "env": unit.env, "task_index": unit.task_index,
                    "task_key": unit.key, "task_spec": unit.task_spec,
                    "success": False, "error": f"{type(exc).__name__}: {exc}",
                    "n_claude_requests": carry["n_claude_requests"],
                    "ts": _now(),
                })
                append_jsonl(self.records_path, rec)
                st.completed_this_invocation += 1
                st.censored_this_invocation += 1
                st.errors.append(f"{unit.key}: {type(exc).__name__}: {exc}")
                self.event("task_exception", task_key=unit.key, error=str(exc)[:400])
                continue

            if res.get("restore_hash_ok") is False:
                # Replay did not reproduce the stored state -> the resumed
                # prefix is not the prefix we promised. Do not pretend it is.
                reason = ("fork_state_mismatch" if unit.key in forked_keys
                          else "resume_state_mismatch")
                rec = finalize_record(unit, res, carry, self.keep_transcript)
                rec["censored"] = True
                rec["censor_reason"] = reason
                append_jsonl(self.records_path, rec)
                partials.pop(unit.key, None)
                write_json_atomic(self.partial_path, partials)
                st.completed_this_invocation += 1
                st.censored_this_invocation += 1
                self.event(reason, task_key=unit.key)
                continue

            if res.get("suspended"):
                merged = merge_carry(carry, res)
                partials[unit.key] = {
                    "key": unit.key,
                    "env": unit.env,
                    "task_index": unit.task_index,
                    "task_spec": unit.task_spec,
                    "checkpoint": {
                        "env_name": unit.env,
                        "task_spec": unit.task_spec,
                        "actions": list(res.get("actions") or []),
                        "step_index": int(res.get("n_steps") or 0),
                        "state_hash": res.get("final_state_hash", ""),
                        "mechanism": "action_replay",
                    },
                    "transcript": res.get("transcript") or [],
                    "carry": merged,
                    "updated": _now(),
                    "reason": "rate_limit" if res.get("rate_limited") else "wall_budget",
                }
                write_json_atomic(self.partial_path, partials)
                st.suspended_this_invocation += 1
                if res.get("rate_limited"):
                    st.rate_limit_hit = True
                    st.rate_limit_reset_hint = res.get("rate_limit_reset_hint")
                    self.event("rate_limit", task_key=unit.key,
                               step=int(res.get("n_steps") or 0),
                               reset_hint=st.rate_limit_reset_hint)
                    break
                self.event("suspended_budget", task_key=unit.key,
                           step=int(res.get("n_steps") or 0))
                st.budget_exhausted = True
                break

            rec = finalize_record(unit, res, carry, self.keep_transcript)
            append_jsonl(self.records_path, rec)
            partials.pop(unit.key, None)
            write_json_atomic(self.partial_path, partials)
            st.completed_this_invocation += 1
            if rec.get("censored"):
                st.censored_this_invocation += 1
            else:
                st.valid_this_invocation += 1
            self.event("task_done", task_key=unit.key, success=rec.get("success"),
                       n_steps=rec.get("n_steps"), censored=rec.get("censored"),
                       censor_reason=rec.get("censor_reason"),
                       n_claude_requests=rec.get("n_claude_requests"))
            if self.verbose:
                print(f"  -> success={rec.get('success')} steps={rec.get('n_steps')} "
                      f"reqs={rec.get('n_claude_requests')} "
                      f"censored={rec.get('censored')}", flush=True)

        st.requests_used = int(getattr(cli, "n_requests", 0) or 0)
        st.wall_s = round(time.time() - t0, 2)
        remaining_units = self.pending(units, resume=True)
        st.n_remaining = len(remaining_units)
        st.work_remaining = bool(remaining_units)
        on_disk = self.load_records()
        st.total_records_on_disk = len(on_disk)
        st.total_valid_on_disk = sum(1 for r in on_disk.values() if not is_censored(r))
        self.event("invocation_end", **{k: v for k, v in st.__dict__.items()
                                        if k != "errors"})
        return st


# ---------------------------------------------------------------------------
# task-spec sources
# ---------------------------------------------------------------------------
def units_from_pilot_json(path: str, env: str,
                          only_indices: list[int] | None = None) -> list[TaskUnit]:
    """Reuse the EXACT task specs already sampled and recorded in a pilot file.

    This guarantees the re-run touches the same task instances as the censored
    originals -- stronger than re-deriving them from the sampler, because it is
    independent of any future change to sampling code.
    """
    with open(path) as f:
        data = json.load(f)
    records = data["records"] if isinstance(data, dict) else data
    units = []
    for r in records:
        idx = int(r["task_index"])
        if only_indices is not None and idx not in only_indices:
            continue
        units.append(TaskUnit(env=env, task_index=idx, task_spec=r["task_spec"]))
    return units


def units_from_manifest(path: str, env: str,
                        only_indices: list[int] | None = None) -> list[TaskUnit]:
    """Read the FROZEN task sample (configs/task_splits.json) in processing order.

    The manifest fixes `processing_order` with seed 42 before any outcome is
    observed, so a capacity-truncated run is a prefix of a pre-registered
    sequence rather than an outcome-dependent subset. `task_index` is kept equal
    to the sampler index so that `task_key` matches the pilot records already on
    disk and `--resume` recognises them as done.
    """
    with open(path) as f:
        data = json.load(f)
    rows = [t for t in data["tasks"] if t["env"] == env]
    rows.sort(key=lambda t: int(t["processing_order"]))
    units = []
    for t in rows:
        idx = int(t["task_index"])
        if only_indices is not None and idx not in only_indices:
            continue
        units.append(TaskUnit(env=env, task_index=idx, task_spec=t["task_spec"]))
    return units


def units_from_sampler(env: str, n: int, seed: int = 42) -> list[TaskUnit]:
    from src.task_sampling import alfworld_tasks, scienceworld_tasks

    specs = alfworld_tasks(n, seed=seed) if env == "alfworld" else scienceworld_tasks(
        n, seed=seed
    )
    return [TaskUnit(env=env, task_index=i, task_spec=s) for i, s in enumerate(specs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["alfworld", "scienceworld"])
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--specs-from", default=None,
                    help="pilot json whose task_specs are reused verbatim")
    ap.add_argument("--manifest", default=None,
                    help="frozen task_splits.json; runs in processing_order")
    ap.add_argument("--keep-transcript", action="store_true",
                    help="persist the observable transcript in each record "
                         "(required for checkpoint reconstruction)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-indices", default=None,
                    help="comma-separated task indices to run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-wall-seconds", type=float, default=540.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    only = (
        [int(x) for x in args.only_indices.split(",") if x.strip() != ""]
        if args.only_indices
        else None
    )
    if args.manifest:
        units = units_from_manifest(args.manifest, args.env, only)
    elif args.specs_from:
        units = units_from_pilot_json(args.specs_from, args.env, only)
    else:
        units = [u for u in units_from_sampler(args.env, args.n, args.seed)
                 if only is None or u.task_index in only]
    drv = RolloutDriver(args.out, verbose=not args.quiet,
                        keep_transcript=args.keep_transcript)
    st = drv.run(units, max_wall_seconds=args.max_wall_seconds, resume=args.resume)
    print(st.to_json(), flush=True)


if __name__ == "__main__":
    main()
