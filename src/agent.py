"""Stateless acting-agent loop over ALFWorld / ScienceWorld.

Why stateless: the counterfactual design forks four arms from ONE saved prefix.
If we relied on CLI session resumption, each arm would inherit hidden
server-side state we cannot inspect or reproduce. Instead we keep the full
*observable* transcript ourselves and re-send it on every step. That gives
byte-exact prefix control, so the only difference between arms is the
intervention text we append.

We deliberately persist ONLY externally observable content: environment
observations, the agent's emitted action lines, and overseer messages. No
hidden chain-of-thought is stored.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.claude_client import ClaudeCLI, CallResult, RateLimitError
from src.env_adapters import BaseAdapter, make_adapter

MAX_PARSE_RETRIES = 2  # capped: never retry forever

SYSTEM_PROMPT = (
    "You are an agent acting in a text-based simulated environment. "
    "On each turn you receive the task, the history of your previous actions and "
    "the environment's observations, and you must choose exactly ONE next action. "
    "Output contract: reply with a single line of the form\n"
    "ACTION: <the action text>\n"
    "Nothing else. No explanation, no reasoning, no markdown, no quotes. "
    "The action must be a command the environment accepts."
)

ALFWORLD_RULES = (
    "Environment: ALFWorld (household tasks). Typical commands:\n"
    "  go to <receptacle>, open <receptacle>, close <receptacle>,\n"
    "  take <object> from <receptacle>, put <object> in/on <receptacle>,\n"
    "  toggle <object>, heat <object> with <receptacle>,\n"
    "  cool <object> with <receptacle>, clean <object> with <receptacle>,\n"
    "  use <object>, examine <object>, look, inventory.\n"
    "Objects and receptacles are numbered (e.g. 'cabinet 3'). You must go to a "
    "receptacle before interacting with it, and open it if it is closed."
)

SCIENCEWORLD_RULES = (
    "Environment: ScienceWorld (elementary-science tasks in a simulated house).\n"
    "Typical commands: open/close <object>, activate/deactivate <object>,\n"
    "  go to <location>, teleport to <location>, look around, look at <object>,\n"
    "  read <object>, pick up <object>, put <object> in/on <object>,\n"
    "  move <object> to <object>, mix <container>, pour <a> into <b>,\n"
    "  focus on <object>, wait1, inventory, task.\n"
    "Note: 'focus on <object>' is a scoring action - use it on the object the "
    "task asks about, and only then."
)

ENV_RULES = {"alfworld": ALFWORLD_RULES, "scienceworld": SCIENCEWORLD_RULES}


@dataclass
class TranscriptEntry:
    kind: str  # "observation" | "action" | "overseer"
    text: str
    step: int


@dataclass
class StepRecord:
    """One acting-agent decision, plus everything needed to rebuild it later.

    The checkpoint-metadata block (``actions_after`` .. ``transcript_len_before``)
    exists so that Step 4 can fork all four intervention arms from this exact
    point WITHOUT re-deriving anything:

    ``actions_after``          the complete action list whose replay reproduces
                               ``state_hash`` -- i.e. the checkpoint itself;
    ``prev_state_hash``        the state BEFORE this action, which is the target
                               arm A3 (ROLLBACK+REPLAN) must restore, and which
                               the replay-fidelity gate verifies;
    ``state_changed``          False when the action was an environment no-op,
                               in which case A3 degenerates to A2-with-other-text
                               and must be reported as such;
    ``reversible``             structural A3 eligibility (>=1 action to undo, a
                               known rollback-target hash, not already terminal).
                               Actual usability still requires the replay gate to
                               pass at fork time -- see src/replay_gate.py;
    ``transcript_len_before``  index into the episode-level observable transcript
                               at which this decision was taken, so the dialogue
                               prefix is reconstructible exactly (transcript is
                               append-only, so transcript[:k] is that prefix).
    """

    step: int
    prompt: str
    raw_reply: str
    parsed_action: str
    invalid_output: bool
    parse_retries: int
    observation: str
    reward: float
    done: bool
    state_hash: str
    obs_hash: str
    # -- checkpoint metadata ------------------------------------------------
    actions_after: list[str]
    prev_state_hash: str
    state_changed: bool
    reversible: bool
    terminal_after: bool
    transcript_len_before: int
    # -----------------------------------------------------------------------
    usage: dict[str, int]
    duration_ms: int
    duration_api_ms: int
    num_turns: int
    session_id: str
    call_ok: bool
    call_attempts: int
    call_errors: list[str]
    wall_s: float
    admissible_n: int
    action_admissible: bool | None


ACTION_RE = re.compile(r"ACTION\s*:\s*(.+)", re.IGNORECASE)


def parse_action(reply: str) -> tuple[str, bool]:
    """Return (action, invalid_flag) under the strict output contract."""
    if not reply or not reply.strip():
        return "", True
    m = ACTION_RE.search(reply)
    if m:
        act = m.group(1).strip().strip("`\"'").strip()
        act = act.split("\n")[0].strip()
        return (act, False) if act else ("", True)
    # Contract violated; salvage a single short line if the model emitted one.
    lines = [ln.strip() for ln in reply.strip().split("\n") if ln.strip()]
    if len(lines) == 1 and len(lines[0]) <= 80 and not lines[0].endswith((".", "?", ":")):
        return lines[0].strip("`\"'"), True
    return "", True


def build_prompt(
    env_name: str,
    task_description: str,
    transcript: list[TranscriptEntry],
    admissible: list[str] | None,
    step: int,
    max_steps: int,
) -> str:
    parts = [ENV_RULES[env_name], ""]
    parts.append(f"TASK: {task_description}")
    parts.append(f"You have used {step} of {max_steps} allowed steps.")
    parts.append("")
    parts.append("--- HISTORY ---")
    for e in transcript:
        if e.kind == "observation":
            parts.append(f"[OBSERVATION] {e.text}")
        elif e.kind == "action":
            parts.append(f"[YOUR ACTION] {e.text}")
        elif e.kind == "overseer":
            parts.append(f"[OVERSEER] {e.text}")
    parts.append("--- END HISTORY ---")
    parts.append("")
    if admissible:
        shown = admissible[:60]
        parts.append("Valid actions right now include:")
        parts.append("; ".join(shown))
        if len(admissible) > len(shown):
            parts.append(f"(...and {len(admissible) - len(shown)} more)")
        parts.append("")
    parts.append("Reply with exactly one line: ACTION: <your next action>")
    return "\n".join(parts)


def run_episode(
    env_name: str,
    task_spec: dict[str, Any],
    cli: ClaudeCLI,
    max_steps: int | None = None,
    interventions: dict[int, str] | None = None,
    restore_from=None,
    prior_transcript: list[TranscriptEntry] | None = None,
    verbose: bool = False,
    deadline: float | None = None,
    expected_state_hash: str | None = None,
) -> dict[str, Any]:
    """Run one episode; optionally start from a restored checkpoint.

    interventions: {step_index: overseer_text} injected BEFORE that step's call.
    restore_from : Checkpoint (or dict) to fork from; when given, the caller must
                   also pass `prior_transcript` so the dialogue prefix matches
                   the environment prefix exactly.
    deadline     : absolute time.time() after which we stop taking new steps and
                   return a *suspended* episode. Because prompting is stateless
                   and the environment is reconstructed by verified action
                   replay, a suspended episode can be resumed later with a
                   byte-identical observable prefix -- so suspension costs
                   nothing scientifically and never burns the task.
    """
    t0 = time.time()
    interventions = interventions or {}
    if restore_from is not None:
        from src.env_adapters import ADAPTERS

        env: BaseAdapter = ADAPTERS[env_name].from_checkpoint(restore_from)
        transcript = list(prior_transcript or [])
        first_obs = env.last_obs
        restore_hash_ok = (
            None if expected_state_hash is None
            else env.state_hash() == expected_state_hash
        )
    else:
        env = make_adapter(env_name, task_spec)
        first_obs = env.reset()
        transcript = [TranscriptEntry("observation", first_obs, 0)]
        restore_hash_ok = None

    limit = max_steps or env.max_steps
    steps: list[StepRecord] = []
    # Hash of the state the episode starts from (fresh reset, or the restored
    # checkpoint). Kept so that step 0 also has a well-defined rollback target.
    prev_state_hash = env.state_hash()
    start_step = env.steps
    injected: list[int] = []
    failure = None
    suspended = False
    rate_limited = False
    reset_hint = None

    while not env.done and env.steps < limit:
        step = env.steps
        if deadline is not None and time.time() >= deadline:
            suspended = True
            break
        if step in interventions:
            transcript.append(TranscriptEntry("overseer", interventions[step], step))
            injected.append(step)

        admissible = list(getattr(env, "admissible", []) or [])
        transcript_len_before = len(transcript)
        prompt = build_prompt(
            env_name, env.task_description, transcript, admissible, step, limit
        )

        action, invalid, retries, res = "", True, 0, None
        try:
            for attempt in range(MAX_PARSE_RETRIES + 1):
                p = prompt if attempt == 0 else (
                    prompt
                    + "\n\nYour previous reply did not follow the contract. "
                    "Reply with EXACTLY one line: ACTION: <action>"
                )
                res = cli.call(p, tag=f"{env_name}:{step}")
                if not res.ok:
                    break
                action, invalid = parse_action(res.text)
                retries = attempt
                if not invalid and action:
                    break
        except RateLimitError as exc:
            # Subscription window exhausted. Stop here and keep the prefix:
            # the episode is suspended, not failed.
            rate_limited = True
            suspended = True
            reset_hint = exc.reset_hint
            failure = {
                "reason": "rate_limit_429",
                "step": step,
                "reset_hint": exc.reset_hint,
                "detail": exc.detail[:800],
            }
            break

        if res is None or not res.ok:
            failure = {
                "reason": "cli_call_failed",
                "step": step,
                "errors": res.errors if res else ["no result"],
            }
            break

        if invalid or not action:
            # Do not crash and do not loop forever: send a no-op the env
            # understands, and record the contract violation.
            action = "look" if env_name == "alfworld" else "look around"

        obs, reward, done, info = env.step(action)
        transcript.append(TranscriptEntry("action", action, step))
        transcript.append(TranscriptEntry("observation", obs, step + 1))

        cur_state_hash = env.state_hash()
        steps.append(
            StepRecord(
                step=step,
                prompt=prompt,
                raw_reply=res.text,
                parsed_action=action,
                invalid_output=bool(invalid),
                parse_retries=retries,
                observation=obs,
                reward=float(reward),
                done=bool(done),
                state_hash=cur_state_hash,
                obs_hash=env.obs_hash(),
                actions_after=list(env.actions),
                prev_state_hash=prev_state_hash,
                state_changed=cur_state_hash != prev_state_hash,
                # A3 needs at least one action to undo and a known target
                # state; a terminal state cannot be forked forward at all.
                reversible=bool(len(env.actions) >= 1 and prev_state_hash
                                and not done),
                terminal_after=bool(done),
                transcript_len_before=transcript_len_before,
                usage=res.usage_vec(),
                duration_ms=res.duration_ms,
                duration_api_ms=res.duration_api_ms,
                num_turns=res.num_turns,
                session_id=res.session_id,
                call_ok=res.ok,
                call_attempts=res.attempts,
                call_errors=res.errors,
                wall_s=res.wall_s,
                admissible_n=len(admissible),
                action_admissible=(action in admissible) if admissible else None,
            )
        )
        prev_state_hash = cur_state_hash
        if verbose:
            print(f"  step {step}: {action!r} -> {obs[:70]!r} score={reward}")

    success = bool(env.won)
    result = {
        "env": env_name,
        "task_spec": task_spec,
        "success": success,
        "final_score": float(env.score),
        "n_steps": env.steps,
        "step_limit": limit,
        "step_limit_reached": env.steps >= limit and not success,
        "actions": list(env.actions),
        "interventions_injected": injected,
        "failure": failure,
        "suspended": suspended,
        "rate_limited": rate_limited,
        "rate_limit_reset_hint": reset_hint,
        "env_done": bool(env.done),
        "segment_start_step": start_step,
        "restore_hash_ok": restore_hash_ok,
        "transcript": [e.__dict__ for e in transcript],
        "wall_s": round(time.time() - t0, 2),
        "n_invalid_outputs": sum(s.invalid_output for s in steps),
        "n_inadmissible_actions": sum(
            1 for s in steps if s.action_admissible is False
        ),
        "totals": {
            k: sum(s.usage.get(k, 0) for s in steps)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        },
        "n_claude_requests": sum(s.parse_retries + 1 for s in steps),
        "total_api_ms": sum(s.duration_api_ms for s in steps),
        "steps": [s.__dict__ for s in steps],
        "final_state_hash": env.state_hash(),
        "transcript_len": len(transcript),
    }
    env.close()
    return result
